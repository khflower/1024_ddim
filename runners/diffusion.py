import os
import re
import sys
import json
import logging
import time
import glob
import shutil
import subprocess

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        self._mem_train_vectors = None
        self._warned_symlink_ckpt = False

    def _resume_ckpt_path(self):
        latest = os.path.join(self.args.log_path, "ckpt_latest.pth")
        default = os.path.join(self.args.log_path, "ckpt.pth")
        if os.path.exists(latest):
            return latest
        return default

    def _save_ckpts(self, states, step):
        step_path = os.path.join(self.args.log_path, f"ckpt_{step}.pth")
        latest_path = os.path.join(self.args.log_path, "ckpt_latest.pth")
        default_path = os.path.join(self.args.log_path, "ckpt.pth")

        torch.save(states, step_path)
        torch.save(states, latest_path)

        # Keep legacy behavior for non-symlinked ckpt.pth.
        # If ckpt.pth is a symlink to a base pretrained checkpoint,
        # avoid overwriting that source file.
        if os.path.islink(default_path):
            if not self._warned_symlink_ckpt:
                logging.info(
                    f"Detected symlinked {default_path}; preserving symlink target. "
                    f"Use {latest_path} for latest resume."
                )
                self._warned_symlink_ckpt = True
            return
        torch.save(states, default_path)

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            resume_path = self._resume_ckpt_path()
            logging.info(f"Resuming from checkpoint: {resume_path}")
            states = torch.load(resume_path)
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    self._save_ckpts(states, step)

                if self._should_eval(step):
                    fid, mem = self.evaluate_metrics(
                        model=model,
                        ema_helper=ema_helper,
                        step=step,
                    )
                    tb_logger.add_scalar("eval/fid", fid, global_step=step)
                    tb_logger.add_scalar("eval/mem_ratio", mem, global_step=step)
                    logging.info(f"step: {step}, fid: {fid}, mem_ratio: {mem}")

                data_start = time.time()
                if step >= self.config.training.n_iters:
                    return

    def _should_eval(self, step):
        eval_cfg = getattr(self.config, "eval", None)
        if eval_cfg is None or not getattr(eval_cfg, "enable", False):
            return False
        freq = int(getattr(eval_cfg, "freq", 0))
        return freq > 0 and step % freq == 0

    def _celeba_eval_transform(self):
        cx = 89
        cy = 121
        x1 = cy - 64
        y1 = cx - 64
        return transforms.Compose(
            [
                transforms.Lambda(lambda img: TF.crop(img, x1, y1, 128, 128)),
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor(),
            ]
        )

    def _load_mem_train_vectors(self):
        if self._mem_train_vectors is not None:
            return self._mem_train_vectors
        if not getattr(self.config.data, "use_img_list_subset", False):
            raise ValueError("MEM eval expects data.use_img_list_subset=true")

        img_dir = self.config.data.subset_img_dir
        list_path = self.config.data.subset_img_list
        with open(list_path, "r") as f:
            files = json.load(f)
        transform = self._celeba_eval_transform()
        tensors = []
        for name in files:
            p = os.path.join(img_dir, name)
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        x = torch.stack(tensors, dim=0).to(self.device)
        x = data_transform(self.config, x).view(x.shape[0], -1)
        self._mem_train_vectors = x
        return self._mem_train_vectors

    def _compute_mem_ratio(self, gen_samples):
        eval_cfg = self.config.eval
        gap_threshold = float(getattr(eval_cfg, "gap_threshold", 0.3333))
        mem_batch = int(getattr(eval_cfg, "mem_batch_size", 128))
        train_vecs = self._load_mem_train_vectors()
        gen_vecs = gen_samples.view(gen_samples.shape[0], -1).to(self.device)
        ratios = []
        with torch.no_grad():
            for i in range(0, gen_vecs.shape[0], mem_batch):
                g = gen_vecs[i : i + mem_batch]
                dist = torch.cdist(g, train_vecs, p=2)
                knn2 = torch.topk(dist, k=2, dim=1, largest=False).values
                ratio = knn2[:, 0] / (knn2[:, 1] + 1e-12)
                ratios.append(ratio)
        ratios = torch.cat(ratios, dim=0)
        return (ratios < gap_threshold).float().mean().item()

    def _compute_fid(self, real_path, gen_path):
        cmd = [sys.executable, "-m", "pytorch_fid", real_path, gen_path, "--device", str(self.device)]
        out = subprocess.check_output(cmd, text=True)
        # Expected output contains "FID: <value>"
        m = re.search(r"FID:\s*([0-9eE+.\-]+)", out)
        if m is None:
            raise RuntimeError(f"Failed to parse FID output: {out}")
        return float(m.group(1))

    def evaluate_metrics(self, model, ema_helper, step):
        eval_cfg = self.config.eval
        n_samples = int(getattr(eval_cfg, "n_samples", 1024))
        batch_size = int(getattr(eval_cfg, "batch_size", self.config.sampling.batch_size))
        use_ema = bool(getattr(eval_cfg, "use_ema_for_eval", True))
        keep_samples = bool(getattr(eval_cfg, "keep_samples", False))

        gen_dir = os.path.join(self.args.exp, "eval_samples", self.args.doc, f"step_{step}")
        os.makedirs(gen_dir, exist_ok=True)

        eval_model = model
        if use_ema and ema_helper is not None:
            eval_model = ema_helper.ema_copy(model)
        eval_model.eval()

        gen_model_space = []
        done = 0
        with torch.no_grad():
            while done < n_samples:
                n = min(batch_size, n_samples - done)
                x = torch.randn(
                    n,
                    self.config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size,
                    device=self.device,
                )
                s = self.sample_image(x, eval_model)
                gen_model_space.append(s.detach().clone())
                s_img = inverse_data_transform(self.config, s)
                for i in range(n):
                    tvu.save_image(s_img[i], os.path.join(gen_dir, f"{done + i:06d}.png"))
                done += n

        gen_model_space = torch.cat(gen_model_space, dim=0)
        real_path = getattr(eval_cfg, "real_images_dir", self.config.data.subset_img_dir)

        fid = float("nan")
        mem = float("nan")
        try:
            fid = self._compute_fid(real_path, gen_dir)
        except Exception as e:
            logging.warning(f"FID evaluation failed at step {step}: {e}")

        try:
            mem = self._compute_mem_ratio(gen_model_space)
        except Exception as e:
            logging.warning(f"MEM evaluation failed at step {step}: {e}")

        if not keep_samples:
            shutil.rmtree(gen_dir, ignore_errors=True)
        return fid, mem

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                ckpt_path = self._resume_ckpt_path()
                states = torch.load(
                    ckpt_path,
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
