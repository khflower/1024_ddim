# DDIM 1024-shot Memorization Runbook (KR/EN)

## 1) 목적 (Goal / Scope)
- Official `ddim` codebase에서 CelebA 64x64 pretrained checkpoint를 기준으로 `resume training`.
- 학습 데이터는 `/data/CelebA/img_list.json` 순서 기준 `first 1024`만 사용.
- 학습 중간에 `FID` + `mem_ratio`를 주기적으로 기록하여 memorization 동향 추적.

실험 범위:
- Dataset: CelebA
- Resolution: 64x64
- Objective/architecture는 official `celeba.yml` 계열 유지
- Data cardinality만 `1024-shot`으로 제한

---

## 2) 실행 환경 (Environment)

### Base image / runtime
- Docker image (initial): `pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel`
- Python: `3.8.5`
- GPU: `A100 40GB x4`

### 주요 패키지 버전 (실측)
- `torch==1.7.1`
- `torchvision==0.8.2`
- `numpy==1.19.2`
- `pyyaml==5.3.1`
- `pillow==8.1.0`
- `tqdm==4.51.0`
- `tensorboard==2.14.0`
- `pandas==1.3.5`
- `scipy==1.9.3`
- `pytorch-fid==0.3.0`
- `protobuf==4.25.3`
- `requests==2.24.0`
- `lmdb==1.4.1`

### container에서 추가 설치한 패키지
```bash
python -m pip install tensorboard pandas lmdb pytorch-fid scipy==1.9.3 protobuf==4.25.3
```

---

## 3) 데이터 준비 (1024 subset)

### Raw source
- Raw image dir: `/data/CelebA/img_align_celeba`
- Order file: `/data/CelebA/img_list.json`

### Subset 생성 스크립트
- Script: `ddim/scripts/extract_celeba_1024.py`
- 생성 산출물:
  - `/kh_code/ddim_kkh/ddim/data/celeba_1024/img_align_celeba`
  - `/kh_code/ddim_kkh/ddim/data/celeba_1024/selected_1024_from_img_list.json`
  - `/kh_code/ddim_kkh/ddim/data/celeba_1024/selected_1024_from_img_list.txt`

Repo include policy:
- `selected_1024_from_img_list.json`는 재현성 위해 Git에 포함
- 실제 이미지(`img_align_celeba/*`)는 용량/라이선스 이슈로 Git 제외

### 실행
```bash
python /kh_code/ddim_kkh/ddim/scripts/extract_celeba_1024.py --mode copy
```

### 검증 포인트
- subset image count = `1024`
- selected json count = `1024`
- `selected_1024_from_img_list.json == img_list.json[:1024]`

핵심 원칙:
- `os.listdir()` 정렬/셔플 결과를 쓰지 않고, 반드시 `img_list.json` 순서를 source of truth로 사용.

---

## 4) 학습 설정 및 실행 (Config + Run)

### 최종 config 파일
- `ddim/configs/celeba_1024_ft.yml`

### 핵심 하이퍼파라미터
- Data
  - `dataset: CELEBA`
  - `image_size: 64`
  - `random_flip: true`
  - `rescaled: true` (input scale to [-1, 1])
  - `use_img_list_subset: true`
  - `subset_img_dir: /kh_code/ddim_kkh/ddim/data/celeba_1024/img_align_celeba`
  - `subset_img_list: /kh_code/ddim_kkh/ddim/data/celeba_1024/selected_1024_from_img_list.json`
- Model
  - `type: simple`
  - `ch: 128`
  - `ch_mult: [1, 2, 2, 2, 4]`
  - `num_res_blocks: 2`
  - `attn_resolutions: [16]`
  - `var_type: fixedlarge`
  - `ema: True`, `ema_rate: 0.9999`
- Diffusion
  - `beta_schedule: linear`
  - `beta_start: 0.0001`
  - `beta_end: 0.02`
  - `num_diffusion_timesteps: 1000`
- Training
  - `batch_size: 128`
  - `n_iters: 999999999` (manual stop intended)
  - `snapshot_freq: 5000`
- Eval
  - `enable: true`
  - `freq: 5000`
  - `n_samples: 1024`
  - `batch_size: 64`
  - `gap_threshold: 0.3333`
- Optimizer
  - `Adam`
  - `lr: 0.0002`
  - `beta1: 0.9`
  - `eps: 1e-8`
  - `weight_decay: 0`

### 실행 전 준비 (공통)
```bash
# 1) run directory
mkdir -p /data/CelebA/ckpt/logs/celeba1024_ft
mkdir -p /data/CelebA/ckpt/tensorboard/celeba1024_ft

# 2) base checkpoint를 run path에 copy (symlink 대신 copy 권장)
cp /kh_code/ddim_kkh/ddim/ckpt.pth /data/CelebA/ckpt/logs/celeba1024_ft/ckpt.pth

# 3) 시작 step 확인 (500000 기대)
python - <<'PY'
import torch
s=torch.load('/data/CelebA/ckpt/logs/celeba1024_ft/ckpt.pth', map_location='cpu')
print('step', s[3], 'epoch', s[2])
PY
```

### Resume training (single GPU)
```bash
cd /kh_code/ddim_kkh/ddim
CUDA_VISIBLE_DEVICES=1 python main.py \
  --config celeba_1024_ft.yml \
  --exp /data/CelebA/ckpt \
  --doc celeba1024_ft \
  --resume_training --ni
```

### Resume training (multi GPU)
```bash
cd /kh_code/ddim_kkh/ddim
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --config celeba_1024_ft.yml \
  --exp /data/CelebA/ckpt \
  --doc celeba1024_ft \
  --resume_training --ni
```

### 새 실험(run name 분리)로 처음부터 다시 시작
기존 run 로그/체크포인트를 건드리지 않으려면 `--doc`를 바꿔서 실행:
```bash
cd /kh_code/ddim_kkh/ddim
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --config celeba_1024_ft.yml \
  --exp /data/CelebA/ckpt \
  --doc celeba1024_ft_v2 \
  --resume_training --ni
```

### 재시작/중단 관련 운영 팁
```bash
# 같은 run의 중복 프로세스 확인
ps -eo pid,etime,cmd | rg "celeba_1024_ft.yml --exp /data/CelebA/ckpt --doc celeba1024_ft"

# 필요 시 종료
pkill -f "python main.py --config celeba_1024_ft.yml --exp /data/CelebA/ckpt --doc celeba1024_ft --resume_training --ni"
```

### 로그/모니터링
```bash
# train log
tail -n 120 /data/CelebA/ckpt/logs/celeba1024_ft/stdout.txt

# tensorboard
tensorboard --logdir /data/CelebA/ckpt/tensorboard --port 6006 --host 0.0.0.0

# checkpoint 생성 확인 (5000 step 간격)
ls -lah /data/CelebA/ckpt/logs/celeba1024_ft | rg "ckpt_.*\\.pth|ckpt_latest\\.pth"
```

---

## 5) 코드 수정 사항 (What we changed)

### A) 1024 subset 데이터 로더
- File: `ddim/datasets/__init__.py`
- Added:
  - `CelebAListSubset` class
  - `data.use_img_list_subset` 분기
- Purpose:
  - 공식 CelebA split 대신, `img_list.json` 기반 1024 subset을 학습/평가 데이터로 강제 사용.

### B) 학습 중 FID + mem_ratio 계산
- File: `ddim/runners/diffusion.py`
- Added:
  - periodic evaluate hook (`eval.freq`)
  - `evaluate_metrics()`
  - `_compute_fid()`
  - `_compute_mem_ratio()`
- Logged tags:
  - `eval/fid`
  - `eval/mem_ratio`

### C) checkpoint 저장 안전성 강화
- File: `ddim/runners/diffusion.py`
- Changed:
  - save output을 `ckpt_<step>.pth` + `ckpt_latest.pth`로 운영
  - `ckpt.pth`가 symlink일 경우 write 금지
  - resume는 `ckpt_latest.pth` 우선, 없으면 `ckpt.pth`
- Purpose:
  - base pretrained checkpoint overwrite 방지
  - run checkpoint lineage 분리

### D) FID parse 안정화
- File: `ddim/runners/diffusion.py`
- Changed:
  - FID parsing regex를 `r"FID:\s*([0-9eE+.\-]+)"`로 고정
- Purpose:
  - 이전 `fid: nan` (파싱 실패) 재발 방지

### E) runbook 기준 재현 파일
- Config: `ddim/configs/celeba_1024_ft.yml`
- Data extractor: `ddim/scripts/extract_celeba_1024.py`
- Main runner: `ddim/runners/diffusion.py`
- Dataset entry: `ddim/datasets/__init__.py`
