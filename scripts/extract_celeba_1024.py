import argparse
import json
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="/data/CelebA/img_align_celeba")
    parser.add_argument("--img-list", default="/data/CelebA/img_list.json")
    parser.add_argument(
        "--out-dir",
        default="/kh_code/ddim_kkh/ddim/data/celeba_1024/img_align_celeba",
    )
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.img_list, "r") as f:
        files = json.load(f)

    selected = files[: args.count]
    if len(selected) < args.count:
        raise ValueError(f"img_list has only {len(selected)} items, requested {args.count}")

    os.makedirs(args.out_dir, exist_ok=True)

    missing = []
    for name in selected:
        src = os.path.join(args.img_dir, name)
        dst = os.path.join(args.out_dir, name)

        if not os.path.exists(src):
            missing.append(name)
            continue

        if os.path.lexists(dst):
            os.remove(dst)

        if args.mode == "copy":
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)

    if missing:
        raise FileNotFoundError(f"missing files: {len(missing)} (example: {missing[:5]})")

    meta_dir = os.path.dirname(args.out_dir)
    with open(os.path.join(meta_dir, "selected_1024_from_img_list.json"), "w") as f:
        json.dump(selected, f)
    with open(os.path.join(meta_dir, "selected_1024_from_img_list.txt"), "w") as f:
        for name in selected:
            f.write(name + "\n")

    print(f"done: {len(selected)} files -> {args.out_dir}")
    print(f"first3: {selected[:3]}")
    print(f"last3: {selected[-3:]}")


if __name__ == "__main__":
    main()
