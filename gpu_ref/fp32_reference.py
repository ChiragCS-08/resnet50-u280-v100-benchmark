#!/usr/bin/env python3
"""FP32 ResNet-50 golden reference (GPU path).

Runs torchvision's pretrained ResNet-50 (IMAGENET1K_V1 weights) in FP32 on a
fixed list of ImageNet validation images and writes:
  * per-image Top-1 / Top-5 labels with softmax scores (CSV)
  * an aggregate summary (JSON) with Top-1 / Top-5 accuracy and environment info
  * the exact image list used, so the FPGA path can run the identical images

Labels: CSV with header-less rows "filename,label", where label is either an
integer class index (0-999) or a WordNet id (e.g. n01440764). WordNet ids need
--synsets, a text file with the 1000 wnids in sorted order, one per line
(first token of each line is used, so LOC_synset_mapping.txt works as-is).
"""
import argparse
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, type=Path, help="directory with validation images")
    p.add_argument("--labels", required=True, type=Path, help="CSV: filename,label (index or wnid)")
    p.add_argument("--synsets", type=Path, help="file with 1000 sorted wnids (needed if labels are wnids)")
    p.add_argument("--num", type=int, default=100, help="number of images to sample (default 100)")
    p.add_argument("--seed", type=int, default=0, help="seed for the image sample (default 0)")
    p.add_argument("--list", type=Path, help="use this image list (one filename per line) instead of sampling")
    p.add_argument("--batch", type=int, default=1, help="batch size (default 1)")
    p.add_argument("--device", default="cuda", help="cuda or cpu (default cuda)")
    p.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    return p.parse_args()


def load_labels(labels_csv, synsets_file):
    """Return {filename: class_index}."""
    wnid_to_idx = None
    if synsets_file is not None:
        wnids = [ln.split()[0] for ln in synsets_file.read_text().splitlines() if ln.strip()]
        if len(wnids) != 1000:
            sys.exit(f"--synsets must list 1000 classes, found {len(wnids)}")
        wnid_to_idx = {w: i for i, w in enumerate(sorted(wnids))}
    labels = {}
    with open(labels_csv, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0].strip().lower() in ("filename", "file", "image"):
                continue
            name, lab = row[0].strip(), row[1].strip()
            if lab.isdigit():
                labels[name] = int(lab)
            elif wnid_to_idx is not None and lab in wnid_to_idx:
                labels[name] = wnid_to_idx[lab]
            else:
                sys.exit(f"Cannot interpret label '{lab}' for {name}; pass --synsets for wnid labels")
    return labels


def pick_images(args, labels):
    if args.list:
        names = [ln.strip() for ln in args.list.read_text().splitlines() if ln.strip()]
    else:
        avail = sorted(p.name for p in args.images.iterdir()
                       if p.suffix.lower() in IMG_EXTS and p.name in labels)
        if len(avail) < args.num:
            sys.exit(f"Only {len(avail)} labelled images found in {args.images}, need {args.num}")
        names = sorted(random.Random(args.seed).sample(avail, args.num))
    missing = [n for n in names if n not in labels or not (args.images / n).exists()]
    if missing:
        sys.exit(f"{len(missing)} listed images missing or unlabelled, e.g. {missing[:3]}")
    return names


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available (check nvidia-smi and the torch cu126 install)")
    device = torch.device(args.device)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    labels = load_labels(args.labels, args.synsets)
    names = pick_images(args, labels)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "image_list.txt").write_text("\n".join(names) + "\n")

    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights).eval().to(device).float()
    # torchvision preprocessing: resize 256, center-crop 224, ImageNet mean/std
    preprocess = weights.transforms()

    rows, top1_ok, top5_ok = [], 0, 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(names), args.batch):
            chunk = names[i:i + args.batch]
            x = torch.stack([preprocess(Image.open(args.images / n).convert("RGB")) for n in chunk]).to(device)
            probs = torch.softmax(model(x), dim=1).cpu()
            p5, i5 = probs.topk(5, dim=1)
            for n, pv, iv in zip(chunk, p5.tolist(), i5.tolist()):
                truth = labels[n]
                ok1, ok5 = iv[0] == truth, truth in iv
                top1_ok += ok1
                top5_ok += ok5
                rows.append([n, truth, *iv, *[f"{v:.6f}" for v in pv], int(ok1), int(ok5)])
    elapsed = time.time() - t0

    header = (["filename", "label"] + [f"top{k}_idx" for k in range(1, 6)] +
              [f"top{k}_prob" for k in range(1, 6)] + ["top1_correct", "top5_correct"])
    with open(args.out / "fp32_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    n = len(names)
    summary = {
        "images": n,
        "top1_accuracy": top1_ok / n,
        "top5_accuracy": top5_ok / n,
        "weights": "torchvision ResNet50_Weights.IMAGENET1K_V1",
        "precision": "FP32",
        "batch": args.batch,
        "seed": None if args.list else args.seed,
        "wall_seconds_incl_preprocess": round(elapsed, 3),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "python": platform.python_version(),
    }
    (args.out / "fp32_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}/fp32_predictions.csv, fp32_summary.json, image_list.txt")


if __name__ == "__main__":
    main()
