"""Compare adaptation strategies for an image foundation encoder on a driving task.

NOT EXECUTED in the repository build environment (no torch). Written against
public PyTorch / transformers / peft APIs; run, fix version drift, record versions.

Strategies (``--strategy``), all sharing data, head, schedule and eval:
    linear_probe    encoder frozen, features cached-in-graph, linear head on CLS+mean-patch
    frozen_head     encoder frozen, small conv head on the patch-token map (dense task)
    lora            LoRA adapters on attention (requires `peft`), head trainable
    partial         last N blocks + head trainable (--unfreeze-last N)
    full            everything trainable, layer-wise LR decay (--llrd)

The task here is deliberately simple and dense: *per-patch vehicle occupancy*
from 2D boxes (does this 14x14 patch overlap a vehicle box?). It isolates
representation quality from detector-head engineering; move to a DETR head
(recipes/2d_detection) once the ranking of strategies is clear.

Dataset contract (--manifest JSONL, one line per image):
    {"image": "path.jpg", "boxes_xyxy": [[x1, y1, x2, y2], ...], "split": "train"|"val"}

Usage:
    python recipes/image_encoders/finetune_encoder.py --manifest data/nuimages_vehicles.jsonl \
        --strategy lora --epochs 10 --out outputs/enc_lora
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class PatchOccupancy(Dataset):
    def __init__(self, manifest: Path, split: str, size_hw: tuple[int, int], patch: int) -> None:
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        self.rows = [r for r in rows if r["split"] == split]
        self.h, self.w = size_hw
        self.patch = patch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rows[i]
        img = Image.open(r["image"]).convert("RGB")
        sx, sy = (
            self.w / img.width,
            self.h / img.height,
        )  # aspect-preserving if size matches source ratio
        x = (
            torch.from_numpy(np.asarray(img.resize((self.w, self.h)))).permute(2, 0, 1).float()
            / 255
        )
        x = (x - MEAN) / STD
        gh, gw = self.h // self.patch, self.w // self.patch
        y = torch.zeros(gh, gw)
        for x1, y1, x2, y2 in r["boxes_xyxy"]:
            c1, c2 = int(x1 * sx // self.patch), math.ceil(x2 * sx / self.patch)
            r1, r2 = int(y1 * sy // self.patch), math.ceil(y2 * sy / self.patch)
            y[max(r1, 0) : min(r2, gh), max(c1, 0) : min(c2, gw)] = 1.0
        return x, y


class DenseHead(nn.Module):
    def __init__(self, dim: int, linear: bool) -> None:
        super().__init__()
        self.net: nn.Module = (
            nn.Conv2d(dim, 1, 1)
            if linear
            else nn.Sequential(nn.Conv2d(dim, 256, 3, padding=1), nn.GELU(), nn.Conv2d(256, 1, 1))
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.net(fmap).squeeze(1)


def patch_map(encoder: nn.Module, x: torch.Tensor, patch: int, n_prefix: int) -> torch.Tensor:
    tok = encoder(pixel_values=x).last_hidden_state[:, n_prefix:]  # drop CLS (+ registers)
    b, _, d = tok.shape
    return tok.transpose(1, 2).reshape(b, d, x.shape[-2] // patch, x.shape[-1] // patch)


def param_groups(encoder: nn.Module, head: nn.Module, args: argparse.Namespace) -> list[dict]:
    """Discriminative LRs. For `full`, layer-wise decay: block k gets lr * llrd^(depth-k)."""
    groups = [{"params": [p for p in head.parameters()], "lr": args.lr_head}]
    enc_params = [(n, p) for n, p in encoder.named_parameters() if p.requires_grad]
    if not enc_params:
        return groups
    if args.strategy != "full":
        groups.append({"params": [p for _, p in enc_params], "lr": args.lr_encoder})
        return groups
    depth = encoder.config.num_hidden_layers
    for n, p in enc_params:
        layer = depth  # embeddings / final norm default
        if ".layer." in n:
            layer = int(n.split(".layer.")[1].split(".")[0]) + 1
        elif "embeddings" in n:
            layer = 0
        scale = args.llrd ** (depth - layer) if layer <= depth else 1.0
        groups.append(
            {
                "params": [p],
                "lr": args.lr_encoder * scale,
                "weight_decay": 0.0 if p.ndim == 1 else 0.05,
            }
        )
    return groups


def configure(encoder: nn.Module, args: argparse.Namespace) -> nn.Module:
    for p in encoder.parameters():
        p.requires_grad = args.strategy == "full"
    if args.strategy == "partial":
        depth = encoder.config.num_hidden_layers
        for n, p in encoder.named_parameters():
            if (
                ".layer." in n
                and int(n.split(".layer.")[1].split(".")[0]) >= depth - args.unfreeze_last
            ):
                p.requires_grad = True
    if args.strategy == "lora":
        from peft import LoraConfig, get_peft_model  # optional dependency

        cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            target_modules=["query", "value"],
            lora_dropout=0.05,
        )
        encoder = get_peft_model(encoder, cfg)
        encoder.print_trainable_parameters()
    return encoder


@torch.no_grad()
def evaluate(
    encoder: nn.Module, head: nn.Module, loader: DataLoader, args: argparse.Namespace
) -> dict[str, float]:
    encoder.eval()
    head.eval()
    tp = fp = fn = 0.0
    for x, y in loader:
        x, y = x.to(args.device), y.to(args.device)
        with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.amp):
            pred = head(patch_map(encoder, x, args.patch, args.n_prefix)).float().sigmoid() > 0.5
        tp += (pred & (y > 0.5)).sum().item()
        fp += (pred & (y < 0.5)).sum().item()
        fn += (~pred & (y > 0.5)).sum().item()
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--revision", default="main")
    ap.add_argument(
        "--strategy",
        choices=["linear_probe", "frozen_head", "lora", "partial", "full"],
        required=True,
    )
    ap.add_argument(
        "--size", type=int, nargs=2, default=[518, 924], help="H W; multiples of the patch size"
    )
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument(
        "--n-prefix", type=int, default=1, help="CLS + register tokens to drop (DINOv2-base: 1)"
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--llrd", type=float, default=0.75)
    ap.add_argument("--unfreeze-last", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    encoder = configure(AutoModel.from_pretrained(args.model, revision=args.revision), args).to(
        args.device
    )
    head = DenseHead(encoder.config.hidden_size, linear=args.strategy == "linear_probe").to(
        args.device
    )
    train = DataLoader(
        PatchOccupancy(args.manifest, "train", tuple(args.size), args.patch),
        args.batch,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val = DataLoader(
        PatchOccupancy(args.manifest, "val", tuple(args.size), args.patch),
        args.batch,
        num_workers=4,
    )

    opt = torch.optim.AdamW(param_groups(encoder, head, args), weight_decay=0.05)
    total = args.epochs * len(train) // args.accum
    warm = max(1, int(args.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))),
    )
    n_train = sum(
        p.numel() for p in list(encoder.parameters()) + list(head.parameters()) if p.requires_grad
    )
    print(f"strategy={args.strategy} trainable params={n_train / 1e6:.2f}M steps={total}")

    best, log = -1.0, []
    for epoch in range(args.epochs):
        encoder.train(args.strategy in {"lora", "partial", "full"})
        head.train()
        for step, (x, y) in enumerate(train):
            x, y = x.to(args.device), y.to(args.device)
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.amp):
                logits = head(patch_map(encoder, x, args.patch, args.n_prefix)).float()
            # positives are rare (vehicles cover few patches) -> weight them
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=torch.tensor(4.0, device=args.device)
            )
            (loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], 1.0
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
        metrics = evaluate(encoder, head, val, args)
        log.append({"epoch": epoch, **metrics})
        print(log[-1])
        if metrics["iou"] > best:
            best = metrics["iou"]
            torch.save(
                {
                    "head": head.state_dict(),
                    "args": vars(args) | {"manifest": str(args.manifest), "out": str(args.out)},
                },
                args.out / "best_head.pt",
            )
    (args.out / "log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
