import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

from headmaster import db
from headmaster.heads import scan_head, scan_all_heads, validate_head

DEFAULT_WORKSPACE = "workspace"


def get_workspace() -> Path:
    ws = Path(os.environ.get("HEADMASTER_WORKSPACE", DEFAULT_WORKSPACE))
    if not ws.is_absolute():
        ws = Path.cwd() / ws
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _require_active_model(ws: Path) -> dict:
    model = db.get_active_model(ws)
    if model is None:
        raise SystemExit("error: no active model — run model-add and model-activate first")
    return model


def _resolve_heads(ws: Path, head_name: str | None) -> list:
    if head_name:
        head_dir = ws / "heads" / head_name
        if not head_dir.is_dir():
            raise SystemExit(f"error: head '{head_name}' not found")
        return [scan_head(head_dir)]
    return scan_all_heads(ws)


# ── Model commands ──────────────────────────────────────────────


def cmd_model_add(args: argparse.Namespace) -> None:
    ws = get_workspace()
    db.init_db(ws)
    db.model_add(ws, args.name, args.path, args.dim)
    print(f"added model '{args.name}'")


def cmd_model_list(args: argparse.Namespace) -> None:
    ws = get_workspace()
    db.init_db(ws)
    models = db.model_list(ws)
    if not models:
        print("no models registered")
        return
    for m in models:
        marker = "*" if m["active"] else " "
        print(f"  {marker} {m['name']:20s} dim={m['embed_dim']}  {m['path']}")


def cmd_model_activate(args: argparse.Namespace) -> None:
    ws = get_workspace()
    db.init_db(ws)
    db.model_activate(ws, args.name)
    print(f"activated model '{args.name}'")


def cmd_model_remove(args: argparse.Namespace) -> None:
    ws = get_workspace()
    db.init_db(ws)
    db.model_remove(ws, args.name)
    print(f"removed model '{args.name}'")


# ── Embed ───────────────────────────────────────────────────────


def cmd_embed(args: argparse.Namespace) -> None:
    from headmaster.embed import embed_head

    ws = get_workspace()
    db.init_db(ws)
    model_info = _require_active_model(ws)
    heads = _resolve_heads(ws, args.head)

    if not heads:
        print("no heads found")
        return

    for head in heads:
        errors, warnings = validate_head(head)
        for w in warnings:
            print(f"warning: {w}")
        if errors:
            for e in errors:
                print(f"error: {e}")
            continue
        print(f"embedding head '{head.name}' ({head.total_images} images)")
        embed_head(ws, head, model_info, workers=args.workers)
        print(f"  done")


# ── Train ───────────────────────────────────────────────────────


def cmd_train(args: argparse.Namespace) -> None:
    from headmaster.embed import embed_head
    from headmaster.train import train_head

    ws = get_workspace()
    db.init_db(ws)
    model_info = _require_active_model(ws)
    heads = _resolve_heads(ws, args.head)

    if not heads:
        print("no heads found")
        return

    for head in heads:
        errors, warnings = validate_head(head)
        for w in warnings:
            print(f"warning: {w}")
        if errors:
            for e in errors:
                print(f"error: {e}")
            continue
        train_head(ws, head, model_info, workers=args.workers)


# ── Status ──────────────────────────────────────────────────────


def cmd_status(args: argparse.Namespace) -> None:
    ws = get_workspace()
    db.init_db(ws)
    heads = _resolve_heads(ws, args.head)

    if not heads:
        print("no heads found")
        return

    if args.head:
        # Detail view for single head
        head = heads[0]
        ckpt_path = ws / "out" / f"{head.name}.pt"
        print(f"Head:       {head.name}")
        print(f"Type:       {head.head_type}")

        bucket_str = ", ".join(f"{b.name} ({len(b.images)})" for b in head.buckets)
        print(f"Buckets:    {bucket_str}")

        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            meta = ckpt.get("metadata", {})
            metrics = meta.get("metrics", {})
            print(f"Model:      {ckpt.get('model', '—')}")
            print(f"Trained:    {meta.get('created_at', '—')[:10]}")

            if ckpt.get("type") == "binary":
                print(f"Accuracy:   {metrics.get('accuracy', '—')}")
                print(f"Precision:  {metrics.get('precision', '—')}")
                print(f"Recall:     {metrics.get('recall', '—')}")
                print(f"F1:         {metrics.get('f1', '—')}")
                print(f"Threshold:  {ckpt.get('threshold', '—')}")
            else:
                print(f"Accuracy:   {metrics.get('accuracy', '—')}")
                per_class = metrics.get("per_class", {})
                for cls, m in sorted(per_class.items()):
                    print(f"  {cls:15s} P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
        else:
            print("Trained:    no")
    else:
        # Summary table
        print(f"{'HEAD':15s} {'TYPE':11s} {'BUCKETS':35s} {'IMAGES':>7s}  {'TRAINED':7s}  {'F1':>5s}")
        for head in heads:
            ckpt_path = ws / "out" / f"{head.name}.pt"
            bucket_str = " ".join(f"{b.name}({len(b.images)})" for b in head.buckets)
            if len(bucket_str) > 33:
                bucket_str = bucket_str[:30] + "..."

            trained = "no"
            f1 = "—"
            if ckpt_path.exists():
                trained = "yes"
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                metrics = ckpt.get("metadata", {}).get("metrics", {})
                if "f1" in metrics:
                    f1 = f"{metrics['f1']:.2f}"
                elif "per_class" in metrics:
                    # Macro average F1 for multiclass
                    per_class = metrics["per_class"]
                    if per_class:
                        f1 = f"{sum(c['f1'] for c in per_class.values()) / len(per_class):.2f}"

            print(f"{head.name:15s} {head.head_type:11s} {bucket_str:35s} {head.total_images:>7d}  {trained:7s}  {f1:>5s}")


# ── Classify ────────────────────────────────────────────────────


def cmd_classify(args: argparse.Namespace) -> None:
    from headmaster.embed import embed_images
    from headmaster.train import ClassifierHead
    from headmaster.heads import IMAGE_EXTS

    ws = get_workspace()
    db.init_db(ws)
    model_info = _require_active_model(ws)

    ckpt_path = ws / "out" / f"{args.head}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"error: no checkpoint for head '{args.head}' — run train first")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    embed_dim = ckpt["input_dim"]
    head_type = ckpt["type"]

    model = ClassifierHead(embed_dim, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.src:
        src_dir = Path(args.src)
    else:
        src_dir = ws / "inbox" / args.head
    if not src_dir.is_dir():
        raise SystemExit(f"error: source directory '{src_dir}' not found")

    dest_dir = Path(args.dest) if args.dest else ws / "classified" / args.head
    images = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    if not images:
        print("no images found in source directory")
        return

    print(f"classifying {len(images)} images with head '{args.head}'")

    embeddings = embed_images(ws, images, model_info)

    from headmaster.embed import hash_file

    for img_path in images:
        h = hash_file(img_path)
        if h not in embeddings:
            continue

        vec = embeddings[h].unsqueeze(0)
        with torch.no_grad():
            out = model(vec)

        if head_type == "binary":
            prob = torch.sigmoid(out).item()
            threshold = ckpt.get("threshold", 0.5)
            if abs(prob - threshold) < 0.1:
                label = "uncertain"
            elif prob >= threshold:
                label = classes[1]  # positive class
            else:
                label = classes[0]  # negative class
        else:
            probs = torch.softmax(out, dim=1)
            max_prob, pred_idx = probs.max(dim=1)
            if max_prob.item() < 0.5:
                label = "uncertain"
            else:
                label = classes[pred_idx.item()]

        out_dir = dest_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, out_dir / img_path.name)

    print(f"  results in {dest_dir}")


# ── Export & Clean ──────────────────────────────────────────────


def cmd_export(args: argparse.Namespace) -> None:
    ws = get_workspace()
    out_dir = ws / "out"
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(out_dir.glob("*.pt"))
    if not checkpoints:
        print("no checkpoints to export")
        return

    for ckpt in checkpoints:
        shutil.copy2(ckpt, dest / ckpt.name)
        print(f"  exported {ckpt.name}")


def cmd_clean(args: argparse.Namespace) -> None:
    ws = get_workspace()
    out_dir = ws / "out"

    if args.head:
        target = out_dir / f"{args.head}.pt"
        if target.exists():
            target.unlink()
            print(f"removed {target.name}")
        else:
            print(f"no checkpoint for head '{args.head}'")
    else:
        checkpoints = sorted(out_dir.glob("*.pt"))
        if not checkpoints:
            print("no checkpoints to remove")
            return
        for ckpt in checkpoints:
            ckpt.unlink()
            print(f"removed {ckpt.name}")


# ── Parser ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="headmaster")
    sub = parser.add_subparsers(dest="command")

    # model-add
    p = sub.add_parser("model-add")
    p.add_argument("--name", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--dim", required=True, type=int)
    p.set_defaults(func=cmd_model_add)

    # model-list
    p = sub.add_parser("model-list")
    p.set_defaults(func=cmd_model_list)

    # model-activate
    p = sub.add_parser("model-activate")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_model_activate)

    # model-remove
    p = sub.add_parser("model-remove")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_model_remove)

    # embed
    p = sub.add_parser("embed")
    p.add_argument("--head", default=None)
    p.add_argument("-j", "--workers", type=int, default=0)
    p.set_defaults(func=cmd_embed)

    # train
    p = sub.add_parser("train")
    p.add_argument("--head", default=None)
    p.add_argument("-j", "--workers", type=int, default=0)
    p.set_defaults(func=cmd_train)

    # status
    p = sub.add_parser("status")
    p.add_argument("--head", default=None)
    p.set_defaults(func=cmd_status)

    # classify
    p = sub.add_parser("classify")
    p.add_argument("--head", required=True)
    p.add_argument("--src", default=None)
    p.add_argument("--dest", default=None)
    p.set_defaults(func=cmd_classify)

    # export
    p = sub.add_parser("export")
    p.add_argument("--dest", required=True)
    p.set_defaults(func=cmd_export)

    # clean
    p = sub.add_parser("clean")
    p.add_argument("--head", default=None)
    p.set_defaults(func=cmd_clean)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)
