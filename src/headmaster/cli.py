import argparse
import sys
from pathlib import Path

from headmaster import db


def get_workspace() -> Path:
    return Path.cwd()


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


# ── Stub commands ───────────────────────────────────────────────


def cmd_embed(args: argparse.Namespace) -> None:
    raise SystemExit("error: embed not implemented yet")


def cmd_train(args: argparse.Namespace) -> None:
    raise SystemExit("error: train not implemented yet")


def cmd_status(args: argparse.Namespace) -> None:
    raise SystemExit("error: status not implemented yet")


def cmd_classify(args: argparse.Namespace) -> None:
    raise SystemExit("error: classify not implemented yet")


def cmd_export(args: argparse.Namespace) -> None:
    raise SystemExit("error: export not implemented yet")


def cmd_clean(args: argparse.Namespace) -> None:
    raise SystemExit("error: clean not implemented yet")


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
    p.set_defaults(func=cmd_embed)

    # train
    p = sub.add_parser("train")
    p.add_argument("--head", default=None)
    p.set_defaults(func=cmd_train)

    # status
    p = sub.add_parser("status")
    p.add_argument("--head", default=None)
    p.set_defaults(func=cmd_status)

    # classify
    p = sub.add_parser("classify")
    p.add_argument("--head", required=True)
    p.add_argument("--src", required=True)
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
