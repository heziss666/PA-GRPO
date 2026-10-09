"""Validated command boundary for run_fake_e2e."""

from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts_permstudy.data import _common as io


def build_parser() -> io.argparse.ArgumentParser:
    parser = io.parser("run_fake_e2e")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixture-root")
    group.add_argument("--split-manifest")
    return parser


def dispatch(args, root):
    raise io.PhaseBoundaryError()


def main(argv: Sequence[str] | None = None) -> int:
    return io.execute(build_parser, dispatch, argv)


if __name__ == "__main__":
    raise SystemExit(main())
