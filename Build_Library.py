"""Build the six-reference SAP library using the available pretrained FECCT."""
import argparse

from Main_DC import (
    LIBRARY_PATH, add_shared_args, derive_entry, load_pretrained, resolve_pretrained_path, setup,
)
from prune.search import checkpoint_digest, load_library, model_config, save_library

REFERENCE_CODES = (
    'BCH_N31_K16', 'BCH_N63_K51', 'NRLDPC_N96_K64',
    'LDPC_N121_K60', 'POLAR_N64_K48', 'POLAR_N128_K86',
)


def build_library(args):
    device = setup(args)
    # A different pruning ratio must never overwrite or search an existing library.
    if LIBRARY_PATH.exists():
        load_library(LIBRARY_PATH, args.model_prun_ratio, model_config(args))
    checkpoint_path = resolve_pretrained_path()
    digest = checkpoint_digest(checkpoint_path)
    model = load_pretrained(args, device, checkpoint_path)
    entries = [derive_entry(model, name, args, device) for name in REFERENCE_CODES]
    save_library({'model_prun_ratio': args.model_prun_ratio, 'model_config': model_config(args),
                  'checkpoint_sha256': digest, 'entries': entries}, LIBRARY_PATH)


def main():
    parser = add_shared_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    try:
        build_library(args)
    except Exception:
        print('Library build failed.')
        return 1
    print('Library build succeeded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
