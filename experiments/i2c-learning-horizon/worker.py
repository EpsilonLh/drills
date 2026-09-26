"""Execute a single fixed group; run.py owns suite provenance and preflight."""
import argparse
from run import GROUPS, group_config, run_group


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=list(GROUPS))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_group(group_config(args.group), args.group, args.resume)
