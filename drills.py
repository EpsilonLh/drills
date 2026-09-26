#!/usr/bin/env python3
# Copyright (c) 2019, SCALE Lab, Brown University. BSD-3-Clause; see LICENSE.
import argparse
from drills.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description='Run the complete FPGA assessment from params.yml')
    parser.add_argument('mode', choices=['train', 'baseline', 'report'])
    parser.add_argument('mapping', choices=['fpga'])
    parser.add_argument('params', nargs='?', default='params.yml')
    parser.add_argument('-l', '--resume', '--load-model', action='store_true', dest='resume')
    args = parser.parse_args()
    run_experiment(args.params, args.mode, args.resume)


if __name__ == '__main__':
    main()
