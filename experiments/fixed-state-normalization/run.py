"""Run the locked study once; resume must be explicitly requested."""
import argparse
from pathlib import Path
import traceback
import subprocess

from study import HERE, SEEDS, NEW_GROUPS, GROUPS, config, prepare, train_task, evaluate_task, execute, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'train', 'evaluate', 'verify', 'analyze', 'package', 'all'])
    parser.add_argument('--task', nargs=2, metavar=('GROUP', 'SEED'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plot-python', help='Python with ReportLab and PyMuPDF; required for all.')
    args = parser.parse_args()
    if args.phase == 'all' and not args.plot_python:
        parser.error('all requires --plot-python for reproducible chart generation.')
    cfg = config()
    if args.task:
        group, raw_seed = args.task
        seed = int(raw_seed)
        allowed = NEW_GROUPS if args.phase == 'train' else GROUPS
        if args.phase not in ('train', 'evaluate') or group not in allowed or seed not in SEEDS or (args.phase == 'evaluate' and group == 'uniform' and seed != 0):
            parser.error('Task outside the locked protocol.')
        folder = Path(cfg['runtime']['output_dir']) / ('training' if args.phase == 'train' else 'evaluation') / group
        folder = folder / 'i2c' / f'seed-{seed}' if args.phase == 'train' else folder / f'seed-{seed}'
        try:
            (train_task if args.phase == 'train' else evaluate_task)(cfg, group, seed, args.resume)
        except Exception as error:
            dump(folder / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
            raise
        return
    phases = ['prepare', 'train', 'evaluate', 'verify', 'analyze', 'package'] if args.phase == 'all' else [args.phase]
    for phase in phases:
        try:
            if phase == 'prepare':
                prepare(cfg)
            elif phase in ('train', 'evaluate'):
                execute(cfg, phase, args.resume)
            else:
                from results import verify, analyze, package
                {'verify': verify, 'analyze': analyze, 'package': package}[phase](cfg)
                if phase == 'analyze' and args.plot_python:
                    subprocess.run([args.plot_python, '-B', str(HERE / 'plot.py')], check=True)
        except Exception as error:
            root=Path(cfg['runtime']['output_dir'])
            if (root/'experiment.json').exists():
                dump(root/(phase+'-failure.json'),dict(error=repr(error),traceback=traceback.format_exc()))
                if phase in ('train','evaluate'):
                    from results import analyze
                    analyze(cfg)
            raise


if __name__ == '__main__':
    main()
