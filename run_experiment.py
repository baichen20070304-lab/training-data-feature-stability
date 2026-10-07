"""Run pilot, calibrated full study, or rebuild analysis from saved checkpoints."""
import argparse
import json
from pathlib import Path

import yaml

from stability.experiment import run_stage, select_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--stage', choices=['all', 'pilot', 'full', 'analysis'], default='all')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = yaml.safe_load((root / args.config).read_text(encoding='utf-8'))
    if args.stage in ['all', 'pilot', 'full'] and 71 not in config.get('sample_sizes', []):
        parser.error('The fixed 512/1024 calibration protocol requires a 71-row training tier in sample_sizes.')
    if args.workers is not None:
        config['workers'] = args.workers
    if args.stage in ['all', 'pilot']:
        directory, pilot = run_stage(config, root, 'pilot', 512)
        selected = select_budget(config, pilot, directory)
        print(f'Calibration selected {selected} samples', flush=True)
    if args.stage in ['all', 'full']:
        calibration = root / config['output'] / 'pilot' / 'calibration.json'
        if not calibration.exists():
            parser.error('Run --stage pilot first to calibrate KernelSHAP.')
        # Revalidate the pilot fingerprint against the current scientific configuration before reusing its decision.
        pilot_directory, pilot = run_stage(config, root, 'pilot', 512)
        selected = select_budget(config, pilot, pilot_directory)
        directory, _ = run_stage(config, root, 'full', selected)
    if args.stage in ['all', 'analysis']:
        from stability.analysis import analyze
        analyze(root / config['output'] / 'full', root, dataset_path=root / config['dataset'])


if __name__ == '__main__':
    main()
