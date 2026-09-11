import re
import yaml
from pathlib import Path

ROOT = Path('/data/lz/public/CBraMod')
DATASETS = ['CHB-MIT', 'TUAB', 'TUEV', 'ISRUC', 'FACED', 'SHU-MI', 'Mumtaz2016', 'MentalArithmetic', 'HMC']
BACKBONES = ['efficientnet_b0', 'convnext_tiny_dinov3', 'vit_small_dinov3']


def safe_name(name):
    return re.sub(r'[^a-zA-Z0-9]+', '_', str(name)).strip('_').lower()


def _pick_metric(results):
    if isinstance(results, dict) and 'mean' in results and isinstance(results['mean'], dict):
        m = results['mean']
        if 'kappa' in m:
            return 'kappa', m['kappa'], results.get('std_population', {}).get('kappa')
        if 'pr_auc' in m:
            return 'pr_auc', m['pr_auc'], results.get('std_population', {}).get('pr_auc')
    if isinstance(results, dict) and 'kappa' in results and isinstance(results['kappa'], dict):
        return 'kappa', results['kappa']['mean'], results['kappa']['std']
    if isinstance(results, dict) and 'pr_auc' in results and isinstance(results['pr_auc'], dict):
        return 'pr_auc', results['pr_auc']['mean'], results['pr_auc']['std']
    return None, None, None


def fold_baseline(backbone, dataset):
    src = ROOT / 'configs' / 'backbones' / backbone / (dataset + '.yaml')
    cfg = yaml.safe_load(src.read_text())
    results = cfg.get('results') or {}

    nested = {k: v for k, v in results.items() if isinstance(v, dict) and ('pr_auc' in v or 'kappa' in v)}
    if nested and ('with_augmentation' in nested or 'no_augmentation' in nested):
        training = cfg.get('training', {})
        aug_on = training.get('mirror_augmentation') or training.get('time_roll_augmentation') or training.get('amplitude_scale_augmentation')
        key = 'with_augmentation' if aug_on else 'no_augmentation'
        return _pick_metric(nested.get(key, {}))

    return _pick_metric(results)


def chunk_results(backbone, dataset):
    log_dir = ROOT / 'experiments' / 'logs' / 'ablation_fold_geometry' / backbone / safe_name(dataset)
    values = {}
    for log in sorted(log_dir.glob('seed*/vision/*/*.log')) if log_dir.exists() else []:
        seed = re.search(r'/seed(\d+)/', str(log))
        if not seed:
            continue
        seed = int(seed.group(1))
        text = log.read_text()
        m = re.search(r'Test Evaluation: ba: ([0-9.]+), (?:kappa|pr_auc): ([0-9.]+)', text)
        if not m:
            continue
        values[seed] = float(m.group(2))
    if not values:
        return None
    nums = [values[s] for s in sorted(values)]
    mean = sum(nums) / len(nums)
    var = sum((x - mean) ** 2 for x in nums) / len(nums)
    return {'n': len(nums), 'mean': mean, 'std': var ** 0.5, 'per_seed': values}


def main():
    print('{:<12} {:<16} {:>10} {:>12} {:>12} {:>12}'.format(
        'backbone', 'dataset', 'metric', 'fold', 'chunk', 'delta'))
    print('-' * 76)
    for backbone in BACKBONES:
        for dataset in DATASETS:
            metric, fmean, fstd = fold_baseline(backbone, dataset)
            chunk = chunk_results(backbone, dataset)
            bname = backbone.replace('_', '-')
            if metric is None:
                print('{:<12} {:<16} {:>10} {:>12}'.format(bname, dataset, '?', 'NO BASELINE'))
                continue
            if chunk is None:
                print('{:<12} {:<16} {:>10} {:>12.5f} {:>12} {:>12}'.format(
                    bname, dataset, metric, fmean, 'pending', ''))
                continue
            delta = chunk['mean'] - fmean
            print('{:<12} {:<16} {:>10} {:>12.5f} {:>12.5f} {:>+12.5f}  ({}/5 seeds)'.format(
                bname, dataset, metric, fmean, chunk['mean'], delta, chunk['n']))


if __name__ == '__main__':
    main()
