from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

from .core import family_delta, load_dataset, ranks, rank_correlation, top_k_credit
from .experiment import load_checkpoint, write_checkpoint


def summarize_results(results: list[dict], models: list[str]) -> dict[str, pd.DataFrame]:
    references = {r['model']: r for r in results if r['is_reference']}
    if set(references) != set(models) or len([r for r in results if r['is_reference']]) != len(models):
        raise ValueError('Incomplete or duplicate full-pool references.')
    features = references[models[0]]['feature_names']
    groups, importance, performance, stability = {}, [], [], []
    seen = set()
    for result in results:
        model = result['model']
        key = (result['size'], result['repeat'], model)
        if key in seen or model not in models or result['feature_names'] != features:
            raise ValueError('Duplicate fit or inconsistent feature/model membership.')
        seen.add(key)
        base = {k: result[k] for k in ['model', 'size', 'repeat', 'is_reference']}
        performance.append({**base, 'mae': result['mae'], 'r2': result['r2'],
                            **{f'{k}_seconds': v for k, v in result['seconds'].items()}})
        vectors = {}
        for method in ['PFI', 'KernelSHAP']:
            values = np.asarray(result['importance'][method])
            ranking = ranks(values)
            credit = top_k_credit(values, min(3, len(values)))
            vectors[method] = ranking
            for feature, value, rank, top3 in zip(features, values, ranking, credit):
                importance.append({**base, 'method': method, 'feature': feature,
                                   'importance': value, 'rank': rank, 'top3_credit': top3})
            if not result['is_reference']:
                rho = rank_correlation(ranking, ranks(references[model]['importance'][method]))
                stability.append({**base, 'method': method, 'rho_reference': rho,
                                  'correlation_defined': np.isfinite(rho), 'mae': result['mae'], 'r2': result['r2']})
        if not result['is_reference']:
            groups.setdefault((result['size'], result['repeat']), {})[model] = vectors
    family = []
    for (size, repeat), vectors in sorted(groups.items()):
        if set(vectors) != set(models):
            raise ValueError(f'Incomplete model set at size={size}, repeat={repeat}.')
        comparison = family_delta(vectors)
        family.append({'size': size, 'repeat': repeat, **comparison,
                       'comparison_defined': np.isfinite(comparison['delta'])})
    imp = pd.DataFrame(importance)
    perturbed = imp.loc[~imp['is_reference']]
    summary = perturbed.groupby(['size', 'model', 'method', 'feature'], sort=False).agg(
        median_rank=('rank', 'median'), q05_rank=('rank', lambda x: x.quantile(.05)),
        q25_rank=('rank', lambda x: x.quantile(.25)), q75_rank=('rank', lambda x: x.quantile(.75)),
        q95_rank=('rank', lambda x: x.quantile(.95)), top3_frequency=('top3_credit', 'mean'),
        observations=('rank', 'size')).reset_index()
    return {'importance': imp, 'performance': pd.DataFrame(performance),
            'stability': pd.DataFrame(stability), 'family': pd.DataFrame(family), 'feature_summary': summary}


COLORS = {'LR': '#2874A6', 'RF': '#229954', 'GBRT': '#D68910', 'SVR': '#884EA0'}


def make_figures(tables: dict, directory: Path, models: list[str], features: list[str]) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    output = []

    def save(name):
        plt.savefig(directory / f'{name}.png', dpi=160, bbox_inches='tight')
        plt.savefig(directory / f'{name}.svg', bbox_inches='tight')
        plt.close()
        output.append(name)

    stability = tables['stability']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, method in zip(axes, ['PFI', 'KernelSHAP']):
        for model in models:
            rows = stability[(stability.model == model) & (stability.method == method)]
            summary = rows.groupby('size').rho_reference.agg(['median', lambda x: x.quantile(.05), lambda x: x.quantile(.95)])
            x = summary.index.to_numpy(dtype=float)
            ax.plot(x, summary.iloc[:, 0], marker='o', color=COLORS[model], label=model)
            ax.fill_between(x, summary.iloc[:, 1], summary.iloc[:, 2], alpha=.12, color=COLORS[model])
        ax.set(title=method, xlabel='Training observations', ylim=(-1.05, 1.05))
        ax.axhline(0, color='#BBBBBB', linewidth=.7)
        ax.legend(ncol=2)
    axes[0].set_ylabel('Signed Spearman vs. 142-row reference')
    fig.suptitle('Ranking stability: median and empirical 5–95% range')
    fig.tight_layout()
    save('stability_curve')

    feature_summary = tables['feature_summary']
    fig, axes = plt.subplots(len(models), 2, figsize=(14, 3.3 * len(models)), sharex=True)
    for i, model in enumerate(models):
        for j, method in enumerate(['PFI', 'KernelSHAP']):
            ax = axes[i, j]
            for offset, (size, grp) in enumerate(feature_summary[(feature_summary.model == model) &
                                                               (feature_summary.method == method)].groupby('size')):
                grp = grp.set_index('feature').loc[features]
                y = np.arange(len(features)) + (offset - 1) * .22
                ax.errorbar(grp.median_rank, y,
                            xerr=[grp.median_rank - grp.q25_rank, grp.q75_rank - grp.median_rank],
                            fmt='o', markersize=3, capsize=1.5, label=f'n={size}', alpha=.85)
            ax.set_yticks(np.arange(len(features)), features, fontsize=7)
            ax.invert_yaxis()
            ax.set(title=f'{model} / {method}', xlabel='Rank (median and IQR; 1 = highest)', xlim=(.5, 14.5))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=3, loc='upper center')
    fig.tight_layout(rect=(0, 0, 1, .97))
    save('rank_distributions')

    sizes = sorted(feature_summary['size'].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(16, 7), sharey=True)
    pipeline_names = [f'{m}/{a}' for m in models for a in ['PFI', 'KernelSHAP']]
    for ax, size in zip(np.atleast_1d(axes), sizes):
        rows = feature_summary[feature_summary['size'] == size].copy()
        rows['pipeline'] = rows.model + '/' + rows.method
        matrix = rows.pivot(index='feature', columns='pipeline', values='top3_frequency').loc[features, pipeline_names]
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        ax.set_xticks(range(len(pipeline_names)), pipeline_names, rotation=55, ha='right', fontsize=8)
        ax.set_yticks(range(len(features)), features, fontsize=9)
        ax.set_title(f'Top-3 frequency: n={size}')
    fig.subplots_adjust(bottom=.22, left=.19, right=.9, wspace=.15)
    fig.colorbar(im, cax=fig.add_axes([.92, .23, .015, .65]), label='Fractional frequency (ties shared)')
    save('top3_frequency')

    family = tables['family']
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [family.loc[family['size'] == n, 'delta'].dropna().to_numpy() for n in sizes]
    ax.boxplot(values, tick_labels=[str(n) for n in sizes], showfliers=False)
    rng = np.random.default_rng(1412)
    for i, vals in enumerate(values):
        ax.scatter(i + 1 + rng.uniform(-.12, .12, len(vals)), vals, alpha=.55, s=18, color='#2874A6')
    ax.axhline(0, linestyle='--', color='#922B21', label='No within-model advantage')
    ax.set(xlabel='Training observations', ylabel='Within-model rho minus across-model rho',
           title='Does the original same-model agreement pattern persist?')
    ax.legend()
    fig.tight_layout()
    save('family_delta')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, method in zip(axes, ['PFI', 'KernelSHAP']):
        for model in models:
            rows = stability[(stability.model == model) & (stability.method == method)]
            ax.scatter(rows.mae, rows.rho_reference, alpha=.45, s=22, color=COLORS[model], label=model)
        ax.set(title=method, xlabel='Evaluation MAE (J/m²)', ylim=(-1.05, 1.05))
        ax.legend(ncol=2)
    axes[0].set_ylabel('Ranking stability vs. reference')
    fig.suptitle('Prediction accuracy and ranking stability are distinct outcomes')
    fig.tight_layout()
    save('accuracy_stability')
    return output


def write_report(tables: dict, metadata: dict, directory: Path) -> None:
    stability, family, features = tables['stability'], tables['family'], tables['feature_summary']
    protocol = metadata['protocol']
    sizes = sorted(stability['size'].unique())
    counts = family.groupby('size').size()
    model_count = len(metadata['models'])
    pair_count = model_count * (model_count - 1) // 2
    header = '| 模型 / 方法 | ' + ' | '.join(f'n={n}' for n in sizes) + ' |'
    separator = '|---|' + '---:|' * len(sizes)
    sampling_description = '，'.join(f'{n}条×{counts[n]}次' for n in sizes)
    lines = ['# 训练数据变化下的特征重要性稳定性：MSI 实验报告', '',
             '## 实验设置', '',
             f"完成 {metadata['expected_fits']} 次训练，{model_count}个模型 × 两种共同解释方法。训练池{len(protocol['pool'])}条，固定评估集{len(protocol['evaluation'])}条；",
             f"抽样设置：{sampling_description}。KernelSHAP预算为{metadata['kernel_nsamples']}，背景{len(protocol['background'])}条，PFI重复{metadata['pfi_repeats']}次。",
             '模型参数来自原仓库并固定；标准化只在当前训练子集拟合。100%模型只作为参考。', '',
             '## 1. 换训练样本后，排名是否保持？', '',
             '下表为相对142条参考模型的带符号Spearman中位数。1表示排名相同，负值表示倾向反向。', '',
             header, separator]
    for (model, method), rows in stability.groupby(['model', 'method']):
        cells = [f'{rows.loc[rows["size"] == n, "rho_reference"].median():.3f}' for n in sizes]
        lines.append(f'| {model} / {method} | ' + ' | '.join(cells) + ' |')
    lines += ['', '![排名稳定性](figures/stability_curve.png)', '',
              '完整特征排名中位数、IQR和5–95%范围见 feature_summary.csv。小样本重复抽样的排名波动描述了对训练样本选择的敏感性。', '',
              '## 2. “同一模型内解释方法更一致”是否保持？', '',
              f'差值 = {model_count}个模型内部PFI–SHAP相关性的均值 − 两种解释方法各自跨模型{pair_count}个配对相关性的等权均值。', '',
              '| 训练样本 | 差值中位数 | 经验5–95%范围 | 差值为正的比例 | 判断 |', '|---|---:|---|---:|---|']
    for size, grp in family.groupby('size'):
        values = grp.delta
        valid = values.dropna()
        low, high = valid.quantile(.05), valid.quantile(.95)
        if len(valid) != len(values):
            verdict = '存在未定义比较，证据不完整'
        elif low > 0:
            verdict = '本档抽样支持原定性结论'
        elif high < 0:
            verdict = '本档抽样显示相反趋势'
        else:
            verdict = '结论依赖训练样本，不能认为普遍保持'
        positive = (valid > 0).mean() if len(valid) else np.nan
        lines.append(f'| {size} | {valid.median():.3f} | [{low:.3f}, {high:.3f}] | {positive:.1%} | {verdict} |')
    lines += ['', '![家族一致性](figures/family_delta.png)', '', '## 3. 表面能是否持续重要？', '',
              '下表为 surface_energy_M 进入前三的频率；边界并列时按比例分配，所有模型分别报告。', '',
              header, separator]
    for (model, method), rows in features[features.feature == 'surface_energy_M'].groupby(['model', 'method']):
        cells = [f'{rows.loc[rows["size"] == n, "top3_frequency"].iloc[0]:.1%}' for n in sizes]
        lines.append(f'| {model} / {method} | ' + ' | '.join(cells) + ' |')
    lines += ['', '![关键特征频率](figures/top3_frequency.png)', '',
              '## 4. 预测性能与稳定性', '',
              '![准确率与稳定性](figures/accuracy_stability.png)', '',
              'MAE/R²完整结果见 performance.csv；表现差的模型也保留。稳定排名不等于模型正确。', '',
              '## 结论范围与复现', '',
              '- 本研究回答固定MSI数据池、评估划分、模型参数和解释背景下的训练样本敏感性。',
              '- 本项目按模型预先分组；并未重建原文九个事后相关性组，也没有复现全部26条解释流程。',
              '- 原文参数可能已利用全数据选出，MAE/R²是辅助诊断，不是完全独立的泛化评估。',
              '- 5–95%是重复抽样的经验波动范围，不是总体置信区间；重复抽样不是相互独立的新材料数据集。',
              '- 当前未研究材料类别偏移、噪声或因果机制；固定SHAP背景来自训练池，可能不包含在某一小训练子集中。',
              '- 未定义相关性显式标记，不删除弱模型或静默改变比较成员。',
              '- 数据及参数出处、哈希与软件版本见 provenance/source.json、manifest.json和requirements-lock.txt。', '',
              '在项目根目录运行 `python run_experiment.py --stage analysis` 可从检查点重新生成全部图表和报告。']
    (directory / 'report_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def make_notebook(root: Path, directory: Path, metadata: dict) -> None:
    relative = directory.relative_to(root).as_posix()
    p = metadata['protocol']
    sizes = sorted({s['size'] for s in p['subsets'] if not s['is_reference']})
    repeats = len({s['repeat'] for s in p['subsets'] if not s['is_reference']})
    notebook = nbformat.v4.new_notebook()
    notebook.metadata['kernelspec'] = {'display_name': 'Python 3 (project .venv)', 'language': 'python', 'name': 'python3'}
    notebook.cells = [
        nbformat.v4.new_markdown_cell('# MSI 训练数据敏感性分析\n\n读取已完成实验；不会重新训练。请使用项目的 Python 环境。'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nroot = Path.cwd()\nif not (root / 'run_experiment.py').exists():\n    root = root.parent\n" + f"results = root / '{relative}'\nassert (results / 'completion.json').exists()"),
        nbformat.v4.new_markdown_cell(f"## 协议与条件限制\n\n{len(p['pool'])}条训练池，{len(p['evaluation'])}条固定评估；训练量{sizes}，重复{repeats}次，背景{len(p['background'])}条；100%仅为参考。区间表示条件抽样波动。"),
        nbformat.v4.new_code_cell("stability = pd.read_csv(results / 'stability.csv')\nfamily = pd.read_csv(results / 'family.csv')\nfeatures = pd.read_csv(results / 'feature_summary.csv')\nperformance = pd.read_csv(results / 'performance.csv')\ndisplay(stability.groupby(['size', 'model', 'method']).rho_reference.agg(['median', 'min', 'max']))"),
        nbformat.v4.new_code_cell("display(family.groupby('size').delta.agg(['median', 'min', 'max']))\ndisplay(features[features.feature == 'surface_energy_M'][['size', 'model', 'method', 'top3_frequency']])"),
        nbformat.v4.new_code_cell("display(performance.groupby(['size', 'model'])[['mae', 'r2']].median())"),
    ]
    for name in ['stability_curve', 'rank_distributions', 'top3_frequency', 'family_delta', 'accuracy_stability']:
        notebook.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(results / 'figures/{name}.png')))"))
    notebook.cells.append(nbformat.v4.new_code_cell("report = (results / 'report_zh.md').read_text(encoding='utf-8')\nfigure_prefix = (results / 'figures').relative_to(root).as_posix()\nreport = report.replace('](figures/', '](' + figure_prefix + '/')\ndisplay(Markdown(report))"))
    nbformat.validate(notebook)
    nbformat.write(notebook, root / 'analysis.ipynb')


def analyze(directory: Path, root: Path, dataset_path: Path | None = None) -> dict:
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    completion = json.loads((directory / 'completion.json').read_text(encoding='utf-8'))
    metadata = manifest['metadata']
    dataset_path = dataset_path or root / 'data/Science_feature_data.xlsx'
    if not dataset_path.is_file() or hashlib.sha256(dataset_path.read_bytes()).hexdigest() != metadata['dataset_sha256']:
        raise ValueError('Dataset hash differs from the experiment manifest; refusing to map indices to changed data.')
    paths = sorted((directory / 'checkpoints').glob('*.json'))
    if (completion['fingerprint'] != manifest['fingerprint'] or not completion['complete'] or
            len(paths) != metadata['expected_fits']):
        raise ValueError('Incomplete experiment; refusing to report partial results as complete.')
    results = [load_checkpoint(p, manifest['fingerprint']) for p in paths]
    models = list(metadata['models'])
    # Check all expected subset/model keys rather than relying only on file counts.
    expected = {(s['size'], s['repeat'], m) for s in metadata['protocol']['subsets'] for m in models}
    observed = {(r['size'], r['repeat'], r['model']) for r in results}
    if observed != expected:
        raise ValueError('Incomplete expected experiment grid.')
    tables = summarize_results(results, models)
    data = load_dataset(dataset_path)
    sample_rows = []
    for subset in metadata['protocol']['subsets']:
        for index in subset['indices']:
            sample_rows.append({'size': subset['size'], 'repeat': subset['repeat'],
                                'is_reference': subset['is_reference'], 'row_index': index, 'sample_id': data.ids[index]})
    tables['samples'] = pd.DataFrame(sample_rows)
    for name, table in tables.items():
        table.to_csv(directory / f'{name}.csv', index=False, encoding='utf-8-sig')
    make_figures(tables, directory / 'figures', models, metadata['feature_names'])
    write_report(tables, metadata, directory)
    make_notebook(root, directory, metadata)
    verification = {'complete': True, 'fits': len(results),
                    'ranking_vectors': len(results) * 2,
                    'importance_rows': len(tables['importance']),
                    'stability_rows': len(tables['stability']), 'family_rows': len(tables['family']),
                    'undefined_correlations': int((~tables['stability'].correlation_defined).sum()),
                    'undefined_family_comparisons': int((~tables['family'].comparison_defined).sum()),
                    'max_shap_additivity_error': max(r['shap_additivity_max_error'] for r in results)}
    write_checkpoint(directory / 'verification.json', verification)
    print(json.dumps(verification, indent=2), flush=True)
    return verification
