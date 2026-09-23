"""Read-only provenance and structured-artifact adapters for the log scrubber."""
import csv
import json
from functools import lru_cache
from pathlib import Path

LABELS = ['test_suite', 'test_name', 'test_root', 'matrix_identity']
BUNDLE_TABLES = ['decisions', 'attempts', 'mode_coverage', 'mode_mechanism', 'transport_costs']
REPORT_TABLES = ['primary_summary', 'all_comparisons', 'pairs', 'summary', 'summary_per_seed',
                 'mechanism_per_seed', 'mechanism_summary', 'vertex_reachability',
                 'concentration_per_solve', 'concentration_summary']
RARE_TABLES = ['weights', 'scene', 'coverage_trials', 'controller_trials', 'attempts', 'plans', 'plan_risk']


@lru_cache(maxsize=None)
def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def context(path, root):
    """Nearest experiment root wins when scrubbing a parent with multiple suites."""
    for parent in [path.parent, *path.parent.parents]:
        if not parent.is_relative_to(root):
            break
        matrix = parent/'matrix.json'
        rare = (parent/'coverage_trials.csv').exists() or (parent/'artifacts/coverage_trials.csv').exists()
        if matrix.exists():
            manifest = read_json(matrix)
            suite = manifest.get('suite', 'analysis_matrix')
            settings = manifest.get('settings', {})
            if suite == 'comparison_matrix':
                if settings.get('overrides', {}).get('safe_horizon_enabled') is False:
                    suite = 'fixed_budget_uncertified'
                elif settings.get('scenario_budgets'):
                    suite = 'sample_efficiency'
            return dict(test_suite=suite, test_name=parent.name, test_root=str(parent),
                        matrix_identity=manifest.get('identity', ''))
        if rare:
            # Prefer the wrapper root over its artifacts child.
            if parent.name == 'artifacts' and (parent.parent/'manifest.json').exists():
                parent = parent.parent
            return dict(test_suite='rare_mode', test_name=parent.name, test_root=str(parent), matrix_identity='')
    return dict(test_suite='legacy_logs', test_name=root.name, test_root=str(root), matrix_identity='')


def enrich_identity(log, root, identity):
    result_path = log.parent/'result.json'
    metadata = read_json(result_path) if result_path.exists() else {}
    identity.update(context(log, root))
    identity['repeat'] = log.stem.removeprefix('repeat_') if log.stem.startswith('repeat_') else ''
    identity['trial_status'] = metadata.get('status', 'UNKNOWN')
    identity['trial_error'] = metadata.get('error', '')
    identity['repeatable'] = metadata.get('repeatable', '')
    for field in ['case', 'solver_style', 'profile', 'scenario_budget', 'environment', 'obstacles', 'classes']:
        if field in metadata: identity[field] = metadata[field]
    if 'pair' in metadata: identity['pair_case'] = metadata['pair']
    # Paired layout: setup/seed_N/controller/repeat_K.log (also works before result.json).
    if 'solver_style' not in identity and log.parent.parent.name.startswith('seed_'):
        identity['solver_style'] = log.parent.name
        identity['case'] = log.parent.name + '_' + identity['pair_case']
    if 'solver_style' not in identity:
        for style in ['sh_mpcc_dro_fallback', 'sh_mpcc_resample', 'sh_mpcc_extra', 'sh_mpcc_dro', 'sh_mpcc']:
            if Path(identity['case']).name.startswith(style+'_'):
                identity['solver_style'] = style
                break
    bundle = log.with_suffix('')
    config = bundle/'resolved_config.yaml'
    if config.exists():
        for line in config.read_text().splitlines():
            key, sep, value = line.partition(':')
            if sep and key in ['safe_horizon_enabled', 'automatically_compute_sample_size', 'num_scenarios', 'nominal_resampling_baseline']:
                identity[key] = value.strip()
        identity['certification_status'] = ('not_requested' if identity.get('safe_horizon_enabled') == 'false'
                                            else 'see_decisions')
    return identity


def artifact_tables(log, identity):
    bundle = log.with_suffix('')
    tables = {}
    for name in BUNDLE_TABLES:
        path = bundle/(name+'.csv')
        if path.exists():
            tables[name] = [dict(row, **{k:v for k,v in identity.items() if k not in row},
                                 source_artifact=str(path)) for row in read_csv(path)]
    return tables


def collect_reports(root, output):
    """Export existing authoritative reports; do not recompute selected-pair statistics."""
    tables = {}
    matrices = sorted(p for p in root.rglob('matrix.json') if not p.is_relative_to(output))
    for manifest in matrices:
        labels = context(manifest, root)
        for name in REPORT_TABLES:
            source = manifest.parent/(name+'.csv')
            if source.exists():
                tables.setdefault('report_'+name, []).extend(
                    dict(row, **labels, source_artifact=str(source)) for row in read_csv(source))
    for marker in sorted(root.rglob('coverage_trials.csv')):
        if marker.is_relative_to(output): continue
        labels = context(marker, root)
        for name in RARE_TABLES:
            source = marker.parent/(name+'.csv')
            if source.exists():
                tables.setdefault('rare_'+name, []).extend(
                    dict(row, **labels, source_artifact=str(source)) for row in read_csv(source))
    return tables
