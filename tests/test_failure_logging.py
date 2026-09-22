#!/usr/bin/env python3
"""Exercise failed-decision diagnostics through the runner and matrix analysis."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

spec = importlib.util.spec_from_file_location('matrix', Path(__file__).with_name('run_analysis_matrix.py'))
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def main():
    runner = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix='failure-logging-') as tmp:
        root = Path(tmp)
        text = (matrix.ROOT/'configs/base_tests/sh_mpcc_1_obstacles.yaml').read_text()
        # Explicitly blocked start exercises rejection; no safety-performance inference.
        text = text.replace('[12.5,0,0,0]', '[0,0,0,0]').replace('rollout_steps: 450', 'rollout_steps: 1')
        config = root/'blocked.yaml'
        config.write_text(text+'\nartifact_write_analysis_csv: true\n')
        command = [str(runner), '--config', str(config), '--seed', '77', '--output',
                   str(root), '--label', 'blocked', '--no-svg', '--no-gif', '--no-rviz']
        run = subprocess.run(command, cwd=matrix.ROOT, capture_output=True, text=True, check=True)
        case = dict(case='blocked', obstacles=1, classes=1, environment='s_curve',
                    modes_per_class=1, solver_style='sh_mpcc')
        metrics = matrix.analyze(root/'blocked', case)
        assert metrics['termination_reason'] == 'no_admissible_control', metrics
        assert metrics['executed_steps'] == 0, metrics
        assert metrics['backup_available'] == 0, metrics
        assert metrics['backup_dro_failed'] == -1, metrics
        assert metrics['braking_collision_feasible'] == -1, metrics
        assert metrics['any_homotopy_geometrically_feasible'] == -1, metrics
        assert metrics['sqp_sampled_collision_feasible'] == 0, metrics
        assert metrics['fallback_sampled_collision_feasible'] == 0, metrics
        assert metrics['last_qp_converged'] in (0, 1), metrics
        assert metrics['failure_class'] in ('sampled_collision',
                    'solver_nonconvergence_and_sampled_collision'), metrics
        summary = matrix.aggregate([{**case, 'status': 'OK',
            'repeats': [{'metrics': metrics}]}], [case], 1)[0]
        assert summary['no_admissible_control_rollouts'] == 1, summary
        assert summary['failure_backup_dro_failed_unknown_rollouts'] == 1, summary
        for line in run.stderr.splitlines():
            if '[FAILURE CLASSIFICATION]' in line:
                print(line)
                for field in matrix.FAILURE_FIELDS:
                    assert f'{field}={metrics[field]}' in line, (field, line)
                break
        else:
            raise AssertionError('missing failure log')
        print('PASS: runner log, decision CSV, matrix metrics and aggregate failure counts')


if __name__ == '__main__':
    main()
