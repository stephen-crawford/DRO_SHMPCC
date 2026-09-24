#!/usr/bin/env python3
"""Report empirical mode-bound distributions, including WDRO mass decreases."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(rows):
    result = []
    for case in ['ALL'] + sorted({r['case'] for r in rows}):
        for shift in ['all', 'increased', 'decreased', 'unchanged']:
            selected = [r for r in rows if (case == 'ALL' or r['case'] == case)
                        and (shift == 'all' or r['mass_shift'] == shift)]
            if not selected:
                continue
            b = np.array([float(r['b_mode']) for r in selected])
            result.append(dict(case=case, mass_shift=shift, count=len(b),
                               fraction_b_lt_one=float(np.mean(b < 1)), median=float(np.median(b)),
                               q90=float(np.quantile(b, .9)), q95=float(np.quantile(b, .95)),
                               max_pair_prob=max(float(r['max_pair_prob']) for r in selected),
                               median_sum_pair_prob=float(np.median([float(r['sum_pair_prob']) for r in selected]))))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    with args.input.open() as f:
        rows = list(csv.DictReader(f))
    result = summarize(rows)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        if args.output.exists():
            p.error('use a new output filename')
        args.output.write_text(text)
    print(text, end='')

if __name__ == '__main__':
    main()
