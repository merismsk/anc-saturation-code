#!/usr/bin/env python3
"""Aggregate multi-seed real-data results from the paper registry + significance tests.

Reads ``outputs/paper_results_registry.json`` (rebuilt by ``train_real_data.py``),
filters by ``--model-type`` and ``--seeds``, loads each run's ``real_data_results.json``
for per-environment NR, prints a Markdown table, runs paired Wilcoxon (hybrid DL vs
FxLMS) per seed on environment-level means, and writes a LaTeX-friendly CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from scipy.stats import wilcoxon
except ImportError as e:
    wilcoxon = None  # type: ignore
    _SCIPY_ERR = e
else:
    _SCIPY_ERR = None


def _load_json(path: str) -> Any:
    with open(path, 'r') as f:
        return json.load(f)


def _resolve_run_path(run_path: str, cwd: str) -> str:
    if os.path.isabs(run_path):
        return run_path
    return os.path.normpath(os.path.join(cwd, run_path))


def _wilcoxon_greater(diffs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Paired one-sided test: median(diff) > 0 (hybrid better than FxLMS)."""
    if wilcoxon is None:
        return None, None
    arr = [float(x) for x in diffs if not math.isnan(x)]
    if len(arr) < 2:
        return None, None
    if all(abs(x) < 1e-12 for x in arr):
        return 0.0, 1.0
    try:
        res = wilcoxon(arr, alternative='greater', zero_method='wilcox', method='auto')
        stat = float(res.statistic) if res.statistic is not None else None
        p = float(res.pvalue) if res.pvalue is not None else None
        return stat, p
    except ValueError:
        return None, None


def _per_env_deltas(payload: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    per = payload.get('per_environment') or {}
    names = sorted(per.keys())
    deltas = []
    for env in names:
        block = per[env]
        dl = block.get('dl_nr_mean')
        fx = block.get('fxlms_nr_mean')
        if dl is None or fx is None:
            deltas.append(float('nan'))
        else:
            deltas.append(float(dl) - float(fx))
    return names, deltas


def _fmt_pm(vals: Sequence[float], nd: int = 3) -> str:
    if not vals:
        return ''
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{statistics.mean(vals):.{nd}f} ± {statistics.stdev(vals):.{nd}f}"


def _registry_row_time(row: Dict[str, Any]) -> datetime:
    s = row.get('registry_updated_utc') or ''
    try:
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--registry',
        default='outputs/paper_results_registry.json',
        help='Registry JSON path (relative to cwd unless absolute)',
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+', required=True,
        help='Seeds to include (e.g. 42 123 456)',
    )
    parser.add_argument(
        '--model-type', required=True,
        help='Filter registry rows by metadata model_type',
    )
    parser.add_argument(
        '--output',
        default='outputs/aggregated',
        help='Output CSV path prefix (writes ``{output}_aggregated.csv`` unless name already ends with _aggregated)',
    )
    parser.add_argument(
        '--run-dir-contains',
        default=None,
        help='If set, only registry rows whose run_dir contains this substring (e.g. mac_dryrun_seed)',
    )
    args = parser.parse_args()

    cwd = os.getcwd()
    reg_path = _resolve_run_path(args.registry, cwd)
    if not os.path.isfile(reg_path):
        print(f"Registry not found: {reg_path}", file=sys.stderr)
        return 1

    if _SCIPY_ERR is not None:
        print(f"Warning: scipy not available ({_SCIPY_ERR}); Wilcoxon p-values omitted.", file=sys.stderr)

    registry = _load_json(reg_path)
    seed_set = set(args.seeds)

    rows_out: List[Dict[str, Any]] = []
    md_lines = [
        f"## Aggregation: ``{args.model_type}`` | seeds={sorted(seed_set)}",
        '',
        '| seed | DL NR | FxLMS NR | MM-FxLMS NR | Δ vs FxLMS | Δ vs MM-FxLMS | W/L/T | Wilcoxon p |',
        '|------|------:|---------:|------------:|-----------:|--------------:|------:|-----------:|',
    ]

    for row in registry:
        if row.get('model_type') != args.model_type:
            continue
        if row.get('seed') not in seed_set:
            continue
        if args.run_dir_contains and args.run_dir_contains not in str(row.get('run_dir', '')):
            continue

        run_json = _resolve_run_path(row['run_path'], cwd)
        if not os.path.isfile(run_json):
            print(f"Skip missing: {run_json}", file=sys.stderr)
            continue
        payload = _load_json(run_json)
        overall = payload.get('overall') or {}

        dl = overall.get('dl_nr_mean')
        fx = overall.get('fxlms_nr_mean')
        mm = overall.get('mm_fxlms_nr_mean')
        d_fx = overall.get('hybrid_minus_fxlms_db')
        d_mm = overall.get('dl_minus_mm_fxlms_db')
        w = overall.get('wins_vs_fxlms')
        l = overall.get('losses_vs_fxlms')
        t = overall.get('ties_vs_fxlms')

        _, deltas = _per_env_deltas(payload)
        stat, p_w = _wilcoxon_greater(deltas)

        p_str = f"{p_w:.4g}" if p_w is not None else '—'
        mm_str = f"{mm:.3f}" if mm is not None else '—'
        d_mm_str = f"{d_mm:+.3f}" if d_mm is not None else '—'

        md_lines.append(
            f"| {row['seed']} | {dl:.3f} | {fx:.3f} | {mm_str} | {d_fx:+.3f} | {d_mm_str} | "
            f"{w}/{l}/{t} | {p_str} |"
        )

        rows_out.append({
            'seed': row['seed'],
            'overall_dl_nr_mean': dl,
            'overall_fxlms_nr_mean': fx,
            'overall_mm_fxlms_nr_mean': mm,
            'hybrid_minus_fxlms_db': d_fx,
            'dl_minus_mm_fxlms_db': d_mm,
            'wins_vs_fxlms': w,
            'losses_vs_fxlms': l,
            'ties_vs_fxlms': t,
            'wilcoxon_statistic': stat,
            'wilcoxon_p_one_sided_greater': p_w,
            'run_path': row.get('run_path'),
            'registry_updated_utc': row.get('registry_updated_utc'),
        })

    if not rows_out:
        print("No matching runs found.", file=sys.stderr)
        return 1

    # One row per seed: keep the registry row with the latest registry_updated_utc.
    by_seed_latest: Dict[int, Dict[str, Any]] = {}
    for r in rows_out:
        s = int(r['seed'])
        prev = by_seed_latest.get(s)
        if prev is None or _registry_row_time(r) > _registry_row_time(prev):
            by_seed_latest[s] = r
    rows_out = [by_seed_latest[s] for s in sorted(by_seed_latest.keys())]

    # Summary statistics
    def pull(key: str) -> List[float]:
        return [float(r[key]) for r in rows_out if r.get(key) is not None]

    d_fx_vals = pull('hybrid_minus_fxlms_db')
    dl_vals = pull('overall_dl_nr_mean')
    fx_vals = pull('overall_fxlms_nr_mean')

    summary = {
        'seed': 'mean_pm_std',
        'overall_dl_nr_mean': _fmt_pm(dl_vals),
        'overall_fxlms_nr_mean': _fmt_pm(fx_vals),
        'overall_mm_fxlms_nr_mean': '',
        'hybrid_minus_fxlms_db': _fmt_pm(d_fx_vals),
        'dl_minus_mm_fxlms_db': '',
        'wins_vs_fxlms': '',
        'losses_vs_fxlms': '',
        'ties_vs_fxlms': '',
        'wilcoxon_statistic': '',
        'wilcoxon_p_one_sided_greater': '',
        'run_path': '',
        'registry_updated_utc': '',
    }
    mm_summ = pull('overall_mm_fxlms_nr_mean')
    if mm_summ:
        summary['overall_mm_fxlms_nr_mean'] = _fmt_pm(mm_summ)
    d_mm_vals = [float(r['dl_minus_mm_fxlms_db']) for r in rows_out if r.get('dl_minus_mm_fxlms_db') is not None]
    if d_mm_vals:
        summary['dl_minus_mm_fxlms_db'] = _fmt_pm(d_mm_vals)

    rows_out.append(summary)

    md_lines.extend(['', '### Summary (mean ± std)', '', f"- **Δ vs FxLMS**: {summary['hybrid_minus_fxlms_db']}", ''])
    print('\n'.join(md_lines))

    out_base = _resolve_run_path(args.output, cwd)
    parent = os.path.dirname(out_base)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if out_base.endswith('_aggregated'):
        csv_path = f"{out_base}.csv"
    else:
        csv_path = f"{out_base}_aggregated.csv"
    fieldnames = list(rows_out[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
