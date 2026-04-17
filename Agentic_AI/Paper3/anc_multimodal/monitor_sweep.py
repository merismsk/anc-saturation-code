#!/usr/bin/env python3
"""Overnight sweep monitor.

Polls the training log every POLL_SEC seconds, parses per-epoch NR snapshots
and completed-seed JSON results, and writes a live status file you can
inspect in the morning:

    outputs/sweep_status.md       — live dashboard, rewritten each tick
    outputs/sweep_history.jsonl   — append-only log of every parsed epoch

Crash detection: if no new log lines appear for STALL_SEC seconds and the
sweep process is gone, writes a CRASH section and posts a macOS notification.

Usage:
    python3 monitor_sweep.py --log outputs/ieee_m4_sweep.log \\
                             --output-root outputs \\
                             --expected-seeds 42 123 456 789 2024 \\
                             --sweep-pid 12345
"""
import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


# ── Parsers ──────────────────────────────────────────────────────────────────

SEED_RE = re.compile(r'Override simulation seed.*?(\d+)|--seed\s+(\d+)|seed[=_]?(\d+)',
                     re.IGNORECASE)
EPOCH_RE = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s*\|\s*Train:\s*([\d.eE+-]+)\s*\|\s*Val:\s*([\d.eE+-]+)'
    r'(?:\s*\|\s*Hybrid NR:\s*([\+\-\d.eE]+)\s*dB\s*\|\s*FxLMS:\s*([\+\-\d.eE]+)\s*dB'
    r'\s*\|\s*Δ:\s*([\+\-\d.eE]+)\s*dB)?'
)
DELTA_RE = re.compile(r'Delta DL vs FxLMS.*?:\s*([\+\-\d.]+)\s*dB.*?W/L/T:\s*(\d+)/(\d+)/(\d+)')
MM_DELTA_RE = re.compile(r'Delta DL vs MM-FxLMS.*?:\s*([\+\-\d.]+)\s*dB.*?W/L/T:\s*(\d+)/(\d+)/(\d+)')
SAVE_DIR_RE = re.compile(r'--save-dir\s+(\S+)')
RESULTS_SAVED_RE = re.compile(r'Results saved to\s+(\S+)/')


def parse_log(path: Path) -> dict:
    """Parse the sweep log incrementally. Returns per-seed state."""
    if not path.exists():
        return {'seeds': [], 'current_seed': None, 'last_mtime': 0}

    state = {
        'seeds': [],              # ordered list of seed dicts
        'current_seed': None,     # seed dict currently training
        'last_mtime': path.stat().st_mtime,
    }
    current = None

    def new_seed(seed_id):
        return {
            'seed': seed_id,
            'save_dir': None,
            'epochs': [],         # list of (epoch, train, val, hybrid_nr, fxlms_nr, delta)
            'data_gen_s': None,
            'fxlms_pretrain_s': None,
            'training_s': None,
            'final_delta_db': None,
            'final_wlt': None,
            'final_mm_delta_db': None,
            'final_mm_wlt': None,
            'status': 'running',  # running | completed | crashed
            'started_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        }

    with path.open() as f:
        for line in f:
            line = line.rstrip('\n')

            # Detect new seed via --save-dir or --seed in command
            m = SAVE_DIR_RE.search(line)
            if m:
                save_dir = m.group(1)
                if 'ieee_a100_seed' in save_dir:
                    seed_id = save_dir.split('ieee_a100_seed')[-1]
                elif 'ieee_m4c_seed' in save_dir:
                    seed_id = save_dir.split('ieee_m4c_seed')[-1]
                elif '/ieee_m4_seed' in save_dir:
                    seed_id = save_dir.split('ieee_m4_seed')[-1]
                else:
                    continue
                if current:
                    state['seeds'].append(current)
                current = new_seed(seed_id)
                current['save_dir'] = save_dir
                continue

            if current is None:
                continue

            # Epoch line
            em = EPOCH_RE.search(line)
            if em:
                ep, tot, train_l, val_l = em.group(1), em.group(2), em.group(3), em.group(4)
                hybrid_nr = em.group(5)
                fxlms_nr = em.group(6)
                delta = em.group(7)
                current['epochs'].append({
                    'epoch': int(ep),
                    'total_epochs': int(tot),
                    'train_loss': float(train_l),
                    'val_loss': float(val_l),
                    'hybrid_nr': float(hybrid_nr) if hybrid_nr else None,
                    'fxlms_nr': float(fxlms_nr) if fxlms_nr else None,
                    'delta_db': float(delta) if delta else None,
                })
                continue

            # Final delta line
            dm = DELTA_RE.search(line)
            if dm:
                current['final_delta_db'] = float(dm.group(1))
                current['final_wlt'] = (int(dm.group(2)), int(dm.group(3)), int(dm.group(4)))
                continue

            mm = MM_DELTA_RE.search(line)
            if mm:
                current['final_mm_delta_db'] = float(mm.group(1))
                current['final_mm_wlt'] = (int(mm.group(2)), int(mm.group(3)), int(mm.group(4)))
                continue

            # Seed completion marker
            if 'Results saved to' in line and current.get('save_dir', '') in line:
                current['status'] = 'completed'
                continue

            # Timing
            if line.startswith('  Data gen: '):
                try:
                    current['data_gen_s'] = float(line.split()[-1].rstrip('s'))
                except ValueError:
                    pass
            elif line.strip().startswith('Training:'):
                try:
                    current['training_s'] = float(line.split()[-1].rstrip('s'))
                except ValueError:
                    pass

    if current:
        state['seeds'].append(current)
        if current['status'] == 'running':
            state['current_seed'] = current
    return state


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def mac_notify(title: str, message: str):
    """Post a macOS notification via osascript (best-effort)."""
    try:
        subprocess.run([
            'osascript', '-e',
            f'display notification "{message}" with title "{title}"'
        ], check=False, timeout=5)
    except Exception:
        pass


# ── Rendering ────────────────────────────────────────────────────────────────

def render_status(state: dict, log_path: Path, sweep_alive: bool,
                  started_at: datetime, expected_seeds: list) -> str:
    now = datetime.utcnow()
    elapsed = now - started_at
    lines = []
    lines.append(f"# Overnight Sweep Monitor")
    lines.append("")
    lines.append(f"**Last update**: {now.isoformat(timespec='seconds')} UTC  ")
    lines.append(f"**Elapsed**:     {str(elapsed).split('.')[0]}  ")
    lines.append(f"**Log file**:    `{log_path}`  ")
    lines.append(f"**Sweep PID**:   {'✅ alive' if sweep_alive else '❌ dead'}  ")
    lines.append("")

    completed = [s for s in state['seeds'] if s['status'] == 'completed']
    running = state.get('current_seed')
    pending = [s for s in expected_seeds if s not in [x['seed'] for x in state['seeds']]]

    lines.append(f"**Progress**: {len(completed)}/{len(expected_seeds)} seeds complete")
    if running:
        lines.append(f"**Currently training**: seed={running['seed']}")
    if pending:
        lines.append(f"**Pending**: {pending}")
    lines.append("")

    # Completed-seed summary table
    if completed:
        lines.append("## Completed seeds")
        lines.append("")
        lines.append("| Seed | Δ vs FxLMS | W/L/T | Δ vs MM-FxLMS | W/L/T | Training |")
        lines.append("|------|------------|-------|----------------|-------|----------|")
        for s in completed:
            d = s['final_delta_db']
            wlt = s['final_wlt']
            md = s['final_mm_delta_db']
            mwlt = s['final_mm_wlt']
            train_s = s.get('training_s')
            lines.append(
                f"| {s['seed']} | "
                f"{d:+.3f} dB | "
                f"{wlt[0]}/{wlt[1]}/{wlt[2]} | "
                f"{md:+.3f} dB | "
                f"{mwlt[0]}/{mwlt[1]}/{mwlt[2]} | "
                f"{train_s:.1f}s |" if d is not None and wlt and md is not None and mwlt and train_s is not None else
                f"| {s['seed']} | (incomplete) |"
            )
        lines.append("")

        # Aggregate stats
        deltas = [s['final_delta_db'] for s in completed if s['final_delta_db'] is not None]
        mm_deltas = [s['final_mm_delta_db'] for s in completed if s['final_mm_delta_db'] is not None]
        if deltas:
            import statistics
            m = statistics.mean(deltas)
            sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            lines.append(f"**Mean Δ vs FxLMS**: {m:+.3f} ± {sd:.3f} dB  ")
        if mm_deltas:
            import statistics
            m = statistics.mean(mm_deltas)
            sd = statistics.stdev(mm_deltas) if len(mm_deltas) > 1 else 0.0
            lines.append(f"**Mean Δ vs MM-FxLMS**: {m:+.3f} ± {sd:.3f} dB  ")
        lines.append("")

    # Current seed — epoch-by-epoch
    if running and running['epochs']:
        lines.append(f"## Current seed={running['seed']} — last 5 epochs")
        lines.append("")
        lines.append("| Epoch | Train | Val | Hybrid NR | FxLMS NR | Δ (probe) |")
        lines.append("|-------|-------|-----|-----------|----------|-----------|")
        for e in running['epochs'][-5:]:
            hn = f"{e['hybrid_nr']:+.2f} dB" if e['hybrid_nr'] is not None else '—'
            fn = f"{e['fxlms_nr']:+.2f} dB" if e['fxlms_nr'] is not None else '—'
            dd = f"{e['delta_db']:+.3f} dB" if e['delta_db'] is not None else '—'
            lines.append(
                f"| {e['epoch']}/{e['total_epochs']} | "
                f"{e['train_loss']:.4f} | {e['val_loss']:.4f} | "
                f"{hn} | {fn} | {dd} |"
            )
        lines.append("")

        # Trend indicator
        deltas = [e['delta_db'] for e in running['epochs'] if e['delta_db'] is not None]
        if len(deltas) >= 2:
            recent = deltas[-1] - deltas[0]
            arrow = '📈' if recent > 0.01 else ('📉' if recent < -0.01 else '➡️')
            lines.append(f"**Trend**: {arrow}  first epoch Δ={deltas[0]:+.3f}, "
                         f"latest Δ={deltas[-1]:+.3f}, change={recent:+.3f} dB")
            lines.append("")

    # Recent log tail for debugging
    try:
        tail = subprocess.run(
            ['tail', '-20', str(log_path)],
            capture_output=True, text=True, timeout=3,
        ).stdout
        lines.append("## Last 20 log lines")
        lines.append("")
        lines.append("```")
        lines.append(tail.rstrip())
        lines.append("```")
    except Exception:
        pass

    return '\n'.join(lines)


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='Path to sweep log file')
    ap.add_argument('--output-root', default='outputs', help='Where to write status files')
    ap.add_argument('--expected-seeds', nargs='+', required=True,
                    help='List of expected seed IDs (for completion tracking)')
    ap.add_argument('--sweep-pid', type=int, default=None,
                    help='PID of the sweep process for liveness check')
    ap.add_argument('--poll-sec', type=int, default=120, help='Polling interval in seconds')
    ap.add_argument('--stall-sec', type=int, default=1800,
                    help='After this many seconds of log silence, mark as stalled')
    args = ap.parse_args()

    log_path = Path(args.log)
    status_path = Path(args.output_root) / 'sweep_status.md'
    history_path = Path(args.output_root) / 'sweep_history.jsonl'
    status_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.utcnow()
    last_log_mtime = 0.0
    last_log_change = time.time()
    crash_notified = False
    stall_notified = False
    completion_notified = False
    seen_completed = set()

    print(f"[monitor] started. log={log_path}  poll={args.poll_sec}s  stall={args.stall_sec}s")
    print(f"[monitor] status will be written to: {status_path}")

    while True:
        try:
            state = parse_log(log_path)

            # Track log freshness
            if state['last_mtime'] > last_log_mtime:
                last_log_mtime = state['last_mtime']
                last_log_change = time.time()

            # Liveness
            sweep_alive = True
            if args.sweep_pid is not None:
                sweep_alive = is_pid_alive(args.sweep_pid)

            # Stall detection
            stall_age = time.time() - last_log_change
            stalled = stall_age > args.stall_sec
            # Notify on stall alone (process may be deadlocked while bash is alive)
            if stalled and not stall_notified:
                mins = int(stall_age / 60)
                mac_notify('ANC sweep STALLED',
                           f'No log progress for {mins} min. Process may be deadlocked.')
                # Also write a marker into the sweep log for visibility
                try:
                    with open(log_path, 'a') as lf:
                        lf.write(f"\n[MONITOR] STALL DETECTED at "
                                 f"{datetime.utcnow().isoformat(timespec='seconds')}Z — "
                                 f"no log progress for {mins} min.\n")
                except Exception:
                    pass
                stall_notified = True
            # Escalate to CRASH label only if process is also dead
            if (stalled and not sweep_alive) and not crash_notified:
                mac_notify('ANC sweep CRASHED',
                           'Sweep process exited without completing.')
                crash_notified = True

            # Newly-completed seeds → append to history + notify
            for s in state['seeds']:
                if s['status'] == 'completed' and s['seed'] not in seen_completed:
                    seen_completed.add(s['seed'])
                    with history_path.open('a') as hf:
                        hf.write(json.dumps({
                            'utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                            'event': 'seed_completed',
                            'seed': s['seed'],
                            'final_delta_db': s['final_delta_db'],
                            'final_mm_delta_db': s['final_mm_delta_db'],
                        }) + '\n')
                    delta_str = (f"Δ={s['final_delta_db']:+.3f} dB"
                                 if s['final_delta_db'] is not None else "no delta")
                    mac_notify('ANC sweep', f"seed={s['seed']} done | {delta_str}")

            # Overall completion
            all_done = len(seen_completed) >= len(args.expected_seeds)
            if all_done and not completion_notified:
                mac_notify('ANC sweep', f'All {len(args.expected_seeds)} seeds complete.')
                completion_notified = True

            # Write dashboard
            dashboard = render_status(
                state=state,
                log_path=log_path,
                sweep_alive=sweep_alive,
                started_at=started_at,
                expected_seeds=args.expected_seeds,
            )
            tmp = status_path.with_suffix('.md.tmp')
            tmp.write_text(dashboard)
            tmp.replace(status_path)

            # Append latest current-epoch to history (for later analysis)
            if state.get('current_seed') and state['current_seed']['epochs']:
                last = state['current_seed']['epochs'][-1]
                with history_path.open('a') as hf:
                    hf.write(json.dumps({
                        'utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                        'event': 'epoch',
                        'seed': state['current_seed']['seed'],
                        **last,
                    }) + '\n')

            if all_done and not sweep_alive:
                print(f"[monitor] all seeds done, sweep exited. Final status in {status_path}")
                break

        except Exception as exc:
            print(f"[monitor] tick error: {exc}")

        time.sleep(args.poll_sec)


if __name__ == '__main__':
    main()
