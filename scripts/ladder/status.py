"""One-screen health check of a running ladder / side-track queue.

usage: python scripts/ladder/status.py [--events 8]

Shows GPU utilisation (averaged over a few samples), each running job with its progress, speed, ETA, latest
val BPB and how long ago its log was written (a stale log means a hung job), the last finished jobs / gates,
and any tracebacks. Reads only nvidia-smi and logs/; needs no GPU libraries.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"


def gpu_samples(n=5, dt=1.0):
    rows = {}
    for _ in range(n):
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            i, u, m, mt, p = [x.strip() for x in line.split(",")]
            rows.setdefault(i, []).append((float(u), float(m), float(mt), float(p)))
        time.sleep(dt)
    for i, s in sorted(rows.items()):
        u = [x[0] for x in s]
        print(f"GPU{i}: util avg {sum(u) / len(u):3.0f}% (min {min(u):3.0f})  mem {s[-1][1] / 1024:5.1f}/"
              f"{s[-1][2] / 1024:.0f} GB  power {sum(x[3] for x in s) / len(s):4.0f} W")


def queue_events():
    ev = []
    for f in sorted(LOGS.glob("queue_*.jsonl")):
        for line in f.read_text(errors="replace").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            d["queue"] = f.stem[len("queue_"):]
            ev.append(d)
    ev.sort(key=lambda d: d.get("time", ""))
    return ev


def last_match(lines, pat):
    for line in reversed(lines):
        if re.search(pat, line):
            return line
    return None


def num(line, key):
    if not line:
        return None
    m = re.search(re.escape(key) + r"=([-+0-9.eE]+|nan|inf)", line)
    return float(m.group(1)) if m else None


def job_status(name):
    f = LOGS / f"{name}.log"
    if not f.exists():
        return "no log yet"
    lines = f.read_text(errors="replace").splitlines()
    age = time.time() - f.stat().st_mtime
    parts = [f"log {age:4.0f}s ago"]
    st = last_match(lines, r"^step \d+/\d+")
    if st:
        m = re.match(r"^step (\d+)/(\d+)", st)
        step, total = int(m.group(1)), int(m.group(2))
        sps = num(st, "perf/sec_per_step")
        mfu = num(st, "perf/mfu")
        parts.insert(0, f"step {step}/{total} ({100 * step / total:.0f}%)")
        if sps:
            parts.append(f"{sps:.3f} s/step, ETA {(total - step) * sps / 60:.0f} min")
        if mfu is not None:
            parts.append(f"MFU {100 * mfu:.0f}%")
        loss = num(st, "train/loss") or num(st, "loss")
        if loss is not None:
            parts.append(f"loss {loss:.4f}")
    ok = [ln for ln in lines if re.match(r"^(OK|FAIL)\s", ln)]
    if ok:
        parts.insert(0, f"{sum(ln.startswith('OK') for ln in ok)} OK / {sum(ln.startswith('FAIL') for ln in ok)} FAIL")
    evl = last_match(lines, r"^\[eval\]")
    if evl:
        parts.append(f"val bpb {num(evl, 'val/bpb')}")
    if any("Traceback" in ln for ln in lines):
        parts.append("TRACEBACK in log")
    if age > 600:
        parts.append("STALE (>10 min without output)")
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=8)
    a = ap.parse_args()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    gpu_samples()
    procs = subprocess.run(["pgrep", "-af", "run_jobs.py"], capture_output=True, text=True).stdout
    qs = sorted({m.group(1) for m in re.finditer(r"scripts/ladder/(\w+)\.txt", procs)})
    print("active queue:", ", ".join(qs) if qs else "NONE (queue not running)")
    ev = queue_events()
    started, ended = {}, set()
    for d in ev:
        if d.get("dry"):
            continue
        if d.get("event") == "start":
            started[d["name"]] = d
        elif d.get("event") in ("end", "cancel"):
            ended.add(d["name"])
    running = [d for n, d in started.items() if n not in ended]
    print(f"running jobs ({len(running)}):")
    for d in running:
        print(f"  [{d['queue']}] {d['name']} on GPU {','.join(map(str, d['gpus']))} since {d['time'][11:]}: "
              f"{job_status(d['name'])}")
    print("recent events:")
    for d in [d for d in ev if d.get("event") in ("end", "gate", "select", "skip", "cancel", "sh_end")][-a.events:]:
        e = d["event"]
        if e == "end":
            s = f"{d['name']} {d['status']} (rc {d['rc']})" + (f": {d['tail'][-160:]}" if d.get("tail") else "")
        elif e in ("gate", "select"):
            s = f"{e} {d['id']} {'PASSED' if d['ok'] else 'FAILED'}"
        elif e == "sh_end":
            s = f"sh rc={d['rc']}: {d['cmd'][:80]}"
        else:
            s = f"{e} {d['name']}: {d.get('reason', '')}"
        print(f"  {d['time'][11:]} [{d['queue']}] {s}")


if __name__ == "__main__":
    main()
