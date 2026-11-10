"""Run a job file on all local GPUs: one job per free GPU (or several GPUs per job via torchrun), with
asynchronous gates that stop the queue as soon as a ladder rung fails.

usage: python scripts/run_jobs.py <jobs.txt> [--gpus 0,1] [--dry]

Job file lines (blank lines and # comments ignored):
  [gpus=N|half|all] python -m latents.train <cfg> name=<run> ...
        a job; gpus defaults to 1; half = max(1, n_gpus // 2). Multi-GPU jobs are started with torchrun.
  gate <ID> [soft]
        once every job above this line whose name starts with L-<ID>- has finished, run
        `python scripts/ladder.py check <ID>`. On failure: kill every job launched after this line, launch
        nothing more, send a notification and exit 1. Jobs below a gate run speculatively in the meantime.
        soft: on failure only skip (and kill) the jobs up to the next barrier, then continue.
        ladder.py exit code 2 means "results not complete yet": the gate is re-checked later.
  select <ID>
        like gate, but runs `python scripts/ladder.py select <ID>` (picks a phase winner, writes its config).
  barrier
        wait until every launched job has finished and every pending gate has been decided.
  sh <command>
        run a shell command synchronously (put a barrier before it if it needs earlier results).
  hold <name>
        if logs/hold_<name> exists: let everything launched so far finish, notify, and wait until the file is
        removed (e.g. to benchmark milestones before the next phase). No file: no-op.

Logs: logs/<run>.log per job, logs/queue_<jobfile>.jsonl with one record per finished job / gate.
Exit codes of jobs: 0 ok, 3 early-killed (latents.train kill_ref), 4 non-finite loss, other = crash.
"""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLL = 10


def notify(title, msg):
    """Pushover notification if PUSHOVER_TOKEN/PUSHOVER_USER (or the setup.sh aliases) are configured."""
    env = dict(os.environ)
    pf = Path.home() / ".claude" / "pushover.env"
    if pf.exists():
        for line in pf.read_text().splitlines():
            if "=" in line and not line.startswith("export"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    tok = env.get("PUSHOVER_TOKEN") or env.get("PUSHOVER_APPLICATION_KEY")
    usr = env.get("PUSHOVER_USER") or env.get("PUSHOVER_KEY")
    print(f"[notify] {title}: {msg}", flush=True)
    if not (tok and usr):
        return
    try:
        data = urllib.parse.urlencode({"token": tok, "user": usr, "title": title, "message": msg[:1000]}).encode()
        urllib.request.urlopen("https://api.pushover.net/1/messages.json", data=data, timeout=10)
    except Exception as e:  # pragma: no cover
        print("notify failed:", e, flush=True)


def detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout.split()
        return [int(x) for x in out]
    except Exception:
        return []


def parse(path):
    items = []
    for i, raw in enumerate(Path(path).read_text().splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\[gpus=(\w+)\]\s*(.*)$", line)
        gpus = "1"
        if m:
            gpus, line = m.group(1), m.group(2)
        word = line.split()[0]
        if word in ("gate", "select"):
            parts = line.split()
            items.append({"kind": word, "id": parts[1], "soft": "soft" in parts[2:], "line": i})
        elif word == "barrier":
            items.append({"kind": "barrier", "line": i})
        elif word == "hold":
            items.append({"kind": "hold", "name": line.split()[1], "line": i})
        elif word == "sh":
            items.append({"kind": "sh", "cmd": line[3:].strip(), "line": i})
        else:
            nm = re.search(r"\bname=(\S+)", line)
            items.append({"kind": "job", "cmd": line, "gpus": gpus, "name": nm.group(1) if nm else f"job{i}", "line": i})
    return items


class Queue:
    def __init__(self, path, gpus, dry=False):
        self.path = Path(path)
        self.items = parse(path)
        self.gpus = gpus
        self.free = list(gpus)
        self.dry = dry
        self.running = {}  # name -> (Popen, gpus, item, logfile)
        self.done = {}  # name -> exit code
        self.pending = []  # gates/selects waiting for their jobs
        self.stopped = False
        self.skip_block = None  # line of a failed soft gate: skip jobs until the next barrier
        self.port = 29500 + (os.getpid() % 500) * 10
        (ROOT / "logs").mkdir(exist_ok=True)
        self.record = open(ROOT / "logs" / f"queue_{self.path.stem}.jsonl", "a")

    def log(self, **kw):
        kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.record.write(json.dumps(kw) + "\n")
        self.record.flush()
        print(json.dumps(kw), flush=True)

    def n_gpus(self, spec):
        n = len(self.gpus)
        if spec == "half":
            return max(1, n // 2)
        if spec == "all":
            return n
        return min(int(spec), n)

    def launch(self, it):
        n = self.n_gpus(it["gpus"])
        ids, self.free = self.free[:n], self.free[n:]
        cmd = it["cmd"]
        if n > 1:
            self.port += 1
            cmd = re.sub(r"^python3? -m latents\.train",
                         f"torchrun --nnodes=1 --nproc_per_node={n} --rdzv-backend=c10d "
                         f"--rdzv-endpoint=localhost:{self.port} -m latents.train", cmd)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, ids)),
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        logf = ROOT / "logs" / f"{it['name']}.log"
        self.log(event="start", name=it["name"], gpus=ids, cmd=cmd)
        if self.dry:
            self.done[it["name"]] = 0
            self.free += ids
            return
        p = subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, stdout=open(logf, "w"), stderr=subprocess.STDOUT,
                             start_new_session=True)
        self.running[it["name"]] = (p, ids, it, logf)

    def reap(self):
        for name, (p, ids, it, logf) in list(self.running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del self.running[name]
            self.free += ids
            self.done[name] = rc
            tail = ""
            if rc not in (0, 3):
                tail = " | ".join(logf.read_text(errors="replace").splitlines()[-3:])
            self.log(event="end", name=name, rc=rc, status={0: "ok", 3: "killed", 4: "nan"}.get(rc, "crash"), tail=tail)
            if rc not in (0, 3):
                notify(f"queue {self.path.stem}", f"{name} exited {rc}: {tail[-300:]}")

    def kill_after(self, line, until=None):
        for name, (p, ids, it, logf) in list(self.running.items()):
            if it["line"] > line and (until is None or it["line"] < until):
                os.killpg(p.pid, signal.SIGTERM)
                self.log(event="cancel", name=name, reason=f"gate at line {line} failed")

    def check_pending(self):
        for g in list(self.pending):
            prefix = f"L-{g['id']}-"
            mine = [it["name"] for it in self.items if it["kind"] == "job" and it["line"] < g["line"]
                    and it["name"].startswith(prefix)]
            if any(n in self.running for n in mine) or any(n not in self.done for n in mine):
                continue
            self.pending.remove(g)
            if self.dry:
                self.log(event=g["kind"], id=g["id"], ok=True, dry=True)
                continue
            r = subprocess.run([sys.executable, "scripts/ladder.py", "check" if g["kind"] == "gate" else "select", g["id"]],
                               cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 2:  # results not complete yet (e.g. a control run still going): retry later
                self.pending.append(g)
                continue
            ok = r.returncode == 0
            self.log(event=g["kind"], id=g["id"], ok=ok, out=r.stdout[-2000:], err=r.stderr[-1000:])
            if not ok and g.get("soft"):
                self.skip_block = g["line"]
                self.kill_after(g["line"])
                notify(f"queue {self.path.stem}: {g['kind']} {g['id']} failed (soft, skipping block)", r.stdout[-600:])
            elif not ok:
                self.stopped = True
                self.kill_after(g["line"])
                notify(f"queue {self.path.stem}: {g['kind']} {g['id']} FAILED", r.stdout[-600:])
            else:
                notify(f"queue {self.path.stem}: {g['kind']} {g['id']} passed", r.stdout[-300:])

    def tick(self):
        self.reap()
        self.check_pending()
        if not self.dry:
            time.sleep(POLL)

    def run(self):
        for it in self.items:
            if self.stopped:
                break
            if it["kind"] == "job":
                if self.n_gpus(it["gpus"]) == 0:
                    raise SystemExit("no GPUs")
                while len(self.free) < self.n_gpus(it["gpus"]) and not self.stopped and self.skip_block is None:
                    self.tick()
                if self.skip_block is not None and it["line"] > self.skip_block:
                    self.log(event="skip", name=it["name"], reason=f"soft gate at line {self.skip_block} failed")
                    continue
                if not self.stopped:
                    self.launch(it)
            elif it["kind"] in ("gate", "select"):
                self.pending.append(it)
                self.check_pending()
            elif it["kind"] == "barrier":
                while (self.running or self.pending) and not self.stopped:
                    self.tick()
                self.skip_block = None
            elif it["kind"] == "hold":
                flag = ROOT / "logs" / f"hold_{it['name']}"
                if flag.exists() and not self.dry:
                    while (self.running or self.pending) and not self.stopped:
                        self.tick()
                    if self.stopped:
                        break
                    self.log(event="hold", name=it["name"], flag=str(flag.relative_to(ROOT)))
                    notify(f"queue {self.path.stem} on hold", f"remove {flag.relative_to(ROOT)} to continue")
                    while flag.exists():
                        time.sleep(POLL)
                    self.log(event="release", name=it["name"])
            elif it["kind"] == "sh":
                self.log(event="sh", cmd=it["cmd"])
                if not self.dry:
                    rc = subprocess.run(it["cmd"], shell=True, cwd=ROOT).returncode
                    self.log(event="sh_end", cmd=it["cmd"], rc=rc)
                    if rc != 0:
                        self.stopped = True
                        notify(f"queue {self.path.stem}", f"sh failed: {it['cmd']}")
        while self.running or (self.pending and not self.stopped):
            self.tick()
        status = "STOPPED" if self.stopped else "finished"
        notify(f"queue {self.path.stem} {status}", ", ".join(f"{k}:{v}" for k, v in self.done.items())[-900:])
        return 1 if self.stopped else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: all visible)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else detect_gpus()
    if a.dry and not gpus:
        gpus = [0, 1]
    print(f"queue {a.jobs} on GPUs {gpus}", flush=True)
    sys.exit(Queue(a.jobs, gpus, a.dry).run())


if __name__ == "__main__":
    main()
