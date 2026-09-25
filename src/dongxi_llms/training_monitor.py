"""Loopback log viewer with optional bounded inference; never training job controls."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, urlsplit

LIMIT = 2 * 1024 * 1024
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def clean(value):
    """Keep wire JSON valid even when a failed experiment logged NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


class LogStore:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)

    def path(self, run, name):
        if not RUN_NAME.fullmatch(run):
            raise ValueError("Invalid run name")
        folder = self.root / run
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Unknown run")
        target = folder / name
        if target.is_symlink() or target.resolve().parent != folder.resolve():
            raise ValueError("Symlinks and external paths are not allowed")
        return target

    def read(self, run, name, lines=False):
        path = self.path(run, name)
        if not path.is_file():
            return ([] if lines else {}), None, False
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > LIMIT:
                if not lines:
                    raise ValueError("Oversized metadata")
                f.seek(-LIMIT, 2)
                f.readline()  # drop the possibly partial first record
            raw = f.read(LIMIT)
        if not lines:
            try:
                value = json.loads(raw)
                return value if isinstance(value, dict) else {}, path.stat().st_mtime, False
            except (ValueError, UnicodeError):
                return {}, path.stat().st_mtime, True
        records = []
        incomplete = size > LIMIT
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                incomplete = True
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    records.append(row)
            except (ValueError, UnicodeError):
                incomplete = True
        return records, path.stat().st_mtime, incomplete

    def runs(self):
        result = []
        for folder in self.root.iterdir():
            if not folder.is_dir() or folder.is_symlink() or not RUN_NAME.fullmatch(folder.name):
                continue
            try:
                path = self.path(folder.name, "run.json")
                if path.is_file():
                    result.append(dict(name=folder.name, modified=path.stat().st_mtime))
            except (ValueError, OSError):
                continue
        return sorted(result, key=lambda r: r["modified"], reverse=True)

    def snapshot(self, run):
        now = time.time()
        config, _, config_partial = self.read(run, "run.json")
        if not config:
            raise ValueError("Run metadata not available yet")
        metrics, update_time, metrics_partial = self.read(run, "metrics.jsonl", True)
        memory, memory_time, memory_partial = self.read(run, "host-memory.jsonl", True)
        summary, _, _ = self.read(run, "memory-summary.json")
        final, _, _ = self.read(run, "final.json")
        snapshots = []
        names = ["initial.json"]
        folder = self.path(run, "run.json").parent
        names += sorted(p.name for p in folder.glob("update-*.json")
                        if re.fullmatch(r"update-\d+\.json", p.name))[-200:]
        names += ["final.json"]
        partial = config_partial or metrics_partial or memory_partial
        for name in names:
            item, _, pending = self.read(run, name)
            partial |= pending
            if item:
                snapshots.append({"name": name, **item})
        snapshots.sort(key=lambda r: r.get("update", -1))
        latest_age = None if memory_time is None else max(0., now-memory_time)
        if summary.get("abort_reason"):
            state = "Aborted: " + str(summary["abort_reason"])
        elif final:
            state = "Final observation saved"
        elif latest_age is not None and latest_age < 15:
            state = "Recent telemetry — process status unverified"
        else:
            state = "Stale / inactive telemetry — inspect training process"
        warnings = []
        if partial:
            warnings.append("Some records are partial, malformed or outside the bounded recent-log window.")
        if config.get("data_manifest", {}).get("subset"):
            warnings.append("Dataset subset: smoke evidence, not a full-corpus learning result.")
        reserve = summary.get("reserve_gib", 25.)
        finite_memory = [r["available_gib"] for r in memory
                         if isinstance(r.get("available_gib"), (int, float)) and math.isfinite(r["available_gib"])]
        if finite_memory and min(finite_memory) < reserve:
            warnings.append("Recorded host memory crossed the reserve threshold.")
        for row in metrics:
            if any(isinstance(row.get(k), (int, float)) and not math.isfinite(row[k])
                   for k in ("loss", "gradient_norm")):
                warnings.append("Nonfinite loss or gradient recorded; inspect the training log.")
                break
        checkpoints = []
        for path in sorted(folder.glob("update-*.pt"))[-200:]:
            if not path.is_symlink() and re.fullmatch(r"update-\d+\.pt", path.name):
                checkpoints.append(dict(name=path.name, bytes=path.stat().st_size))
        # Memory history can be dense: bin while preserving min/max, never invent points.
        stride = max(1, math.ceil(len(memory)/600))
        memory_plot = []
        for start in range(0, len(memory), stride):
            rows = [r for r in memory[start:start+stride] if isinstance(r.get("available_gib"), (int, float))]
            if rows:
                pair = [min(rows, key=lambda r: r["available_gib"]), max(rows, key=lambda r: r["available_gib"])]
                memory_plot.extend(sorted(pair, key=lambda r: r.get("seconds", 0)))
        return clean(dict(run=run, state=state, warnings=warnings, config=config,
                          metrics=metrics, snapshots=snapshots, memory=memory_plot,
                          memory_summary=summary, checkpoints=checkpoints,
                          reserve_gib=reserve, memory_age_seconds=latest_age,
                          update_age_seconds=None if update_time is None else max(0., now-update_time),
                          generated_at=now, recent_log_limit_bytes=LIMIT))


def handler(store, assets, playground=None):
    assets = Path(assets)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if urlsplit(self.path).path != '/api/generate' or playground is None:
                self.send_error(501)
                return
            host = urlsplit('http://' + self.headers.get('Host', '')).hostname
            origin = self.headers.get('Origin')
            if (host not in ('localhost', '127.0.0.1', '::1') or
                (origin and urlsplit(origin).hostname not in ('localhost', '127.0.0.1', '::1')) or
                self.headers.get('X-Playground-Request') != '1' or
                self.headers.get('Content-Type', '').split(';')[0] != 'application/json'):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384:
                    raise ValueError('Request must be at most16 KiB.')
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(size))
                result = playground.generate(body)
                status = 200
            except (ValueError, UnicodeError) as error:
                result, status = {'error': str(error)}, 400
            except BlockingIOError as error:
                result, status = {'error': str(error)}, 409
            except Exception:
                result, status = {'error': 'Generation could not complete safely. Inspect the local service log.'}, 503
            payload = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            host = urlsplit("http://" + self.headers.get("Host", "")).hostname
            origin = self.headers.get("Origin")
            if host not in ("127.0.0.1", "localhost", "::1") or (origin and urlsplit(origin).hostname not in ("127.0.0.1", "localhost", "::1")):
                self.send_error(403)
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == '/api/playground':
                    payload = json.dumps(playground.info() if playground else {'enabled': False}).encode()
                    kind = 'application/json'
                elif parsed.path == "/api/runs":
                    payload = json.dumps(store.runs()).encode()
                    kind = "application/json"
                elif parsed.path == "/api/run":
                    run = parse_qs(parsed.query).get("name", [""])[0]
                    payload = json.dumps(store.snapshot(run), allow_nan=False).encode()
                    kind = "application/json"
                elif parsed.path in ("/", "/monitor.css", "/monitor.js", "/playground.css"):
                    name = "index.html" if parsed.path == "/" else parsed.path[1:]
                    payload = (assets/name).read_bytes()
                    kind = {"index.html": "text/html", "monitor.css": "text/css", "playground.css": "text/css", "monitor.js": "text/javascript"}[name]
                else:
                    self.send_error(404)
                    return
            except (ValueError, OSError):
                self.send_error(404, "Run or asset unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="outputs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument('--playground-run', help='Opt in to inference using this completed run only')
    args = parser.parse_args()
    assets = Path(__file__).resolve().parents[2]/"visuals"/"training-monitor"
    store = LogStore(args.runs)
    playground = None
    if args.playground_run:
        from dongxi_llms.stories_playground import Playground
        from dongxi_llms.stories_data import digest
        config, _, _ = store.read(args.playground_run, 'run.json')
        completion, _, _ = store.read(args.playground_run, 'completion.json')
        if not completion.get('schedule_complete'):
            raise ValueError('Only a completed training schedule may be loaded into the playground')
        checkpoint = store.path(args.playground_run, f"update-{completion['completed_updates']:06d}.pt")
        # Explicit server-side choice; clients cannot submit checkpoint or tokenizer paths.
        import subprocess
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
            raise RuntimeError('Another GPU process is active; refuse automatic model loading')
        tokenizer_path = Path('data/cache/day09-full-v2/tokenizer.json').resolve()
        if digest(tokenizer_path) != config['data_manifest']['tokenizer_sha256']:
            raise ValueError('Tokenizer hash mismatch')
        playground = Playground(checkpoint, tokenizer_path, args.playground_run)
        print(json.dumps(playground.info()), flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(store, assets, playground))
    print(f"Training monitor: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
