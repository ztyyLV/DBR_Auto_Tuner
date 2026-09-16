"""Local web UI: ``dbr-autotune ui``.

A single-page front end over :mod:`autotune.api`, served by the standard
library so the package stays dependency-free. Tuning runs in a worker thread and
streams progress to the browser over Server-Sent Events.

Security posture - this server can read directories and start long jobs, so:

* it binds to 127.0.0.1 only, never to a routable address;
* every API call must carry a token minted at startup and printed in the URL,
  which stops another page in the same browser from driving it (a web page can
  POST to localhost, but cannot read our token);
* requests whose ``Origin`` is a real website are refused outright.

None of that makes it safe to expose publicly. Do not put it behind a tunnel.
"""
from __future__ import annotations

import json
import mimetypes
import os
import queue
import secrets
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from . import dataset as ds
from .api import TuneRequest, tune
from .engine import image_stats
from .config import IMAGE_EXT
from .crossval import cross_validate
from .repro import environment_fingerprint

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


@dataclass
class Run:
    id: str
    kind: str                                   # "tune" | "crossval"
    request: TuneRequest
    k: int = 5
    status: str = "running"                     # running | done | failed | cancelled
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    _listeners: List["queue.Queue"] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def publish(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener.put_nowait(event)
            except Exception:
                pass

    def subscribe(self) -> "queue.Queue":
        listener: queue.Queue = queue.Queue()
        with self._lock:
            backlog = list(self.events)
            self._listeners.append(listener)
        for event in backlog:                  # late joiners see the whole story
            listener.put_nowait(event)
        return listener

    def unsubscribe(self, listener: "queue.Queue") -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def meta(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "started": self.started, "finished": self.finished,
                "images": self.request.images, "out": self.request.out,
                "error": self.error,
                "elapsed_s": round((self.finished or time.time()) - self.started, 1)}


class Cancelled(RuntimeError):
    pass


class RunManager:
    def __init__(self) -> None:
        self.runs: Dict[str, Run] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, request: TuneRequest, k: int = 5) -> Run:
        run = Run(id=secrets.token_hex(6), kind=kind, request=request, k=k)
        with self._lock:
            self.runs[run.id] = run
        threading.Thread(target=self._work, args=(run,), daemon=True).start()
        return run

    def get(self, run_id: str) -> Optional[Run]:
        return self.runs.get(run_id)

    def list(self) -> List[Dict[str, Any]]:
        return sorted((r.meta() for r in self.runs.values()),
                      key=lambda m: m["started"], reverse=True)

    def _work(self, run: Run) -> None:
        def log(message: str = "") -> None:
            for line in str(message).split("\n"):
                run.logs.append(line)
                run.publish({"event": "log", "line": line})

        def progress(event: Dict[str, Any]) -> None:
            if run.cancel.is_set():
                raise Cancelled()
            run.publish(event)

        try:
            if run.kind == "crossval":
                outcome = cross_validate(run.request, k=run.k, log=log, progress=progress)
                run.summary = outcome.to_json()
                if run.request.out:
                    os.makedirs(run.request.out, exist_ok=True)
                    path = os.path.join(run.request.out, "cross-validation.json")
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump(run.summary, handle, indent=2, ensure_ascii=False)
            else:
                result = tune(run.request, log=log, progress=progress)
                run.summary = result.summary
            run.status = "done"
        except Cancelled:
            run.status = "cancelled"
            log("run cancelled")
        except Exception as exc:
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            log(run.error)
            log(traceback.format_exc())
        finally:
            run.finished = time.time()
            run.publish({"event": "finished", "status": run.status,
                         "summary": run.summary, "error": run.error})


def _drives() -> List[str]:
    """Roots to offer in the picker: drive letters on Windows, / elsewhere."""
    if os.name != "nt":
        return ["/"]
    import string
    return [f"{letter}:\\" for letter in string.ascii_uppercase
            if os.path.exists(f"{letter}:\\")]


def _shortcuts() -> List[Dict[str, str]]:
    """The handful of folders people actually keep images in."""
    home = os.path.expanduser("~")
    candidates = [("Home", home)]
    for name in ("Desktop", "Downloads", "Pictures", "Documents"):
        candidates.append((name, os.path.join(home, name)))
    return [{"name": name, "path": path}
            for name, path in candidates if os.path.isdir(path)]


def _count_images(path: str, cap: int = 4000) -> int:
    """Image files directly inside ``path``. Not recursive - this runs for every
    subfolder of the directory being listed, so it has to stay cheap."""
    total = 0
    try:
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= cap:
                    break
                if (entry.is_file(follow_symlinks=False)
                        and os.path.splitext(entry.name)[1].lower() in IMAGE_EXT):
                    total += 1
    except (OSError, PermissionError):
        return 0
    return total


def browse(path: Optional[str], show_hidden: bool = False) -> Dict[str, Any]:
    """Directory listing for the folder picker.

    Every subfolder is reported with the number of images directly inside it, so
    you can see where your pictures are without walking into each one. Image
    files in the current folder are listed too - a run can target individual
    files, not only whole folders.
    """
    if not path:
        shortcuts = _shortcuts()
        # Start somewhere with images in it rather than dumping the user in a
        # home directory full of dot-folders.
        best = max((s for s in shortcuts), key=lambda s: _count_images(s["path"]),
                   default=None)
        path = best["path"] if best and _count_images(best["path"]) else os.path.expanduser("~")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        return {"error": f"not a directory: {path}", "path": path,
                "shortcuts": _shortcuts(), "drives": _drives()}

    folders: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                hidden = entry.name.startswith(".")
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if hidden and not show_hidden:
                            continue
                        folders.append({"name": entry.name,
                                        "path": entry.path,
                                        "images": _count_images(entry.path)})
                    elif os.path.splitext(entry.name)[1].lower() in IMAGE_EXT:
                        files.append({"name": entry.name, "path": entry.path,
                                      "kb": round(entry.stat().st_size / 1024)})
                except (OSError, PermissionError):
                    continue
    except PermissionError:
        return {"error": f"permission denied: {path}", "path": path,
                "shortcuts": _shortcuts(), "drives": _drives()}

    folders.sort(key=lambda f: f["name"].lower())
    files.sort(key=lambda f: f["name"].lower())
    hidden_count = 0
    if not show_hidden:
        try:
            hidden_count = sum(1 for n in os.listdir(path)
                               if n.startswith(".") and os.path.isdir(os.path.join(path, n)))
        except OSError:
            hidden_count = 0

    parent = os.path.dirname(path)
    return {"path": path, "parent": parent if parent != path else None,
            "folders": folders[:500], "files": files[:500],
            "images_here": len(files), "hidden_folders": hidden_count,
            "shortcuts": _shortcuts(), "drives": _drives()}


# Measured on two real runs (12 MP phone photos and small sample scans) with
# the default worker count: cost per page-decode is a fixed overhead plus a
# term proportional to resolution.
_SECONDS_PER_PAGE_DECODE = 0.05          # fixed, per page per configuration
_SECONDS_PER_MEGAPIXEL = 0.023
_TRIALS = {"quick": 60, "full": 115, "deep": 210}


def recommend(pages: int, files: int, megapixels: float,
              effort: str = "full") -> Dict[str, Any]:
    """Settings a first-time user should not have to think about.

    Holding pages back is how overfitting gets caught, but it also takes them
    away from the search. Below roughly twenty files the loss outweighs the
    check, and the honest page-coverage figure already tells the same story, so
    it is only switched on once there is enough data for it to mean something.
    """
    if files >= 20:
        holdout, why = 0.25, f"{files} files is enough to hold a quarter back as a check"
    else:
        holdout, why = 0.0, (f"only {files} file(s) - all of them are used for tuning, "
                             f"since holding any back would cost more than it proves")

    trials = _TRIALS.get(effort, _TRIALS["full"])
    per_decode = _SECONDS_PER_PAGE_DECODE + _SECONDS_PER_MEGAPIXEL * max(megapixels, 0.1)
    tuning_pages = max(1, int(pages * (1 - holdout)))
    seconds = trials * tuning_pages * per_decode
    return {"holdout": holdout, "holdout_why": why,
            "estimate_seconds": int(seconds),
            "estimate_text": _duration(seconds)}


def _duration(seconds: float) -> str:
    if seconds < 90:
        return "under a minute"
    minutes = seconds / 60.0
    if minutes < 10:
        return f"about {round(minutes)} minutes"
    if minutes < 60:
        return f"about {int(round(minutes / 5.0) * 5)} minutes"
    return f"about {minutes / 60.0:.1f} hours"


def preview(images: List[str], recursive: bool = True,
            limit: Optional[int] = None, effort: str = "full") -> Dict[str, Any]:
    """What a run would actually process, before committing to it."""
    try:
        data = ds.discover(images, recursive=recursive, limit=limit)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    if not len(data):
        return {"pages": 0, "files": 0, "skipped": len(data.skipped),
                "uncounted": [], "sample_labels": []}

    profile = image_stats(data)
    megapixels = profile.get("megapixels") or 1.0
    files = len({s.path for s in data.samples})
    return {
        "pages": len(data),
        "files": files,
        "megapixels": megapixels,
        "skipped": len(data.skipped),
        "uncounted": [os.path.basename(p) for p in data.uncounted],
        "sample_labels": [s.label for s in data.samples[:12]],
        "recommended": recommend(len(data), files, megapixels, effort),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "dbr-autotune"
    manager: RunManager
    token: str

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:        # keep the console clean
        pass

    def _authorised(self, params: Dict[str, List[str]]) -> bool:
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
            return False
        supplied = (self.headers.get("X-Autotune-Token")
                    or (params.get("token") or [""])[0])
        return secrets.compare_digest(supplied or "", self.token)

    def _send(self, code: int, payload: Any, content_type: str = "application/json") -> None:
        body = (json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if content_type == "application/json" else payload)
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:                               # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        route = parsed.path

        if route in ("/", "/index.html"):
            return self._serve_page()
        if route.startswith("/static/"):
            return self._serve_static(route[len("/static/"):])

        if not self._authorised(params):
            return self._send(403, {"error": "bad or missing token"})

        if route == "/api/browse":
            return self._send(200, browse(
                (params.get("path") or [None])[0],
                show_hidden=(params.get("hidden") or ["0"])[0] == "1"))
        if route == "/api/preview":
            images = [i for i in (params.get("images") or []) if i.strip()]
            if not images:
                return self._send(200, {"error": "no path given"})
            return self._send(200, preview(
                images,
                (params.get("recursive") or ["1"])[0] != "0",
                effort=(params.get("effort") or ["full"])[0]))
        if route == "/api/runs":
            return self._send(200, {"runs": self.manager.list(),
                                    "environment": environment_fingerprint()})
        if route.startswith("/api/run/"):
            run = self.manager.get(route.rsplit("/", 1)[-1])
            if not run:
                return self._send(404, {"error": "no such run"})
            return self._send(200, {**run.meta(), "summary": run.summary,
                                    "logs": run.logs[-400:]})
        if route.startswith("/api/events/"):
            return self._stream(route.rsplit("/", 1)[-1])
        if route.startswith("/api/report/"):
            return self._serve_report(route.rsplit("/", 1)[-1])
        return self._send(404, {"error": "not found"})

    def do_POST(self) -> None:                              # noqa: N802
        parsed = urlparse(self.path)
        if not self._authorised(parse_qs(parsed.query)):
            return self._send(403, {"error": "bad or missing token"})
        body = self._body()

        if parsed.path == "/api/start":
            try:
                request = TuneRequest.from_json(body.get("request") or {})
            except TypeError as exc:
                return self._send(400, {"error": f"bad request: {exc}"})
            if not request.images:
                return self._send(400, {"error": "no images given"})
            kind = "crossval" if body.get("cross_validate") else "tune"
            run = self.manager.start(kind, request, k=int(body.get("k") or 5))
            return self._send(200, run.meta())

        if parsed.path.startswith("/api/cancel/"):
            run = self.manager.get(parsed.path.rsplit("/", 1)[-1])
            if not run:
                return self._send(404, {"error": "no such run"})
            run.cancel.set()
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})

    # -- responses ---------------------------------------------------------

    def _serve_page(self) -> None:
        try:
            with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as handle:
                html = handle.read()
        except FileNotFoundError:
            return self._send(500, {"error": "web/index.html is missing"})
        html = html.replace("__AUTOTUNE_TOKEN__", self.token)
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

    def _serve_static(self, name: str) -> None:
        safe = os.path.normpath(name).replace("\\", "/")
        if safe.startswith("..") or os.path.isabs(safe):
            return self._send(403, {"error": "forbidden"})
        path = os.path.join(WEB_DIR, safe)
        if not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        kind = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as handle:
            self._send(200, handle.read(), kind)

    def _serve_report(self, run_id: str) -> None:
        run = self.manager.get(run_id)
        if not run or not run.request.out:
            return self._send(404, {"error": "no report for this run"})
        path = os.path.join(run.request.out, "report.html")
        if not os.path.isfile(path):
            return self._send(404, {"error": "report not written yet"})
        with open(path, "rb") as handle:
            self._send(200, handle.read(), "text/html; charset=utf-8")

    def _stream(self, run_id: str) -> None:
        run = self.manager.get(run_id)
        if not run:
            return self._send(404, {"error": "no such run"})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        listener = run.subscribe()
        try:
            while True:
                try:
                    event = listener.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")   # stop proxies timing out
                    self.wfile.flush()
                    if run.status != "running":
                        break
                    continue
                payload = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                if event.get("event") == "finished":
                    break
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        finally:
            run.unsubscribe(listener)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            "refusing to bind to a non-loopback address: this server browses the "
            "filesystem and starts jobs, and has no authentication beyond a local "
            "token. Put it behind your own authenticated proxy if you need remote "
            "access.")

    handler = type("BoundHandler", (Handler,),
                   {"manager": RunManager(), "token": secrets.token_urlsafe(16)})
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{httpd.server_address[1]}/?token={handler.token}"
    print(f"dbr-autotune UI  ->  {url}")
    print("  loopback only; the token in the URL authorises every call.")
    print("  Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
