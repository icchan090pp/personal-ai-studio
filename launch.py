"""Start the localhost app and open the browser when its actual API is ready."""

import argparse
import http.cookiejar
import ipaddress
import json
import os
import secrets
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch DeeL Pyess")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--lan", action="store_true", help="Pair a phone on the same trusted Wi-Fi")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("Choose a port between 1 and 65535")
    root = Path(__file__).resolve().parent
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        raise SystemExit("Run Install-Windows.bat or ./install.sh first")
    with socket.socket() as probe:
        try:
            probe.bind(("0.0.0.0" if args.lan else "127.0.0.1", args.port))
        except OSError as exc:
            raise SystemExit(
                "This port is already in use. Close the old app or use --port."
            ) from exc
    child_env = os.environ.copy()
    token = secrets.token_urlsafe(32) if args.lan else ""
    child_env["STUDIO_ACCESS_TOKEN"] = token
    process = subprocess.Popen(
        [
            str(python),
            "-m",
            "uvicorn",
            "studio.app:app",
            "--host",
            "0.0.0.0" if args.lan else "127.0.0.1",
            "--port",
            str(args.port),
            "--no-access-log",
        ],
        cwd=root,
        env=child_env,
        start_new_session=os.name != "nt",
    )
    url = f"http://127.0.0.1:{args.port}"
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    try:
        ready = False
        deadline = time.monotonic() + 30
        while process.poll() is None and time.monotonic() < deadline:
            try:
                probe_url = url + ("/connect?token=" + token if args.lan else "/api/models")
                if args.lan:
                    with opener.open(probe_url, timeout=1) as response:
                        ready = response.status == 200
                    if ready:
                        break
                with opener.open(probe_url, timeout=1) as response:
                    models = json.load(response)
                if len(models) == 2 and {item["kind"] for item in models} == {"image", "video"}:
                    ready = True
                    break
            except (OSError, ValueError, urllib.error.URLError):
                time.sleep(0.2)
        if not ready:
            raise SystemExit("Could not start the application. See the error above.")
        print(f"DeeL Pyess: {url}\nPress Ctrl+C to stop.", flush=True)
        browser_url = url
        if args.lan:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
            }
            addresses = sorted(
                ip
                for ip in addresses
                if ipaddress.ip_address(ip).is_private and not ipaddress.ip_address(ip).is_loopback
            )
            browser_url = url + "/connect?token=" + token
            for ip in addresses:
                print(
                    f"Phone (same Wi-Fi): http://{ip}:{args.port}/connect?token={token}", flush=True
                )
            if not addresses:
                print(
                    "Use this computer's Wi-Fi IPv4 address with the /connect URL shown in your browser.",
                    flush=True,
                )
        if not args.no_browser:
            webbrowser.open(browser_url)
        process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
