import json
import select
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path


class DevelopmentWorkflows(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cloudspace-workflow-")
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)

    def run_command(self, *command):
        result = subprocess.run(
            command,
            cwd=self.workspace,
            text=True,
            capture_output=True,
            timeout=180,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Command failed: {command}\n{result.stdout}\n{result.stderr}",
        )
        return result.stdout

    def test_python_dependencies_and_file_operations(self):
        environment = self.workspace / ".venv"
        self.run_command(sys.executable, "-m", "venv", str(environment))
        python = str(environment / "bin" / "python")
        self.run_command(
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            "packaging",
        )
        result = self.run_command(
            python,
            "-c",
            """
import json
from pathlib import Path
import sqlite3
import sys
from packaging.version import Version

assert sys.prefix != sys.base_prefix
assert Version("2.0") > Version("1.9")
text = "\u65e5\u672c\u8a9e"
path = Path("roundtrip.txt")
path.write_text(text, encoding="utf-8")
assert path.read_text(encoding="utf-8") == text
path.rename("renamed.txt")
assert not path.exists()
with sqlite3.connect("example.sqlite") as database:
    database.execute("create table example (value integer)")
    database.execute("insert into example values (10)")
with sqlite3.connect("example.sqlite") as database:
    assert database.execute("select value from example").fetchone()[0] == 10
print(json.dumps({"result": "ok"}))
""",
        )
        self.assertEqual(json.loads(result), {"result": "ok"})

    def test_python_editable_package_installation(self):
        project = self.workspace / "example-project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["setuptools>=61"]\n'
            'build-backend = "setuptools.build_meta"\n\n'
            '[project]\nname = "cloudspace-example"\nversion = "0.0.0"\n\n'
            '[tool.setuptools]\npy-modules = ["example"]\n',
            encoding="utf-8",
        )
        module = project / "example.py"
        module.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        environment = self.workspace / ".venv"
        self.run_command(sys.executable, "-m", "venv", str(environment))
        python = str(environment / "bin" / "python")
        self.run_command(
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            "--no-deps",
            "--editable",
            str(project),
        )
        self.run_command(python, "-c", "import example; assert example.add(2, 3) == 5")
        module.write_text("def add(a, b):\n    return a + b + 1\n", encoding="utf-8")
        self.run_command(python, "-c", "import example; assert example.add(2, 3) == 6")

    def test_typescript_build_and_clean_dependency_install(self):
        (self.workspace / "package.json").write_text(
            json.dumps({"name": "environment-check", "private": True}),
            encoding="utf-8",
        )
        (self.workspace / "example.ts").write_text(
            "const values: number[] = [2, 3, 5];\n"
            "const total: number = values.reduce((sum, value) => sum + value, 0);\n"
            "console.log(JSON.stringify({ total }));\n",
            encoding="utf-8",
        )
        self.run_command(
            "npm",
            "install",
            "--ignore-scripts",
            "--no-audit",
            "--no-fund",
            "--save-dev",
            "typescript",
        )
        lockfile = (self.workspace / "package-lock.json").read_bytes()
        for clean_install in (False, True):
            with self.subTest(clean_install=clean_install):
                if clean_install:
                    self.run_command("npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund")
                self.run_command(
                    str(self.workspace / "node_modules" / ".bin" / "tsc"),
                    "--strict",
                    "--target",
                    "ES2022",
                    "--module",
                    "commonjs",
                    "--outDir",
                    "dist",
                    "example.ts",
                )
                result = self.run_command("node", "dist/example.js")
                self.assertEqual(json.loads(result), {"total": 10})
                self.assertEqual((self.workspace / "package-lock.json").read_bytes(), lockfile)

    def check_http_restarts(self, command):
        client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for attempt in range(2):
            with self.subTest(start=attempt + 1):
                process = subprocess.Popen(
                    command,
                    cwd=self.workspace,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    ready, _, _ = select.select([process.stdout], [], [], 15)
                    self.assertTrue(ready, "HTTP server did not announce readiness")
                    line = process.stdout.readline()
                    self.assertTrue(line, "HTTP server exited before becoming ready")
                    port = int(line)
                    with client.open(f"http://127.0.0.1:{port}/health", timeout=10) as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(json.load(response), {"status": "ok"})
                    self.assertIsNone(process.poll())
                finally:
                    process.terminate()
                    try:
                        process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate(timeout=10)

    def test_python_http_start_and_restart(self):
        self.check_http_restarts(
            [
                sys.executable,
                "-u",
                "-c",
                """
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
""",
            ]
        )

    def test_node_http_start_and_restart(self):
        self.check_http_restarts(
            [
                "node",
                "--input-type=module",
                "-e",
                """
import { createServer } from 'node:http';
const server = createServer((request, response) => {
  response.writeHead(request.url === '/health' ? 200 : 404, {
    'Content-Type': 'application/json',
  });
  response.end(JSON.stringify({ status: 'ok' }));
});
server.listen(0, '127.0.0.1', () => console.log(server.address().port));
""",
            ]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
