"""Inspect real hardware separately from optional Python packages."""

import argparse
import ctypes
import importlib.util
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

DRIVER_URL = "https://www.nvidia.com/Download/index.aspx"


def fallback_state(root):
    try:
        value = json.loads((Path(root) / "runtime-fallback.json").read_text(encoding="utf-8"))
        return value if value.get("backend") in {"cuda", "directml"} else {}
    except (OSError, ValueError, AttributeError):
        return {}


def remember_fallback(root, backend, reason):
    path = Path(root) / "runtime-fallback.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"backend": backend, "reason": reason}, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def selected_backend(root):
    try:
        value = json.loads((Path(root) / "runtime-choice.json").read_text(encoding="utf-8"))
        return value if value in ("cpu", "cuda", "directml") else "auto"
    except (OSError, ValueError):
        return "auto"


def memory_guidance(ram):
    if ram is None:
        return "メモリ量を取得できませんでした。軽量画像はRAM 8GB以上が目安です。"
    if ram < 8:
        return "RAM 8GB未満です。画像生成はメモリ不足になる可能性があります。画像整理・コースから始め、不要なアプリを終了してください。"
    if ram < 16:
        return "RAM 8〜16GBのPCではTiny-SD・ノーマルを推奨します。他のアプリを閉じ、1件ずつ生成してください。"
    return "軽量画像の目安はRAM 8GB以上です。16GBでも大型モデル・動画には別途十分なGPUメモリが必要です。"


def run_json(command, timeout=20):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env={key: value for key, value in os.environ.items() if key.lower() != "psmodulepath"},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:] or "診断プロセスが失敗しました")
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise ValueError("診断結果が空です")
    return json.loads(lines[-1])


def adapter_names():
    if os.name != "nt":
        return []
    powershell = (
        Path(os.environ.get("SystemRoot", r"C:\Windows"))
        / "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    try:
        names = run_json(
            [
                str(powershell),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) | ConvertTo-Json -Compress",
            ]
        )
        return [names] if isinstance(names, str) else names or []
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
        return []


def memory_gb():
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (key, ctypes.c_ulonglong)
                for key in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.total / 1024**3, 1)
    elif hasattr(os, "sysconf"):
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    return None


def directml_probe():
    try:
        import torch
        import torch_directml

        names = [torch_directml.device_name(i) for i in range(torch_directml.device_count())]
        # A software adapter must not be advertised as hardware GPU acceleration.
        choices = [
            i
            for i, name in enumerate(names)
            if not any(word in name.lower() for word in ("basic render", "software", "warp"))
        ]
        if not choices:
            return {"ready": False, "devices": names, "reason": "DirectML対応のGPUが見つかりません"}
        index = choices[0]
        device = torch_directml.device(index)
        assert (torch.ones(4).to(device) * 2).cpu().tolist() == [2, 2, 2, 2]
        return {"ready": True, "devices": names, "index": index, "name": names[index]}
    except Exception as exc:
        return {"ready": False, "devices": [], "reason": str(exc)[:500]}


def describe(names, libraries, cuda, cpu, dml, cuda_build=None):
    if cuda:
        code, label, message = (
            "cuda_ready",
            "NVIDIA CUDA",
            "PCのNVIDIA GPUで画像・動画の学習と生成を実行できます。必要なメモリ量はモデルによって異なります。",
        )
    elif dml.get("ready") and cpu:
        code, label, message = (
            "directml_ready",
            "DirectML GPU",
            "軽量画像（Tiny-SD）はDirectML GPUを使用します。演算非対応・メモリ不足時はCPUで再試行します。動画・大型モデル・画像LoRA学習はCUDA GPUが必要です。",
        )
    elif not libraries or not cpu:
        code, label = "setup_required", "AI機能の準備が必要"
        message = "PCのGPUを確認しました。『このPCのAI機能をセットアップ』で必要なライブラリをアプリ内に導入してください。"
    elif any("nvidia" in name.lower() for name in names):
        code, label = "cuda_unavailable", "NVIDIA GPUの準備が必要"
        message = (
            "NVIDIA GPUはありますが、CUDA演算を開始できません。"
            + (
                "CUDA対応ライブラリをセットアップしてください。"
                if not cuda_build
                else "NVIDIAのドライバを更新してPCを再起動し、再診断してください。"
            )
            + "軽量画像はCPUで利用できます。"
        )
    else:
        code, label = "cpu_ready", "CPU互換モード"
        message = "軽量画像（Tiny-SD）と日本語音声はCPUで利用できます。AMD・Intel GPUはCUDA非対応です。DirectMLを導入して対応GPUを確認できます。動画・大型モデル・画像LoRA学習にはCUDA GPUが必要です。"
    return {
        "code": code,
        "label": label,
        "message": message,
        "nvidia_detected": any("nvidia" in name.lower() for name in names),
    }


@lru_cache(maxsize=1)
def inspect_system():
    names = adapter_names()
    libraries = all(
        importlib.util.find_spec(name) for name in ("torch", "diffusers", "peft", "transformers")
    )
    cpu, cuda, cuda_build, cuda_error = False, False, None, None
    try:
        import torch

        cpu = (torch.ones(1) + 1).item() == 2
        cuda_build = torch.version.cuda
        if torch.cuda.is_available():
            torch.ones(1, device="cuda").cpu()
            cuda = True
            if not names:
                names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        cuda_error = str(exc)[:500]
    dml = {"ready": False, "devices": []}
    dml_python = os.environ.get("STUDIO_DML_PYTHON")
    if dml_python and Path(dml_python).is_file():
        try:
            dml = run_json([dml_python, "-m", "studio.hardware", "--directml-probe"])
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            dml["reason"] = str(exc)[:500]
    ram = memory_gb()
    return {
        **describe(names, libraries, cuda and libraries, cpu and libraries, dml, cuda_build),
        "adapters": names,
        "ram_gb": ram,
        "memory_guidance": memory_guidance(ram),
        "libraries_ready": libraries,
        "cpu_ready": cpu and libraries,
        "cuda_ready": cuda and libraries,
        "cuda_build": cuda_build,
        "cuda_error": cuda_error,
        "directml": dml,
        "driver_url": DRIVER_URL,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directml-probe", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            directml_probe() if args.directml_probe else inspect_system(), ensure_ascii=False
        )
    )
