#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
operation="${1:-setup}"
if (( $# > 0 )); then
    shift
fi

case "$operation" in
    setup|verify)
        if (( $# != 0 )); then
            printf 'The %s command does not take arguments.\n' "$operation" >&2
            exit 2
        fi
        ;;
    run)
        if (( $# == 0 )); then
            printf 'Usage: bash scripts/cloud-environment.sh run COMMAND [ARG...]\n' >&2
            exit 2
        fi
        ;;
    *)
        printf 'Usage: bash scripts/cloud-environment.sh {setup|verify|run COMMAND [ARG...]}\n' >&2
        exit 2
        ;;
esac

# Repository-local defaults also work when the cloud home directory is read-only.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-"$repository_root/.cache"}"
export npm_config_cache="${npm_config_cache:-"$XDG_CACHE_HOME/npm"}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-"$repository_root/.cache/ms-playwright"}"
mkdir -p -- "$XDG_CACHE_HOME" "$npm_config_cache"
for cache_directory in "$XDG_CACHE_HOME" "$npm_config_cache"; do
    if [[ ! -w "$cache_directory" ]]; then
        printf 'Cache directory is not writable: %s\n' "$cache_directory" >&2
        exit 1
    fi
done

environment_python="$repository_root/.venv/bin/python"
if [[ "$operation" == setup || "$operation" == verify ]]; then
    if [[ ! -x "$environment_python" ]]; then
        python3 -m venv "$repository_root/.venv"
    fi
    "$environment_python" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"'
    "$environment_python" -m pip install --disable-pip-version-check \
        -r "$repository_root/requirements-dev.txt" \
        -r "$repository_root/requirements-app.txt"
    # The numerical product test uses CPU PyTorch and the same pinned diffusion stack.
    # An already-installed compatible CUDA build is retained by pip.
    "$environment_python" -m pip install --disable-pip-version-check \
        torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
    "$environment_python" -m pip install --disable-pip-version-check \
        -r "$repository_root/requirements-gpu.txt"
    if ! command -v chromium >/dev/null 2>&1 && ! command -v google-chrome >/dev/null 2>&1; then
        "$environment_python" -m playwright install --with-deps --only-shell chromium
    fi
elif [[ ! -x "$environment_python" ]]; then
    printf 'Run bash scripts/cloud-environment.sh setup first.\n' >&2
    exit 1
fi

export VIRTUAL_ENV="$repository_root/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if [[ "$operation" == run ]]; then
    exec "$@"
fi

cd -- "$repository_root"
if [[ "$operation" == setup ]]; then
    exec bash scripts/check-environment.sh
fi

for attempt in 1 2 3 4 5; do
    printf '\nVerification %s/5\n' "$attempt"
    report_directory="$repository_root/artifacts/verification/run-$attempt"
    mkdir -p -- "$report_directory"
    bash scripts/check-environment.sh
    python3 -m ruff check .
    python3 -m ruff format --check .
    CLOUDSPACE_CHECK_RUN="$attempt" python3 -m pytest -q --durations=10 \
        --junitxml="$report_directory/results.xml"
    python3 scripts/check-results.py "$report_directory/results.xml"
    printf 'PASS: verification %s/5\n' "$attempt"
done
printf '\nPASS: all 5 consecutive verification runs\n'
