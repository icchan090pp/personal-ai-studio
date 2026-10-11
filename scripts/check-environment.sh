#!/usr/bin/env bash
set -euo pipefail

check_dir="$(mktemp -d)"
trap 'rm -rf -- "$check_dir"' EXIT

python3 --version
python3 -m pytest --version
python3 -m ruff --version
node --version
npm --version
gcc --version
g++ --version

python3 -m venv "$check_dir/venv"
"$check_dir/venv/bin/python" -m pip --version
"$check_dir/venv/bin/python" - <<'PY'
import json
import sqlite3
import ssl

values = json.loads('[2, 3, 5]')
assert sum(values) == 10
with sqlite3.connect(":memory:") as database:
    assert database.execute("select 2 + 3 + 5").fetchone()[0] == 10
assert ssl.create_default_context().check_hostname
print("PASS: Python, virtual environments, SQLite and SSL")
PY

node --input-type=module - <<'JS'
import assert from 'node:assert/strict';
const values = JSON.parse('[2, 3, 5]');
assert.equal(values.reduce((sum, value) => sum + value, 0), 10);
console.log('PASS: JavaScript and Node.js');
JS

gcc -std=c17 -Wall -Wextra -Werror -x c -o "$check_dir/c-check" - <<'C'
#include <stdio.h>
int main(void) {
    const int values[] = {2, 3, 5};
    int sum = 0;
    for (int i = 0; i < 3; ++i) sum += values[i];
    if (sum != 10) return 1;
    puts("PASS: C compilation and execution");
    return 0;
}
C
"$check_dir/c-check"

g++ -std=c++20 -Wall -Wextra -Werror -x c++ -o "$check_dir/cpp-check" - <<'CPP'
#include <iostream>
#include <numeric>
#include <vector>
int main() {
    const std::vector<int> values{2, 3, 5};
    if (std::accumulate(values.begin(), values.end(), 0) != 10) return 1;
    std::cout << "PASS: C++ compilation and execution\n";
    return 0;
}
CPP
"$check_dir/cpp-check"
