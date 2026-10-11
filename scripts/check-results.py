"""Require exactly 100 successful product cases; no failures, errors or skips."""

import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
counts = {
    key: sum(int(suite.get(key, "0")) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
assert counts == {"tests": 100, "failures": 0, "errors": 0, "skipped": 0}, counts
print("Validated: exactly 100 passed, 0 failed, 0 errors, 0 skipped")
