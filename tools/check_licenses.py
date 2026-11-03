#!/usr/bin/env python3
"""License allowlist gate for stockpred dependencies.

Reads ``pyproject.toml`` (``[project].dependencies`` plus **every** group under
``[project.optional-dependencies]``) and checks each dependency package against
``tools/license_allowlist.toml``, whose ``[packages]`` table maps a normalized
package name to a statically verified SPDX identifier.

Rules
-----
* dependency not registered in the allowlist  -> FAIL (unknown license)
* registered SPDX id outside ALLOWED_SPDX     -> FAIL (denied license)
* separate config problems (missing/invalid TOML) -> exit code 2

Output: one ``OK <pkg> <SPDX>`` / ``FAIL <pkg> <reason>`` line per dependency,
then ``RESULT: PASS`` or ``RESULT: FAIL (n)``.

Exit codes: 0 = PASS, 1 = license failure(s), 2 = config/IO error.

Standard library only (Python 3.11: tomllib). No network access: licenses are
declared by hand in the allowlist and must be verified by a human.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# --- allowlist policy -------------------------------------------------------
# Permissive-only: MIT / Apache-2.0 / BSD-2-Clause / BSD-3-Clause / PSF-2.0 / ISC.
# Copyleft and source-available licenses (AGPL/GPL/LGPL/Commons Clause/SSPL/BUSL)
# are deliberately absent, so registering one of them always fails the gate.
ALLOWED_SPDX: frozenset[str] = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "PSF-2.0",
        "ISC",
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
ALLOWLIST = Path(__file__).resolve().parent / "license_allowlist.toml"

# PEP 508: the distribution name is the leading run of [A-Za-z0-9._-] characters,
# so extras / specifiers / markers (``pandas[all]>=2; python_version<"3.12"``)
# and direct references (``pkg @ https://...``) are all handled by one match.
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_NOT_NAME_RE = re.compile(r"[-_.]+")


def normalize(name: str) -> str:
    """Normalize a distribution name to its PEP 503 form."""
    return _NOT_NAME_RE.sub("-", name.strip()).lower()


def load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file, tolerating a UTF-8 BOM (``tomllib`` rejects it)."""
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return tomllib.loads(data.decode("utf-8"))


def parse_requirement_name(requirement: str) -> str | None:
    """Extract the raw distribution name from a PEP 508 requirement string."""
    match = _NAME_RE.match(requirement)
    return match.group(1) if match else None


def collect_requirements(pyproject_path: Path) -> list[str]:
    """Return dependency strings: core first, then each optional group in file order."""
    data = load_toml(pyproject_path)

    project: dict[str, Any] = data.get("project") or {}
    requirements: list[str] = [str(item) for item in project.get("dependencies") or []]

    optional: dict[str, Any] = project.get("optional-dependencies") or {}
    for group in optional.values():  # every group, in declaration order
        requirements.extend(str(item) for item in group or [])

    return requirements


def load_allowlist(
    allowlist_path: Path, extra_allowed: Iterable[str]
) -> tuple[dict[str, str], set[str]]:
    """Return (normalized package -> SPDX id, allowed SPDX set)."""
    data = load_toml(allowlist_path)

    raw_packages: dict[str, Any] = data.get("packages") or {}
    if not isinstance(raw_packages, dict):
        raise ValueError("[packages] in license_allowlist.toml must be a table")

    packages: dict[str, str] = {}
    for name, spdx in raw_packages.items():
        if not isinstance(spdx, str) or not spdx.strip():
            raise ValueError(f"allowlist entry {name!r} must be a non-empty SPDX string")
        packages[normalize(name)] = spdx.strip()

    allowed = set(ALLOWED_SPDX) | {item.strip() for item in extra_allowed if item.strip()}
    return packages, allowed


def evaluate(
    requirements: Iterable[str],
    allowlist: dict[str, str],
    allowed: set[str],
) -> tuple[list[tuple[bool, str, str]], list[str]]:
    """Return ([(ok, pkg, detail), ...] deduplicated by package, unknown packages)."""
    results: list[tuple[bool, str, str]] = []
    unknown: list[str] = []
    seen: set[str] = set()

    for requirement in requirements:
        raw_name = parse_requirement_name(requirement)
        if raw_name is None:
            results.append(
                (False, requirement.strip() or "<empty>", "unparsable requirement string")
            )
            continue

        pkg = normalize(raw_name)
        if pkg in seen:  # same package in several groups: report once
            continue
        seen.add(pkg)

        spdx = allowlist.get(pkg)
        if spdx is None:
            unknown.append(pkg)
            results.append((False, pkg, "not registered in license_allowlist.toml"))
        elif spdx not in allowed:
            known = ", ".join(sorted(allowed))
            results.append((False, pkg, f'SPDX "{spdx}" not allowed (allowed: {known})'))
        else:
            results.append((True, pkg, spdx))

    return results, unknown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check dependencies against the license allowlist."
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=PYPROJECT,
        help=f"path to pyproject.toml (default: {PYPROJECT})",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=ALLOWLIST,
        help=f"path to license_allowlist.toml (default: {ALLOWLIST})",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="SPDX",
        help="extra SPDX id to allow on top of the built-in set (repeatable)",
    )
    args = parser.parse_args(argv)

    for label, path in (("pyproject.toml", args.pyproject), ("allowlist", args.allowlist)):
        if not path.is_file():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    try:
        requirements = collect_requirements(args.pyproject)
        allowlist, allowed = load_allowlist(args.allowlist, args.allow)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not requirements:
        print("ERROR: no dependencies found in [project].dependencies", file=sys.stderr)
        return 2

    results, unknown = evaluate(requirements, allowlist, allowed)

    failures = 0
    for ok, pkg, detail in results:
        if ok:
            print(f"OK {pkg} {detail}")
        else:
            failures += 1
            print(f"FAIL {pkg} {detail}")

    if unknown:
        print(
            f"hint: register these packages in {args.allowlist.name} after verifying the license:"
        )
        print("      " + ", ".join(f'"{name}" = "<SPDX-ID>"' for name in unknown))

    if failures:
        print(f"RESULT: FAIL ({failures})")
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
