from __future__ import annotations

import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
OUTPUT_DIR = ROOT / "reproducibility_outputs" / "first_line"
EXPECTED_DIR = ROOT / "expected"
SCRIPT = CODE_DIR / "study_preserve_final_refinement.py"
EXPECTED_FILES = [
    "variant_summary.csv",
    "preserve_purity_table.csv",
    "robustness_check.csv",
]
ABS_TOL = 1e-9


def _as_float(value: str):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(out):
        return "nan"
    return out


def _load_csv(path: Path) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized: list[dict[str, object]] = []
    for row in rows:
        normalized.append({key: _as_float(value) for key, value in row.items()})
    return normalized


def _compare_csv(actual_path: Path, expected_path: Path) -> list[str]:
    actual = _load_csv(actual_path)
    expected = _load_csv(expected_path)
    mismatches: list[str] = []
    if len(actual) != len(expected):
        mismatches.append(f"row_count: actual={len(actual)} expected={len(expected)}")
        return mismatches
    for row_idx, (a_row, e_row) in enumerate(zip(actual, expected), start=1):
        if list(a_row.keys()) != list(e_row.keys()):
            mismatches.append(f"row_{row_idx}: columns differ")
            continue
        for key in a_row.keys():
            a_val = a_row[key]
            e_val = e_row[key]
            if isinstance(a_val, float) and isinstance(e_val, float):
                if not math.isclose(a_val, e_val, rel_tol=0.0, abs_tol=ABS_TOL):
                    mismatches.append(f"row_{row_idx}.{key}: actual={a_val} expected={e_val}")
            else:
                if a_val != e_val:
                    mismatches.append(f"row_{row_idx}.{key}: actual={a_val} expected={e_val}")
    return mismatches


def main() -> int:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(SCRIPT), "--outdir", str(OUTPUT_DIR)]
    subprocess.run(cmd, cwd=ROOT, check=True)

    expected_map = {
        "variant_summary.csv": OUTPUT_DIR / "variant_summary.csv",
        "preserve_purity_table.csv": OUTPUT_DIR / "preserve_purity_table.csv",
        "robustness_check.csv": OUTPUT_DIR / "robustness_check.csv",
    }

    all_mismatches: list[str] = []
    for name, actual_path in expected_map.items():
        expected_path = EXPECTED_DIR / name
        mismatches = _compare_csv(actual_path, expected_path)
        all_mismatches.extend([f"{name}: {item}" for item in mismatches])

    if all_mismatches:
        print("REPRODUCIBILITY_CHECK=FAIL")
        for item in all_mismatches[:20]:
            print(item)
        return 1

    summary = _load_csv(OUTPUT_DIR / "variant_summary.csv")
    refined = next(row for row in summary if row["variant"] == "refined_preserve_variant")
    control = next(row for row in summary if row["variant"] == "control")
    print("REPRODUCIBILITY_CHECK=PASS")
    print(f"control_total_return={control['total_return']}")
    print(f"refined_total_return={refined['total_return']}")
    print(f"control_max_drawdown={control['max_drawdown']}")
    print(f"refined_max_drawdown={refined['max_drawdown']}")
    print(f"outputs_dir={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
