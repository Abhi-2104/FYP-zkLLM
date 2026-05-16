#!/usr/bin/env python3
"""Generate proof/verification metrics tables and CSV for PPT.

Notes:
- Excludes ppgen and commitment phases by design.
- Verification Overhead Ratio:
    - If normal_infer_s is provided: overhead = (proof_gen_s + verify_s) / normal_infer_s
    - Otherwise uses a proxy baseline: overhead = proof_gen_s / verify_s
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

DEFAULT_MODELS = [
    {
        "model": "7b",
        "params": 7_000_000_000,
        "proof_gen_s": 2987,
        "verify_s": 362,
        "proof_size_mb": 588,
    },
    {
        "model": "13b",
        "params": 13_000_000_000,
        "proof_gen_s": 3952,
        "verify_s": 692,
        "proof_size_mb": 906,
    },
]


def _compute_metrics(row: Dict) -> Dict:
    proof_size_bytes = row["proof_size_mb"] * 1024 * 1024
    proof_eff = row["params"] / proof_size_bytes if proof_size_bytes else 0.0
    normal_infer_s = row.get("normal_infer_s")
    if normal_infer_s:
        overhead_ratio = (row["proof_gen_s"] + row["verify_s"]) / normal_infer_s
        overhead_note = "uses normal_infer_s"
    else:
        overhead_ratio = row["proof_gen_s"] / row["verify_s"] if row["verify_s"] else 0.0
        overhead_note = "proxy (verify_s baseline)"
    total_time = row["proof_gen_s"] + row["verify_s"]

    return {
        **row,
        "proof_size_bytes": int(proof_size_bytes),
        "proof_eff_params_per_byte": proof_eff,
        "overhead_ratio": overhead_ratio,
        "overhead_ratio_note": overhead_note,
        "total_zk_time_s": total_time,
    }


def _write_csv(path: Path, rows: List[Dict]) -> None:
    headers = [
        "model",
        "params",
        "proof_gen_s",
        "verify_s",
        "normal_infer_s",
        "proof_size_mb",
        "proof_size_bytes",
        "proof_eff_params_per_byte",
        "overhead_ratio",
        "overhead_ratio_note",
        "total_zk_time_s",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_table(path: Path, rows: List[Dict]) -> None:
    lines = []
    lines.append("| Model | Proof Gen Time (s) | Verification Time (s) | Verification Overhead Ratio (proxy) | Proof Efficiency (params/byte) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['proof_gen_s']} | {row['verify_s']} | {row['overhead_ratio']:.2f} | {row['proof_eff_params_per_byte']:.2f} |"
        )
    lines.append("")
    lines.append("Note: Overhead ratio uses verification time as a proxy baseline unless normal_infer_s is provided.")
    path.write_text("\n".join(lines))


def _write_summary(path: Path, rows: List[Dict]) -> None:
    lines = []
    lines.append("Metrics Summary (proof/verification only)")
    lines.append("- Excludes ppgen and commitment phases")
    lines.append("- Overhead ratio uses normal_infer_s if provided, otherwise verification time proxy")
    lines.append("")
    for row in rows:
        lines.append(f"Model {row['model']}")
        lines.append(f"  Proof gen time (s): {row['proof_gen_s']}")
        lines.append(f"  Verification time (s): {row['verify_s']}")
        if row.get("normal_infer_s"):
            lines.append(f"  Normal inference time (s): {row['normal_infer_s']}")
        lines.append(f"  Proof size (MB): {row['proof_size_mb']}")
        lines.append(f"  Proof efficiency (params/byte): {row['proof_eff_params_per_byte']:.4f}")
        lines.append(f"  Overhead ratio: {row['overhead_ratio']:.4f} ({row['overhead_ratio_note']})")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate proof/verification metrics CSV and PPT-ready table."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="Optional JSON file with model metrics list. If omitted, defaults are used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for CSV and tables.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_json:
        data = json.loads(Path(args.input_json).read_text())
        if not isinstance(data, list):
            raise ValueError("input_json must be a list of model entries")
        rows_in = data
    else:
        rows_in = DEFAULT_MODELS

    rows = [_compute_metrics(row) for row in rows_in]

    csv_path = output_dir / "metrics_results.csv"
    md_path = output_dir / "metrics_table.md"
    summary_path = output_dir / "metrics_summary.txt"

    _write_csv(csv_path, rows)
    _write_markdown_table(md_path, rows)
    _write_summary(summary_path, rows)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
