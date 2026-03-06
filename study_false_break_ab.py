"""A/B study runner for regime-aware false-break strategy variants.

Runs full study pipeline for baseline + A + B variants, then builds comparison artifacts.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

from study_event_first import run_study


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _normalize_variant_overrides(raw_variant_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in raw_variant_cfg.items():
        if not k.endswith("_overrides"):
            continue
        section = k[: -len("_overrides")]
        out[section] = copy.deepcopy(v)
    return out


def _collect_variant_metrics(variant_dir: Path) -> Dict[str, Any]:
    tf = pd.read_csv(variant_dir / "trades_false_break.csv")
    tc = pd.read_csv(variant_dir / "trades_combined.csv")
    ef = pd.read_csv(variant_dir / "equity_false_break.csv")
    ec = pd.read_csv(variant_dir / "equity_combined.csv")

    def _safe_total(curve: pd.DataFrame) -> float:
        if curve.empty or "equity" not in curve.columns:
            return 0.0
        return float(curve["equity"].iloc[-1] - 1.0)

    def _safe_dd(curve: pd.DataFrame) -> float:
        if curve.empty or "drawdown" not in curve.columns:
            return 0.0
        return float(curve["drawdown"].min())

    def _win_rate(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        col = "net_return" if "net_return" in df.columns else "gross_return"
        return float((df[col] > 0).mean())

    def _mean_ret(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        col = "net_return" if "net_return" in df.columns else "gross_return"
        return float(df[col].mean())

    out = {
        "false_break_trades": int(len(tf)),
        "false_break_win_rate": _win_rate(tf),
        "false_break_mean_ret": _mean_ret(tf),
        "false_break_total": _safe_total(ef),
        "false_break_maxdd": _safe_dd(ef),
        "combined_trades": int(len(tc)),
        "combined_win_rate": _win_rate(tc),
        "combined_mean_ret": _mean_ret(tc),
        "combined_total": _safe_total(ec),
        "combined_maxdd": _safe_dd(ec),
        "exit_time_stop_early": int((tf["exit_reason"] == "time_stop_early").sum()) if "exit_reason" in tf.columns else 0,
        "exit_price_stop": int((tf["exit_reason"] == "price_stop").sum()) if "exit_reason" in tf.columns else 0,
        "exit_time_stop_max": int((tf["exit_reason"] == "time_stop_max").sum()) if "exit_reason" in tf.columns else 0,
    }
    return out


def _plot_equity_compare(curves: Dict[str, pd.DataFrame], output_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, c in curves.items():
        if c.empty:
            continue
        ax.plot(pd.to_datetime(c["timestamp"], utc=True), c["equity"], label=name, linewidth=1.7)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_png, dpi=130)
    plt.close(fig)


def _build_compare_html(
    variants: Iterable[str],
    out_root: Path,
    summary: pd.DataFrame,
    output_html: Path,
) -> None:
    fig = go.Figure()
    for name in variants:
        curve_path = out_root / name / "equity_false_break.csv"
        if not curve_path.exists():
            continue
        c = pd.read_csv(curve_path)
        if c.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(c["timestamp"], utc=True),
                y=c["equity"],
                mode="lines",
                name=f"{name} false_break",
            )
        )

    fig.update_layout(
        title="A/B False-break Equity Comparison",
        xaxis_title="Time",
        yaxis_title="Equity",
        template="plotly_white",
        legend=dict(orientation="h"),
    )

    table_rows = "\n".join(
        [
            "<tr>"
            f"<td>{r['variant']}</td>"
            f"<td>{int(r['false_break_trades'])}</td>"
            f"<td>{r['false_break_win_rate']:.2%}</td>"
            f"<td>{r['false_break_mean_ret']:.4%}</td>"
            f"<td>{r['false_break_total']:.2%}</td>"
            f"<td>{r['false_break_maxdd']:.2%}</td>"
            f"<td>{int(r['exit_time_stop_early'])}</td>"
            f"<td>{int(r['exit_price_stop'])}</td>"
            f"<td>{int(r['exit_time_stop_max'])}</td>"
            f"<td><a href='{r['variant']}/master_event_first.html'>master</a></td>"
            f"<td><a href='{r['variant']}/event_family_false_break.html'>false_break</a></td>"
            "</tr>"
            for _, r in summary.iterrows()
        ]
    )

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>A/B False-break Compare</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .chart {{ margin-bottom: 18px; }}
  </style>
</head>
<body>
  <h2>A/B False-break Compare</h2>
  <div class="chart">{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
  <table>
    <thead>
      <tr>
        <th>Variant</th>
        <th>FB Trades</th>
        <th>FB Win</th>
        <th>FB Mean</th>
        <th>FB Total</th>
        <th>FB MaxDD</th>
        <th>time_stop_early</th>
        <th>price_stop</th>
        <th>time_stop_max</th>
        <th>Master HTML</th>
        <th>FB HTML</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def _write_report(summary: pd.DataFrame, out_path: Path, regime_mode: str) -> None:
    lines = [
        "# False-break A/B Report",
        "",
        f"- Regime mode: `{regime_mode}`",
        "",
        "| Variant | FB Trades | FB Win | FB Mean | FB Total | FB MaxDD | EarlyStop | PriceStop | TimeMax | Combined Total | Combined MaxDD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            "| {v} | {t} | {w:.2%} | {m:.4%} | {tot:.2%} | {dd:.2%} | {es} | {ps} | {tm} | {ct:.2%} | {cd:.2%} |".format(
                v=r["variant"],
                t=int(r["false_break_trades"]),
                w=float(r["false_break_win_rate"]),
                m=float(r["false_break_mean_ret"]),
                tot=float(r["false_break_total"]),
                dd=float(r["false_break_maxdd"]),
                es=int(r["exit_time_stop_early"]),
                ps=int(r["exit_price_stop"]),
                tm=int(r["exit_time_stop_max"]),
                ct=float(r["combined_total"]),
                cd=float(r["combined_maxdd"]),
            )
        )

    lines += [
        "",
        "## Variant Definitions",
        "- `baseline_regime`: regime-aware false-break with base config.",
        "- `A_late_early_progress`: same stops, but delayed/looser early-progress check.",
        "- `B_strict_entry_confirm`: stricter false-break entry confirmation before execution.",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ab_study(
    config_path: str,
    outdir: str,
    regime_mode_override: str | None,
    limit_rows: int | None,
) -> Path:
    base_cfg = _load_yaml(config_path)
    out_root = Path(outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    ab_cfg = base_cfg.get("ab_study", {})
    variants: list[tuple[str, Dict[str, Any]]] = []

    if bool(ab_cfg.get("run_baseline", True)):
        variants.append((str(ab_cfg.get("base_variant_name", "baseline_regime")), {}))

    v_a = ab_cfg.get("variant_a", {})
    if isinstance(v_a, dict) and v_a:
        variants.append((str(v_a.get("name", "A_late_early_progress")), _normalize_variant_overrides(v_a)))

    v_b = ab_cfg.get("variant_b", {})
    if isinstance(v_b, dict) and v_b:
        variants.append((str(v_b.get("name", "B_strict_entry_confirm")), _normalize_variant_overrides(v_b)))

    if not variants:
        raise ValueError("No variants configured in ab_study.")

    summary_rows: list[Dict[str, Any]] = []
    false_break_curves: Dict[str, pd.DataFrame] = {}
    combined_curves: Dict[str, pd.DataFrame] = {}

    for name, overrides in variants:
        vdir = out_root / name
        vdir.mkdir(parents=True, exist_ok=True)

        vcfg = copy.deepcopy(base_cfg)
        if regime_mode_override:
            vcfg.setdefault("execution", {})["false_break_regime_mode"] = regime_mode_override
        _deep_merge(vcfg, overrides)

        vcfg_path = vdir / "config_variant.yaml"
        _save_yaml(vcfg, vcfg_path)

        run_study(config_path=str(vcfg_path), outdir_override=str(vdir), limit_rows=limit_rows)

        metrics = _collect_variant_metrics(vdir)
        metrics["variant"] = name
        summary_rows.append(metrics)

        ef = pd.read_csv(vdir / "equity_false_break.csv")
        ec = pd.read_csv(vdir / "equity_combined.csv")
        false_break_curves[name] = ef
        combined_curves[name] = ec

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("variant").reset_index(drop=True)
    summary.to_csv(out_root / "summary_ab.csv", index=False)

    _plot_equity_compare(false_break_curves, out_root / "equity_false_break_ab.png", "A/B False-break Equity")
    _plot_equity_compare(combined_curves, out_root / "equity_combined_ab.png", "A/B Combined Equity")

    regime_mode = regime_mode_override or str(base_cfg.get("execution", {}).get("false_break_regime_mode", "down_up"))
    _write_report(summary, out_root / "report_ab_false_break.md", regime_mode)
    _build_compare_html([v[0] for v in variants], out_root, summary, out_root / "ab_compare_false_break.html")

    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B study for regime-aware false-break variants.")
    parser.add_argument("--config", default="config.yaml", help="Base config path")
    parser.add_argument("--outdir", default="outputs/ab_false_break", help="Output root for A/B study")
    parser.add_argument(
        "--regime-mode",
        default=None,
        choices=["all", "down_only", "down_up"],
        help="Optional override for false_break_regime_mode across all variants",
    )
    parser.add_argument("--limit_rows", type=int, default=None, help="Optional trailing-row limit")
    args = parser.parse_args()

    out = run_ab_study(
        config_path=args.config,
        outdir=args.outdir,
        regime_mode_override=args.regime_mode,
        limit_rows=args.limit_rows,
    )
    print(f"A/B study complete. Outputs saved to: {out}")


if __name__ == "__main__":
    main()
