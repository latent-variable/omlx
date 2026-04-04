# SPDX-License-Identifier: Apache-2.0
"""Report formatting helpers for harness cache benchmarks."""

from __future__ import annotations

import html
import math
import statistics

from .schema import BenchmarkResult


def render_markdown(result: BenchmarkResult) -> str:
    lines = [
        f"# Harness Cache Benchmark: {result.config.harness}",
        "",
        f"- Model: `{result.config.model_id}`",
        f"- Workload: `{result.config.workload_id}`",
        f"- Block size: `{result.config.block_size}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in result.summary.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Turns",
            "",
            "| Turn | Prompt chars | Prefix chars | Block-aligned prefix | Reprocessed est. | Msgs | After abort |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for turn in result.turns:
        lines.append(
            f"| {turn.turn_index} | {turn.prompt_chars_total} | {turn.common_prefix_chars} | "
            f"{turn.block_aligned_prefix_chars} | {turn.reprocessed_chars_estimate} | "
            f"{turn.effective_message_count} | {turn.likely_replay_after_abort} |"
        )
    return "\n".join(lines) + "\n"


def render_html_dashboard(results: list[BenchmarkResult], *, title: str = "oMLX Harness Benchmark Dashboard") -> str:
    if not results:
        raise ValueError("results must not be empty")

    def fmt_int(value: float | int) -> str:
        return f"{int(round(value)):,}"

    def fmt_pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    def turn_reuse_ratio(prompt_chars_total: int, block_aligned_prefix_chars: int) -> float:
        if prompt_chars_total <= 0:
            return 0.0
        return min(max(block_aligned_prefix_chars / prompt_chars_total, 0.0), 1.0)

    def median(values: list[float]) -> float:
        return statistics.median(values) if values else 0.0

    def series_svg(
        series: list[tuple[str, list[float], str]],
        *,
        width: int = 920,
        height: int = 280,
        y_label: str,
        event_positions: list[int] | None = None,
    ) -> str:
        left = 54
        right = 18
        top = 18
        bottom = 30
        plot_width = width - left - right
        plot_height = height - top - bottom
        points_count = max((len(values) for _, values, _ in series), default=0)
        max_y = max((max(values) for _, values, _ in series if values), default=1.0)
        max_y = max(max_y, 1.0)
        step_x = plot_width / max(points_count - 1, 1)

        def point_coords(values: list[float]) -> str:
            coords: list[str] = []
            for idx, value in enumerate(values):
                x = left + idx * step_x
                y = top + plot_height - ((value / max_y) * plot_height)
                coords.append(f"{x:.1f},{y:.1f}")
            return " ".join(coords)

        grid_lines: list[str] = []
        for tick in range(5):
            ratio = tick / 4
            y = top + plot_height - ratio * plot_height
            tick_value = max_y * ratio
            grid_lines.append(
                f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_width}' y2='{y:.1f}' class='grid' />"
                f"<text x='{left - 8}' y='{y + 4:.1f}' class='axis-label axis-left'>{html.escape(fmt_int(tick_value))}</text>"
            )

        x_labels: list[str] = []
        for idx in range(points_count):
            if points_count <= 1 or idx in {0, points_count - 1} or idx % max(math.ceil(points_count / 8), 1) == 0:
                x = left + idx * step_x
                x_labels.append(f"<text x='{x:.1f}' y='{height - 8}' class='axis-label axis-center'>{idx}</text>")

        markers = ""
        if event_positions:
            marker_bits: list[str] = []
            for idx in event_positions:
                if 0 <= idx < points_count:
                    x = left + idx * step_x
                    marker_bits.append(
                        f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + plot_height}' class='event-line' />"
                    )
            markers = "".join(marker_bits)

        polylines = "".join(
            f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{point_coords(values)}' />"
            for _, values, color in series
            if values
        )

        legend = "".join(
            f"<div class='legend-item'><span class='legend-swatch' style='background:{color}'></span>{html.escape(label)}</div>"
            for label, _, color in series
        )

        return (
            "<div class='chart-card'>"
            f"<div class='chart-meta'><h3>{html.escape(y_label)}</h3><div class='legend'>{legend}</div></div>"
            f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(y_label)} chart'>"
            + "".join(grid_lines)
            + markers
            + polylines
            + "".join(x_labels)
            + f"<text x='10' y='14' class='axis-label axis-left'>{html.escape(y_label)}</text>"
            + f"<text x='{left + plot_width}' y='{height - 8}' class='axis-label axis-right'>Turn</text>"
            + "</svg></div>"
        )

    def bar_chart_svg(entries: list[tuple[str, float]], *, width: int = 920, height: int = 300, label: str) -> str:
        if not entries:
            return ""
        left = 200
        right = 24
        top = 18
        bottom = 28
        plot_width = width - left - right
        plot_height = height - top - bottom
        max_value = max((value for _, value in entries), default=1.0)
        max_value = max(max_value, 1.0)
        row_height = plot_height / max(len(entries), 1)
        bars: list[str] = []
        for idx, (entry_label, value) in enumerate(entries):
            y = top + idx * row_height + row_height * 0.15
            bar_height = row_height * 0.7
            bar_width = (value / max_value) * plot_width
            bars.append(
                f"<text x='{left - 10}' y='{y + bar_height * 0.65:.1f}' class='axis-label axis-left'>{html.escape(entry_label)}</text>"
                f"<rect x='{left}' y='{y:.1f}' width='{bar_width:.1f}' height='{bar_height:.1f}' rx='8' fill='var(--accent)' />"
                f"<text x='{left + bar_width + 8:.1f}' y='{y + bar_height * 0.65:.1f}' class='bar-value'>{html.escape(fmt_int(value))}</text>"
            )
        return (
            "<div class='chart-card'>"
            f"<div class='chart-meta'><h3>{html.escape(label)}</h3></div>"
            f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(label)} bar chart'>"
            + "".join(bars)
            + "</svg></div>"
        )

    def result_label(result: BenchmarkResult) -> str:
        return f"{result.config.harness} / {result.config.workload_id} / {result.config.block_size}"

    comparison_rows: list[dict[str, str | int | float]] = []
    for result in results:
        turns = result.turns[1:] if len(result.turns) > 1 else result.turns
        reuse_ratios = [turn_reuse_ratio(turn.prompt_chars_total, turn.block_aligned_prefix_chars) for turn in turns]
        reprocess_fractions = [
            (turn.reprocessed_chars_estimate / turn.prompt_chars_total) if turn.prompt_chars_total else 0.0
            for turn in turns
        ]
        comparison_rows.append(
            {
                "label": result_label(result),
                "harness": result.config.harness,
                "model": result.config.model_id,
                "workload": result.config.workload_id,
                "block_size": result.config.block_size,
                "turns": len(result.turns),
                "median_reprocessed": result.summary.get("median_reprocessed_chars_estimate", 0),
                "max_reprocessed": result.summary.get("max_reprocessed_chars_estimate", 0),
                "turns_after_abort": result.summary.get("turns_after_abort", 0),
                "median_reuse": median(reuse_ratios),
                "median_reprocessed_fraction": median(reprocess_fractions),
            }
        )

    best_reuse = max(comparison_rows, key=lambda row: float(row["median_reuse"]))
    best_reprocess = min(comparison_rows, key=lambda row: float(row["median_reprocessed"]))
    summary_cards = [
        ("Reports", fmt_int(len(results)), "Combined benchmark result sets in this dashboard."),
        ("Best median reuse", fmt_pct(float(best_reuse["median_reuse"])), f"{best_reuse['label']}"),
        ("Lowest median recompute", fmt_int(float(best_reprocess["median_reprocessed"])), f"{best_reprocess['label']} chars"),
        (
            "Worst single-turn recompute",
            fmt_int(max(float(row["max_reprocessed"]) for row in comparison_rows)),
            "Largest estimated recompute across all loaded runs.",
        ),
    ]

    comparison_table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['harness']))}</td>"
        f"<td>{html.escape(str(row['workload']))}</td>"
        f"<td>{html.escape(str(row['model']))}</td>"
        f"<td>{html.escape(fmt_int(float(row['block_size'])))}</td>"
        f"<td>{html.escape(fmt_int(float(row['turns'])))}</td>"
        f"<td>{html.escape(fmt_int(float(row['median_reprocessed'])))}</td>"
        f"<td>{html.escape(fmt_pct(float(row['median_reuse'])))}</td>"
        f"<td>{html.escape(fmt_pct(float(row['median_reprocessed_fraction'])))}</td>"
        f"<td>{html.escape(fmt_int(float(row['turns_after_abort'])))}</td>"
        "</tr>"
        for row in comparison_rows
    )

    comparison_chart = ""
    if len(comparison_rows) > 1:
        comparison_chart = (
            "<section class='panel'>"
            "<div class='panel-header'><h2>Run Comparison</h2><p>Use these to compare block sizes or later, different harness adapters.</p></div>"
            + bar_chart_svg(
                [(str(row["label"]), float(row["median_reprocessed"])) for row in comparison_rows],
                label="Median estimated recompute chars",
            )
            + bar_chart_svg(
                [(str(row["label"]), float(row["max_reprocessed"])) for row in comparison_rows],
                label="Worst single-turn recompute chars",
            )
            + "</section>"
        )

    result_sections: list[str] = []
    palette = {
        "prompt": "#0f766e",
        "prefix": "#2563eb",
        "reprocessed": "#ef4444",
        "reuse": "#d97706",
    }
    for result in results:
        turns = result.turns
        prompt_values = [turn.prompt_chars_total for turn in turns]
        prefix_values = [turn.block_aligned_prefix_chars for turn in turns]
        reprocessed_values = [turn.reprocessed_chars_estimate for turn in turns]
        reuse_values = [turn_reuse_ratio(turn.prompt_chars_total, turn.block_aligned_prefix_chars) * 100 for turn in turns]
        replay_turns = [turn.turn_index for turn in turns if turn.likely_replay_after_abort]
        turn_rows = "".join(
            "<tr>"
            f"<td>{turn.turn_index}</td>"
            f"<td>{fmt_int(turn.prompt_chars_total)}</td>"
            f"<td>{fmt_int(turn.common_prefix_chars)}</td>"
            f"<td>{fmt_int(turn.block_aligned_prefix_chars)}</td>"
            f"<td>{fmt_int(turn.reprocessed_chars_estimate)}</td>"
            f"<td>{fmt_pct(turn_reuse_ratio(turn.prompt_chars_total, turn.block_aligned_prefix_chars))}</td>"
            f"<td>{turn.effective_message_count}</td>"
            f"<td>{'yes' if turn.likely_replay_after_abort else ''}</td>"
            "</tr>"
            for turn in turns
        )
        result_sections.append(
            "<section class='panel'>"
            f"<div class='panel-header'><h2>{html.escape(result_label(result))}</h2>"
            f"<p>Model <code>{html.escape(result.config.model_id)}</code> with block size <code>{result.config.block_size}</code>.</p></div>"
            "<div class='metrics-grid'>"
            f"<div class='metric'><span class='metric-label'>Turns</span><strong>{fmt_int(len(turns))}</strong></div>"
            f"<div class='metric'><span class='metric-label'>Median recompute</span><strong>{fmt_int(result.summary.get('median_reprocessed_chars_estimate', 0))}</strong></div>"
            f"<div class='metric'><span class='metric-label'>Max recompute</span><strong>{fmt_int(result.summary.get('max_reprocessed_chars_estimate', 0))}</strong></div>"
            f"<div class='metric'><span class='metric-label'>Replay-after-abort turns</span><strong>{fmt_int(result.summary.get('turns_after_abort', 0))}</strong></div>"
            "</div>"
            "<div class='chart-grid'>"
            + series_svg(
                [
                    ("Prompt chars", prompt_values, palette["prompt"]),
                    ("Block-aligned prefix", prefix_values, palette["prefix"]),
                    ("Reprocessed estimate", reprocessed_values, palette["reprocessed"]),
                ],
                y_label="Chars per turn",
                event_positions=replay_turns,
            )
            + series_svg(
                [("Reuse ratio", reuse_values, palette["reuse"])],
                y_label="Reuse ratio (%)",
                event_positions=replay_turns,
            )
            + "</div>"
            "<div class='table-wrap'><table>"
            "<thead><tr><th>Turn</th><th>Prompt chars</th><th>Common prefix</th><th>Block-aligned prefix</th><th>Reprocessed est.</th><th>Reuse</th><th>Msgs</th><th>After abort</th></tr></thead>"
            f"<tbody>{turn_rows}</tbody></table></div>"
            "</section>"
        )

    cards_html = "".join(
        "<div class='metric-card'>"
        f"<span class='metric-card-label'>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        f"<p>{html.escape(description)}</p>"
        "</div>"
        for label, value, description in summary_cards
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4efe8;
      --panel: #fffdfa;
      --ink: #1f2937;
      --muted: #6b7280;
      --border: #ddd6cb;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --grid: #e5e7eb;
      --event: #b91c1c;
      --shadow: 0 18px 35px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 22rem),
        radial-gradient(circle at bottom right, rgba(217, 119, 6, 0.08), transparent 18rem),
        var(--bg);
    }}
    .shell {{ max-width: 1480px; margin: 0 auto; padding: 32px 24px 60px; }}
    header {{
      padding: 28px 30px;
      background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(245,247,250,0.94));
      border: 1px solid var(--border);
      border-radius: 28px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 700; }}
    h1 {{ font-size: clamp(2rem, 5vw, 3.6rem); line-height: 1; letter-spacing: -0.04em; }}
    h2 {{ font-size: 1.4rem; }}
    h3 {{ font-size: 1rem; }}
    p {{ margin: 0; color: var(--muted); }}
    code {{
      font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
      font-size: 0.92em;
      background: #f1f5f9;
      border-radius: 6px;
      padding: 0.12rem 0.32rem;
    }}
    .lede {{ margin-top: 10px; max-width: 64rem; line-height: 1.5; }}
    .metric-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin: 22px 0 28px;
    }}
    .metric-card, .panel, .chart-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .metric-card {{
      padding: 18px 18px 16px;
    }}
    .metric-card-label, .metric-label {{
      display: block;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .metric-card strong {{
      display: block;
      font-size: 2rem;
      letter-spacing: -0.04em;
      margin-bottom: 8px;
    }}
    .metric-card p {{ line-height: 1.4; }}
    .panel {{ padding: 22px; margin-bottom: 22px; }}
    .panel-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric {{
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.92));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .metric strong {{
      font-size: 1.6rem;
      letter-spacing: -0.04em;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .chart-card {{
      padding: 16px;
    }}
    .chart-meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    svg {{ width: 100%; height: auto; display: block; }}
    .grid {{
      stroke: var(--grid);
      stroke-width: 1;
      stroke-dasharray: 4 6;
    }}
    .event-line {{
      stroke: var(--event);
      stroke-width: 2;
      stroke-dasharray: 6 6;
      opacity: 0.65;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 12px;
      font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
    }}
    .axis-left {{ text-anchor: end; }}
    .axis-center {{ text-anchor: middle; }}
    .axis-right {{ text-anchor: end; }}
    .bar-value {{
      fill: var(--ink);
      font-size: 12px;
      font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(255,255,255,0.88);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f8fafc;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 0.75rem;
    }}
    tbody tr:nth-child(even) {{ background: rgba(248,250,252,0.8); }}
    @media (max-width: 980px) {{
      .chart-grid {{ grid-template-columns: 1fr; }}
      .shell {{ padding: 18px 14px 40px; }}
      header, .panel {{ border-radius: 20px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>{html.escape(title)}</h1>
      <p class="lede">This dashboard is generated from additive harness benchmark reports. It highlights prompt growth, block-aligned reuse, estimated recompute waste, and replay-after-abort risk so we can compare harness behavior without changing core oMLX runtime logic.</p>
    </header>
    <section class="metric-cards">{cards_html}</section>
    <section class="panel">
      <div class="panel-header">
        <h2>Comparison Table</h2>
        <p>Lower recompute and higher reuse are better. Replay-after-abort turns are marked from trace evidence.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Harness</th>
              <th>Workload</th>
              <th>Model</th>
              <th>Block</th>
              <th>Turns</th>
              <th>Median recompute</th>
              <th>Median reuse</th>
              <th>Median recompute fraction</th>
              <th>After abort</th>
            </tr>
          </thead>
          <tbody>{comparison_table_rows}</tbody>
        </table>
      </div>
    </section>
    {comparison_chart}
    {''.join(result_sections)}
  </div>
</body>
</html>
"""
