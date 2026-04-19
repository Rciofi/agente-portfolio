"""
Dashboard Expandido — Agente de Gestão de Portfólio
Inclui o 4º benchmark DLS (Zhang et al., 2021) e novos painéis:
  - Painel 10: Comparativo de performance das 4 estratégias
  - Painel 11: Evolução das alocações dinâmicas (DLS ao longo do tempo)
  - Painel 12: Sensitivity analysis das features de input
  - Painel 13: Tabela de métricas

Complementa os 9 painéis originais (Monte Carlo, heatmap, etc.)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Paleta consistente com os painéis originais
COLORS = {
    "DLS":        "#00B4D8",
    "DeepStatArb":"#06D6A0",
    "Markowitz":  "#FFB703",
    "BuyHold":    "#EF476F",
    "neutral":    "#8D99AE",
    "bg":         "#0F172A",
    "card":       "#1E293B",
    "grid":       "#334155",
    "text":       "#E2E8F0",
    "subtext":    "#94A3B8",
}

# Cores de preenchimento rgba (sem hex+alpha que o Plotly não aceita)
FILL_COLORS = {
    "DLS":        "rgba(0,180,216,0.13)",
    "DeepStatArb":"rgba(6,214,160,0.13)",
    "Markowitz":  "rgba(255,183,3,0.13)",
    "BuyHold":    "rgba(239,71,111,0.13)",
}

# Cores de fundo de célula para tabela
CELL_BG = {
    "DLS":        "rgba(0,180,216,0.08)",
    "DeepStatArb":"rgba(6,214,160,0.08)",
    "Markowitz":  "rgba(255,183,3,0.08)",
    "BuyHold":    "rgba(239,71,111,0.08)",
}


# ─────────────────────────────────────────────
# PAINEL 10 — COMPARATIVO DE PERFORMANCE
# ─────────────────────────────────────────────

def plot_performance_comparison(
    cumulative_returns: dict,
    metrics: dict,
    title: str = "Comparativo de Performance — 4 Estratégias",
) -> go.Figure:

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Retorno Acumulado",
            "Sharpe Ratio",
            "Volatilidade Anualizada",
            "Maximum Drawdown",
        ],
        row_heights=[0.55, 0.45],
        specs=[[{"colspan": 2}, None], [{}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    strategy_order = ["DLS", "DeepStatArb", "Markowitz", "BuyHold"]
    labels = {
        "DLS":         "DLS (Zhang et al.)",
        "DeepStatArb": "DeepStatArb (CNN-Transformer)",
        "Markowitz":   "Markowitz (Corrigido)",
        "BuyHold":     "Buy & Hold",
    }

    for key in strategy_order:
        if key not in cumulative_returns:
            continue
        series = cumulative_returns[key]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=labels.get(key, key),
                line=dict(color=COLORS.get(key, "#fff"), width=2.5),
                hovertemplate=f"<b>{labels.get(key, key)}</b><br>"
                              f"Data: %{{x|%Y-%m-%d}}<br>"
                              f"Retorno Acum.: %{{y:.2f}}x<extra></extra>",
            ),
            row=1, col=1,
        )

    fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["neutral"],
                  line_width=1, row=1, col=1)

    sharpe_keys = [k for k in strategy_order if k in metrics]
    sharpe_vals = [metrics[k].get("sharpe_ratio", 0) for k in sharpe_keys]

    fig.add_trace(
        go.Bar(
            x=[labels[k] for k in sharpe_keys],
            y=sharpe_vals,
            marker_color=[COLORS[k] for k in sharpe_keys],
            text=[f"{v:.2f}" for v in sharpe_vals],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=11, family="monospace"),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )

    vol_vals = [metrics[k].get("annual_volatility", 0) * 100 for k in sharpe_keys]
    mdd_vals = [abs(metrics[k].get("max_drawdown", 0)) * 100 for k in sharpe_keys]

    fig.add_trace(
        go.Bar(
            x=[labels[k] for k in sharpe_keys],
            y=vol_vals,
            marker_color=[COLORS[k] for k in sharpe_keys],
            opacity=0.85,
            text=[f"{v:.1f}%" for v in vol_vals],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=10),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Vol: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=2,
    )

    for k, mdd in zip(sharpe_keys, mdd_vals):
        fig.add_trace(
            go.Scatter(
                x=[labels[k]],
                y=[mdd],
                mode="markers+text",
                marker=dict(symbol="diamond", size=10, color=COLORS[k],
                            line=dict(color="#fff", width=1.5)),
                text=[f"MDD: {mdd:.1f}%"],
                textposition="top center",
                textfont=dict(color=COLORS["subtext"], size=9),
                showlegend=False,
                hovertemplate=f"<b>{labels[k]}</b><br>MDD: {mdd:.1f}%<extra></extra>",
            ),
            row=2, col=2,
        )

    _apply_dark_theme(fig, title)
    fig.update_yaxes(title_text="Retorno Acumulado (×1)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
    fig.update_yaxes(title_text="% a.a.", row=2, col=2)
    return fig


# ─────────────────────────────────────────────
# PAINEL 11 — ALOCAÇÕES DINÂMICAS
# ─────────────────────────────────────────────

def plot_dynamic_allocations(
    weights_df: pd.DataFrame,
    strategy_name: str = "DLS",
    top_n: int = 10,
    title: str = "Evolução das Alocações Dinâmicas",
) -> go.Figure:

    mean_weights = weights_df.mean().sort_values(ascending=False)
    top_assets = mean_weights.head(top_n).index.tolist()
    other_assets = [c for c in weights_df.columns if c not in top_assets]

    plot_df = weights_df[top_assets].copy()
    if other_assets:
        plot_df["Outros"] = weights_df[other_assets].sum(axis=1)

    colors_pool = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
    fig = go.Figure()

    for i, col in enumerate(plot_df.columns):
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df[col].values * 100,
                mode="lines",
                stackgroup="one",
                name=col,
                line=dict(width=0.5),
                fillcolor=colors_pool[i % len(colors_pool)],
                hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>Peso: %{{y:.1f}}%<extra></extra>",
            )
        )

    _apply_dark_theme(fig, f"{title} — {strategy_name}", height=420)
    fig.update_yaxes(title_text="Alocação (%)", range=[0, 100])
    fig.update_xaxes(title_text="Data")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.35,
                    xanchor="center", x=0.5, font=dict(size=10))
    )
    return fig


# ─────────────────────────────────────────────
# PAINEL 12 — SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────

def plot_sensitivity_analysis(
    sensitivity: np.ndarray,
    tickers: list,
    lookback: int = 50,
    title: str = "Sensitivity Analysis das Features de Input",
) -> go.Figure:

    n = len(tickers)
    price_sens = sensitivity[:, :n]
    return_sens = sensitivity[:, n:]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Sensibilidade — Preços Normalizados", "Sensibilidade — Retornos"],
        horizontal_spacing=0.08,
    )

    time_axis = list(range(-lookback + 1, 1))

    fig.add_trace(go.Heatmap(
        z=price_sens.T, x=time_axis, y=tickers,
        colorscale="Blues", showscale=False,
        hovertemplate="Ativo: <b>%{y}</b><br>Lag: %{x}<br>Sensibilidade: %{z:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=return_sens.T, x=time_axis, y=tickers,
        colorscale="Oranges", showscale=True,
        colorbar=dict(
            title=dict(text="Importância", font=dict(color=COLORS["subtext"], size=10)),
            tickfont=dict(color=COLORS["subtext"]),
            x=1.02,
        ),
        hovertemplate="Ativo: <b>%{y}</b><br>Lag: %{x}<br>Sensibilidade: %{z:.3f}<extra></extra>",
    ), row=1, col=2)

    _apply_dark_theme(fig, title, height=450)
    fig.update_xaxes(title_text="Lag (dias, 0=mais recente)")
    fig.update_yaxes(title_text="Ativo", row=1, col=1)
    fig.add_annotation(
        text="★ Observações recentes (lag=0) têm maior importância — consistente com Zhang et al. (2021)",
        xref="paper", yref="paper", x=0.5, y=-0.12,
        showarrow=False,
        font=dict(color=COLORS["subtext"], size=10, style="italic"),
        align="center",
    )
    return fig


# ─────────────────────────────────────────────
# PAINEL 13 — TABELA DE MÉTRICAS
# ─────────────────────────────────────────────

def plot_metrics_table(
    metrics: dict,
    title: str = "Métricas Comparativas — Todas as Estratégias",
) -> go.Figure:

    strategy_order = ["DLS", "DeepStatArb", "Markowitz", "BuyHold"]
    labels = {
        "DLS":         "DLS (Zhang et al.)",
        "DeepStatArb": "DeepStatArb",
        "Markowitz":   "Markowitz",
        "BuyHold":     "Buy & Hold",
    }
    present = [k for k in strategy_order if k in metrics]

    columns = [
        ("sharpe_ratio",      "Sharpe Ratio",      ".3f", True),
        ("annual_return",     "Retorno Anual",      ".1%", True),
        ("annual_volatility", "Volatilidade",       ".1%", False),
        ("max_drawdown",      "Max Drawdown",       ".1%", False),
        ("cumulative_return", "Retorno Acumulado",  ".1%", True),
    ]

    header_values = ["Estratégia"] + [c[1] for c in columns]
    cell_values = [[labels[k] for k in present]]

    for key, label, fmt, higher_is_better in columns:
        vals = [metrics[k].get(key, 0) for k in present]
        best_idx = vals.index(max(vals) if higher_is_better else min(vals))
        formatted = []
        for i, v in enumerate(vals):
            fv = format(v, fmt)
            if i == best_idx:
                fv = f"★ {fv}"
            formatted.append(fv)
        cell_values.append(formatted)

    # Cores rgba — sem hex+alpha
    fill_colors = [
        [CELL_BG.get(k, "rgba(30,41,59,1)") for k in present]
    ] * len(header_values)

    fig = go.Figure(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in header_values],
                fill_color=COLORS["card"],
                font=dict(color=COLORS["text"], size=12, family="monospace"),
                align="center",
                height=36,
                line=dict(color=COLORS["grid"], width=1),
            ),
            cells=dict(
                values=cell_values,
                fill_color=fill_colors,
                font=dict(color=COLORS["text"], size=11, family="monospace"),
                align="center",
                height=32,
                line=dict(color=COLORS["grid"], width=0.5),
            ),
        )
    )

    _apply_dark_theme(fig, title, height=250)
    return fig


# ─────────────────────────────────────────────
# DASHBOARD COMPLETO
# ─────────────────────────────────────────────

def build_dls_dashboard(
    cumulative_returns: dict,
    metrics: dict,
    weights_dls: pd.DataFrame,
    tickers: list,
    sensitivity: Optional[np.ndarray] = None,
    output_path: str = "dashboard_dls.html",
) -> str:

    figs = []
    figs.append(("performance_comparison",
                 plot_performance_comparison(cumulative_returns, metrics)))
    figs.append(("dynamic_allocations",
                 plot_dynamic_allocations(weights_dls, strategy_name="DLS")))

    if sensitivity is not None and len(sensitivity) > 0:
        figs.append(("sensitivity_analysis",
                     plot_sensitivity_analysis(sensitivity, tickers)))

    figs.append(("metrics_table",
                 plot_metrics_table(metrics)))

    html_parts = [_html_header()]
    for panel_id, fig in figs:
        html_parts.append(
            f'<div class="panel" id="{panel_id}">'
            + fig.to_html(full_html=False, include_plotlyjs=False, config=_plotly_config())
            + "</div>"
        )
    html_parts.append(_html_footer())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logger.info(f"[Dashboard] Salvo em: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _apply_dark_theme(fig: go.Figure, title: str, height: int = 500):
    fig.update_layout(
        title=dict(text=title,
                   font=dict(color=COLORS["text"], size=14, family="Inter, sans-serif"),
                   x=0.01),
        height=height,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False),
        legend=dict(bgcolor=COLORS["card"], bordercolor=COLORS["grid"],
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=60, r=40, t=55, b=50),
        hovermode="x unified",
    )


def _plotly_config() -> dict:
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }


def _html_header() -> str:
    return """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agente de Portfólio — Dashboard DLS</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0F172A; color: #E2E8F0;
           font-family: 'Inter', 'Segoe UI', sans-serif; padding: 24px; }
    h1 { font-size: 1.4rem; color: #00B4D8; margin-bottom: 8px; }
    .subtitle { font-size: 0.85rem; color: #94A3B8; margin-bottom: 24px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 20px;
            max-width: 1400px; margin: 0 auto; }
    .panel { background: #1E293B; border: 1px solid #334155;
             border-radius: 12px; padding: 12px; overflow: hidden; }
    .panel:hover { border-color: #00B4D8; transition: border-color 0.2s; }
    .badge { display: inline-block; background: rgba(0,180,216,0.12);
             color: #00B4D8; border: 1px solid rgba(0,180,216,0.25);
             border-radius: 6px; padding: 2px 10px; font-size: 0.75rem;
             font-family: monospace; margin-bottom: 16px; }
  </style>
</head>
<body>
  <h1>Agente de Gestão de Portfólio — Dashboard de Deep Learning</h1>
  <p class="subtitle">
    Comparativo: DeepStatArb (CNN-Transformer) vs DLS (LSTM, Zhang et al. 2021)
    vs Markowitz vs Buy &amp; Hold
  </p>
  <span class="badge">Backtest fora da amostra · 2015–2017 · 725 dias</span>
  <div class="grid">
"""


def _html_footer() -> str:
    return """
  </div>
  <p style="text-align:center; color:#475569; font-size:0.75rem; margin-top:24px;">
    Agente de Gestão de Portfólio · Zhang et al. (2021) · Markowitz (1952)
  </p>
</body>
</html>"""