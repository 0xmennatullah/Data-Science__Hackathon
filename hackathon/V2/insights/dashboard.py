"""
Vehicle Sales — Interactive Plotly Dash Dashboard
Run:  python dashboard.py
Open: http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback
import warnings
warnings.filterwarnings("ignore")

# ─── Load & prepare data ───────────────────────────────────────────────────
RAW_PATH = "/home/mennatullah/Documents/repos/li/AI/instant/hackathon/VehicleSales.parquet"

df = pd.read_parquet(RAW_PATH)

# Same cleaning as notebook
df["Transmission"] = df["Transmission"].replace({"sedan": "automatic", "Sedan": "automatic"})
df["SaleDate"]     = pd.to_datetime(df["SaleDate"], errors="coerce")
df.dropna(subset=["SaleDate", "SellingPrice", "MMR"], inplace=True)
Q1, Q3 = df["SellingPrice"].quantile([0.01, 0.99])
df = df[(df["SellingPrice"] >= Q1) & (df["SellingPrice"] <= Q3)]
for col in ["Make", "Model", "Body", "Color", "State"]:
    df[col] = df[col].str.strip().str.title()
df["Transmission"] = df["Transmission"].str.strip().str.lower()
df["SaleYear"]     = df["SaleDate"].dt.year
df["SaleMonth"]    = df["SaleDate"].dt.to_period("M").astype(str)
df["VehicleAge"]   = df["SaleYear"] - df["Year"]
df["PriceDiff"]    = df["SellingPrice"] - df["MMR"]
df["PriceDiffPct"] = (df["PriceDiff"] / df["MMR"]) * 100

bins   = [0, 5000, 10000, 15000, 20000, 30000, 50000, 999999]
labels = ["<$5k", "$5-10k", "$10-15k", "$15-20k", "$20-30k", "$30-50k", ">$50k"]
df["PriceBand"] = pd.cut(df["SellingPrice"], bins=bins, labels=labels)

# Filter options
all_makes  = sorted(df["Make"].dropna().unique())
all_bodies = sorted(df["Body"].dropna().unique())
all_states = sorted(df["State"].dropna().unique())
year_min, year_max = int(df["Year"].min()), int(df["Year"].max())

# ─── Colour palette ────────────────────────────────────────────────────────
COLORS = ["#1E3A5F", "#2D6A9F", "#4CA3DD", "#7EC8E3", "#FFB347",
          "#FF6B35", "#A3C4BC", "#94A3B8", "#CBD5E1"]
ACCENT = "#FF6B35"
BG     = "#F7F9FC"
CARD   = "#FFFFFF"

# ─── App ──────────────────────────────────────────────────────────────────
app = Dash(__name__, title="🚗 Vehicle Sales Dashboard")

# ── Sidebar filters ────────────────────────────────────────────────────────
sidebar = html.Div([
    html.Div([
        html.H2("🚗 Vehicle Sales", style={"color": "#1E3A5F", "marginBottom": "4px"}),
        html.P("Interactive Dashboard", style={"color": "#64748B", "fontSize": "13px"}),
    ], style={"borderBottom": "2px solid #FF6B35", "paddingBottom": "12px", "marginBottom": "20px"}),

    html.Label("Make", style={"fontWeight": "600", "color": "#1E3A5F"}),
    dcc.Dropdown(
        id="filter-make",
        options=[{"label": m, "value": m} for m in all_makes],
        multi=True, placeholder="All makes",
        style={"marginBottom": "16px"}
    ),

    html.Label("Body Type", style={"fontWeight": "600", "color": "#1E3A5F"}),
    dcc.Dropdown(
        id="filter-body",
        options=[{"label": b, "value": b} for b in all_bodies],
        multi=True, placeholder="All body types",
        style={"marginBottom": "16px"}
    ),

    html.Label("State", style={"fontWeight": "600", "color": "#1E3A5F"}),
    dcc.Dropdown(
        id="filter-state",
        options=[{"label": s, "value": s} for s in all_states],
        multi=True, placeholder="All states",
        style={"marginBottom": "16px"}
    ),

    html.Label("Transmission", style={"fontWeight": "600", "color": "#1E3A5F"}),
    dcc.Checklist(
        id="filter-trans",
        options=[{"label": " Automatic", "value": "automatic"},
                 {"label": " Manual",    "value": "manual"}],
        value=["automatic", "manual"],
        style={"marginBottom": "16px", "fontSize": "14px"}
    ),

    html.Label("Model Year Range", style={"fontWeight": "600", "color": "#1E3A5F"}),
    dcc.RangeSlider(
        id="filter-year",
        min=2005, max=year_max, step=1,
        value=[2010, year_max],
        marks={y: str(y) for y in range(2005, year_max+1, 2)},
        tooltip={"placement": "bottom"},
    ),

    html.Div(id="record-count", style={
        "marginTop": "24px", "padding": "10px", "background": "#EFF6FF",
        "borderRadius": "8px", "textAlign": "center",
        "color": "#1E3A5F", "fontWeight": "600"
    }),
], style={
    "width": "240px", "minWidth": "240px", "background": CARD,
    "padding": "24px 16px", "height": "100vh", "overflowY": "auto",
    "boxShadow": "2px 0 8px rgba(0,0,0,0.07)", "position": "fixed",
    "top": "0", "left": "0", "zIndex": "100"
})


def kpi_card(title, id_val, bg="#1E3A5F"):
    return html.Div([
        html.P(title, style={"margin": "0 0 4px 0", "fontSize": "11px",
                              "color": "#94A3B8", "textTransform": "uppercase",
                              "letterSpacing": "0.05em"}),
        html.H3(id=id_val, style={"margin": "0", "color": "white", "fontSize": "22px"}),
    ], style={
        "background": bg, "borderRadius": "12px",
        "padding": "16px 20px", "flex": "1", "minWidth": "140px"
    })


main = html.Div([
    # KPI row
    html.Div([
        kpi_card("Total Vehicles",   "kpi-total",   "#1E3A5F"),
        kpi_card("Median Price",     "kpi-median",  "#2D6A9F"),
        kpi_card("Avg Odometer",     "kpi-odo",     "#4CA3DD"),
        kpi_card("% Above MMR",      "kpi-above",   "#FF6B35"),
        kpi_card("Avg Age at Sale",  "kpi-age",     "#FFB347"),
    ], style={"display": "flex", "gap": "14px", "marginBottom": "20px", "flexWrap": "wrap"}),

    # Row 1
    html.Div([
        html.Div(dcc.Graph(id="chart-price-dist"),   style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "12px"}),
        html.Div(dcc.Graph(id="chart-top-makes"),    style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "12px"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

    # Row 2
    html.Div([
        html.Div(dcc.Graph(id="chart-body-pie"),     style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "12px"}),
        html.Div(dcc.Graph(id="chart-price-body"),   style={"flex": "1.5", "background": CARD, "borderRadius": "12px", "padding": "12px"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

    # Row 3
    html.Div([
        html.Div(dcc.Graph(id="chart-mmr-scatter"),  style={"flex": "1.5", "background": CARD, "borderRadius": "12px", "padding": "12px"}),
        html.Div(dcc.Graph(id="chart-cond-price"),   style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "12px"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

    # Row 4 - full width time series
    html.Div(dcc.Graph(id="chart-timeseries"),
             style={"background": CARD, "borderRadius": "12px", "padding": "12px", "marginBottom": "16px"}),

    # Row 5
    html.Div([
        html.Div(dcc.Graph(id="chart-yr-price"),     style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "12px"}),
        html.Div(dcc.Graph(id="chart-mmr-diff"),     style={"flex": "1", "background": CARD, "borderRadius": "12px", "padding": "12px"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

    # Row 6 - heatmap
    html.Div(dcc.Graph(id="chart-heatmap"),
             style={"background": CARD, "borderRadius": "12px", "padding": "12px", "marginBottom": "16px"}),

], style={"marginLeft": "272px", "padding": "24px"})


app.layout = html.Div([sidebar, main],
    style={"fontFamily": "'Segoe UI', Arial, sans-serif", "background": BG, "minHeight": "100vh"})


# ─── Shared filter helper ─────────────────────────────────────────────────
def apply_filters(makes, bodies, states, trans, yr_range):
    d = df.copy()
    if makes:   d = d[d["Make"].isin(makes)]
    if bodies:  d = d[d["Body"].isin(bodies)]
    if states:  d = d[d["State"].isin(states)]
    if trans:   d = d[d["Transmission"].isin(trans)]
    d = d[(d["Year"] >= yr_range[0]) & (d["Year"] <= yr_range[1])]
    return d


INPUTS = [
    Input("filter-make",  "value"),
    Input("filter-body",  "value"),
    Input("filter-state", "value"),
    Input("filter-trans", "value"),
    Input("filter-year",  "value"),
]

LAYOUT_BASE = dict(
    paper_bgcolor=CARD, plot_bgcolor=BG,
    font=dict(family="Segoe UI, Arial", color="#1E3A5F"),
    margin=dict(t=40, b=30, l=30, r=20),
    colorway=COLORS,
)


# ── KPIs ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-total",  "children"),
    Output("kpi-median", "children"),
    Output("kpi-odo",    "children"),
    Output("kpi-above",  "children"),
    Output("kpi-age",    "children"),
    Output("record-count", "children"),
    *INPUTS
)
def update_kpis(makes, bodies, states, trans, yr):
    d = apply_filters(makes, bodies, states, trans, yr)
    n = len(d)
    above = (d["PriceDiff"] > 0).mean() * 100 if n else 0
    return (
        f"{n:,}",
        f"${d['SellingPrice'].median():,.0f}" if n else "—",
        f"{d['Odometer'].mean():,.0f} mi"      if n else "—",
        f"{above:.1f}%"                         if n else "—",
        f"{d['VehicleAge'].mean():.1f} yrs"    if n else "—",
        f"📋 {n:,} records"
    )


# ── Price distribution ────────────────────────────────────────────────────
@app.callback(Output("chart-price-dist", "figure"), *INPUTS)
def price_dist(makes, bodies, states, trans, yr):
    d = apply_filters(makes, bodies, states, trans, yr)
    fig = px.histogram(d, x="SellingPrice", nbins=80,
                       title="Selling Price Distribution",
                       color_discrete_sequence=[COLORS[2]])
    fig.add_vline(x=d["SellingPrice"].median(), line_dash="dash",
                  line_color=ACCENT, annotation_text=f"Median ${d['SellingPrice'].median():,.0f}")
    fig.update_xaxes(tickprefix="$", title="Price")
    fig.update_layout(**LAYOUT_BASE)
    return fig


# ── Top makes bar ─────────────────────────────────────────────────────────
@app.callback(Output("chart-top-makes", "figure"), *INPUTS)
def top_makes_chart(makes, bodies, states, trans, yr):
    d = apply_filters(makes, bodies, states, trans, yr)
    top = d["Make"].value_counts().head(12).reset_index()
    top.columns = ["Make", "Count"]
    fig = px.bar(top, x="Count", y="Make", orientation="h",
                 title="Top Makes by Volume",
                 color="Count", color_continuous_scale=["#7EC8E3", "#1E3A5F"])
    fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False,
                      yaxis=dict(categoryorder="total ascending"))
    return fig


# ── Body pie ──────────────────────────────────────────────────────────────
@app.callback(Output("chart-body-pie", "figure"), *INPUTS)
def body_pie(makes, bodies, states, trans, yr):
    d = apply_filters(makes, bodies, states, trans, yr)
    top = d["Body"].value_counts().head(8).reset_index()
    top.columns = ["Body", "Count"]
    fig = px.pie(top, values="Count", names="Body", title="Body Type Share",
                 hole=0.45, color_discrete_sequence=COLORS)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(**LAYOUT_BASE)
    return fig


# ── Price by body type ────────────────────────────────────────────────────
@app.callback(Output("chart-price-body", "figure"), *INPUTS)
def price_body(makes, bodies, states, trans, yr):
    d = apply_filters(makes, bodies, states, trans, yr)
    top6 = d["Body"].value_counts().head(6).index
    d2   = d[d["Body"].isin(top6)]
    fig  = px.box(d2, x="Body", y="SellingPrice",
                  title="Price Distribution by Body Type",
                  color="Body", color_discrete_sequence=COLORS,
                  points=False)
    fig.update_yaxes(tickprefix="$", title="Selling Price")
    fig.update_layout(**LAYOUT_BASE, showlegend=False)
    return fig


# ── MMR scatter ───────────────────────────────────────────────────────────
@app.callback(Output("chart-mmr-scatter", "figure"), *INPUTS)
def mmr_scatter(makes, bodies, states, trans, yr):
    d = apply_filters(makes, bodies, states, trans, yr).dropna(subset=["MMR"])
    samp = d.sample(min(6000, len(d)), random_state=42)
    fig = px.scatter(samp, x="MMR", y="SellingPrice",
                     color="PriceDiffPct",
                     color_continuous_scale="RdYlGn",
                     range_color=[-25, 25],
                     opacity=0.4, size_max=4,
                     title="MMR vs Selling Price (colored by % diff from market)",
                     labels={"PriceDiffPct": "% vs MMR"})
    fig.add_shape(type="line", x0=0, y0=0, x1=70000, y1=70000,
                  line=dict(color="#475569", dash="dash", width=1.5))
    fig.update_xaxes(tickprefix="$", range=[0, 70000])
    fig.update_yaxes(tickprefix="$", range=[0, 70000])
    fig.update_layout(**LAYOUT_BASE)
    return fig


# ── Condition vs price ────────────────────────────────────────────────────
@app.callback(Output("chart-cond-price", "figure"), *INPUTS)
def cond_price_chart(makes, bodies, states, trans, yr):
    d = apply_filters(makes, bodies, states, trans, yr).dropna(subset=["ConditionValue"])
    d["CondBin"] = pd.cut(d["ConditionValue"], bins=[0,10,20,30,40,50],
                           labels=["1-10","11-20","21-30","31-40","41-50"])
    cp = d.groupby("CondBin", observed=True)["SellingPrice"].median().reset_index()
    fig = px.bar(cp, x="CondBin", y="SellingPrice",
                 title="Median Price by Condition Score",
                 color="SellingPrice",
                 color_continuous_scale=["#7EC8E3","#1E3A5F"])
    fig.update_yaxes(tickprefix="$", title="Median Price")
    fig.update_xaxes(title="Condition Band")
    fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False)
    return fig


# ── Time series ───────────────────────────────────────────────────────────
@app.callback(Output("chart-timeseries", "figure"), *INPUTS)
def timeseries(makes, bodies, states, trans, yr):
    d   = apply_filters(makes, bodies, states, trans, yr)
    vol = d.groupby("SaleMonth").size().reset_index(name="Volume")
    rev = d.groupby("SaleMonth")["SellingPrice"].sum().reset_index()
    rev["Revenue_M"] = rev["SellingPrice"] / 1e6

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=vol["SaleMonth"], y=vol["Volume"],
                              fill="tozeroy", name="Volume",
                              line=dict(color=COLORS[0], width=2.5),
                              fillcolor="rgba(44,106,159,0.15)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=rev["SaleMonth"], y=rev["Revenue_M"],
                              name="Revenue ($M)", mode="lines+markers",
                              line=dict(color=ACCENT, dash="dash", width=2),
                              marker=dict(size=5)), secondary_y=True)
    fig.update_layout(title="Monthly Sales Volume & Revenue",
                      **LAYOUT_BASE, legend=dict(x=0.01, y=0.99))
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="# Vehicles",  secondary_y=False)
    fig.update_yaxes(title_text="Revenue ($M)", secondary_y=True)
    return fig


# ── Median price by model year ────────────────────────────────────────────
@app.callback(Output("chart-yr-price", "figure"), *INPUTS)
def yr_price_chart(makes, bodies, states, trans, yr):
    d  = apply_filters(makes, bodies, states, trans, yr)
    yp = d[d["Year"] >= 2005].groupby("Year")["SellingPrice"].median().reset_index()
    fig = px.area(yp, x="Year", y="SellingPrice",
                  title="Median Price by Model Year",
                  color_discrete_sequence=[COLORS[0]])
    fig.update_yaxes(tickprefix="$", title="Median Price")
    fig.update_layout(**LAYOUT_BASE)
    return fig


# ── Above/below MMR by make ───────────────────────────────────────────────
@app.callback(Output("chart-mmr-diff", "figure"), *INPUTS)
def mmr_diff_chart(makes, bodies, states, trans, yr):
    d = apply_filters(makes, bodies, states, trans, yr)
    top12 = d["Make"].value_counts().head(12).index
    diff  = (d[d["Make"].isin(top12)]
             .groupby("Make")["PriceDiffPct"].median()
             .sort_values().reset_index())
    diff.columns = ["Make", "PctDiff"]
    diff["Color"] = diff["PctDiff"].apply(lambda x: ACCENT if x > 0 else COLORS[0])
    fig = px.bar(diff, x="PctDiff", y="Make", orientation="h",
                 title="Median % Above / Below MMR by Make",
                 color="Color", color_discrete_map="identity")
    fig.add_vline(x=0, line_color="#475569", line_dash="dash", line_width=1.5)
    fig.update_xaxes(title="% diff from MMR")
    fig.update_layout(**LAYOUT_BASE, showlegend=False)
    return fig


# ── Make × Body heatmap ───────────────────────────────────────────────────
@app.callback(Output("chart-heatmap", "figure"), *INPUTS)
def heatmap_chart(makes, bodies, states, trans, yr):
    d     = apply_filters(makes, bodies, states, trans, yr)
    top8m = d["Make"].value_counts().head(8).index
    top6b = d["Body"].value_counts().head(6).index
    heat  = (d[d["Make"].isin(top8m) & d["Body"].isin(top6b)]
             .groupby(["Make","Body"])["SellingPrice"].median()
             .unstack().fillna(0) / 1000)
    fig = px.imshow(heat, text_auto=".0f",
                    color_continuous_scale="Blues",
                    title="Median Price ($k): Top Makes × Body Types",
                    aspect="auto")
    fig.update_coloraxes(colorbar_title="$k")
    fig.update_layout(**LAYOUT_BASE)
    return fig


# ─── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚗  Vehicle Sales Dashboard")
    print("  Open → http://127.0.0.1:8050")
    print("="*55 + "\n")
    app.run(debug=False, host="127.0.0.1", port=8050)