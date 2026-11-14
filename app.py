"""Field X valuation - Streamlit front end.

Main file for Streamlit Cloud. Needs field_x_valuation.py, field_x_production.py and
requirements.txt in the same folder. Run locally with:  streamlit run app.py
"""
import io
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import field_x_valuation as fx

st.set_page_config(page_title="Field X valuation", layout="wide")

CARBON_PATHS = {"None": None, "Moderate (IEA NZE, EMDE)": fx.CARBON_MOD,
                "Severe (IEA NZE, net-zero pledges)": fx.CARBON_SEV}
LINE_ITEMS = ["REV", "OPEX", "CAPEX", "ROY", "HCT", "IT", "DL", "HCDT", "NDDC", "FP", "CC", "DCF", "TD", "FCF", "GFR"]
LABELS = {"REV": "Revenue", "OPEX": "Opex", "CAPEX": "Capex", "ROY": "Royalty", "HCT": "Hydrocarbon tax",
          "IT": "Income tax", "DL": "Development levy", "HCDT": "HCDT", "NDDC": "NDDC levy",
          "FP": "Flare payments", "CC": "Carbon cost", "DCF": "Decom. contribution",
          "TD": "Terminal decom.", "FCF": "Free cash flow", "GFR": "Gov. fiscal receipts"}

# ------------------------------------------------------------------ controls
DEFAULTS = dict(price=50.0, carbon="None", flare_out=False, hct=30, licence=2037,
                renewal_stress=False, fp_rate=2.00, r_company=15.0)


def scenario_to_controls(over):
    c = dict(DEFAULTS)
    c["price"] = float(over.get("price", c["price"]))
    path = over.get("carbon")
    c["carbon"] = next(k for k, v in CARBON_PATHS.items() if v == path)
    c["flare_out"] = bool(over.get("flare_out", False))
    c["hct"] = int(round(100 * over.get("hct_rate", 0.30)))
    c["licence"] = int(over.get("licence", 2037))
    c["renewal_stress"] = bool(over.get("renewal_stress", False))
    return c


def apply_preset():
    for k, v in scenario_to_controls(fx.SCENARIOS[st.session_state.preset]).items():
        st.session_state[k] = v


for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.header("Inputs")
    st.selectbox("Scenario preset", list(fx.SCENARIOS), key="preset", on_change=apply_preset,
                 help="Loads the scenario's settings into the controls below. You can then adjust them.")
    case = st.radio("Field case (deterministic view)", ["Low", "Base", "High"], index=1, horizontal=True)
    st.slider("Oil price, 2026 (US$/bbl)", 10.0, 120.0, step=1.0, key="price")
    st.selectbox("Carbon-price path", list(CARBON_PATHS), key="carbon")
    st.checkbox("Flare-out project (2028-29)", key="flare_out")
    st.slider("Hydrocarbon tax rate (%)", 20, 50, step=1, key="hct")
    st.radio("Licence end", [2037, 2045], horizontal=True, key="licence")
    st.checkbox("Renewal-terms stress (flat 15% royalty after 2037)", key="renewal_stress",
                disabled=st.session_state.licence == 2037)
    st.slider("Flare payment (US$/Mscf)", 0.0, 3.5, step=0.25, key="fp_rate")
    st.slider("Company discount rate (%)", 8.0, 25.0, step=0.5, key="r_company")

over = dict(price=st.session_state.price, carbon=CARBON_PATHS[st.session_state.carbon],
            flare_out=st.session_state.flare_out, hct_rate=st.session_state.hct / 100,
            licence=st.session_state.licence,
            renewal_stress=st.session_state.renewal_stress and st.session_state.licence == 2045,
            fp_rate=st.session_state.fp_rate, r_company=st.session_state.r_company / 100)


def encode(o):
    """Cache key: carbon dicts have int keys, so store the path name instead."""
    o = dict(o)
    o["carbon"] = next(k for k, v in CARBON_PATHS.items() if v == o.get("carbon"))
    return json.dumps(o, sort_keys=True)


def decode(key):
    o = json.loads(key)
    o["carbon"] = CARBON_PATHS[o["carbon"]]
    return o


over_key = encode(over)


# ------------------------------------------------------------------ cached computation
@st.cache_data(show_spinner=False)
def deterministic(case, over_key):
    v = fx.run_case(case, **decode(over_key))
    a = v.pop("annual")
    annual = pd.DataFrame({k: a[k] for k in ["year", "Q"] + LINE_ITEMS})
    annual["Q"] = annual["Q"] / 1e6
    for k in LINE_ITEMS:
        annual[k] = annual[k] / 1e6
    annual = annual[annual["year"] <= v["T_s"]]
    return v, annual


@st.cache_resource(show_spinner=False)
def mc_profiles(n, seed):
    s = fx.sample_inputs(n, seed)
    out = []
    for i in range(n):
        x = {k: float(v[i]) for k, v in s.items()}
        out.append((fx.profile(**x, monthly=True)["monthly"], x["R0"], x["A"]))
    return out


@st.cache_data(show_spinner=False)
def mc_run(n, seed, over_key):
    o = decode(over_key)
    res = [fx.valuate(m, R0, A, **o) for m, R0, A in mc_profiles(n, seed)]
    return pd.DataFrame({k: [r[k] for r in res] for k in
                         ("npv", "npv_nnpc", "npv_private", "gfr15", "gt15", "T_s", "econ_prod")})


def musd(x):
    return f"US${x / 1e6:,.0f}m"


def to_excel(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


# ------------------------------------------------------------------ page
st.title("Field X integrated valuation")
st.caption("Onshore Niger Delta UJV (NNPC 55%, private 45%). Valuation date 1 January 2026, nominal US$, "
           "costs escalated at 2% a year. Historical capex phasing and four tax treatments are placeholders "
           "(see Chapter 3).")

tab_det, tab_mc, tab_all = st.tabs(["Deterministic", "Monte Carlo", "All scenarios"])

with tab_det:
    v, annual = deterministic(case, over_key)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Project NPV @ {over['r_company']:.1%}", musd(v["npv"]))
    c2.metric("NNPC share (55%)", musd(v["npv_nnpc"]))
    c3.metric("Private share (45%)", musd(v["npv_private"]))
    c4.metric("Government take", f"{v['gt15']:.0%}" if np.isfinite(v["gt15"]) else "n/a")
    c5.metric("Closure year", v["T_s"], help=f"Economic limit {v['T_el']}")
    st.caption(f"{case} case: economic production {v['econ_prod']:.1f} MMstb, "
               f"oil left at closure {v['remaining']:.1f} MMstb.")

    left, right = st.columns(2)
    with left:
        st.subheader("Production (MMstb/yr)")
        st.bar_chart(annual.set_index("year")["Q"])
    with right:
        st.subheader("Free cash flow vs government receipts (US$m)")
        long = annual.melt("year", ["FCF", "GFR"], var_name="item", value_name="US$m")
        long["item"] = long["item"].map(LABELS)
        st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("year:O", title=None), y="US$m:Q", color=alt.Color("item:N", title=None)),
            width="stretch")

    st.subheader("Annual cash flow (US$m; production in MMstb)")
    show = annual.rename(columns={**LABELS, "Q": "Production", "year": "Year"})
    st.dataframe(show.style.format({c: "{:,.1f}" for c in show.columns if c != "Year"}),
                 hide_index=True, width="stretch")
    st.download_button("Download annual cash flow (Excel)", to_excel({"Annual": show}),
                       f"field_x_{case.lower()}_annual.xlsx")

with tab_mc:
    st.write("Samples remaining oil, well rate, opening rate, plateau fraction, availability, "
             "on-stream date and water cut (triangular, rank correlation 0.5), then values every "
             "sample under the current inputs.")
    a, b = st.columns(2)
    n = a.select_slider("Iterations", [500, 1000, 2000, 5000, 10000], value=1000,
                        help="10,000 matches the thesis; the first run takes about a minute.")
    seed = b.number_input("Random seed", value=2026, step=1)
    if st.button("Run Monte Carlo", type="primary"):
        st.session_state.mc_args = (int(n), int(seed), over_key)
    if "mc_args" in st.session_state:
        n_, seed_, key_ = st.session_state.mc_args
        if key_ != over_key:
            st.info("Inputs have changed since the last run. Press Run Monte Carlo to update.")
        with st.spinner(f"Running {n_:,} iterations..."):
            df = mc_run(n_, seed_, key_)
        npv = df["npv"] / 1e6
        p10, p50, p90 = np.percentile(npv, [10, 50, 90])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Mean NPV", f"US${npv.mean():,.0f}m")
        m2.metric("P10", f"US${p10:,.0f}m")
        m3.metric("P50", f"US${p50:,.0f}m")
        m4.metric("P90", f"US${p90:,.0f}m")
        m5.metric("P(NPV < 0)", f"{(npv < 0).mean():.1%}")
        st.caption("P10 is the low value (90% of outcomes exceed it).")
        hist = pd.DataFrame({"NPV (US$m)": npv})
        chart = alt.Chart(hist).mark_bar().encode(
            x=alt.X("NPV (US$m):Q", bin=alt.Bin(maxbins=50)), y=alt.Y("count():Q", title="Iterations"))
        rules = alt.Chart(pd.DataFrame({"x": [p10, p50, p90], "label": ["P10", "P50", "P90"]})).mark_rule(
            strokeDash=[4, 3]).encode(x="x:Q", tooltip=["label", alt.Tooltip("x:Q", format=",.0f")])
        st.altair_chart(chart + rules, width="stretch")
        out = df.assign(**{c: df[c] / 1e6 for c in ("npv", "npv_nnpc", "npv_private", "gfr15")})
        out.columns = ["NPV US$m", "NNPC US$m", "Private US$m", "GFR@15% US$m", "Gov take", "Closure year",
                       "Econ. production MMstb"]
        st.download_button("Download iterations (Excel)", to_excel({"Iterations": out}),
                           "field_x_monte_carlo.xlsx")

with tab_all:
    st.write("Deterministic NPV of the twelve thesis scenarios for the three field cases, at the "
             "discount rate and flare payment set in the sidebar. Other sidebar inputs are ignored here.")
    rows = []
    for name, sc in fx.SCENARIOS.items():
        o = {**sc, "fp_rate": over["fp_rate"], "r_company": over["r_company"]}
        for c in ("Low", "Base", "High"):
            r, _ = deterministic(c, encode(o))
            rows.append(dict(Scenario=name, Case=c, NPV=r["npv"] / 1e6, NNPC=r["npv_nnpc"] / 1e6,
                             Private=r["npv_private"] / 1e6, GovTake=r["gt15"], Closure=r["T_s"]))
    tbl = pd.DataFrame(rows)
    st.altair_chart(alt.Chart(tbl).mark_bar().encode(
        y=alt.Y("Scenario:N", sort=list(fx.SCENARIOS), title=None),
        x=alt.X("NPV:Q", title="NPV (US$m)"), color=alt.Color("Case:N", sort=["Low", "Base", "High"]),
        yOffset=alt.YOffset("Case:N", sort=["Low", "Base", "High"]),
        tooltip=["Scenario", "Case", alt.Tooltip("NPV:Q", format=",.0f")]).properties(height=520),
        width="stretch")
    st.dataframe(tbl.style.format({"NPV": "{:,.0f}", "NNPC": "{:,.0f}", "Private": "{:,.0f}",
                                   "GovTake": "{:.0%}"}), hide_index=True, width="stretch")
    st.download_button("Download scenario table (Excel)", to_excel({"Scenarios": tbl}), "field_x_scenarios.xlsx")
