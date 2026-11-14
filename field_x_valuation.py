"""Field X - integrated valuation engine (production, fiscal cash flow, economic limit, Monte Carlo).

Implements Chapter 3 of the thesis. Production comes from field_x_production.profile().
All money in nominal US$; costs stated in 2026 dollars escalate at 2%/yr; discounting
is end-of-year to 1 January 2026.

Requires: field_x_production.py in the same folder; numpy, scipy, pandas, openpyxl.
Tested with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0 (record your own versions in the thesis archive).

Run:  python field_x_valuation.py           -> deterministic cases + 10,000-iteration Monte Carlo
      python field_x_valuation.py --quick    -> 1,000 iterations (for testing)

ASSUMPTIONS THAT ARE NOT YET FIXED BY THE THESIS (change them in PARAMS below):
  * HIST_CAPEX: development capex incurred before 2026 by year (PLACEHOLDER profile of the
    US$1.5bn; replace with the operator's records). Its unrelieved capital allowances reduce
    hydrocarbon tax and income tax from 2026.
  * Capital allowances: 20% a year straight line over 5 years from the year incurred.
  * Cost-price ratio: applies to operating costs, HCDT, NDDC, decommissioning and capital
    allowances (not royalty or production allowance); excess carried forward.
  * Development levy: 4% of income-tax assessable profit (before capital allowances).
  * Flare payments and carbon cost are not tax-deductible.
  * Decommissioning: US$62m (2026$) escalated to the planned closure year (licence end or
    technical end) and funded per barrel over planned production; if the field closes
    earlier on economics, the unfunded balance is paid at closure. Opening fund balance 0.
"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_x_production import profile, CASES, tri, NP_HIST

YEARS = np.arange(2026, 2046)
NY = len(YEARS)

PARAMS = dict(
    price=50.0, esc=0.02, r_company=0.15, r_gov=0.08,
    opex_fixed=50e6, opex_var=5.0, opex_2025=50e6,
    dev_capex={2026: 220e6, 2027: 68e6, 2028: 12e6}, sust_capex=2e6,
    # PLACEHOLDER (replace with operator records): US$1.5bn spent 2021-2025 as 150/300/400/400/250 US$m;
    # spending before first production (2023) is treated as qualifying from 2023.
    hist_capex={2023: 850e6, 2024: 400e6, 2025: 250e6},
    ca_rate=0.20, ca_years=5,
    hct_rate=0.30, it_rate=0.30, dl_rate=0.04, cpr=0.65,
    pa_cap=2.50, pa_share=0.20,
    nddc=0.03, hcdt=0.03,
    decom_2026=62e6, decom_fund_2026=0.0,   # unit-of-production funding to the closure year (Section 3.4.1)
    fi=5000 / 44670, ef_flare=54.4, ei_other=13.0, fp_rate=2.00,
    carbon=None,                         # dict year->US$/t (2026 dollars) or None
    flare_out=False, fo_capex=33e6, fo_opex=2e6, fo_cut=0.90,
    licence=2037, renewal_stress=False,
    nnpc=0.55,
)
CARBON_MOD = {2030: 15, 2035: 25, 2040: 35, 2045: 45}
CARBON_SEV = {2030: 90, 2035: 125, 2040: 160, 2045: 180}

SCENARIOS = {
    "Reference (US$50)":            {},
    "Moderate price (US$35)":       dict(price=35.0),
    "Severe price (US$20)":         dict(price=20.0),
    "Moderate carbon":              dict(carbon=CARBON_MOD),
    "Severe carbon":                dict(carbon=CARBON_SEV),
    "Flare elimination":            dict(flare_out=True),
    "Combined transition":          dict(price=20.0, carbon=CARBON_SEV, flare_out=True),
    "High price (US$75)":           dict(price=75.0),
    "High price (US$90)":           dict(price=90.0),
    "Fiscal tightening (HCT 40%)":  dict(hct_rate=0.40),
    "Renewal to 2045":              dict(licence=2045),
    "Renewal-terms stress":         dict(licence=2045, renewal_stress=True),
}


def carbon_price(path, year):
    if not path: return 0.0
    ys = sorted(path)
    if year < ys[0]: return 0.0
    return float(np.interp(year, ys, [path[y] for y in ys]))


def royalty_month(P, q, prod_days, year, flat15=False):
    """Production-based (tranched) + price-based royalty for one month, US$."""
    if flat15:
        vroy = 0.15 * q
    else:
        vroy = 0.05 * min(q, 5000) + 0.075 * min(max(q - 5000, 0), 5000) + 0.15 * max(q - 10000, 0)
    prod = P * vroy * prod_days
    L, M, U = (x * 1.02 ** (year - 2020) for x in (50, 100, 150))
    if P < L: r = 0.0
    elif P < M: r = 0.05 * (P - L) / (M - L)
    elif P < U: r = 0.05 + 0.05 * (P - M) / (U - M)
    else: r = 0.10
    return prod + P * q * prod_days * r


def ca_schedule(capex_by_year, rate, nyears):
    """Capital allowances by model year from a dict {year: amount}."""
    ca = np.zeros(NY)
    for y, amt in capex_by_year.items():
        for k in range(nyears):
            yy = y + k
            if 2026 <= yy <= 2045:
                ca[yy - 2026] += amt * rate
    return ca


def valuate(monthly, R0, A, p=PARAMS, **over):
    p = {**p, **over}
    e = (1 + p["esc"]) ** (YEARS - 2026)
    # ---- annual production and royalty from monthly profile ----
    Q = np.zeros(NY); ROY = np.zeros(NY)
    for t, q, vol, _ in monthly:
        y = int(t); i = y - 2026
        if vol <= 0: continue
        Q[i] += vol * 1e6
        days = 365.0 / 12.0
        flat = p["renewal_stress"] and y >= 2038
        ROY[i] += royalty_month(p["price"], q, days * A, y, flat)
    REV = p["price"] * Q
    # ---- costs ----
    fo = p["flare_out"]
    fo_capex = np.where((YEARS == 2028) | (YEARS == 2029), p["fo_capex"] / 2, 0.0) * e if fo else np.zeros(NY)
    fo_opex = np.where(YEARS >= 2030, p["fo_opex"], 0.0) * e if fo else np.zeros(NY)
    OPEX = (p["opex_fixed"] + p["opex_var"] * Q) * e + fo_opex
    DCAPEX = np.array([p["dev_capex"].get(int(y), 0.0) for y in YEARS])
    SCAPEX = p["sust_capex"] * e
    CAPEX = DCAPEX + SCAPEX + fo_capex
    HCDT = p["hcdt"] * np.concatenate([[p["opex_2025"]], OPEX[:-1]])
    NDDC = p["nddc"] * (OPEX + CAPEX)
    fi = np.where((YEARS >= 2030) & fo, p["fi"] * (1 - p["fo_cut"]), p["fi"])
    FP = p["fp_rate"] * fi * Q
    E_t = Q * (fi * p["ef_flare"] + p["ei_other"]) / 1000.0
    CC = E_t * np.array([carbon_price(p["carbon"], int(y)) for y in YEARS]) * e
    # ---- economic limit ----
    ELCF = REV - OPEX - SCAPEX - ROY - HCDT - NDDC - FP - CC
    cum = np.cumsum(ELCF)
    T_el = int(YEARS[int(np.argmax(cum))])
    T_dep = int(YEARS[np.nonzero(Q > 0)[0][-1]]) if (Q > 0).any() else 2026
    T_s = min(T_el, T_dep, p["licence"], 2045)
    live = YEARS <= T_s
    # ---- decommissioning (Section 3.4.1) ----
    # D&A plan: cost escalated to the planned closure year (licence end or technical end),
    # funded per barrel over planned production. Actual closure T_s may come earlier;
    # the unfunded balance of the cost escalated to T_s is then paid at T_s.
    T_plan = min(T_dep, p["licence"], 2045)
    q_plan = np.where(YEARS <= T_plan, Q, 0.0)
    DC_plan = p["decom_2026"] * (1 + p["esc"]) ** (T_plan - 2026)
    rate = max(DC_plan - p["decom_fund_2026"], 0.0) / q_plan.sum() if q_plan.sum() > 0 else 0.0
    DCF = np.where(live, rate * Q, 0.0)
    DC = p["decom_2026"] * (1 + p["esc"]) ** (T_s - 2026)
    TD = np.zeros(NY)
    TD[T_s - 2026] = max(DC - p["decom_fund_2026"] - DCF.sum(), 0.0)   # shortfall only
    # truncate operating flows after closure
    for arr in (Q, REV, ROY, OPEX, CAPEX, SCAPEX, HCDT, NDDC, FP, CC, E_t):
        arr[~live] = 0.0
    # ---- capital allowances (historical + forecast pools) ----
    CA = ca_schedule({**p["hist_capex"], **{int(y): c for y, c in zip(YEARS, CAPEX) if c}}, p["ca_rate"], p["ca_years"])
    CA[~live] = 0.0
    # ---- hydrocarbon tax ----
    pa_unit = min(p["pa_cap"], p["pa_share"] * p["price"])
    PA = np.where((YEARS >= 2038) & p["renewal_stress"], 0.0, pa_unit * Q)
    hct_rate = p["hct_rate"]
    HCT = np.zeros(NY); IT = np.zeros(NY); DL = np.zeros(NY)
    cpr_cf = 0.0; hct_loss = 0.0; it_loss = 0.0
    for i in range(NY):
        if not live[i]: break
        costs = OPEX[i] + HCDT[i] + NDDC[i] + DCF[i] + TD[i] + CA[i] + cpr_cf
        cap = p["cpr"] * REV[i]
        allowed = min(costs, cap); cpr_cf = costs - allowed
        cp = REV[i] - ROY[i] - PA[i] - allowed - hct_loss
        if cp < 0: hct_loss = -cp; cp = 0.0
        else: hct_loss = 0.0
        HCT[i] = hct_rate * cp
        ap = REV[i] - ROY[i] - OPEX[i] - HCDT[i] - NDDC[i] - DCF[i] - TD[i]   # assessable profit
        DL[i] = p["dl_rate"] * max(ap, 0.0)
        tp = ap - CA[i] - it_loss
        if tp < 0: it_loss = -tp; tp = 0.0
        else: it_loss = 0.0
        IT[i] = p["it_rate"] * tp
    # ---- cash flows ----
    FCF = REV - OPEX - CAPEX - ROY - HCT - IT - DL - HCDT - NDDC - FP - DCF - TD - CC
    GFR = ROY + HCT + IT + DL + NDDC
    PFCF = REV - OPEX - CAPEX - HCDT - FP - DCF - TD - CC
    df = lambda r: (1 + r) ** -(YEARS - 2025.0)
    d15, d8 = df(p["r_company"]), df(p["r_gov"])
    npv = float((FCF * d15).sum())
    econ = Q.sum() / 1e6
    return dict(
        npv=npv, npv_nnpc=p["nnpc"] * npv, npv_private=(1 - p["nnpc"]) * npv,
        gfr15=float((GFR * d15).sum()), gfr8=float((GFR * d8).sum()),
        gt15=float((GFR * d15).sum() / (PFCF * d15).sum()) if (PFCF * d15).sum() > 0 else float("nan"),
        gt8=float((GFR * d8).sum() / (PFCF * d8).sum()) if (PFCF * d8).sum() > 0 else float("nan"),
        T_el=T_el, T_s=T_s, life=T_s - 2025, econ_prod=econ, remaining=R0 - econ,
        annual=dict(year=YEARS, Q=Q, REV=REV, ROY=ROY, OPEX=OPEX, CAPEX=CAPEX, HCT=HCT, IT=IT, DL=DL,
                    HCDT=HCDT, NDDC=NDDC, FP=FP, CC=CC, DCF=DCF, TD=TD, FCF=FCF, GFR=GFR, ELCF=ELCF),
    )


# ---------------------------------------------------------------- deterministic
def run_case(case="Base", **over):
    c = CASES[case]
    r = profile(**c, monthly=True)
    return valuate(r["monthly"], c["R0"], c["A"], **over)


def deterministic_table():
    rows = []
    for name, over in SCENARIOS.items():
        for case in ("Low", "Base", "High"):
            v = run_case(case, **over)
            rows.append((name, case, v))
    return rows


# ---------------------------------------------------------------- Monte Carlo
def sample_inputs(n, seed=2026, rho=0.5):
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    C = np.full((4, 4), rho); np.fill_diagonal(C, 1.0)
    C = 2 * np.sin(np.pi * C / 6)                       # rank -> normal correlation
    z = rng.standard_normal((n, 4)) @ np.linalg.cholesky(C).T
    u = norm.cdf(z)
    return dict(
        R0=tri(u[:, 0], 159.0, 171.3, 229.0), q_well=tri(u[:, 1], 1_650, 3_300, 4_000),
        q0=tri(u[:, 2], 52_000, 53_700, 55_000), phi=tri(u[:, 3], 0.50, 0.55, 0.60),
        A=tri(rng.random(n), 0.90, 0.95, 0.97), t_w=tri(rng.random(n), 2027.0, 2027.5, 2028.0),
        wc=tri(rng.random(n), 0.02, 0.05, 0.10),
    )


def monte_carlo(n=10_000, seed=2026, rho=0.5, scenarios=SCENARIOS):
    s = sample_inputs(n, seed, rho)
    profiles = []
    for i in range(n):                                   # production is scenario-independent
        x = {k: float(v[i]) for k, v in s.items()}
        profiles.append((profile(**x, monthly=True)["monthly"], x["R0"], x["A"]))
    out = {}
    for name, over in scenarios.items():
        res = [valuate(m, R0, A, **over) for m, R0, A in profiles]
        g = lambda k: np.array([r[k] for r in res], dtype=float)
        out[name] = {k: g(k) for k in ("npv", "npv_nnpc", "npv_private", "gfr15", "gfr8", "gt15", "gt8",
                                       "T_el", "T_s", "econ_prod", "remaining")}
    return out


def summarise(mc):
    lines = []
    for name, d in mc.items():
        npv = d["npv"] / 1e6
        p10, p50, p90 = np.percentile(npv, [10, 50, 90])
        lines.append(dict(scenario=name, mean=npv.mean(), P10=p10, P50=p50, P90=p90,
                          p_neg=(npv < 0).mean(), nnpc_mean=d["npv_nnpc"].mean() / 1e6,
                          private_mean=d["npv_private"].mean() / 1e6,
                          gfr15=d["gfr15"].mean() / 1e6, gfr8=d["gfr8"].mean() / 1e6,
                          gt15=np.nanmean(d["gt15"]), gt8=np.nanmean(d["gt8"]),
                          Ts_P50=np.percentile(d["T_s"], 50), econ_prod=d["econ_prod"].mean(),
                          remaining=d["remaining"].mean()))
    return lines


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n = 1_000 if quick else 10_000
    import pandas as pd
    det = deterministic_table()
    dt = pd.DataFrame([dict(scenario=s, case=c, npv_musd=v["npv"] / 1e6, nnpc=v["npv_nnpc"] / 1e6,
                            private=v["npv_private"] / 1e6, gfr15=v["gfr15"] / 1e6, gfr8=v["gfr8"] / 1e6,
                            gt15=v["gt15"], gt8=v["gt8"], T_el=v["T_el"], T_s=v["T_s"],
                            econ_prod=v["econ_prod"], remaining=v["remaining"]) for s, c, v in det])
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(dt.round(2).to_string(index=False))
    mc = monte_carlo(n=n)
    ms = pd.DataFrame(summarise(mc))
    print(ms.round(2).to_string(index=False))
    # reproducibility check
    mc2 = monte_carlo(n=200, scenarios={"Reference (US$50)": {}})
    mc3 = monte_carlo(n=200, scenarios={"Reference (US$50)": {}})
    print("Reproducible with seed 2026:", np.array_equal(mc2["Reference (US$50)"]["npv"], mc3["Reference (US$50)"]["npv"]))
    with pd.ExcelWriter("field_x_results.xlsx") as xw:
        dt.to_excel(xw, sheet_name="Deterministic", index=False)
        ms.to_excel(xw, sheet_name="MonteCarlo_summary", index=False)
        base = run_case("Base")["annual"]
        pd.DataFrame({k: (v if k == "year" else v / (1e6 if k != "Q" else 1e6)) for k, v in base.items()}).to_excel(
            xw, sheet_name="Base_annual_USDm_MMbbl", index=False)
    print("Wrote field_x_results.xlsx")
