"""Field X - reserves-consistent production model (reference implementation).

Monthly time step from 1 Jan 2026. Rates are producing-day (stream-day) rates;
volumes = rate x days x operational availability.

Profile logic
  1. Pre-well period: q = q0 (opening producing-day rate).
  2. New wells on stream at t_w: q = qp = min(q0 + n_wells*q_well, export cap,
     liquid cap / (1 - WC)).
  3. Plateau holds until cumulative recovery (historical + forecast) reaches
     phi * EUR_case.
  4. Exponential decline thereafter at nominal rate D derived from reserves:
         D = (Qp - Qab) / R_dec
     where Qp, Qab are annualised plateau and technical-abandonment volumes and
     R_dec is the recoverable volume left when decline starts. The decline tail
     therefore recovers exactly R_dec by the time the rate reaches q_ab, so the
     remaining-recoverable constraint is honoured without a hard cut-off.
"""
import math
import numpy as np

NP_HIST = 74.7          # MMstb produced to 31 Dec 2025 (check against records)
EXPORT_CAP = 70_000     # bopd
LIQUID_CAP = 80_000     # blpd
N_WELLS = 5
Q_AB = 2_000            # bopd technical abandonment rate (producing-day)
START = 2026
END = 2045              # renewal horizon; licence truncation applied in reporting

CASES = {
    # Opening rates are December 2025 producing-day potentials (median daily net rate, 53,700 bopd base);
    # availability from 2025 records (net production / monthly median potential = 0.95).
    #        q0     q_well  A     R0     phi   wells_on (decimal year)  WC
    "Low":  dict(q0=52_000, q_well=1_650, A=0.90, R0=159.0, phi=0.50, t_w=2028.0, wc=0.10),
    "Base": dict(q0=53_700, q_well=3_300, A=0.95, R0=171.3, phi=0.55, t_w=2027.5, wc=0.05),
    "High": dict(q0=55_000, q_well=4_000, A=0.97, R0=229.0, phi=0.60, t_w=2027.0, wc=0.02),
}

def profile(q0, q_well, A, R0, phi, t_w, wc, np_hist=NP_HIST, monthly=False):
    eur = np_hist + R0
    qp = min(q0 + N_WELLS * q_well, EXPORT_CAP, LIQUID_CAP * (1 - wc))
    q0c = min(q0, qp)                                   # opening rate cannot exceed plateau
    trigger = max(phi * eur, np_hist)                   # cumulative at which decline starts
    cum = 0.0                                           # forecast cumulative, MMstb
    months = []
    t_d = None; D = None; q = q0c
    for k in range((END - START + 1) * 12):
        t = START + k / 12.0                            # start of month (decimal year)
        days = 365.0 / 12.0
        if t_d is None:
            q = q0c if t < t_w else qp
            # decline starts once wells are on stream AND the recovery trigger is reached
            if t >= t_w - 1e-9 and np_hist + cum >= trigger:
                t_d = t
                r_dec = R0 - cum
                Qp = q * 365 * A / 1e6
                Qab = Q_AB * 365 * A / 1e6
                D = (Qp - Qab) / r_dec if r_dec > 0 else float("inf")
                q_d = q
        if t_d is not None:
            # exact monthly integral of exponential decline
            a, b = t - t_d, t - t_d + 1 / 12.0
            rate_avg = q_d * (math.exp(-D * a) - math.exp(-D * b)) / (D * (1 / 12.0))
            q = rate_avg
            if q_d * math.exp(-D * a) <= Q_AB + 1e-6:
                q = 0.0
        vol = q * days * A / 1e6
        vol = min(vol, R0 - cum)                        # numerical guard only
        cum += vol
        months.append((t, q, vol, cum))
    m = np.array(months)
    years = np.arange(START, END + 1)
    annual = np.array([m[(m[:, 0] >= y) & (m[:, 0] < y + 1), 2].sum() for y in years])
    out = dict(qp=qp, D=D, t_d=t_d, years=years, annual=annual, cum=np.cumsum(annual),
               eur=eur, R0=R0, trigger=trigger,
               max_rate=m[:, 1].max(), max_liquid=m[:, 1].max() / (1 - wc))
    if monthly:
        out["monthly"] = m
    return out


def report_deterministic():
    print("Historical: 50.5 MMstb before 1995 + 24.2 MMstb net 2023-2025 = 74.7 MMstb")
    for name, p in CASES.items():
        r = profile(**p)
        lic = r["cum"][r["years"] == 2037][0]
        tail = r["cum"][-1]
        print(f"\n{name}: qp={r['qp']:,.0f} bopd  decline starts {r['t_d']:.2f}  "
              f"D_nominal={r['D']:.3f} (effective {1-math.exp(-r['D']):.1%}/yr)  "
              f"EUR={r['eur']:.1f}")
        print("  " + "  ".join(f"{y}:{v:5.1f}" for y, v in zip(r["years"], r["annual"])))
        print(f"  cum to 2037 = {lic:.1f}  left at 2037 = {r['R0']-lic:.1f}  "
              f"cum to 2045 = {tail:.1f}  left at 2045 = {r['R0']-tail:.1f}  "
              f"max rate {r['max_rate']:,.0f}  max liquid {r['max_liquid']:,.0f}")


def tri(u, a, c, b):
    """Inverse CDF of triangular(min a, mode c, max b) for uniform u."""
    fc = (c - a) / (b - a)
    return np.where(u < fc, a + np.sqrt(u * (b - a) * (c - a)),
                    b - np.sqrt((1 - u) * (b - a) * (b - c)))


def monte_carlo(n=10_000, seed=2026, rho=0.5):
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    # Subsurface variables correlated (R0, q_well, q0, phi); availability and
    # well timing sampled independently (operational, not geological).
    names = ["R0", "q_well", "q0", "phi"]
    C = np.full((4, 4), rho); np.fill_diagonal(C, 1.0)
    # convert target rank correlation to normal correlation (Pearson-Spearman)
    C = 2 * np.sin(np.pi * C / 6)
    z = rng.standard_normal((n, 4)) @ np.linalg.cholesky(C).T
    u = norm.cdf(z)
    s = dict(
        R0=tri(u[:, 0], 159.0, 171.3, 229.0),
        q_well=tri(u[:, 1], 1_650, 3_300, 4_000),
        q0=tri(u[:, 2], 52_000, 53_700, 55_000),
        phi=tri(u[:, 3], 0.50, 0.55, 0.60),
        A=tri(rng.random(n), 0.90, 0.95, 0.97),
        t_w=tri(rng.random(n), 2027.0, 2027.5, 2028.0),
        wc=tri(rng.random(n), 0.02, 0.05, 0.10),
    )
    viol = dict(cum_gt_R0=0, rate_gt_export=0, liquid_gt_cap=0, negative=0)
    D, td, left37, cum37 = [], [], [], []
    for i in range(n):
        r = profile(**{k: float(v[i]) for k, v in s.items()})
        viol["cum_gt_R0"] += r["cum"][-1] > r["R0"] + 1e-6
        viol["rate_gt_export"] += r["max_rate"] > EXPORT_CAP + 1e-6
        viol["liquid_gt_cap"] += r["max_liquid"] > LIQUID_CAP + 1e-6
        viol["negative"] += (r["annual"] < -1e-9).any()
        D.append(1 - math.exp(-r["D"])); td.append(r["t_d"])
        c37 = r["cum"][r["years"] == 2037][0]; cum37.append(c37); left37.append(r["R0"] - c37)
    pct = lambda x: np.percentile(x, [10, 50, 90])
    print(f"\nMonte Carlo n={n}, seed={seed}, rank corr={rho}")
    print("  violations:", viol)
    print("  effective decline P10/P50/P90:", np.round(pct(D), 3))
    print("  decline-start year P10/P50/P90:", np.round(pct(td), 2))
    print("  cum to 2037 P10/P50/P90 (MMstb):", np.round(pct(cum37), 1))
    print("  recoverable left at 2037 P10/P50/P90:", np.round(pct(left37), 1))


if __name__ == "__main__":
    report_deterministic()
    monte_carlo()
