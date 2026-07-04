"""
Econometric Methods Comparison: Title Insurance as a Refinance Friction
=======================================================================
Simulates county-level mortgage panel data and estimates the effect of
title-insurance costs on refinancing activity using six identification
strategies. Compares point estimates, standard errors, and assumptions.

Data-generating process mirrors the proposed empirical design:
  - Counties in two states (high vs. low title cost)
  - Border-county pairs share local economic conditions
  - National rate shocks create exogenous refinancing incentives
  - Title cost friction suppresses refi response (true beta = -0.4)

Methods compared:
  1. Pooled OLS (naive)
  2. Two-way Fixed Effects (TWFE) DiD
  3. Triple-Difference (state x purpose x rate-shock)
  4. Event Study (TWFE with leads/lags)
  5. Propensity-Score Matching (cross-section)
  6. Synthetic Control (Iowa vs. synthetic Iowa)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Optional: install with pip install linearmodels
try:
    from linearmodels.panel import PanelOLS, BetweenOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("linearmodels not installed — TWFE will use statsmodels dummies instead.")

import statsmodels.formula.api as smf
import statsmodels.api as sm

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. DATA GENERATION
# ---------------------------------------------------------------------------

def generate_data(
    n_counties=120,          # 60 per state, 30 border pairs each side
    n_periods=32,            # quarterly, 2017Q1–2024Q4
    true_beta=-0.40,         # true triple-diff coefficient
    border_corr=0.70,        # within-pair correlation in local shocks
):
    """
    Generate county x purpose x quarter panel.
    Treatment = HighTitleCost state (state B vs state A).
    RefIncentive = national rate-gap measure (exogenous).
    Outcome = log refinance (or purchase) applications.
    """
    rng = np.random.default_rng(42)

    periods = pd.period_range("2017Q1", periods=n_periods, freq="Q")
    n_border_pairs = n_counties // 2  # each pair has one county per state

    # ---- national rate incentive (exogenous, common) ----------------------
    # Simulate a rate-gap series: rises in 2020-21 (refi boom), drops after
    base_rate = np.linspace(0, 0, n_periods)
    rate_shock = np.zeros(n_periods)
    rate_shock[12:20] = np.linspace(0.5, 1.2, 8)   # 2020-21 boom
    rate_shock[20:24] = np.linspace(1.2, 0.3, 4)   # 2022 taper
    rate_shock += rng.normal(0, 0.05, n_periods)
    refi_incentive = pd.Series(base_rate + rate_shock, index=periods)

    # ---- county characteristics -------------------------------------------
    pairs = np.arange(n_border_pairs)
    pair_shock = rng.normal(0, 0.3, n_border_pairs)  # shared within-pair

    rows = []
    for pair_id in pairs:
        for state in ["A_low_cost", "B_high_cost"]:
            treated = (state == "B_high_cost")
            title_cost = 900 if treated else 350   # dollars, refi policy
            county_fe = pair_shock[pair_id] + rng.normal(0, 0.1)

            for t, period in enumerate(periods):
                for purpose in ["refi", "purchase"]:
                    is_refi = (purpose == "refi")

                    # baseline log applications
                    y = (
                        4.0                                          # intercept
                        + county_fe                                  # county FE
                        + 0.05 * t                                   # time trend
                        + 1.2 * refi_incentive.iloc[t] * is_refi    # rate -> refi
                        # true treatment: title cost suppresses refi response
                        + true_beta * (title_cost / 1000)
                          * refi_incentive.iloc[t] * is_refi
                        + rng.normal(0, 0.15)                       # idio error
                    )

                    rows.append({
                        "county_id": f"pair{pair_id:02d}_{state[:1]}",
                        "pair_id": pair_id,
                        "state": state,
                        "treated": int(treated),
                        "title_cost": title_cost,
                        "period": str(period),
                        "t": t,
                        "purpose": purpose,
                        "is_refi": int(is_refi),
                        "refi_incentive": refi_incentive.iloc[t],
                        "log_apps": y,
                        # post = period where incentive is meaningfully positive
                        "post": int(refi_incentive.iloc[t] > 0.3),
                        "period_num": t,
                    })

    df = pd.DataFrame(rows)
    df["county_purpose"] = df["county_id"] + "_" + df["purpose"]
    return df, refi_incentive


df, refi_incentive = generate_data()
print(f"Panel shape: {df.shape}  |  Unique county-purposes: {df['county_purpose'].nunique()}")
print(df[["county_id","state","treated","title_cost","purpose","log_apps"]].head(8))

# ---------------------------------------------------------------------------
# 2. METHOD 1 — POOLED OLS (naive)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("METHOD 1: Pooled OLS")
print("="*60)

refi_df = df[df["is_refi"] == 1].copy()
ols_model = smf.ols(
    "log_apps ~ treated * refi_incentive + title_cost + t",
    data=refi_df
).fit(cov_type="HC3")

ols_coef = ols_model.params["treated:refi_incentive"]
ols_se   = ols_model.bse["treated:refi_incentive"]
print(f"  Coeff (treated x refi_incentive): {ols_coef:.4f}  SE: {ols_se:.4f}")
print(f"  [Bias: omits county FE and purchase-loan comparison]")

# ---------------------------------------------------------------------------
# 3. METHOD 2 — TWO-WAY FIXED EFFECTS DiD
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("METHOD 2: Two-Way Fixed Effects DiD")
print("="*60)

# Add county and period dummies (within refi loans only)
refi_df2 = refi_df.copy()
refi_df2["county_id_cat"] = refi_df2["county_id"].astype("category")
refi_df2["period_cat"]    = refi_df2["period"].astype("category")

twfe_formula = (
    "log_apps ~ treated:refi_incentive + C(county_id) + C(period)"
)
twfe_model = smf.ols(twfe_formula, data=refi_df2).fit(
    cov_type="cluster", cov_kwds={"groups": refi_df2["county_id"]}
)
twfe_coef = twfe_model.params["treated:refi_incentive"]
twfe_se   = twfe_model.bse["treated:refi_incentive"]
print(f"  Coeff (treated x refi_incentive): {twfe_coef:.4f}  SE: {twfe_se:.4f}")
print(f"  [Absorbs county and time FE; still only uses refi loans]")

# ---------------------------------------------------------------------------
# 4. METHOD 3 — TRIPLE DIFFERENCE
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("METHOD 3: Triple-Difference (state x purpose x rate-shock)")
print("="*60)

# Use full panel (refi + purchase)
# beta = coeff on treated x refi_incentive x is_refi
triple_formula = (
    "log_apps ~ treated:refi_incentive:is_refi "
    "+ treated:refi_incentive + treated:is_refi "
    "+ refi_incentive:is_refi + treated + refi_incentive + is_refi "
    "+ C(county_purpose) + C(period)"
)
triple_model = smf.ols(triple_formula, data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["county_id"]}
)
triple_coef = triple_model.params["treated:refi_incentive:is_refi"]
triple_se   = triple_model.bse["treated:refi_incentive:is_refi"]
print(f"  Coeff (treated x refi_incentive x is_refi): {triple_coef:.4f}  SE: {triple_se:.4f}")
print(f"  [Uses purchase loans as within-county control; most credible]")

# ---------------------------------------------------------------------------
# 5. METHOD 4 — EVENT STUDY (leads and lags around rate-shock onset)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("METHOD 4: Event Study (TWFE with relative-time dummies)")
print("="*60)

# Define event as t=12 (start of rate boom); relative time = t - 12
# Bin: leads -4..-1, event 0, lags 1..8, absorb rest
event_df = refi_df.copy()
event_df["rel_time"] = event_df["t"] - 12
event_df["rel_bin"] = event_df["rel_time"].clip(-4, 8)
# Omit rel_time = -1 (baseline)
event_df = event_df[event_df["rel_bin"] != -1]

# Rename bins to valid Python identifiers (patsy can't parse "rt_-4")
def rt_name(v):
    return f"rt_m{abs(v)}" if v < 0 else f"rt_p{v}"

event_df["rel_bin_label"] = event_df["rel_bin"].apply(rt_name)
es_dummies = pd.get_dummies(event_df["rel_bin_label"], drop_first=False)
event_df = pd.concat([event_df, es_dummies], axis=1)

rt_cols = sorted([c for c in event_df.columns if c.startswith("rt_")],
                 key=lambda x: int(x[4:]) * (-1 if "m" in x else 1))

# Interact each dummy with treated via matrix algebra to avoid patsy issues
X_parts = [event_df[["log_apps", "treated", "county_id", "period"]].copy()]
for col in rt_cols:
    event_df[f"tx_{col}"] = event_df["treated"] * event_df[col]

tx_cols = [f"tx_{c}" for c in rt_cols]
es_formula = (
    "log_apps ~ "
    + " + ".join(tx_cols)
    + " + C(county_id) + C(period)"
)
es_model = smf.ols(es_formula, data=event_df).fit(
    cov_type="cluster", cov_kwds={"groups": event_df["county_id"]}
)

es_coefs = {
    col: (
        es_model.params.get(f"tx_{col}", np.nan),
        es_model.bse.get(f"tx_{col}", np.nan)
    )
    for col in rt_cols
}
def rt_sort_key(k):
    part = k[4:]
    return -int(part) if k.startswith("rt_m") else int(part)

print("  Relative-time coefficients (treated x rt_*):")
for k, (c, s) in sorted(es_coefs.items(), key=lambda x: rt_sort_key(x[0])):
    sig = "*" if abs(c / s) > 1.96 else " "
    print(f"    {k:>8s}: {c:+.3f}  SE {s:.3f} {sig}")
print("  [Pre-trend coefficients near zero => parallel trends plausible]")

# ---------------------------------------------------------------------------
# 6. METHOD 5 — PROPENSITY-SCORE MATCHING (cross-sectional)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("METHOD 5: Propensity-Score Matching")
print("="*60)

# Use a single high-incentive period (t=16) cross-section of refi loans
match_df = refi_df[refi_df["t"] == 16].copy()

# Estimate propensity score
ps_model = smf.logit("treated ~ pair_id", data=match_df).fit(disp=False)
match_df["ps"] = ps_model.predict()

# Nearest-neighbor 1:1 matching within caliper
treated_  = match_df[match_df["treated"] == 1].sort_values("ps").reset_index(drop=True)
control_  = match_df[match_df["treated"] == 0].sort_values("ps").reset_index(drop=True)

matched_pairs = []
used = set()
for _, row in treated_.iterrows():
    dists = (control_["ps"] - row["ps"]).abs()
    dists[list(used)] = np.inf
    best = dists.idxmin()
    if dists[best] < 0.05:   # caliper
        matched_pairs.append((row["log_apps"], control_.loc[best, "log_apps"]))
        used.add(best)

if matched_pairs:
    diffs = [t - c for t, c in matched_pairs]
    att = np.mean(diffs)
    att_se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    print(f"  ATT (matched pairs={len(matched_pairs)}): {att:.4f}  SE: {att_se:.4f}")
    print(f"  [Cross-sectional; no time variation; weaker identification]")
else:
    print("  No matches within caliper.")

# ---------------------------------------------------------------------------
# 7. METHOD 6 — SYNTHETIC CONTROL (Iowa analog)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("METHOD 6: Synthetic Control")
print("="*60)

# Treat one 'state A' county as 'Iowa' (low cost, quasi-policy unit)
# and construct synthetic from state B counties pre-shock
from scipy.optimize import minimize

# Aggregate to state-period level for simplicity
state_panel = (
    df[df["is_refi"] == 1]
    .groupby(["state", "t"])["log_apps"]
    .mean()
    .reset_index()
)

iowa_series   = state_panel[state_panel["state"] == "A_low_cost"]["log_apps"].values
donor_series  = state_panel[state_panel["state"] == "B_high_cost"]["log_apps"].values
T_pre = 12   # pre-shock periods

# Find weights minimizing pre-period MSE
def synth_loss(w):
    synth = w[0] * donor_series[:T_pre]
    return np.mean((iowa_series[:T_pre] - synth) ** 2)

res = minimize(synth_loss, x0=[0.5], bounds=[(0, 1)], method="L-BFGS-B")
w_opt = res.x[0]

synth = w_opt * donor_series
gap   = iowa_series - synth
pre_rmse  = np.sqrt(np.mean(gap[:T_pre]**2))
post_gap  = gap[T_pre:].mean()

print(f"  Optimal donor weight: {w_opt:.3f}")
print(f"  Pre-period RMSE: {pre_rmse:.4f}  (lower = better fit)")
print(f"  Post-shock avg gap (Iowa - Synthetic): {post_gap:.4f}")
print(f"  [Positive gap = Iowa refis HIGHER than synthetic = less friction]")

# ---------------------------------------------------------------------------
# 8. RESULTS SUMMARY TABLE
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("RESULTS SUMMARY  (True beta = -0.40)")
print("="*60)

results = {
    "1. Pooled OLS":          (ols_coef,    ols_se,    "Low",    "Omits county FE, selection"),
    "2. TWFE DiD":            (twfe_coef,   twfe_se,   "Medium", "Parallel trends in refi only"),
    "3. Triple-Difference":   (triple_coef, triple_se, "High",   "Most credible; uses purchase as control"),
    "4. Event Study":         (np.nan,      np.nan,    "High",   "Pre-trend test; see plot"),
    "5. PS Matching":         (att,         att_se,    "Low",    "Cross-section; no dynamics"),
    "6. Synthetic Control":   (post_gap,    np.nan,    "Medium", "Aggregate; good for Iowa case"),
}

print(f"  {'Method':<28} {'Coeff':>8} {'SE':>7} {'ID strength':>12}  Notes")
print("  " + "-"*80)
for method, (coef, se, strength, note) in results.items():
    coef_str = f"{coef:+.4f}" if not np.isnan(coef) else "   (see plot)"
    se_str   = f"{se:.4f}"   if not np.isnan(se)   else "     —"
    print(f"  {method:<28} {coef_str:>8} {se_str:>7} {strength:>12}  {note}")

# ---------------------------------------------------------------------------
# 9. PLOTS
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# (a) Refi incentive series
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(refi_incentive.values, color="#2166ac", lw=2)
ax0.axvline(12, color="red", ls="--", lw=1.2, label="Rate shock onset")
ax0.axvline(20, color="orange", ls="--", lw=1.2, label="Taper")
ax0.set_title("National Refi Incentive (simulated)", fontsize=11)
ax0.set_xlabel("Quarter"); ax0.set_ylabel("Rate gap index")
ax0.legend(fontsize=8)

# (b) Mean refi apps by state over time
ax1 = fig.add_subplot(gs[0, 1])
for state, color, lbl in [
    ("A_low_cost",  "#2166ac", "State A (low cost)"),
    ("B_high_cost", "#d6604d", "State B (high cost)"),
]:
    s = df[(df["state"] == state) & (df["is_refi"] == 1)].groupby("t")["log_apps"].mean()
    ax1.plot(s.values, color=color, lw=2, label=lbl)
ax1.axvline(12, color="gray", ls="--", lw=1)
ax1.set_title("Mean Log Refi Apps by State", fontsize=11)
ax1.set_xlabel("Quarter"); ax1.set_ylabel("Log applications")
ax1.legend(fontsize=8)

# (c) Triple-diff: refi minus purchase gap by state
ax2 = fig.add_subplot(gs[0, 2])
for state, color, lbl in [
    ("A_low_cost",  "#2166ac", "State A (low cost)"),
    ("B_high_cost", "#d6604d", "State B (high cost)"),
]:
    r = df[(df["state"] == state) & (df["is_refi"] == 1)].groupby("t")["log_apps"].mean()
    p = df[(df["state"] == state) & (df["is_refi"] == 0)].groupby("t")["log_apps"].mean()
    ax2.plot((r - p).values, color=color, lw=2, label=lbl)
ax2.axvline(12, color="gray", ls="--", lw=1)
ax2.set_title("Refi – Purchase Gap (Triple-Diff Logic)", fontsize=11)
ax2.set_xlabel("Quarter"); ax2.set_ylabel("Refi – Purchase log apps")
ax2.legend(fontsize=8)

# (d) Event study plot
ax3 = fig.add_subplot(gs[1, 0])
es_x, es_y, es_err = [], [], []
for k, (c, s) in sorted(es_coefs.items(), key=lambda x: rt_sort_key(x[0])):
    rt = -int(k[4:]) if k.startswith("rt_m") else int(k[4:])
    es_x.append(rt); es_y.append(c); es_err.append(1.96 * s)
ax3.axhline(0, color="gray", lw=0.8)
ax3.axvline(-0.5, color="red", ls="--", lw=1, label="Event (t=0)")
ax3.errorbar(es_x, es_y, yerr=es_err, fmt="o-", color="#2166ac",
             capsize=4, lw=1.5, ms=5)
ax3.set_title("Event Study: Treated × Relative Time", fontsize=11)
ax3.set_xlabel("Quarters relative to rate shock"); ax3.set_ylabel("Coefficient")
ax3.legend(fontsize=8)

# (e) Synthetic control
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(iowa_series,  color="#2166ac", lw=2, label="State A (actual)")
ax4.plot(synth,        color="#d6604d", lw=2, ls="--", label="Synthetic A")
ax4.axvline(T_pre, color="gray", ls="--", lw=1, label="Rate shock")
ax4.set_title("Synthetic Control: State A vs Synthetic", fontsize=11)
ax4.set_xlabel("Quarter"); ax4.set_ylabel("Mean log refi apps")
ax4.legend(fontsize=8)

# (f) Coefficient comparison (point + 95% CI)
ax5 = fig.add_subplot(gs[1, 2])
comp_methods = ["Pooled OLS", "TWFE DiD", "Triple-Diff", "PS Matching"]
comp_coefs   = [ols_coef, twfe_coef, triple_coef, att]
comp_ses     = [ols_se,   twfe_se,   triple_se,   att_se]
colors_bar   = ["#b2182b", "#f4a582", "#2166ac", "#92c5de"]
y_pos = range(len(comp_methods))
ax5.barh(y_pos, comp_coefs, xerr=[1.96*s for s in comp_ses],
         color=colors_bar, alpha=0.85, capsize=4)
ax5.axvline(0, color="gray", lw=0.8)
ax5.axvline(-0.40, color="black", lw=1.5, ls="--", label="True β = −0.40")
ax5.set_yticks(list(y_pos)); ax5.set_yticklabels(comp_methods, fontsize=9)
ax5.set_xlabel("Estimated coefficient")
ax5.set_title("Point Estimates ± 95% CI\n(vs. true β = −0.40)", fontsize=11)
ax5.legend(fontsize=8)

fig.suptitle(
    "Title Insurance & Refinancing: Econometric Methods Comparison\n"
    "(Simulated data, DGP true β = −0.40)",
    fontsize=13, fontweight="bold"
)
plt.savefig("econometrics_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to econometrics_comparison.png")

# ---------------------------------------------------------------------------
# 10. IDENTIFICATION ASSUMPTIONS CHECKLIST
# ---------------------------------------------------------------------------

print("""
IDENTIFICATION ASSUMPTIONS BY METHOD
=====================================
Method              | Key assumption                          | Testable?
--------------------|------------------------------------------|----------
Pooled OLS          | No unobserved county heterogeneity       | No
TWFE DiD            | Parallel trends (treated vs control)     | Pre-trend test
Triple-Difference   | Purchase loans are valid counterfactual  | Placebo test
Event Study         | No anticipation; stable unit treatment   | Pre-trend plot
PS Matching         | Conditional independence (on observables)| Covariate balance
Synthetic Control   | Convex hull; no interference             | Placebo states
""")
