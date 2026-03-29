"""
Coarsened Exact Matching (CEM) for HAB Study
==============================================
Match weir-influenced reaches to non-weir control reaches.

CEM = primary matching estimator (preregistered, locked)
PSM = sensitivity analysis (appendix)

Covariates for matching:
  - Watershed area (km²)
  - Mean slope (degrees)
  - Agricultural land fraction
  - Drainage area (km²)
  - Stream order

Balance diagnostic: Standardized Mean Difference (SMD) < 0.1

References:
  - Iacus, King & Porro (2012). Causal Inference without Balance Checking:
    Coarsened Exact Matching. Political Analysis.
"""

import numpy as np
import pandas as pd
from itertools import product


# ─────────────────────────────────────────────
# CEM Implementation
# ─────────────────────────────────────────────

def coarsen_variable(values, n_bins=None, breaks=None):
    """Coarsen a continuous variable into bins.

    Args:
        values: array-like of continuous values
        n_bins: number of equal-width bins (default: Sturges' rule)
        breaks: explicit bin edges (overrides n_bins)

    Returns:
        array of bin labels (integers)
    """
    values = np.asarray(values, dtype=float)

    if breaks is not None:
        bins = np.digitize(values, breaks)
    else:
        if n_bins is None:
            n_bins = max(2, int(np.ceil(np.log2(len(values)) + 1)))  # Sturges
        _, bin_edges = np.histogram(values, bins=n_bins)
        bins = np.digitize(values, bin_edges[:-1])

    return bins


def cem_match(df, treatment_col, covariates, n_bins=None, coarsen_spec=None):
    """Perform Coarsened Exact Matching.

    Args:
        df: DataFrame with treatment indicator and covariates
        treatment_col: column name for treatment (1=treated, 0=control)
        covariates: list of covariate column names
        n_bins: default number of bins (can be overridden per variable)
        coarsen_spec: dict of {covariate: n_bins or breaks} for fine control

    Returns:
        matched_df: DataFrame with matched units + CEM weights
        diagnostics: dict with balance statistics
    """
    if coarsen_spec is None:
        coarsen_spec = {}

    # Step 1: Coarsen each covariate
    coarsened = pd.DataFrame(index=df.index)
    for cov in covariates:
        spec = coarsen_spec.get(cov, {})
        bins = spec.get("bins", n_bins)
        breaks = spec.get("breaks", None)
        coarsened[f"{cov}_bin"] = coarsen_variable(df[cov].values, n_bins=bins, breaks=breaks)

    # Step 2: Create strata (unique combinations of coarsened values)
    bin_cols = [f"{c}_bin" for c in covariates]
    coarsened["stratum"] = coarsened[bin_cols].apply(
        lambda row: tuple(row.values), axis=1
    )

    df_work = df.copy()
    df_work["stratum"] = coarsened["stratum"]

    # Step 3: Keep only strata with both treated and control units
    strata_counts = df_work.groupby(["stratum", treatment_col]).size().unstack(fill_value=0)

    if 0 not in strata_counts.columns or 1 not in strata_counts.columns:
        print("⚠️ No strata with both treated and control units!")
        return pd.DataFrame(), {"n_matched": 0}

    valid_strata = strata_counts[(strata_counts[0] > 0) & (strata_counts[1] > 0)].index
    matched = df_work[df_work["stratum"].isin(valid_strata)].copy()

    # Step 4: Compute CEM weights
    # Weight = (N_treated_total / N_treated_stratum) * (N_control_stratum / N_control_total)
    n_treated_total = (matched[treatment_col] == 1).sum()
    n_control_total = (matched[treatment_col] == 0).sum()

    weights = []
    for _, row in matched.iterrows():
        stratum = row["stratum"]
        n_t_s = strata_counts.loc[stratum, 1]
        n_c_s = strata_counts.loc[stratum, 0]
        if row[treatment_col] == 1:
            weights.append(1.0)
        else:
            weights.append(n_t_s / n_c_s if n_c_s > 0 else 0)

    matched["cem_weight"] = weights

    # Step 5: Diagnostics
    diagnostics = compute_balance(df, matched, treatment_col, covariates)
    diagnostics["n_total"] = len(df)
    diagnostics["n_matched"] = len(matched)
    diagnostics["n_treated_matched"] = (matched[treatment_col] == 1).sum()
    diagnostics["n_control_matched"] = (matched[treatment_col] == 0).sum()
    diagnostics["n_strata"] = len(valid_strata)
    diagnostics["n_pruned"] = len(df) - len(matched)

    return matched, diagnostics


# ─────────────────────────────────────────────
# Balance Diagnostics
# ─────────────────────────────────────────────

def compute_smd(treated_vals, control_vals):
    """Compute Standardized Mean Difference (SMD).

    SMD = |mean_T - mean_C| / sqrt((var_T + var_C) / 2)

    Good balance: SMD < 0.1
    Acceptable: SMD < 0.25
    """
    mean_t = np.nanmean(treated_vals)
    mean_c = np.nanmean(control_vals)
    var_t = np.nanvar(treated_vals)
    var_c = np.nanvar(control_vals)

    pooled_sd = np.sqrt((var_t + var_c) / 2)
    if pooled_sd < 1e-10:
        return 0.0

    return abs(mean_t - mean_c) / pooled_sd


def compute_balance(df_full, df_matched, treatment_col, covariates):
    """Compute covariate balance before and after matching.

    Returns dict with SMD for each covariate, before and after.
    """
    balance = {}

    for cov in covariates:
        # Before matching
        t_before = df_full.loc[df_full[treatment_col] == 1, cov].values
        c_before = df_full.loc[df_full[treatment_col] == 0, cov].values
        smd_before = compute_smd(t_before, c_before)

        # After matching
        if len(df_matched) > 0:
            t_after = df_matched.loc[df_matched[treatment_col] == 1, cov].values
            c_after = df_matched.loc[df_matched[treatment_col] == 0, cov].values
            smd_after = compute_smd(t_after, c_after)
        else:
            smd_after = np.nan

        balance[cov] = {
            "smd_before": smd_before,
            "smd_after": smd_after,
            "improved": smd_after < smd_before if not np.isnan(smd_after) else False,
            "balanced": smd_after < 0.1 if not np.isnan(smd_after) else False,
        }

    return {"covariate_balance": balance}


def print_balance_table(diagnostics):
    """Pretty-print balance diagnostics."""
    print("\n" + "=" * 65)
    print("CEM Matching Balance Report")
    print("=" * 65)
    print(f"Total units: {diagnostics['n_total']}")
    print(f"Matched units: {diagnostics['n_matched']} "
          f"(T={diagnostics['n_treated_matched']}, "
          f"C={diagnostics['n_control_matched']})")
    print(f"Pruned: {diagnostics['n_pruned']}")
    print(f"Strata: {diagnostics['n_strata']}")
    print()
    print(f"{'Covariate':<25} {'SMD Before':>12} {'SMD After':>12} {'Balanced?':>10}")
    print("-" * 65)

    balance = diagnostics.get("covariate_balance", {})
    all_balanced = True
    for cov, stats in balance.items():
        balanced = "✅" if stats["balanced"] else "❌"
        if not stats["balanced"]:
            all_balanced = False
        print(f"{cov:<25} {stats['smd_before']:>12.4f} "
              f"{stats['smd_after']:>12.4f} {balanced:>10}")

    print("-" * 65)
    overall = "✅ ALL BALANCED (SMD < 0.1)" if all_balanced else "⚠️ SOME IMBALANCED"
    print(f"Overall: {overall}")
    print("=" * 65)


# ─────────────────────────────────────────────
# PSM (Sensitivity Analysis — Appendix)
# ─────────────────────────────────────────────

def psm_match(df, treatment_col, covariates, caliper=0.2, n_matches=1):
    """Propensity Score Matching (for sensitivity analysis).

    Uses logistic regression to estimate propensity scores,
    then nearest-neighbor matching with caliper.

    This is for APPENDIX ONLY — CEM is the primary estimator.
    """
    from sklearn.linear_model import LogisticRegression

    X = df[covariates].values
    y = df[treatment_col].values

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]  # P(treatment=1)

    df_work = df.copy()
    df_work["propensity_score"] = ps

    treated = df_work[df_work[treatment_col] == 1]
    control = df_work[df_work[treatment_col] == 0]

    matched_pairs = []
    used_controls = set()

    for t_idx, t_row in treated.iterrows():
        t_ps = t_row["propensity_score"]

        # Find nearest control within caliper
        distances = abs(control["propensity_score"] - t_ps)
        within_caliper = distances[distances < caliper * ps.std()]
        within_caliper = within_caliper.drop(index=within_caliper.index.intersection(used_controls))

        if not within_caliper.empty:
            matches = within_caliper.nsmallest(n_matches)
            for c_idx in matches.index:
                matched_pairs.append((t_idx, c_idx))
                used_controls.add(c_idx)

    if not matched_pairs:
        return pd.DataFrame(), {"n_matched": 0}

    matched_indices = set()
    for t, c in matched_pairs:
        matched_indices.add(t)
        matched_indices.add(c)

    matched_df = df_work.loc[list(matched_indices)].copy()
    diagnostics = compute_balance(df, matched_df, treatment_col, covariates)
    diagnostics["n_matched"] = len(matched_df)
    diagnostics["n_pairs"] = len(matched_pairs)
    diagnostics["method"] = "PSM"

    return matched_df, diagnostics


# ─────────────────────────────────────────────
# DID Estimation
# ─────────────────────────────────────────────

def did_estimate(df, outcome_col, treatment_col, post_col, weights_col=None,
                 cluster_col=None):
    """Simple DID estimator with optional CEM weights and clustered SE.

    Model: Y = β0 + β1*Treatment + β2*Post + β3*Treatment×Post + ε

    β3 = Average Treatment Effect on the Treated (ATT)

    For full analysis, use statsmodels with proper FE and clustered SE.
    """
    import statsmodels.formula.api as smf

    df_work = df.copy()
    df_work["treat_post"] = df_work[treatment_col] * df_work[post_col]

    formula = f"{outcome_col} ~ {treatment_col} + {post_col} + treat_post"

    if weights_col and weights_col in df_work.columns:
        model = smf.wls(formula, data=df_work, weights=df_work[weights_col])
    else:
        model = smf.ols(formula, data=df_work)

    result = model.fit(
        cov_type="cluster" if cluster_col else "HC3",
        cov_kwds={"groups": df_work[cluster_col]} if cluster_col else {}
    )

    return {
        "att": result.params["treat_post"],
        "se": result.bse["treat_post"],
        "pvalue": result.pvalues["treat_post"],
        "ci_lower": result.conf_int().loc["treat_post", 0],
        "ci_upper": result.conf_int().loc["treat_post", 1],
        "n_obs": int(result.nobs),
        "r_squared": result.rsquared,
        "summary": str(result.summary()),
    }


def event_study(df, outcome_col, treatment_col, year_col, event_year,
                cluster_col=None, weights_col=None):
    """Event-study specification for parallel trends + dynamic effects.

    Creates year-relative-to-event dummies and interacts with treatment.
    Omits t=-1 as reference period.
    """
    import statsmodels.formula.api as smf

    df_work = df.copy()
    df_work["rel_year"] = df_work[year_col] - event_year

    # Create dummies for each relative year (omit -1)
    rel_years = sorted(df_work["rel_year"].unique())
    ref_year = -1

    for ry in rel_years:
        if ry == ref_year:
            continue
        col_name = f"ry_{ry}" if ry >= 0 else f"ry_m{abs(ry)}"
        df_work[col_name] = (df_work["rel_year"] == ry).astype(int)
        df_work[f"treat_x_{col_name}"] = df_work[col_name] * df_work[treatment_col]

    # Build formula
    interaction_terms = [f"treat_x_ry_{ry}" if ry >= 0 else f"treat_x_ry_m{abs(ry)}"
                         for ry in rel_years if ry != ref_year]
    interaction_terms = [t for t in interaction_terms if t in df_work.columns]

    formula = f"{outcome_col} ~ {treatment_col} + " + " + ".join(interaction_terms)

    if weights_col and weights_col in df_work.columns:
        model = smf.wls(formula, data=df_work, weights=df_work[weights_col])
    else:
        model = smf.ols(formula, data=df_work)

    result = model.fit(
        cov_type="cluster" if cluster_col else "HC3",
        cov_kwds={"groups": df_work[cluster_col]} if cluster_col else {}
    )

    # Extract event-study coefficients
    es_coeffs = {}
    for term in interaction_terms:
        ry = term.replace("treat_x_ry_", "").replace("m", "-")
        try:
            ry_int = int(ry)
        except ValueError:
            ry_int = ry
        es_coeffs[ry_int] = {
            "coef": result.params[term],
            "se": result.bse[term],
            "pvalue": result.pvalues[term],
            "ci_lower": result.conf_int().loc[term, 0],
            "ci_upper": result.conf_int().loc[term, 1],
        }

    return {
        "event_study_coefficients": es_coeffs,
        "parallel_trends_test": all(
            es_coeffs[ry]["pvalue"] > 0.05
            for ry in es_coeffs if isinstance(ry, int) and ry < 0
        ),
        "summary": str(result.summary()),
    }


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("CEM Matching module loaded.")
    print("Primary: cem_match() → CEM (preregistered)")
    print("Sensitivity: psm_match() → PSM (appendix only)")
    print("Estimation: did_estimate() + event_study()")
