from pathlib import Path
import math
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# RA-IFS-TOPSIS | 2025 IRENA PROJECT-EVIDENCE PIPELINE
# ============================================================================
# Purpose:
#   Reproducible implementation of the mathematical framework reported in the
#   manuscript. The main experiment uses only 2025 IRENA project-level data for
#   which the report provides a global weighted-average value and explicit
#   5th-95th percentile bounds for ALL THREE selected criteria:
#       1) LCOE
#       2) Total Installed Cost (TIC)
#       3) Capacity Factor
#
# The resulting core alternatives are:
#       Solar PV, Onshore Wind, Offshore Wind, Bioenergy
#
# This is deliberately a methodological proof-of-concept. The code does not
# claim that the final ranking is a universal or policy-optimal ranking.
#
# Mathematical chain:
#   central values -> direction-aware normalization -> Shannon entropy weights
#   evidence [L,U] -> direction-aware interval normalization
#               -> IFN (mu, nu, pi)
#   pi -> criterion hesitation H -> reliability R -> adjusted weights w*
#   w* + IFNs -> IF-TOPSIS -> closeness coefficient -> ranking
#
# Source basis: IRENA (2026), Renewable Power Generation Costs in 2025.
# ============================================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ALTERNATIVES = ["Solar PV", "Onshore Wind", "Offshore Wind", "Bioenergy"]
CRITERIA = ["LCOE", "Total Installed Cost", "Capacity Factor"]
DIRECTIONS = {
    "LCOE": "cost",
    "Total Installed Cost": "cost",
    "Capacity Factor": "benefit",
}

# Published 2025 global project evidence from the IRENA report.
# Each tuple is (central weighted-average, lower=P5, upper=P95).
# All ranges are explicitly reported as 5th-95th percentile project ranges.
DATA_ROWS = [
    # Solar PV: LCOE Fig. 3.6; TIC Fig. 3.1; CF Fig. 3.4.
    ["Solar PV", "LCOE", "cost", 44.0, 28.0, 97.0, "USD/MWh",
     "2025 global projects; 5th-95th percentiles", "IRENA (2026), Fig. 3.6, p. 71"],
    ["Solar PV", "Total Installed Cost", "cost", 667.0, 383.0, 1211.0, "USD/kW DC",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 3.1, p. 62"],
    ["Solar PV", "Capacity Factor", "benefit", 16.5, 11.9, 20.1, "% AC-DC",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 3.4, p. 69"],

    # Onshore wind: LCOE Fig. 2.10; TIC Fig. 2.3; CF Fig. 2.7.
    ["Onshore Wind", "LCOE", "cost", 33.0, 21.0, 80.0, "USD/MWh",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 2.10, p. 56"],
    ["Onshore Wind", "Total Installed Cost", "cost", 976.0, 632.0, 2208.0, "USD/kW",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 2.3, p. 46"],
    ["Onshore Wind", "Capacity Factor", "benefit", 36.0, 28.0, 50.0, "% AC-AC",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 2.7, p. 51"],

    # Offshore wind: LCOE Fig. 4.9; TIC Fig. 4.3; CF Fig. 4.6.
    ["Offshore Wind", "LCOE", "cost", 78.0, 46.0, 144.0, "USD/MWh",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 4.9, p. 88"],
    ["Offshore Wind", "Total Installed Cost", "cost", 2931.0, 1321.0, 5800.0, "USD/kW",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 4.3, p. 82"],
    ["Offshore Wind", "Capacity Factor", "benefit", 41.0, 35.0, 49.0, "% AC-AC",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 4.6, p. 85"],

    # Bioenergy: LCOE Fig. 8.8; TIC Fig. 8.2; CF Fig. 8.6.
    ["Bioenergy", "LCOE", "cost", 86.0, 49.0, 206.0, "USD/MWh",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 8.8, p. 135"],
    ["Bioenergy", "Total Installed Cost", "cost", 3606.0, 1396.0, 10273.0, "USD/kW",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 8.2, p. 130"],
    ["Bioenergy", "Capacity Factor", "benefit", 78.0, 71.0, 90.0, "% AC-AC",
     "2025 global newly commissioned projects; 5th-95th percentiles", "IRENA (2026), Fig. 8.6, p. 133"],
]
COLUMNS = ["alternative", "criterion", "direction", "central", "lower", "upper",
           "unit", "range_semantics", "source_location"]
SOURCE_URL = "https://www.irena.org/Publications/2026/Jul/Renewable-Power-Generation-Costs-in-2025"

# Optional exclusion note for transparency. These technologies are not used in
# the main matrix because the report does not provide the same complete set of
# 2025 global 5th-95th percentile bounds across all three selected criteria.
EXCLUDED_TECHNOLOGIES = {
    "Hydropower": "2025 global capacity-factor percentiles are reported, but the 2025 global LCOE range is not presented as a 5th-95th percentile pair in the same form.",
    "Geothermal": "2025 TIC and LCOE are reported as observed high-low project ranges and capacity factor is described over 2010-2025, not as a complete 2025 global 5th-95th percentile triplet.",
    "CSP": "The 2025 report does not provide a complete 2025 global 5th-95th percentile range set for LCOE, TIC, and capacity factor in the same form used for the selected technologies.",
}


def dataset_df() -> pd.DataFrame:
    return pd.DataFrame(DATA_ROWS, columns=COLUMNS)


def validate_dataset(df: pd.DataFrame) -> None:
    if list(df.columns) != COLUMNS:
        raise ValueError("Dataset schema mismatch.")
    if len(df) != len(ALTERNATIVES) * len(CRITERIA):
        raise ValueError("Expected 12 alternative-criterion observations.")
    if set(df.alternative) != set(ALTERNATIVES):
        raise ValueError("Alternative set mismatch.")
    if set(df.criterion) != set(CRITERIA):
        raise ValueError("Criterion set mismatch.")
    x = df[["central", "lower", "upper"]].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite numerical value found.")
    if np.any(df.lower.to_numpy(float) > df.upper.to_numpy(float)):
        raise ValueError("Lower bound exceeds upper bound.")
    if np.any(df.central.to_numpy(float) < df.lower.to_numpy(float)) or np.any(df.central.to_numpy(float) > df.upper.to_numpy(float)):
        raise ValueError("Central value is outside the published evidence range.")
    if not df.range_semantics.str.contains("5th-95th percentiles", regex=False).all():
        raise ValueError("All evidence ranges must be explicitly identified as 5th-95th percentile ranges.")
    if any(df.direction.iloc[k] != DIRECTIONS[df.criterion.iloc[k]] for k in range(len(df))):
        raise ValueError("Criterion direction metadata is inconsistent with DIRECTIONS.")


def normalize_series(x, direction):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if np.isclose(lo, hi):
        return np.full_like(x, 0.5)
    if direction == "benefit":
        return (x - lo) / (hi - lo)
    return (hi - x) / (hi - lo)


def central_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=ALTERNATIVES, columns=CRITERIA, dtype=float)
    for c in CRITERIA:
        s = df[df.criterion == c].set_index("alternative").loc[ALTERNATIVES]
        out[c] = normalize_series(s.central.to_numpy(float), DIRECTIONS[c])
    return out


def central_matrix_vector(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=ALTERNATIVES, columns=CRITERIA, dtype=float)
    for c in CRITERIA:
        s = df[df.criterion == c].set_index("alternative").loc[ALTERNATIVES]
        x = s.central.to_numpy(float)
        den = np.linalg.norm(x)
        if den <= 1e-15:
            raise ValueError(f"Cannot vector-normalize {c}.")
        r = x / den
        out[c] = r if DIRECTIONS[c] == "benefit" else 1.0 - r
    return out


def entropy_weights_from_Z(Z: pd.DataFrame):
    m = len(ALTERNATIVES)
    P = pd.DataFrame(index=ALTERNATIVES, columns=CRITERIA, dtype=float)
    E = {}
    G = {}
    for c in CRITERIA:
        v = Z[c].to_numpy(float)
        s = v.sum()
        p = np.full(m, 1.0 / m) if s <= 1e-15 else v / s
        P[c] = p
        mask = p > 0
        terms = np.zeros_like(p)
        terms[mask] = p[mask] * np.log(p[mask])
        # Standard convention: p*ln(p) = 0 at p=0.
        e = -terms.sum() / math.log(m)
        E[c] = float(e)
        G[c] = float(1.0 - e)
    g = np.array([G[c] for c in CRITERIA], dtype=float)
    w = g / g.sum() if g.sum() > 1e-15 else np.full(len(CRITERIA), 1.0 / len(CRITERIA))
    return P, pd.Series(E), pd.Series(G), pd.Series(w, index=CRITERIA)


def entropy_weights(df: pd.DataFrame):
    Z = central_matrix(df)
    P, E, G, w = entropy_weights_from_Z(Z)
    return Z, P, E, G, w


def interval_to_ifs(df: pd.DataFrame, collapse=False, normalization="minmax") -> pd.DataFrame:
    rows = []
    for c in CRITERIA:
        s = df[df.criterion == c].set_index("alternative").loc[ALTERNATIVES]
        lo = s.lower.to_numpy(float)
        hi = s.upper.to_numpy(float)
        direction = DIRECTIONS[c]
        if normalization == "minmax":
            gmin = lo.min()
            gmax = hi.max()
            if np.isclose(gmin, gmax):
                raise ValueError(f"Degenerate evidence envelope: {c}")
            if direction == "benefit":
                l = (lo - gmin) / (gmax - gmin)
                u = (hi - gmin) / (gmax - gmin)
            else:
                l = (gmax - hi) / (gmax - gmin)
                u = (gmax - lo) / (gmax - gmin)
        elif normalization == "vector":
            central = s.central.to_numpy(float)
            den = np.linalg.norm(central)
            if den <= 1e-15:
                raise ValueError(f"Cannot vector-normalize interval for {c}.")
            if direction == "benefit":
                l = lo / den
                u = hi / den
            else:
                l = 1.0 - hi / den
                u = 1.0 - lo / den
            l = np.clip(l, 0.0, 1.0)
            u = np.clip(u, 0.0, 1.0)
        else:
            raise ValueError("Unknown normalization.")
        if collapse:
            mid = (l + u) / 2.0
            l, u = mid, mid
        for i, a in enumerate(ALTERNATIVES):
            mu = float(l[i])
            nu = float(1.0 - u[i])
            pi = float(u[i] - l[i])
            if min(mu, nu, pi) < -1e-10 or abs(mu + nu + pi - 1.0) > 1e-9:
                raise ValueError(f"Invalid IFN for {a}/{c}: {(mu, nu, pi)}")
            rows.append([a, c, mu, nu, pi, pi, (1.0 + mu - nu) / 2.0])
    return pd.DataFrame(rows, columns=["alternative", "criterion", "mu", "nu", "pi", "hesitation", "score"])


def matrix_tensor(ifs_df: pd.DataFrame) -> np.ndarray:
    D = np.zeros((len(ALTERNATIVES), len(CRITERIA), 3), dtype=float)
    for i, a in enumerate(ALTERNATIVES):
        for j, c in enumerate(CRITERIA):
            r = ifs_df[(ifs_df.alternative == a) & (ifs_df.criterion == c)].iloc[0]
            D[i, j] = [r.mu, r.nu, r.pi]
    return D


def weighted_ifs_topsis(D, weights):
    w = np.asarray(weights, dtype=float)
    if len(w) != D.shape[1] or not np.isclose(w.sum(), 1.0):
        raise ValueError("Weights must match criteria and sum to 1.")
    S = (1.0 + D[:, :, 0] - D[:, :, 1]) / 2.0
    pos = np.zeros((D.shape[1], 3), dtype=float)
    neg = np.zeros((D.shape[1], 3), dtype=float)
    for j in range(D.shape[1]):
        ip = np.argmax(S[:, j])
        im = np.argmin(S[:, j])
        pos[j] = D[ip, j]
        neg[j] = D[im, j]
    dpos = np.zeros(D.shape[0], dtype=float)
    dneg = np.zeros(D.shape[0], dtype=float)
    for i in range(D.shape[0]):
        sqp = 0.0
        sqn = 0.0
        for j in range(D.shape[1]):
            djp = np.linalg.norm(D[i, j] - pos[j]) / math.sqrt(2.0)
            djn = np.linalg.norm(D[i, j] - neg[j]) / math.sqrt(2.0)
            sqp += w[j] * djp * djp
            sqn += w[j] * djn * djn
        dpos[i] = math.sqrt(max(sqp, 0.0))
        dneg[i] = math.sqrt(max(sqn, 0.0))
    cc = dneg / (dpos + dneg + 1e-15)
    return dpos, dneg, cc, pos, neg


def classical_topsis(Z, weights):
    X = Z.to_numpy(float)
    w = np.asarray(weights, dtype=float)
    V = X * w[None, :]
    pos = V.max(axis=0)
    neg = V.min(axis=0)
    dp = np.linalg.norm(V - pos, axis=1)
    dn = np.linalg.norm(V - neg, axis=1)
    cc = dn / (dp + dn + 1e-15)
    return dp, dn, cc


def adjusted_weights(base, ifs_df, alpha=1.0):
    H = ifs_df.groupby("criterion")["hesitation"].mean().reindex(CRITERIA)
    R = np.clip(1.0 - alpha * H.to_numpy(float), 0.0, 1.0)
    raw = base.to_numpy(float) * R
    if raw.sum() <= 1e-15:
        raise ValueError("All adjusted weights are zero.")
    w = raw / raw.sum()
    return H, pd.Series(R, index=CRITERIA), pd.Series(w, index=CRITERIA)


def kendall_tau(a, b):
    a = list(a)
    b = list(b)
    rb = {v: i for i, v in enumerate(b)}
    ranks = [rb[v] for v in a]
    conc = 0
    disc = 0
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            if ranks[i] < ranks[j]:
                conc += 1
            elif ranks[i] > ranks[j]:
                disc += 1
    den = conc + disc
    return 1.0 if den == 0 else (conc - disc) / den


def rank_series(values):
    order = np.argsort(-np.asarray(values))
    return [ALTERNATIVES[i] for i in order]


def run_all():
    df = dataset_df()
    validate_dataset(df)
    df.to_csv(RESULTS / "embedded_dataset.csv", index=False)

    audit_cols = ["alternative", "criterion", "central", "lower", "upper", "unit", "range_semantics", "source_location"]
    df[audit_cols].to_csv(RESULTS / "dataset_audit.csv", index=False)
    pd.DataFrame([
        [k, v, SOURCE_URL] for k, v in EXCLUDED_TECHNOLOGIES.items()
    ], columns=["excluded_technology", "reason", "source"]).to_csv(RESULTS / "exclusion_audit.csv", index=False)

    # ------------------------------ Baseline ------------------------------
    Z, P, E, G, w = entropy_weights(df)
    Z.to_csv(RESULTS / "normalized_central_matrix.csv")
    pd.DataFrame({"entropy_E": E, "divergence_1_minus_E": G, "base_weight": w}).to_csv(
        RESULTS / "criterion_weights_base.csv"
    )

    ifs = interval_to_ifs(df, collapse=False, normalization="minmax")
    ifs0 = interval_to_ifs(df, collapse=True, normalization="minmax")
    ifs.to_csv(RESULTS / "ifs_decision_matrix.csv", index=False)
    ifs0.to_csv(RESULTS / "ifs_ablation_matrix.csv", index=False)
    D = matrix_tensor(ifs)
    D0 = matrix_tensor(ifs0)

    _, _, cc_classical = classical_topsis(Z, w)
    _, _, cc_deg, _, _ = weighted_ifs_topsis(D0, w)
    _, _, cc_std, _, _ = weighted_ifs_topsis(D, w)
    H, R, w_adj = adjusted_weights(w, ifs, alpha=1.0)
    _, _, cc_ra, _, _ = weighted_ifs_topsis(D, w_adj)

    # ------------------------ Mathematical checks -------------------------
    max_ifs_residual = float(np.max(np.abs(D.sum(axis=2) - 1.0)))
    if max_ifs_residual > 1e-9:
        raise AssertionError("IFS feasibility check failed.")
    if not np.isclose(w.sum(), 1.0, atol=1e-10):
        raise AssertionError("Base weights do not sum to one.")
    if not np.isclose(w_adj.sum(), 1.0, atol=1e-10):
        raise AssertionError("Adjusted weights do not sum to one.")
    if np.any((R.to_numpy() < -1e-12) | (R.to_numpy() > 1 + 1e-12)):
        raise AssertionError("Reliability bounds violated.")

    validation = pd.DataFrame([
        ["max_IF_feasibility_residual", max_ifs_residual, max_ifs_residual <= 1e-9],
        ["base_weight_sum", float(w.sum()), bool(np.isclose(w.sum(), 1.0, atol=1e-10))],
        ["adjusted_weight_sum", float(w_adj.sum()), bool(np.isclose(w_adj.sum(), 1.0, atol=1e-10))],
        ["reliability_bounds", bool(np.all((R.to_numpy() >= -1e-12) & (R.to_numpy() <= 1 + 1e-12))), True],
    ], columns=["check", "value", "passed"])
    validation.to_csv(RESULTS / "validation_checks.csv", index=False)

    ranking = pd.DataFrame({
        "alternative": ALTERNATIVES,
        "Classical_TOPSIS": cc_classical,
        "Degenerate_IF_TOPSIS": cc_deg,
        "Standard_IF_TOPSIS": cc_std,
        "Proposed_RA_IF_TOPSIS": cc_ra,
    })
    for col in ranking.columns[1:]:
        ranking[col + "_rank"] = ranking[col].rank(ascending=False, method="min").astype(int)
    ranking.to_csv(RESULTS / "ranking_comparison.csv", index=False)

    cw = pd.DataFrame({
        "base_weight": w,
        "mean_hesitation_H": H,
        "reliability_R": R,
        "adjusted_weight": w_adj,
    })
    cw.to_csv(RESULTS / "criterion_weights_hesitation_reliability.csv")

    abl = ranking[["alternative", "Classical_TOPSIS", "Degenerate_IF_TOPSIS", "Standard_IF_TOPSIS", "Proposed_RA_IF_TOPSIS"]].copy()
    abl.to_csv(RESULTS / "ifs_ablation.csv", index=False)

    # ---------------------- Hesitation sensitivity -------------------------
    sens_rows = []
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        Hx, Rx, wx = adjusted_weights(w, ifs, alpha=alpha)
        _, _, ccx, _, _ = weighted_ifs_topsis(D, wx)
        order = rank_series(ccx)
        sens_rows.append({
            "alpha": alpha,
            "top_alternative": order[0],
            "rank_order": " > ".join(order),
            "top_score": float(ccx[np.argmax(ccx)]),
            **{f"w_{c}": float(wx[c]) for c in CRITERIA},
        })
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(RESULTS / "hesitation_sensitivity.csv", index=False)

    # --------------------- Normalization sensitivity ----------------------
    Zv = central_matrix_vector(df)
    _, Ev, Gv, wv = entropy_weights_from_Z(Zv)
    ifsv = interval_to_ifs(df, collapse=False, normalization="vector")
    Dv = matrix_tensor(ifsv)
    Hv, Rv, wv_adj = adjusted_weights(wv, ifsv, alpha=1.0)
    _, _, cc_vector, _, _ = weighted_ifs_topsis(Dv, wv_adj)
    norm_sens = pd.DataFrame({
        "alternative": ALTERNATIVES,
        "minmax_RA_IF": cc_ra,
        "vector_RA_IF": cc_vector,
    })
    norm_sens["minmax_rank"] = norm_sens["minmax_RA_IF"].rank(ascending=False, method="min").astype(int)
    norm_sens["vector_rank"] = norm_sens["vector_RA_IF"].rank(ascending=False, method="min").astype(int)
    norm_sens.to_csv(RESULTS / "normalization_sensitivity.csv", index=False)
    pd.DataFrame({
        "vector_base_weight": wv,
        "vector_adjusted_weight": wv_adj,
        "vector_mean_hesitation": Hv,
    }).to_csv(RESULTS / "normalization_vector_weights.csv")

    # ---------------------- Criterion-weight robustness -------------------
    base_order = rank_series(cc_ra)
    wr = []
    for c in CRITERIA:
        j = CRITERIA.index(c)
        for factor in [0.90, 1.00, 1.10]:
            wp = w_adj.to_numpy(float).copy()
            wp[j] *= factor
            wp /= wp.sum()
            _, _, ccp, _, _ = weighted_ifs_topsis(D, wp)
            order = rank_series(ccp)
            wr.append({
                "perturbed_criterion": c,
                "factor": factor,
                "perturbed_weight": float(wp[j]),
                "top_alternative": order[0],
                "rank_order": " > ".join(order),
                "kendall_tau_to_baseline": kendall_tau(base_order, order),
            })
    weight_robustness = pd.DataFrame(wr)
    weight_robustness.to_csv(RESULTS / "criterion_weight_robustness.csv", index=False)

    # ------------------------- Ranking agreement --------------------------
    orders = {k: rank_series(ranking[k].to_numpy()) for k in [
        "Classical_TOPSIS", "Degenerate_IF_TOPSIS", "Standard_IF_TOPSIS", "Proposed_RA_IF_TOPSIS"
    ]}
    pair = []
    keys = list(orders)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pair.append({"method_1": keys[i], "method_2": keys[j], "kendall_tau": kendall_tau(orders[keys[i]], orders[keys[j]])})
    pd.DataFrame(pair).to_csv(RESULTS / "ranking_agreement.csv", index=False)

    # ------------------------------ Figures -------------------------------
    plt.rcParams.update({"figure.dpi": 150})

    # Fig 1: principal model comparison.
    # Include all four models evaluated in the manuscript.
    rplot = ranking.set_index("alternative")[[
        "Classical_TOPSIS",
        "Degenerate_IF_TOPSIS",
        "Standard_IF_TOPSIS",
        "Proposed_RA_IF_TOPSIS",
    ]].copy()
    rplot.columns = [
        "Classical TOPSIS",
        "Degenerate IF-TOPSIS",
        "Standard IF-TOPSIS",
        "Proposed RA-IF-TOPSIS",
    ]
    ax = rplot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Closeness coefficient")
    ax.set_title("Comparison of principal TOPSIS models")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS / "01_ranking_comparison.png")
    plt.close(ax.figure)

    # Fig 2: weights.
    wplot = cw[["base_weight", "adjusted_weight"]].copy()
    wplot.columns = ["Baseline weight", "Adjusted weight"]
    ax = wplot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Criterion weight")
    ax.set_title("Baseline and reliability-adjusted criterion weights")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS / "02_weight_adjustment.png")
    plt.close(ax.figure)

    # Fig 3: hesitation.
    ax = cw["mean_hesitation_H"].plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Mean hesitation H_j")
    ax.set_title("Evidence-induced hesitation by criterion")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS / "03_hesitation_by_criterion.png")
    plt.close(ax.figure)

    # Fig 4: ablation.
    ax = abl.set_index("alternative")[["Degenerate_IF_TOPSIS", "Standard_IF_TOPSIS"]].plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Closeness coefficient")
    ax.set_title("IFS ablation: zero-hesitation midpoint vs full interval-induced IFS")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS / "04_ifs_ablation.png")
    plt.close(ax.figure)

    # Fig 5: alpha sensitivity of adjusted weights.
    splot = sens.set_index("alpha")[[f"w_{c}" for c in CRITERIA]].copy()
    splot.columns = ["LCOE", "Total Installed Cost", "Capacity Factor"]
    ax = splot.plot(marker="o", figsize=(10, 5))
    ax.set_xlabel("Hesitation attenuation factor alpha")
    ax.set_ylabel("Adjusted criterion weight")
    ax.set_title("Sensitivity to hesitation attenuation")
    ax.grid(alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS / "05_hesitation_sensitivity.png")
    plt.close(ax.figure)

    # Fig 6: normalization sensitivity.
    nplot = norm_sens.set_index("alternative")[["minmax_RA_IF", "vector_RA_IF"]].copy()
    nplot.columns = ["Baseline min-max", "Vector normalization"]
    ax = nplot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("RA-IF-TOPSIS closeness")
    ax.set_title("Normalization sensitivity")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS / "06_normalization_sensitivity.png")
    plt.close(ax.figure)

    # Fig 7: criterion-weight robustness (Kendall tau).
    pivot = weight_robustness.pivot(index="perturbed_criterion", columns="factor", values="kendall_tau_to_baseline")
    pivot.columns = ["90%", "100%", "110%"]
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Perturbed criterion")
    ax.set_ylabel("Kendall rank correlation")
    ax.set_title("Ranking robustness to +/-10% criterion-weight perturbation")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS / "07_weight_robustness.png")
    plt.close(ax.figure)

    summary_lines = [
        "RA-IFS-TOPSIS 2025 IRENA PIPELINE: SUCCESS",
        "Source: IRENA (2026), Renewable Power Generation Costs in 2025.",
        "Core alternatives: " + ", ".join(ALTERNATIVES),
        "Core criteria: " + ", ".join(CRITERIA),
        "Top proposed alternative: " + rank_series(cc_ra)[0],
        "Proposed ranking: " + " > ".join(rank_series(cc_ra)),
        "Base weights: " + str(w.to_dict()),
        "Adjusted weights: " + str(w_adj.to_dict()),
        "Mean hesitation: " + str(H.to_dict()),
        "Max IFS feasibility residual: " + f"{max_ifs_residual:.3e}",
        "Vector-normalization ranking: " + " > ".join(rank_series(cc_vector)),
    ]
    (RESULTS / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    manifest = {}
    for f in sorted(RESULTS.iterdir()):
        if f.is_file() and f.name not in {"reproducibility_manifest.json", "results_bundle.zip"}:
            manifest[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    main_script = Path(__file__)
    (RESULTS / "reproducibility_manifest.json").write_text(json.dumps({
        "main_script": main_script.name,
        "main_script_sha256": hashlib.sha256(main_script.read_bytes()).hexdigest(),
        "source_url": SOURCE_URL,
        "generated_files": manifest,
    }, indent=2), encoding="utf-8")

    bundle = RESULTS / "results_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(RESULTS.iterdir()):
            if f.is_file() and f.name != bundle.name:
                zf.write(f, arcname=f.name)

    print("=" * 78)
    print("RA-IFS-TOPSIS 2025 IRENA PIPELINE: SUCCESS")
    print("=" * 78)
    print("\nRanking comparison:")
    print(ranking.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nWeights / hesitation / reliability:")
    print(cw.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nProposed ranking:", " > ".join(rank_series(cc_ra)))
    print("\nNormalization sensitivity:")
    print(norm_sens.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nWeight robustness:")
    print(weight_robustness.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return ranking, cw, ifs, sens, norm_sens, weight_robustness


if __name__ == "__main__":
    run_all()

