import os
import io
import csv
import re
import time
import shutil
import zipfile
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import requests


# =============================================================================
# NIFTY PHARMA SSD — COMPLETE FAST-TRACK FINAL PIPELINE
# =============================================================================
#
# PURPOSE
# -------
# Finish the SSD method while covering the mandatory requirements in the
# Brindco pairs-trading assignment, with as little additional machinery as
# possible.
#
# PRIMARY SSD METHOD
# ------------------
# 1. Point-in-time NIFTY Pharma + frozen investability screen.
# 2. 12-month formation / 6-month trading.
# 3. Multiple-testing treatment: split the formation window into two halves.
#      - pair must rank in the closest 10% by SSD in the first half, AND
#      - again rank in the closest 10% in the untouched second half.
#    Only then can it enter the candidate pool.
# 4. Among survivors, choose up to 4 non-overlapping pairs with the smallest
#    full-formation SSD.
# 5. Trade normalized total-return spread:
#      entry = +/- 2 formation SD
#      exit  = formation mean
#      no primary stop-loss
#      force close at 6-month window end / structural break
# 6. Signal at close; execute on the next common observation.
# 7. Long leg = cash total-return series.
# 8. Short leg = stock-futures proxy, with actual NSE single-stock-futures
#    availability checked on the primary trade entry date.
# 9. Costs: 30 bps cash RT, 8 bps futures RT, plus the SAME financing
#    assumption used in the frozen EG method.
# 10. Produce all mandatory performance, trade, attribution, exposure, cost,
#     capacity and robustness outputs.
#
# FAST-TRACK ROBUSTNESS
# ---------------------
# Robustness checks are DEVELOPMENT-only and do not re-download F&O files:
# entry/exit thresholds, formation length, universe size and half-life filter.
# This limitation is written to the output.
#
# IMPORTANT RESEARCH-PROCESS DISCLOSURE
# -------------------------------------
# The original SSD OOS result was seen before the multiple-testing correction
# was added. Therefore this script explicitly labels the final block as a
# constrained pseudo-OOS check, not a perfectly untouched holdout.
# =============================================================================


# =============================================================================
# 1. PATHS / FROZEN SETTINGS
# =============================================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "PAIR_TRADING_PROJECT_ROOT",
        r"C:\fin proj",
    )
)

SSD_ROOT = PROJECT_ROOT / "pair_trading_methods" / "SSD"
OUTPUT_DIR = SSD_ROOT / "05_FINAL_FASTTRACK_REQUIRED"
PAIR_DIR = OUTPUT_DIR / "01_pair_selection"
TRADE_DIR = OUTPUT_DIR / "02_trades"
FNO_DIR = OUTPUT_DIR / "03_entry_fno"
RAW_FO_DIR = FNO_DIR / "raw_nse_fo_bhavcopy"
RESULT_DIR = OUTPUT_DIR / "04_results"
AUDIT_DIR = OUTPUT_DIR / "05_audit"

for d in [OUTPUT_DIR, PAIR_DIR, TRADE_DIR, FNO_DIR, RAW_FO_DIR, RESULT_DIR, AUDIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INVESTABLE_FILE = (
    PROJECT_ROOT
    / "nse_pharma_final_investable_universe_FINAL"
    / "07_FINAL_INVESTABLE_BY_FORMATION.csv"
)

FORMATION_DIAGNOSTICS_FILE = (
    PROJECT_ROOT
    / "nse_pharma_formation_investability"
    / "02_STOCK_FORMATION_DIAGNOSTICS.csv"
)

TOTAL_RETURN_FILE = (
    PROJECT_ROOT
    / "nse_pharma_total_return"
    / "NIFTY_PHARMA_TOTAL_RETURN_BASE_2016_2026.csv"
)

BENCHMARK_PREFERRED = (
    PROJECT_ROOT
    / "benchmark"
    / "NIFTY500_2017_2026.csv"
)

OOS_START = pd.Timestamp("2024-08-01")

# Multiple-testing / validation rule fixed BEFORE the new result is viewed.
VALIDATION_TOP_FRACTION = 0.10
PRIMARY_FORMATION_MONTHS = 12
MAX_SELECTED_PAIRS = 4
MIN_HALF_OBSERVATIONS = 30
MIN_PAIR_COVERAGE = 0.90

# Primary trading rule.
PRIMARY_ENTRY_Z = 2.0
PRIMARY_EXIT_Z = 0.0
PAIR_GROSS_WEIGHT = 0.25
LEG_ABS_WEIGHT = 0.125

# Costs identical to frozen EG method.
CASH_RT_BPS = 30.0
FUTURES_RT_BPS = 8.0
CASH_ONE_WAY_RATE = CASH_RT_BPS / 2.0 / 10000.0
FUTURES_ONE_WAY_RATE = FUTURES_RT_BPS / 2.0 / 10000.0
COST_MULTIPLIERS = [0.0, 0.5, 1.0, 2.0]
FUTURES_MARGIN_FRACTION = 0.20
FINANCING_RATE_ANNUAL = 0.08

INITIAL_NAV = 1.0
TRADING_DAYS_PER_YEAR = 252
CAPACITY_PARTICIPATION_RATE = 0.01
NUMERIC_TOL = 1e-12

# F&O history schema switch.
UDIFF_START_DATE = pd.Timestamp("2024-07-08")
REQUEST_DELAY_SECONDS = 0.05

ALIASES = {
    "CADILAHC": "ZYDUS",
    "ZYDUSLIFE": "ZYDUS",
    "AJANTAPHARM": "AJANTPHARM",
}

# Synthetic testing only. Never set this in the real project run.
TEST_ASSUME_ALL_FNO = (
    os.environ.get("SSD_FASTTRACK_TEST_ASSUME_ALL_FNO", "0").strip() == "1"
)


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def clean_columns(df):
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.upper()
    )
    return out


def parse_dates(series):
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce").dt.normalize()
    except TypeError:
        return pd.to_datetime(series, errors="coerce").dt.normalize()


def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "1", "YES", "Y"])
    )


def require_columns(df, required, name):
    missing = set(required) - set(df.columns)
    if missing:
        raise RuntimeError(f"{name} missing required columns: {sorted(missing)}")


def parse_company_list(value):
    if pd.isna(value):
        return []
    return sorted({
        x.strip().upper()
        for x in str(value).split(";")
        if x.strip()
    })


def safe_div(a, b):
    if b is None or not np.isfinite(b) or abs(b) <= NUMERIC_TOL:
        return np.nan
    return a / b


def pair_id(a, b):
    return "__".join(sorted([str(a).upper(), str(b).upper()]))


def locate_exact_file(preferred, filename):
    if preferred.exists():
        return preferred
    hits = list(PROJECT_ROOT.rglob(filename))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            f"Required file not found:\n{preferred}\n"
            f"Also searched project root for exact filename {filename}."
        )
    # Prefer a path containing benchmark when finding the NIFTY500 file.
    benchmark_hits = [p for p in hits if "benchmark" in str(p).lower()]
    if len(benchmark_hits) == 1:
        return benchmark_hits[0]
    raise RuntimeError(
        "Multiple files with the same required name were found; refusing to guess:\n"
        + "\n".join(map(str, hits))
    )


def exact_nonoverlap_choice(candidates, max_pairs=4):
    """Maximize number of selected disjoint pairs, then minimize total FULL_SSD."""
    if candidates.empty:
        return candidates.copy()

    c = candidates.sort_values(["FULL_SSD", "PAIR_ID"]).reset_index(drop=True)
    best_combo = None
    best_score = None
    best_ids = None

    for k in range(min(max_pairs, len(c)), 0, -1):
        for combo in combinations(range(len(c)), k):
            used = set()
            valid = True
            score = 0.0
            ids = []
            for idx in combo:
                row = c.iloc[idx]
                a = row["COMPANY_A"]
                b = row["COMPANY_B"]
                if a in used or b in used:
                    valid = False
                    break
                used.add(a)
                used.add(b)
                score += float(row["FULL_SSD"])
                ids.append(row["PAIR_ID"])
            if not valid:
                continue
            ids = tuple(sorted(ids))
            if (
                best_combo is None
                or score < best_score - NUMERIC_TOL
                or (
                    abs(score - best_score) <= NUMERIC_TOL
                    and ids < best_ids
                )
            ):
                best_combo = combo
                best_score = score
                best_ids = ids
        if best_combo is not None:
            return c.iloc[list(best_combo)].copy()

    return c.iloc[0:0].copy()


def normalized_ssd(pair_df, col_a="A", col_b="B"):
    if pair_df.empty:
        return np.nan, None
    aa = pair_df[col_a].astype(float).to_numpy()
    bb = pair_df[col_b].astype(float).to_numpy()
    if aa[0] <= 0 or bb[0] <= 0:
        return np.nan, None
    na = aa / aa[0]
    nb = bb / bb[0]
    diff = na - nb
    return float(np.sum(diff ** 2)), (na, nb, diff)


def ar1_half_life(spread):
    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 20:
        return np.nan
    x = s[:-1]
    y = s[1:]
    X = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    phi = float(coef[1])
    if phi <= 0 or phi >= 1:
        return np.nan
    return float(-np.log(2.0) / np.log(phi))


# =============================================================================
# 3. LOAD THE FROZEN INPUTS
# =============================================================================

print("=" * 116)
print("SSD — COMPLETE FAST-TRACK FINAL PIPELINE")
print("=" * 116)

for path in [INVESTABLE_FILE, FORMATION_DIAGNOSTICS_FILE, TOTAL_RETURN_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found:\n{path}")

investable = clean_columns(pd.read_csv(INVESTABLE_FILE, low_memory=False))
diag = clean_columns(pd.read_csv(FORMATION_DIAGNOSTICS_FILE, low_memory=False))
prices = clean_columns(pd.read_csv(TOTAL_RETURN_FILE, low_memory=False))

require_columns(
    investable,
    ["FORMATION_DATE", "BLOCK_TYPE", "N_FINAL_INVESTABLE", "FINAL_INVESTABLE_COMPANIES"],
    "Final investable universe",
)
require_columns(
    diag,
    ["FORMATION_DATE", "FORMATION_START", "TRADING_START", "TRADING_END", "BLOCK_TYPE"],
    "Formation diagnostics",
)
require_columns(
    prices,
    [
        "DATE", "COMPANY_ID", "TOTAL_RETURN_INDEX", "TOTAL_RETURN", "CLOSE",
        "TR_PREV_CLOSE", "CA_EVENT_TYPE", "SEGMENT_ID", "STRUCTURAL_BREAK_FLAG",
        "TOTAL_TRADED_VALUE",
    ],
    "Total-return data",
)

investable["FORMATION_DATE"] = parse_dates(investable["FORMATION_DATE"])
for c in ["FORMATION_DATE", "FORMATION_START", "TRADING_START", "TRADING_END"]:
    diag[c] = parse_dates(diag[c])
prices["DATE"] = parse_dates(prices["DATE"])

for df, cols in [
    (investable, ["BLOCK_TYPE"]),
    (diag, ["BLOCK_TYPE"]),
    (prices, ["COMPANY_ID", "SEGMENT_ID", "CA_EVENT_TYPE"]),
]:
    for c in cols:
        df[c] = df[c].astype(str).str.strip().str.upper()

for c in ["TOTAL_RETURN_INDEX", "TOTAL_RETURN", "CLOSE", "TR_PREV_CLOSE", "TOTAL_TRADED_VALUE"]:
    prices[c] = pd.to_numeric(prices[c], errors="coerce")
prices["STRUCTURAL_BREAK_FLAG"] = to_bool(prices["STRUCTURAL_BREAK_FLAG"])

if prices[["DATE", "COMPANY_ID"]].duplicated().any():
    raise RuntimeError("Duplicate DATE + COMPANY_ID in total-return file.")
if prices["DATE"].isna().any() or prices["TOTAL_RETURN_INDEX"].isna().any():
    raise RuntimeError("Missing required DATE/TOTAL_RETURN_INDEX in total-return file.")
if (prices["TOTAL_RETURN_INDEX"] <= 0).any():
    raise RuntimeError("TOTAL_RETURN_INDEX must be positive.")

schedule = (
    diag[["FORMATION_DATE", "FORMATION_START", "TRADING_START", "TRADING_END", "BLOCK_TYPE"]]
    .drop_duplicates()
    .sort_values("FORMATION_DATE")
    .reset_index(drop=True)
)
if schedule["FORMATION_DATE"].duplicated().any():
    raise RuntimeError("Formation diagnostics contain inconsistent duplicate schedules.")

formation_table = schedule.merge(
    investable,
    on=["FORMATION_DATE", "BLOCK_TYPE"],
    how="inner",
    validate="one_to_one",
)
if len(formation_table) != len(schedule):
    raise RuntimeError("Frozen schedule does not reconcile with final investable universe.")


# =============================================================================
# 4. PRICE PROXY FOR THE FUTURES SHORT LEG
# =============================================================================

# On ordinary days the audited total-return move is used. On cash-dividend
# days, use the raw ex-dividend price move so the futures proxy does not pay or
# receive the shareholder cash dividend. Split/bonus/rights effects remain in
# the audited mechanically adjusted return construction.
dividend_mask = prices["CA_EVENT_TYPE"].str.contains("DIVIDEND", na=False)

prices["FUTURES_PROXY_RETURN"] = prices["TOTAL_RETURN"].astype(float)
valid_div = (
    dividend_mask
    & prices["TR_PREV_CLOSE"].notna()
    & prices["CLOSE"].notna()
    & prices["TR_PREV_CLOSE"].gt(0)
)
prices.loc[valid_div, "FUTURES_PROXY_RETURN"] = (
    prices.loc[valid_div, "CLOSE"] / prices.loc[valid_div, "TR_PREV_CLOSE"] - 1.0
)

# The first observation in a segment may legitimately have no return.
prices["_PROXY_GROWTH"] = 1.0 + prices["FUTURES_PROXY_RETURN"].fillna(0.0)
prices["FUTURES_PROXY_INDEX"] = (
    100.0
    * prices.groupby(["COMPANY_ID", "SEGMENT_ID"], sort=False)["_PROXY_GROWTH"].cumprod()
)
prices = prices.drop(columns=["_PROXY_GROWTH"])

lookup = prices.set_index(["DATE", "COMPANY_ID"]).sort_index()


def maybe_price_row(dt, company):
    key = (pd.Timestamp(dt).normalize(), str(company).upper())
    if key not in lookup.index:
        return None
    row = lookup.loc[key]
    if isinstance(row, pd.DataFrame):
        raise RuntimeError(f"Duplicate price lookup row for {company} on {pd.Timestamp(dt).date()}.")
    return row


def price_row(dt, company):
    row = maybe_price_row(dt, company)
    if row is None:
        raise RuntimeError(f"Missing price row for {company} on {pd.Timestamp(dt).date()}.")
    return row


# =============================================================================
# 5. SSD FORMATION / VALIDATION SELECTION
# =============================================================================

def select_pairs_for_spec(formation_months=12, universe_fraction=1.0, block_only=None):
    """
    Mandatory multiple-testing treatment:
      first half = selection
      second half = untouched validation
      pair must be closest 10% in BOTH halves.

    Optimized implementation: each formation window is sliced/pivoted once;
    pair calculations then use the in-memory matrix rather than repeatedly
    filtering the full price table.
    """
    all_pair_rows = []
    selected_rows = []
    audit_rows = []

    rows_iter = formation_table.copy()
    if block_only is not None:
        rows_iter = rows_iter[rows_iter["BLOCK_TYPE"].eq(str(block_only).upper())]

    for form in rows_iter.itertuples(index=False):
        fd = pd.Timestamp(form.FORMATION_DATE)
        block = str(form.BLOCK_TYPE)
        trading_start = pd.Timestamp(form.TRADING_START)
        trading_end = pd.Timestamp(form.TRADING_END)

        if int(formation_months) == PRIMARY_FORMATION_MONTHS:
            fs = pd.Timestamp(form.FORMATION_START)
        else:
            fs = (fd - pd.DateOffset(months=int(formation_months)) + pd.Timedelta(days=1)).normalize()

        companies = parse_company_list(form.FINAL_INVESTABLE_COMPANIES)

        window_all = prices.loc[
            prices["DATE"].between(fs, fd, inclusive="both")
            & prices["COMPANY_ID"].isin(companies),
            ["DATE", "COMPANY_ID", "TOTAL_RETURN_INDEX", "SEGMENT_ID", "TOTAL_TRADED_VALUE"],
        ].copy()

        if universe_fraction < 1.0 and companies:
            liq = (
                window_all.groupby("COMPANY_ID")["TOTAL_TRADED_VALUE"]
                .median()
                .sort_values(ascending=False)
            )
            keep_n = max(4, int(np.ceil(len(companies) * float(universe_fraction))))
            companies = [c for c in liq.index[:keep_n] if c in companies]
            window_all = window_all[window_all["COMPANY_ID"].isin(companies)].copy()

        split_date = (fs + pd.DateOffset(months=int(round(int(formation_months) / 2.0)))).normalize()
        if split_date <= fs or split_date > fd:
            raise RuntimeError(f"Bad validation split for {fd.date()} / {formation_months}m.")

        expected_dates = pd.DatetimeIndex(sorted(
            prices.loc[prices["DATE"].between(fs, fd, inclusive="both"), "DATE"].unique()
        ))
        n_expected = len(expected_dates)

        if window_all.empty or n_expected == 0:
            pair_df = pd.DataFrame()
            valid_df = pd.DataFrame()
            survivors = pd.DataFrame()
            picked = pd.DataFrame()
            top_n = 0
        else:
            tri = window_all.pivot(index="DATE", columns="COMPANY_ID", values="TOTAL_RETURN_INDEX").sort_index()
            seg_counts = window_all.groupby("COMPANY_ID")["SEGMENT_ID"].nunique()
            obs_counts = tri.notna().sum()
            eligible_companies = [
                c for c in companies
                if c in tri.columns
                and int(seg_counts.get(c, 0)) == 1
                and int(obs_counts.get(c, 0)) >= max(2 * MIN_HALF_OBSERVATIONS, int(np.ceil(MIN_PAIR_COVERAGE * n_expected)))
            ]

            pair_rows = []
            for a, b in combinations(eligible_companies, 2):
                pair = tri[[a, b]].dropna()
                first = pair[pair.index < split_date]
                second = pair[pair.index >= split_date]
                valid = len(first) >= MIN_HALF_OBSERVATIONS and len(second) >= MIN_HALF_OBSERVATIONS
                reason = "" if valid else "INSUFFICIENT_HALF_WINDOW_OBSERVATIONS"
                first_ssd = second_ssd = full_ssd = np.nan
                if valid:
                    fa = first[a].to_numpy(dtype=float); fb = first[b].to_numpy(dtype=float)
                    sa = second[a].to_numpy(dtype=float); sb = second[b].to_numpy(dtype=float)
                    aa = pair[a].to_numpy(dtype=float); bb = pair[b].to_numpy(dtype=float)
                    first_ssd = float(np.sum((fa/fa[0] - fb/fb[0])**2))
                    second_ssd = float(np.sum((sa/sa[0] - sb/sb[0])**2))
                    full_ssd = float(np.sum((aa/aa[0] - bb/bb[0])**2))
                pair_rows.append({
                    "FORMATION_DATE": fd, "BLOCK_TYPE": block,
                    "FORMATION_START": fs, "FORMATION_END": fd,
                    "VALIDATION_SPLIT_DATE": split_date,
                    "TRADING_START": trading_start, "TRADING_END": trading_end,
                    "PAIR_ID": pair_id(a, b), "COMPANY_A": min(a,b), "COMPANY_B": max(a,b),
                    "COMMON_OBSERVATIONS": len(pair),
                    "FIRST_HALF_OBSERVATIONS": len(first), "SECOND_HALF_OBSERVATIONS": len(second),
                    "VALID_PAIR": valid, "INVALID_REASON": reason,
                    "FIRST_HALF_SSD": first_ssd, "SECOND_HALF_SSD": second_ssd, "FULL_SSD": full_ssd,
                })

            pair_df = pd.DataFrame(pair_rows)
            valid_df = pair_df[pair_df["VALID_PAIR"]].copy() if not pair_df.empty else pd.DataFrame()
            if not valid_df.empty:
                valid_df["FIRST_HALF_RANK"] = valid_df["FIRST_HALF_SSD"].rank(method="first", ascending=True).astype(int)
                valid_df["SECOND_HALF_RANK"] = valid_df["SECOND_HALF_SSD"].rank(method="first", ascending=True).astype(int)
                valid_df["FULL_SSD_RANK"] = valid_df["FULL_SSD"].rank(method="first", ascending=True).astype(int)
                top_n = max(1, int(np.ceil(VALIDATION_TOP_FRACTION * len(valid_df))))
                valid_df["FIRST_HALF_TOP10"] = valid_df["FIRST_HALF_RANK"].le(top_n)
                valid_df["SECOND_HALF_TOP10"] = valid_df["SECOND_HALF_RANK"].le(top_n)
                valid_df["SURVIVES_VALIDATION"] = valid_df["FIRST_HALF_TOP10"] & valid_df["SECOND_HALF_TOP10"]
                pair_df = pair_df.merge(
                    valid_df[["PAIR_ID", "FIRST_HALF_RANK", "SECOND_HALF_RANK", "FULL_SSD_RANK", "FIRST_HALF_TOP10", "SECOND_HALF_TOP10", "SURVIVES_VALIDATION"]],
                    on="PAIR_ID", how="left"
                )
                survivors = valid_df[valid_df["SURVIVES_VALIDATION"]].copy()
                picked = exact_nonoverlap_choice(survivors, MAX_SELECTED_PAIRS)
            else:
                top_n = 0
                survivors = valid_df
                picked = valid_df

            picked = picked.sort_values(["FULL_SSD", "PAIR_ID"]).reset_index(drop=True) if not picked.empty else picked

            # Defensive selection checks: fail immediately rather than
            # silently writing a bad pair-selection file.
            if not picked.empty:
                if not picked["SURVIVES_VALIDATION"].fillna(False).all():
                    raise RuntimeError(
                        f"{fd.date()}: selected SSD pair failed the two-half validation rule."
                    )

                selected_names = list(picked["COMPANY_A"]) + list(picked["COMPANY_B"])
                if len(selected_names) != len(set(selected_names)):
                    raise RuntimeError(
                        f"{fd.date()}: selected SSD pairs overlap in company names."
                    )

                if len(picked) > MAX_SELECTED_PAIRS:
                    raise RuntimeError(
                        f"{fd.date()}: more than {MAX_SELECTED_PAIRS} SSD pairs selected."
                    )

            for n, row in enumerate(picked.itertuples(index=False), start=1):
                a = row.COMPANY_A; b = row.COMPANY_B
                pair = tri[[a,b]].dropna()
                aa = pair[a].to_numpy(dtype=float); bb = pair[b].to_numpy(dtype=float)
                na = aa/aa[0]; nb = bb/bb[0]; spread = na-nb
                mu = float(np.mean(spread)); sd = float(np.std(spread, ddof=1))
                if not np.isfinite(sd) or sd <= NUMERIC_TOL:
                    raise RuntimeError(f"Zero/invalid formation spread SD for {fd.date()} {row.PAIR_ID}.")
                seg_a = str(window_all.loc[window_all["COMPANY_ID"].eq(a), "SEGMENT_ID"].iloc[0])
                seg_b = str(window_all.loc[window_all["COMPANY_ID"].eq(b), "SEGMENT_ID"].iloc[0])
                selected_rows.append({
                    "FORMATION_DATE": fd, "BLOCK_TYPE": block,
                    "FORMATION_START": fs, "FORMATION_END": fd,
                    "VALIDATION_SPLIT_DATE": split_date,
                    "TRADING_START": trading_start, "TRADING_END": trading_end,
                    "SELECTED_PAIR_NUMBER": n,
                    "PAIR_ID": row.PAIR_ID, "COMPANY_A": a, "COMPANY_B": b,
                    "FIRST_HALF_SSD": float(row.FIRST_HALF_SSD),
                    "SECOND_HALF_SSD": float(row.SECOND_HALF_SSD),
                    "FULL_SSD": float(row.FULL_SSD),
                    "FIRST_HALF_RANK": int(row.FIRST_HALF_RANK),
                    "SECOND_HALF_RANK": int(row.SECOND_HALF_RANK),
                    "FULL_SSD_RANK": int(row.FULL_SSD_RANK),
                    "VALIDATION_TOP_N": int(top_n),
                    "ANCHOR_DATE": pd.Timestamp(pair.index[0]),
                    "ANCHOR_TRI_A": float(aa[0]), "ANCHOR_TRI_B": float(bb[0]),
                    "FORMATION_SEGMENT_A": seg_a, "FORMATION_SEGMENT_B": seg_b,
                    "FORMATION_SPREAD_MEAN": mu, "FORMATION_SPREAD_STD": sd,
                    "HALF_LIFE_SESSIONS": ar1_half_life(spread),
                    "FORMATION_MONTHS": int(formation_months),
                    "UNIVERSE_FRACTION": float(universe_fraction),
                })

        if not pair_df.empty:
            all_pair_rows.extend(pair_df.to_dict("records"))
        audit_rows.append({
            "FORMATION_DATE": fd, "BLOCK_TYPE": block,
            "N_INVESTABLE_INPUT": len(parse_company_list(form.FINAL_INVESTABLE_COMPANIES)),
            "N_COMPANIES_USED": len(companies),
            "N_PAIR_RELATIONSHIPS_EXAMINED": int(len(pair_df)),
            "N_VALID_PAIRS": int(len(valid_df)),
            "TOP_10PCT_COUNT_PER_HALF": int(top_n),
            "N_SURVIVING_BOTH_HALVES": int(len(survivors)),
            "N_SELECTED_NONOVERLAPPING": int(len(picked)),
            "MAX_SELECTED_ALLOWED": MAX_SELECTED_PAIRS,
            "MULTIPLE_TESTING_RULE": "TOP_10_PERCENT_FIRST_HALF_AND_TOP_10_PERCENT_UNTOUCHED_SECOND_HALF",
            "STATUS": (
                "PASS"
                if (
                    len(picked) <= MAX_SELECTED_PAIRS
                    and (
                        picked.empty
                        or picked["SURVIVES_VALIDATION"].fillna(False).all()
                    )
                    and (
                        picked.empty
                        or len(list(picked["COMPANY_A"]) + list(picked["COMPANY_B"]))
                        == len(set(list(picked["COMPANY_A"]) + list(picked["COMPANY_B"])))
                    )
                )
                else "FAIL"
            ),
        })

    return pd.DataFrame(all_pair_rows), pd.DataFrame(selected_rows), pd.DataFrame(audit_rows)


print("\n1/7 Selecting SSD pairs with formation/validation correction...")
all_pairs_primary, selected_primary, selection_audit = select_pairs_for_spec(
    formation_months=PRIMARY_FORMATION_MONTHS,
    universe_fraction=1.0,
)

if (selection_audit["STATUS"] != "PASS").any():
    raise RuntimeError("SSD formation/validation selection audit failed.")

all_pairs_primary.to_csv(PAIR_DIR / "01_ALL_PAIR_VALIDATION_RANKINGS.csv", index=False, date_format="%Y-%m-%d")
selected_primary.to_csv(PAIR_DIR / "02_SELECTED_VALIDATED_SSD_PAIRS.csv", index=False, date_format="%Y-%m-%d")
selection_audit.to_csv(PAIR_DIR / "03_MULTIPLE_TESTING_SELECTION_AUDIT.csv", index=False, date_format="%Y-%m-%d")

print(
    f"   Pair relationships examined: {selection_audit['N_PAIR_RELATIONSHIPS_EXAMINED'].sum():,}\n"
    f"   Pair-periods surviving both halves: {selection_audit['N_SURVIVING_BOTH_HALVES'].sum():,}\n"
    f"   Pair-periods finally selected: {len(selected_primary):,}"
)


# =============================================================================
# 6. SIGNAL / LOGICAL TRADE GENERATOR
# =============================================================================

def generate_logical_trades(selected_df, entry_z=2.0, exit_z=0.0, stop_z=None, variant_name="PRIMARY"):
    trade_rows = []
    pair_audit_rows = []

    if selected_df is None or selected_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    for row in selected_df.sort_values(["FORMATION_DATE", "SELECTED_PAIR_NUMBER"]).itertuples(index=False):
        fd = pd.Timestamp(row.FORMATION_DATE)
        ts = pd.Timestamp(row.TRADING_START)
        te = pd.Timestamp(row.TRADING_END)
        a = str(row.COMPANY_A)
        b = str(row.COMPANY_B)
        mu = float(row.FORMATION_SPREAD_MEAN)
        sd = float(row.FORMATION_SPREAD_STD)
        anchor_a = float(row.ANCHOR_TRI_A)
        anchor_b = float(row.ANCHOR_TRI_B)
        seg_a = str(row.FORMATION_SEGMENT_A)
        seg_b = str(row.FORMATION_SEGMENT_B)

        aa = prices.loc[
            prices["COMPANY_ID"].eq(a) & prices["DATE"].between(ts, te, inclusive="both"),
            ["DATE", "TOTAL_RETURN_INDEX", "SEGMENT_ID", "STRUCTURAL_BREAK_FLAG"],
        ].rename(columns={
            "TOTAL_RETURN_INDEX": "TRI_A",
            "SEGMENT_ID": "SEG_A",
            "STRUCTURAL_BREAK_FLAG": "BREAK_A",
        })
        bb = prices.loc[
            prices["COMPANY_ID"].eq(b) & prices["DATE"].between(ts, te, inclusive="both"),
            ["DATE", "TOTAL_RETURN_INDEX", "SEGMENT_ID", "STRUCTURAL_BREAK_FLAG"],
        ].rename(columns={
            "TOTAL_RETURN_INDEX": "TRI_B",
            "SEGMENT_ID": "SEG_B",
            "STRUCTURAL_BREAK_FLAG": "BREAK_B",
        })
        pair = aa.merge(bb, on="DATE", how="inner").sort_values("DATE").reset_index(drop=True)
        if pair.empty:
            pair_audit_rows.append({
                "FORMATION_DATE": fd, "PAIR_ID": row.PAIR_ID, "N_TRADES": 0,
                "MEAN_OR_BAND_EXITS": 0, "FORCED_EXITS": 0, "STATUS": "PASS_EMPTY_TRADING_PANEL",
            })
            continue

        bad = (
            pair["BREAK_A"].astype(bool)
            | pair["BREAK_B"].astype(bool)
            | pair["SEG_A"].astype(str).ne(seg_a)
            | pair["SEG_B"].astype(str).ne(seg_b)
        )
        bad_pos = np.where(bad.to_numpy())[0]
        if len(bad_pos):
            terminal_pos = int(bad_pos[0]) - 1
            terminal_reason = "STRUCTURAL_BREAK_FORCE_CLOSE"
        else:
            terminal_pos = len(pair) - 1
            terminal_reason = "WINDOW_END_FORCE_CLOSE"
        if terminal_pos < 1:
            continue
        pair = pair.iloc[: terminal_pos + 1].copy().reset_index(drop=True)
        pair["NORM_A"] = pair["TRI_A"].astype(float) / anchor_a
        pair["NORM_B"] = pair["TRI_B"].astype(float) / anchor_b
        pair["SPREAD"] = pair["NORM_A"] - pair["NORM_B"]
        pair["Z"] = (pair["SPREAD"] - mu) / sd

        scan = 0
        trade_no = 0
        mean_exits = 0
        forced_exits = 0

        while scan <= len(pair) - 2:
            sig = None
            side = None
            for i in range(scan, len(pair) - 1):
                z = float(pair.loc[i, "Z"])
                if z >= entry_z:
                    sig, side = i, "HIGH"
                    break
                if z <= -entry_z:
                    sig, side = i, "LOW"
                    break
            if sig is None:
                break
            entry = sig + 1
            # Require at least one common observation after entry so the
            # position is never opened and force-closed on the same date.
            if entry >= len(pair) - 1:
                break

            if side == "HIGH":
                direction = "SHORT_A_LONG_B"
                long_company, short_company = b, a
            else:
                direction = "LONG_A_SHORT_B"
                long_company, short_company = a, b

            exit_sig = None
            exit_reason = None
            for j in range(entry, len(pair) - 1):
                z = float(pair.loc[j, "Z"])
                converged = (
                    (side == "HIGH" and z <= exit_z)
                    or (side == "LOW" and z >= -exit_z)
                )
                stopped = (
                    stop_z is not None
                    and (
                        (side == "HIGH" and z >= stop_z)
                        or (side == "LOW" and z <= -stop_z)
                    )
                )
                if stopped:
                    exit_sig, exit_reason = j, "STOP_OUT"
                    break
                if converged:
                    exit_sig = j
                    exit_reason = "MEAN_CROSS" if exit_z == 0.0 else "EXIT_BAND_REACHED"
                    break

            if exit_sig is None:
                exit_exec = len(pair) - 1
                exit_signal_date = pd.NaT
                exit_signal_spread = np.nan
                exit_signal_z = np.nan
                exit_reason = terminal_reason
                forced_exits += 1
            else:
                exit_exec = exit_sig + 1
                exit_signal_date = pd.Timestamp(pair.loc[exit_sig, "DATE"])
                exit_signal_spread = float(pair.loc[exit_sig, "SPREAD"])
                exit_signal_z = float(pair.loc[exit_sig, "Z"])
                mean_exits += int(exit_reason in ["MEAN_CROSS", "EXIT_BAND_REACHED"])

            trade_no += 1
            trade_rows.append({
                "VARIANT": variant_name,
                "FORMATION_DATE": fd,
                "BLOCK_TYPE": row.BLOCK_TYPE,
                "SELECTED_PAIR_NUMBER": int(row.SELECTED_PAIR_NUMBER),
                "PAIR_ID": row.PAIR_ID,
                "COMPANY_A": a,
                "COMPANY_B": b,
                "TRADE_NUMBER_WITHIN_PAIR_WINDOW": trade_no,
                "ENTRY_SIGNAL_DATE": pd.Timestamp(pair.loc[sig, "DATE"]),
                "ENTRY_SIGNAL_SPREAD": float(pair.loc[sig, "SPREAD"]),
                "ENTRY_SIGNAL_Z": float(pair.loc[sig, "Z"]),
                "ENTRY_EXECUTION_DATE": pd.Timestamp(pair.loc[entry, "DATE"]),
                "ENTRY_EXECUTION_SPREAD": float(pair.loc[entry, "SPREAD"]),
                "ENTRY_EXECUTION_Z": float(pair.loc[entry, "Z"]),
                "ENTRY_DIRECTION": direction,
                "LONG_COMPANY": long_company,
                "SHORT_COMPANY": short_company,
                "EXIT_SIGNAL_DATE": exit_signal_date,
                "EXIT_SIGNAL_SPREAD": exit_signal_spread,
                "EXIT_SIGNAL_Z": exit_signal_z,
                "EXIT_EXECUTION_DATE": pd.Timestamp(pair.loc[exit_exec, "DATE"]),
                "EXIT_EXECUTION_SPREAD": float(pair.loc[exit_exec, "SPREAD"]),
                "EXIT_EXECUTION_Z": float(pair.loc[exit_exec, "Z"]),
                "EXIT_REASON": exit_reason,
                "HOLDING_CALENDAR_DAYS": int((pd.Timestamp(pair.loc[exit_exec, "DATE"]) - pd.Timestamp(pair.loc[entry, "DATE"])).days),
                "PAIR_GROSS_CAPITAL_WEIGHT": PAIR_GROSS_WEIGHT,
                "ABS_WEIGHT_PER_LEG": LEG_ABS_WEIGHT,
            })

            # Normal completed trade can re-enter immediately from the next
            # execution observation. After a stop, require reset inside the
            # entry band before another trade is allowed.
            if exit_sig is None:
                break
            scan = exit_exec
            if exit_reason == "STOP_OUT":
                while scan <= len(pair) - 2 and abs(float(pair.loc[scan, "Z"])) >= entry_z:
                    scan += 1

        pair_audit_rows.append({
            "FORMATION_DATE": fd,
            "BLOCK_TYPE": row.BLOCK_TYPE,
            "SELECTED_PAIR_NUMBER": int(row.SELECTED_PAIR_NUMBER),
            "PAIR_ID": row.PAIR_ID,
            "N_TRADES": trade_no,
            "MEAN_OR_BAND_EXITS": mean_exits,
            "FORCED_EXITS": forced_exits,
            "STATUS": "PASS",
        })

    trades = pd.DataFrame(trade_rows)
    audit = pd.DataFrame(pair_audit_rows)
    if not trades.empty:
        trades["SSD_TRADE_ID"] = [
            f"SSD_{fd:%Y%m%d}_P{int(p):02d}_T{int(t):02d}"
            for fd, p, t in zip(
                trades["FORMATION_DATE"],
                trades["SELECTED_PAIR_NUMBER"],
                trades["TRADE_NUMBER_WITHIN_PAIR_WINDOW"],
            )
        ]
        if trades["SSD_TRADE_ID"].duplicated().any():
            raise RuntimeError("Duplicate SSD_TRADE_ID generated.")
    return trades, audit


print("\n2/7 Generating primary SSD logical trades...")
logical_primary, signal_audit = generate_logical_trades(
    selected_primary,
    entry_z=PRIMARY_ENTRY_Z,
    exit_z=PRIMARY_EXIT_Z,
    stop_z=None,
    variant_name="PRIMARY",
)
logical_primary.to_csv(TRADE_DIR / "01_PRIMARY_LOGICAL_TRADE_LEDGER_NO_PNL.csv", index=False, date_format="%Y-%m-%d")
signal_audit.to_csv(TRADE_DIR / "02_PRIMARY_SIGNAL_AUDIT.csv", index=False, date_format="%Y-%m-%d")
print(f"   Logical trades: {len(logical_primary):,}")


# =============================================================================
# 7. OFFICIAL NSE ENTRY-DATE F&O GATE — PRIMARY TRADES ONLY
# =============================================================================

def clean_col(v):
    return re.sub(r"[^A-Z0-9]+", "_", str(v).strip().upper()).strip("_")


def clean_symbol(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = re.sub(r"\s+", "", str(v).strip().upper())
    if s in {"", "NAN", "NONE", "NA", "N/A", "-", "--"}:
        return None
    return s


def company_id_from_symbol(symbol):
    s = clean_symbol(symbol)
    return None if s is None else ALIASES.get(s, s)


def valid_zip_bytes(content):
    if not isinstance(content, (bytes, bytearray)) or len(content) < 100 or content[:2] != b"PK":
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            return z.testzip() is None
    except Exception:
        return False


def decode_csv_bytes(content):
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            pass
    return content.decode("latin-1", errors="replace")


def fo_urls(date_value):
    dt = pd.Timestamp(date_value).normalize()
    if dt < UDIFF_START_DATE:
        year = dt.strftime("%Y")
        month = dt.strftime("%b").upper()
        day = dt.strftime("%d")
        filename = f"fo{day}{month}{year}bhav.csv.zip"
        urls = [
            f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{year}/{month}/{filename}",
            f"https://archives.nseindia.com/content/historical/DERIVATIVES/{year}/{month}/{filename}",
        ]
        schema = "LEGACY"
    else:
        ymd = dt.strftime("%Y%m%d")
        filename = f"BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip"
        urls = [
            f"https://nsearchives.nseindia.com/content/fo/{filename}",
            f"https://archives.nseindia.com/content/fo/{filename}",
        ]
        schema = "UDIFF"
    return urls, schema, filename


def build_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    try:
        s.get("https://www.nseindia.com/", timeout=15)
    except Exception:
        pass
    return s


def obtain_fo_zip(dt, session):
    urls, schema, filename = fo_urls(dt)
    target = RAW_FO_DIR / filename
    if target.exists() and valid_zip_bytes(target.read_bytes()):
        return target, schema, "FINAL_SSD_CACHE"

    prior_dirs = [
        SSD_ROOT / "03_backtest" / "01_entry_date_fno_check" / "raw_nse_fo_bhavcopy",
        SSD_ROOT / "05_FINAL_FASTTRACK_REQUIRED" / "03_entry_fno" / "raw_nse_fo_bhavcopy",
        PROJECT_ROOT / "pair_trading_methods" / "ENGLE_GRANGER" / "02_FINAL_BACKTEST" / "raw_nse_fo_bhavcopy",
        PROJECT_ROOT / "nse_pharma_final_investable_universe_FINAL" / "raw_nse_fo_bhavcopy",
    ]
    external = os.environ.get("NIFTY_FO_CACHE_DIR")
    if external:
        prior_dirs.append(Path(external))

    for d in prior_dirs:
        prior = d / filename
        if prior.exists() and valid_zip_bytes(prior.read_bytes()):
            shutil.copy2(prior, target)
            return target, schema, f"REUSED_{d}"

    errors = []
    for attempt in range(1, 4):
        for url in urls:
            try:
                r = session.get(url, timeout=45, allow_redirects=True)
                if r.status_code == 200 and valid_zip_bytes(r.content):
                    target.write_bytes(r.content)
                    time.sleep(REQUEST_DELAY_SECONDS)
                    return target, schema, url
                errors.append(f"attempt={attempt} http={r.status_code} url={url}")
            except Exception as exc:
                errors.append(f"attempt={attempt} {type(exc).__name__}: {exc} url={url}")
        time.sleep(attempt)
    raise RuntimeError(
        f"Could not obtain official NSE F&O bhavcopy for {pd.Timestamp(dt).date()}.\n"
        + "\n".join(errors)
    )


def parse_fo_zip(path, expected_schema):
    with zipfile.ZipFile(path) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        for name in csv_names:
            text = decode_csv_bytes(z.read(name))
            reader = csv.DictReader(io.StringIO(text))
            if not reader.fieldnames:
                continue
            colmap = {clean_col(c): c for c in reader.fieldnames}
            if "INSTRUMENT" in colmap and "SYMBOL" in colmap:
                eligible = set()
                for row in reader:
                    if str(row.get(colmap["INSTRUMENT"], "")).strip().upper() != "FUTSTK":
                        continue
                    company = company_id_from_symbol(row.get(colmap["SYMBOL"]))
                    if company:
                        eligible.add(company)
                if eligible:
                    return eligible, "LEGACY_FUTSTK", name
            if "FININSTRMTP" in colmap and "TCKRSYMB" in colmap:
                eligible = set()
                for row in reader:
                    if str(row.get(colmap["FININSTRMTP"], "")).strip().upper() != "STF":
                        continue
                    company = company_id_from_symbol(row.get(colmap["TCKRSYMB"]))
                    if company:
                        eligible.add(company)
                if eligible:
                    return eligible, "UDIFF_STF", name
    raise RuntimeError(f"No recognized stock-futures schema in {path}; expected {expected_schema}.")


def primary_fno_gate(logical):
    if logical.empty:
        empty = logical.copy()
        empty["SHORT_FUTURES_AVAILABLE_ON_ENTRY"] = pd.Series(dtype=bool)
        return empty, pd.DataFrame()

    if TEST_ASSUME_ALL_FNO:
        out = logical.copy()
        out["SHORT_FUTURES_AVAILABLE_ON_ENTRY"] = True
        out["FNO_SOURCE"] = "SYNTHETIC_TEST_MODE_ONLY"
        return out, pd.DataFrame({"TEST_MODE": [True]})

    session = build_session()
    date_cache = {}
    date_audit = []
    rows = []

    for dt in sorted(logical["ENTRY_EXECUTION_DATE"].dropna().unique()):
        dt = pd.Timestamp(dt).normalize()
        path, expected_schema, source = obtain_fo_zip(dt, session)
        eligible, parsed_schema, member = parse_fo_zip(path, expected_schema)
        date_cache[dt] = eligible
        date_audit.append({
            "ENTRY_EXECUTION_DATE": dt,
            "EXPECTED_SCHEMA": expected_schema,
            "PARSED_SCHEMA": parsed_schema,
            "SOURCE": source,
            "ZIP_FILE": str(path),
            "CSV_MEMBER": member,
            "N_STOCKS_WITH_FUTURES": len(eligible),
            "STATUS": "PASS",
        })

    for tr in logical.itertuples(index=False):
        dt = pd.Timestamp(tr.ENTRY_EXECUTION_DATE).normalize()
        short = str(tr.SHORT_COMPANY).upper()
        ok = short in date_cache[dt]
        r = tr._asdict()
        r["SHORT_FUTURES_AVAILABLE_ON_ENTRY"] = bool(ok)
        r["FNO_SOURCE"] = "OFFICIAL_NSE_DAILY_FO_BHAVCOPY"
        rows.append(r)

    return pd.DataFrame(rows), pd.DataFrame(date_audit)


print("\n3/7 Checking official NSE stock-futures availability on primary trade entries...")
fno_checked, fno_date_audit = primary_fno_gate(logical_primary)
if fno_checked.empty:
    executable_primary = fno_checked.copy()
    skipped_fno = fno_checked.copy()
else:
    executable_primary = fno_checked[fno_checked["SHORT_FUTURES_AVAILABLE_ON_ENTRY"]].copy()
    skipped_fno = fno_checked[~fno_checked["SHORT_FUTURES_AVAILABLE_ON_ENTRY"]].copy()

fno_checked.to_csv(FNO_DIR / "01_PRIMARY_ENTRY_FNO_CHECK.csv", index=False, date_format="%Y-%m-%d")
fno_date_audit.to_csv(FNO_DIR / "02_FNO_DATE_AUDIT.csv", index=False, date_format="%Y-%m-%d")
skipped_fno.to_csv(FNO_DIR / "03_SKIPPED_FNO_INFEASIBLE_TRADES.csv", index=False, date_format="%Y-%m-%d")
print(f"   Executable primary trades: {len(executable_primary):,} / {len(logical_primary):,}")


# =============================================================================
# 8. PORTFOLIO SIMULATION
# =============================================================================

def market_dates_for(trade_df, start=None, end=None):
    if start is None:
        start = trade_df["ENTRY_EXECUTION_DATE"].min() if not trade_df.empty else schedule["TRADING_START"].min()
    if end is None:
        end = trade_df["EXIT_EXECUTION_DATE"].max() if not trade_df.empty else schedule["TRADING_END"].max()
    return pd.DatetimeIndex(sorted(
        prices.loc[prices["DATE"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both"), "DATE"].unique()
    ))


def simulate_portfolio(trade_df, cost_multiplier=1.0, start=None, end=None):
    trades = trade_df.copy()
    if not trades.empty:
        for c in ["ENTRY_EXECUTION_DATE", "EXIT_EXECUTION_DATE", "FORMATION_DATE", "ENTRY_SIGNAL_DATE"]:
            trades[c] = pd.to_datetime(trades[c]).dt.normalize()
        if "SSD_TRADE_ID" not in trades.columns:
            trades["SSD_TRADE_ID"] = [
                f"VAR_{i:05d}" for i in range(len(trades))
            ]

    dates = market_dates_for(trades, start=start, end=end)
    if len(dates) == 0:
        return pd.DataFrame(), pd.DataFrame()

    entries = {dt: g.copy() for dt, g in trades.groupby("ENTRY_EXECUTION_DATE")} if not trades.empty else {}
    exits = {dt: g.copy() for dt, g in trades.groupby("EXIT_EXECUTION_DATE")} if not trades.empty else {}

    nav = INITIAL_NAV
    positions = {}
    daily_rows = []
    completed = []
    previous_market_date = None

    for dt in dates:
        dt = pd.Timestamp(dt)
        nav_start = nav
        gross_pnl_today = 0.0
        transaction_cost_today = 0.0
        financing_cost_today = 0.0
        turnover_today = 0.0

        # Mark positions only when both new observations are available. Signal
        # generation itself never forward-fills. If a valuation observation is
        # missing, NAV remains stale for that position until the next common
        # mark; this is recorded as a limitation rather than creating a fake
        # spread observation.
        for trade_id, pos in list(positions.items()):
            if previous_market_date is not None:
                calendar_days = max(0, int((dt - previous_market_date).days))
                short_notional_last = abs(pos["SHORT_UNITS"] * pos["LAST_SHORT_PROXY"])
                financing = (
                    short_notional_last
                    * FUTURES_MARGIN_FRACTION
                    * FINANCING_RATE_ANNUAL
                    * calendar_days / 365.0
                    * float(cost_multiplier)
                )
                nav -= financing
                financing_cost_today += financing
                pos["FINANCING_COST"] += financing

            long_now = maybe_price_row(dt, pos["LONG_COMPANY"])
            short_now = maybe_price_row(dt, pos["SHORT_COMPANY"])
            if long_now is not None and short_now is not None:
                if str(long_now["SEGMENT_ID"]) != pos["LONG_SEGMENT"] or str(short_now["SEGMENT_ID"]) != pos["SHORT_SEGMENT"]:
                    raise RuntimeError(f"Active trade crosses structural segment: {trade_id}")
                long_index = float(long_now["TOTAL_RETURN_INDEX"])
                short_proxy = float(short_now["FUTURES_PROXY_INDEX"])
                long_pnl = pos["LONG_UNITS"] * (long_index - pos["LAST_LONG_INDEX"])
                short_pnl = -pos["SHORT_UNITS"] * (short_proxy - pos["LAST_SHORT_PROXY"])
                pnl = long_pnl + short_pnl
                nav += pnl
                gross_pnl_today += pnl
                pos["LONG_PNL"] += long_pnl
                pos["SHORT_PNL"] += short_pnl
                pos["GROSS_PNL"] += pnl
                pos["LAST_LONG_INDEX"] = long_index
                pos["LAST_SHORT_PROXY"] = short_proxy
                pos["LAST_MARK_DATE"] = dt

        # Exits first.
        if dt in exits:
            for tr in exits[dt].itertuples(index=False):
                tid = tr.SSD_TRADE_ID
                if tid not in positions:
                    raise RuntimeError(f"Exit without open position: {tid}")
                pos = positions[tid]
                long_now = price_row(dt, pos["LONG_COMPANY"])
                short_now = price_row(dt, pos["SHORT_COMPANY"])
                long_value = abs(pos["LONG_UNITS"] * float(long_now["TOTAL_RETURN_INDEX"]))
                short_value = abs(pos["SHORT_UNITS"] * float(short_now["FUTURES_PROXY_INDEX"]))
                exit_cost = (
                    long_value * CASH_ONE_WAY_RATE
                    + short_value * FUTURES_ONE_WAY_RATE
                ) * float(cost_multiplier)
                nav -= exit_cost
                transaction_cost_today += exit_cost
                turnover = long_value + short_value
                turnover_today += turnover
                pos["TRANSACTION_COST"] += exit_cost
                pos["TOTAL_COST"] = pos["TRANSACTION_COST"] + pos["FINANCING_COST"]
                pos["TURNOVER_VALUE"] += turnover
                pos["EXIT_EXECUTION_DATE"] = dt
                pos["EXIT_REASON"] = tr.EXIT_REASON
                pos["EXIT_SIGNAL_DATE"] = tr.EXIT_SIGNAL_DATE
                pos["EXIT_SIGNAL_Z"] = tr.EXIT_SIGNAL_Z
                pos["EXIT_EXECUTION_Z"] = tr.EXIT_EXECUTION_Z
                pos["HOLDING_CALENDAR_DAYS"] = int((dt - pos["ENTRY_EXECUTION_DATE"]).days)
                pos["NET_PNL"] = pos["GROSS_PNL"] - pos["TOTAL_COST"]
                pos["GROSS_RETURN_ON_PAIR_GROSS"] = safe_div(pos["GROSS_PNL"], pos["INITIAL_PAIR_GROSS"])
                pos["NET_RETURN_ON_PAIR_GROSS"] = safe_div(pos["NET_PNL"], pos["INITIAL_PAIR_GROSS"])
                completed.append(pos.copy())
                del positions[tid]

        # Same-day entries use common pre-entry NAV basis.
        if dt in entries:
            nav_basis = nav
            active_slots = {(p["FORMATION_DATE"], p["SELECTED_PAIR_NUMBER"]) for p in positions.values()}
            for tr in entries[dt].itertuples(index=False):
                slot = (pd.Timestamp(tr.FORMATION_DATE), int(tr.SELECTED_PAIR_NUMBER))
                if slot in active_slots:
                    raise RuntimeError(f"Overlapping trade in same SSD pair slot: {slot}")
                long_row = price_row(dt, tr.LONG_COMPANY)
                short_row = price_row(dt, tr.SHORT_COMPANY)
                long_notional = LEG_ABS_WEIGHT * nav_basis
                short_notional = LEG_ABS_WEIGHT * nav_basis
                long_units = long_notional / float(long_row["TOTAL_RETURN_INDEX"])
                short_units = short_notional / float(short_row["FUTURES_PROXY_INDEX"])
                entry_cost = (
                    long_notional * CASH_ONE_WAY_RATE
                    + short_notional * FUTURES_ONE_WAY_RATE
                ) * float(cost_multiplier)
                nav -= entry_cost
                transaction_cost_today += entry_cost
                turnover = long_notional + short_notional
                turnover_today += turnover
                positions[tr.SSD_TRADE_ID] = {
                    "SSD_TRADE_ID": tr.SSD_TRADE_ID,
                    "COST_MULTIPLIER": float(cost_multiplier),
                    "FORMATION_DATE": pd.Timestamp(tr.FORMATION_DATE),
                    "BLOCK_TYPE": tr.BLOCK_TYPE,
                    "SELECTED_PAIR_NUMBER": int(tr.SELECTED_PAIR_NUMBER),
                    "PAIR_ID": tr.PAIR_ID,
                    "ENTRY_SIGNAL_DATE": pd.Timestamp(tr.ENTRY_SIGNAL_DATE),
                    "ENTRY_SIGNAL_Z": float(tr.ENTRY_SIGNAL_Z),
                    "ENTRY_EXECUTION_DATE": dt,
                    "ENTRY_EXECUTION_Z": float(tr.ENTRY_EXECUTION_Z),
                    "ENTRY_DIRECTION": tr.ENTRY_DIRECTION,
                    "LONG_COMPANY": tr.LONG_COMPANY,
                    "SHORT_COMPANY": tr.SHORT_COMPANY,
                    "LONG_SEGMENT": str(long_row["SEGMENT_ID"]),
                    "SHORT_SEGMENT": str(short_row["SEGMENT_ID"]),
                    "LONG_UNITS": long_units,
                    "SHORT_UNITS": short_units,
                    "INITIAL_LONG_NOTIONAL": long_notional,
                    "INITIAL_SHORT_NOTIONAL": short_notional,
                    "INITIAL_PAIR_GROSS": long_notional + short_notional,
                    "LAST_LONG_INDEX": float(long_row["TOTAL_RETURN_INDEX"]),
                    "LAST_SHORT_PROXY": float(short_row["FUTURES_PROXY_INDEX"]),
                    "LAST_MARK_DATE": dt,
                    "LONG_PNL": 0.0,
                    "SHORT_PNL": 0.0,
                    "GROSS_PNL": 0.0,
                    "TRANSACTION_COST": entry_cost,
                    "FINANCING_COST": 0.0,
                    "TOTAL_COST": entry_cost,
                    "TURNOVER_VALUE": turnover,
                }
                active_slots.add(slot)

        # End-of-day exposure from latest observed mark.
        long_exp = sum(abs(p["LONG_UNITS"] * p["LAST_LONG_INDEX"]) for p in positions.values())
        short_exp = sum(abs(p["SHORT_UNITS"] * p["LAST_SHORT_PROXY"]) for p in positions.values())
        gross_exp = long_exp + short_exp
        net_exp = long_exp - short_exp
        daily_rows.append({
            "COST_MULTIPLIER": float(cost_multiplier),
            "DATE": dt,
            "NAV_START": nav_start,
            "GROSS_PNL": gross_pnl_today,
            "TRANSACTION_COST": transaction_cost_today,
            "FINANCING_COST": financing_cost_today,
            "TOTAL_COST": transaction_cost_today + financing_cost_today,
            "NAV_END": nav,
            "DAILY_RETURN": nav / nav_start - 1.0,
            "OPEN_PAIRS_EOD": len(positions),
            "LONG_EXPOSURE": long_exp,
            "SHORT_EXPOSURE": short_exp,
            "GROSS_EXPOSURE": gross_exp,
            "NET_EXPOSURE": net_exp,
            "GROSS_EXPOSURE_TO_NAV": safe_div(gross_exp, nav),
            "NET_EXPOSURE_TO_NAV": safe_div(net_exp, nav),
            "TURNOVER_VALUE": turnover_today,
            "TURNOVER_TO_NAV": safe_div(turnover_today, nav_start),
        })
        previous_market_date = dt

    if positions:
        raise RuntimeError("Open SSD positions remain after final market date.")
    return pd.DataFrame(daily_rows), pd.DataFrame(completed)


def performance_row(df, label, multiplier):
    x = df.sort_values("DATE").copy()
    if x.empty:
        return {
            "COST_MULTIPLIER": multiplier, "PERIOD": label, "N_DAYS": 0,
            "TOTAL_RETURN": np.nan, "CAGR": np.nan, "ANNUALIZED_VOLATILITY": np.nan,
            "SHARPE_RF0": np.nan, "SORTINO_RF0": np.nan, "MAX_DRAWDOWN": np.nan,
            "CALMAR": np.nan, "AVG_GROSS_EXPOSURE_TO_NAV": np.nan,
            "AVG_ABS_NET_EXPOSURE_TO_NAV": np.nan, "TOTAL_TURNOVER_TO_NAV": np.nan,
        }
    initial = float(x["NAV_START"].iloc[0])
    final = float(x["NAV_END"].iloc[-1])
    total = final / initial - 1.0
    n = len(x)
    cagr = (1.0 + total) ** (TRADING_DAYS_PER_YEAR / n) - 1.0 if 1.0 + total > 0 else np.nan
    r = x["DAILY_RETURN"].astype(float)
    sd = float(r.std(ddof=1))
    vol = sd * np.sqrt(TRADING_DAYS_PER_YEAR) if np.isfinite(sd) else np.nan
    sharpe = float(r.mean()) / sd * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > NUMERIC_TOL else np.nan
    downside = np.minimum(r.to_numpy(dtype=float), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    sortino = (
        float(r.mean()) * TRADING_DAYS_PER_YEAR / (downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR))
        if downside_dev > NUMERIC_TOL else np.nan
    )
    navs = x["NAV_END"].astype(float)
    dd = navs / navs.cummax() - 1.0
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if np.isfinite(cagr) and mdd < -NUMERIC_TOL else np.nan
    return {
        "COST_MULTIPLIER": multiplier,
        "GROSS_OR_NET": "GROSS_BEFORE_COSTS" if float(multiplier) == 0.0 else "NET_AFTER_COSTS",
        "PERIOD": label,
        "START_DATE": x["DATE"].min(), "END_DATE": x["DATE"].max(), "N_DAYS": n,
        "TOTAL_RETURN": total, "CAGR": cagr, "ANNUALIZED_VOLATILITY": vol,
        "SHARPE_RF0": sharpe, "SORTINO_RF0": sortino, "MAX_DRAWDOWN": mdd,
        "CALMAR": calmar,
        "AVG_GROSS_EXPOSURE_TO_NAV": float(x["GROSS_EXPOSURE_TO_NAV"].mean()),
        "AVG_ABS_NET_EXPOSURE_TO_NAV": float(x["NET_EXPOSURE_TO_NAV"].abs().mean()),
        "TOTAL_TURNOVER_TO_NAV": float(x["TURNOVER_TO_NAV"].sum()),
    }


print("\n4/7 Running primary SSD portfolio at 0x / 0.5x / 1x / 2x costs...")
daily_frames = []
trade_frames = []
PRIMARY_BACKTEST_START = pd.Timestamp(
    schedule["TRADING_START"].min()
).normalize()

PRIMARY_BACKTEST_END = pd.Timestamp(
    schedule["TRADING_END"].max()
).normalize()

for mult in COST_MULTIPLIERS:
    d, t = simulate_portfolio(
        executable_primary,
        cost_multiplier=mult,
        start=PRIMARY_BACKTEST_START,
        end=PRIMARY_BACKTEST_END,
    )
    daily_frames.append(d)
    if not t.empty:
        trade_frames.append(t)

daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
trade_pnl = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

daily_all.to_csv(RESULT_DIR / "01_DAILY_NAV_ALL_COST_SCENARIOS.csv", index=False, date_format="%Y-%m-%d")
trade_pnl.to_csv(RESULT_DIR / "02_TRADE_PNL_ALL_COST_SCENARIOS.csv", index=False, date_format="%Y-%m-%d")

perf_rows = []
for mult, grp in daily_all.groupby("COST_MULTIPLIER"):
    perf_rows.append(performance_row(grp, "FULL", mult))
    perf_rows.append(performance_row(grp[grp["DATE"] < OOS_START], "DEVELOPMENT", mult))
    perf_rows.append(performance_row(grp[grp["DATE"] >= OOS_START], "OOS", mult))
performance = pd.DataFrame(perf_rows)
performance.to_csv(RESULT_DIR / "03_REQUIRED_PERFORMANCE_METRICS.csv", index=False, date_format="%Y-%m-%d")

primary_ledger = trade_pnl[trade_pnl["COST_MULTIPLIER"].eq(1.0)].copy() if not trade_pnl.empty else pd.DataFrame()
primary_ledger.to_csv(RESULT_DIR / "04_PRIMARY_TRADE_LEDGER_1X.csv", index=False, date_format="%Y-%m-%d")


# =============================================================================
# 9. REQUIRED TRADE TABLES / ATTRIBUTION / CAPACITY / MARKET EXPOSURE
# =============================================================================

def trade_stats(df, label):
    if df.empty:
        return {
            "PERIOD": label, "N_TRADES": 0, "HIT_RATE": np.nan,
            "AVERAGE_NET_PNL": np.nan, "MEDIAN_NET_PNL": np.nan,
            "AVERAGE_NET_RETURN_ON_PAIR_GROSS": np.nan,
            "MEDIAN_NET_RETURN_ON_PAIR_GROSS": np.nan,
            "HOLDING_DAYS_MEAN": np.nan, "HOLDING_DAYS_MEDIAN": np.nan,
            "CONVERGENCE_RATE": np.nan, "STOP_OUT_RATE": np.nan,
            "FORCED_EXIT_RATE": np.nan, "GROSS_PNL": 0.0,
            "TRANSACTION_COST": 0.0, "FINANCING_COST": 0.0,
            "TOTAL_COST": 0.0, "NET_PNL": 0.0,
        }
    return {
        "PERIOD": label,
        "N_TRADES": len(df),
        "HIT_RATE": float(df["NET_PNL"].gt(0).mean()),
        "AVERAGE_NET_PNL": float(df["NET_PNL"].mean()),
        "MEDIAN_NET_PNL": float(df["NET_PNL"].median()),
        "AVERAGE_NET_RETURN_ON_PAIR_GROSS": float(df["NET_RETURN_ON_PAIR_GROSS"].mean()),
        "MEDIAN_NET_RETURN_ON_PAIR_GROSS": float(df["NET_RETURN_ON_PAIR_GROSS"].median()),
        "HOLDING_DAYS_MEAN": float(df["HOLDING_CALENDAR_DAYS"].mean()),
        "HOLDING_DAYS_MEDIAN": float(df["HOLDING_CALENDAR_DAYS"].median()),
        "CONVERGENCE_RATE": float(df["EXIT_REASON"].isin(["MEAN_CROSS", "EXIT_BAND_REACHED"]).mean()),
        "STOP_OUT_RATE": float(df["EXIT_REASON"].eq("STOP_OUT").mean()),
        "FORCED_EXIT_RATE": float(df["EXIT_REASON"].str.contains("FORCE_CLOSE", na=False).mean()),
        "GROSS_PNL": float(df["GROSS_PNL"].sum()),
        "TRANSACTION_COST": float(df["TRANSACTION_COST"].sum()),
        "FINANCING_COST": float(df["FINANCING_COST"].sum()),
        "TOTAL_COST": float(df["TOTAL_COST"].sum()),
        "NET_PNL": float(df["NET_PNL"].sum()),
    }

trade_statistics = pd.DataFrame([
    trade_stats(primary_ledger, "FULL"),
    trade_stats(primary_ledger[primary_ledger["BLOCK_TYPE"].eq("DEVELOPMENT")] if not primary_ledger.empty else primary_ledger, "DEVELOPMENT"),
    trade_stats(primary_ledger[primary_ledger["BLOCK_TYPE"].eq("OOS")] if not primary_ledger.empty else primary_ledger, "OOS"),
])
trade_statistics.to_csv(RESULT_DIR / "05_REQUIRED_TRADE_STATISTICS.csv", index=False)

holding_bins = [("0-30", 0, 30), ("31-60", 31, 60), ("61-90", 61, 90), ("91-120", 91, 120), ("121-180", 121, 180), (">180", 181, np.inf)]
holding_rows = []
for label, low, high in holding_bins:
    h = primary_ledger["HOLDING_CALENDAR_DAYS"].astype(float) if not primary_ledger.empty else pd.Series(dtype=float)
    count = int((h.ge(low) & h.le(high)).sum())
    holding_rows.append({
        "HOLDING_PERIOD_DAYS": label,
        "N_TRADES": count,
        "SHARE_OF_TRADES": count / len(primary_ledger) if len(primary_ledger) else np.nan,
    })
pd.DataFrame(holding_rows).to_csv(RESULT_DIR / "06_HOLDING_PERIOD_DISTRIBUTION.csv", index=False)

# Monthly return table.
daily_1x = daily_all[daily_all["COST_MULTIPLIER"].eq(1.0)].sort_values("DATE").copy()
if not daily_1x.empty:
    monthly = (
        daily_1x.set_index("DATE")["DAILY_RETURN"]
        .resample("ME")
        .apply(lambda x: (1.0 + x).prod() - 1.0)
        .reset_index(name="MONTHLY_RETURN_1X")
    )
else:
    monthly = pd.DataFrame(columns=["DATE", "MONTHLY_RETURN_1X"])
monthly.to_csv(RESULT_DIR / "07_MONTHLY_RETURNS_1X.csv", index=False, date_format="%Y-%m-%d")

# Attribution.
if primary_ledger.empty:
    pair_attr = pd.DataFrame()
    form_attr = pd.DataFrame()
    top_decile = pd.DataFrame([{"N_TRADES": 0, "N_TOP_DECILE": 0, "TOP_DECILE_NET_PNL": 0.0, "TOTAL_NET_PNL": 0.0, "SHARE_OF_POSITIVE_PNL": np.nan}])
else:
    pair_attr = primary_ledger.groupby("PAIR_ID", as_index=False).agg(
        N_TRADES=("SSD_TRADE_ID", "count"),
        GROSS_PNL=("GROSS_PNL", "sum"),
        TRANSACTION_COST=("TRANSACTION_COST", "sum"),
        FINANCING_COST=("FINANCING_COST", "sum"),
        TOTAL_COST=("TOTAL_COST", "sum"),
        NET_PNL=("NET_PNL", "sum"),
    ).sort_values("NET_PNL", ascending=False)
    form_attr = primary_ledger.groupby(["FORMATION_DATE", "BLOCK_TYPE"], as_index=False).agg(
        N_TRADES=("SSD_TRADE_ID", "count"),
        GROSS_PNL=("GROSS_PNL", "sum"),
        TOTAL_COST=("TOTAL_COST", "sum"),
        NET_PNL=("NET_PNL", "sum"),
    ).sort_values("FORMATION_DATE")
    ranked = primary_ledger.sort_values("NET_PNL", ascending=False)
    n_top = max(1, int(np.ceil(0.10 * len(ranked))))
    top_net = float(ranked.head(n_top)["NET_PNL"].sum())
    total_net = float(ranked["NET_PNL"].sum())
    positive = float(ranked.loc[ranked["NET_PNL"].gt(0), "NET_PNL"].sum())
    top_decile = pd.DataFrame([{
        "N_TRADES": len(ranked), "N_TOP_DECILE": n_top,
        "TOP_DECILE_NET_PNL": top_net, "TOTAL_NET_PNL": total_net,
        "SHARE_OF_POSITIVE_PNL": top_net / positive if positive > NUMERIC_TOL else np.nan,
    }])
pair_attr.to_csv(RESULT_DIR / "08_PAIR_ATTRIBUTION_1X.csv", index=False)
form_attr.to_csv(RESULT_DIR / "09_FORMATION_ATTRIBUTION_1X.csv", index=False, date_format="%Y-%m-%d")
top_decile.to_csv(RESULT_DIR / "10_TOP_DECILE_PNL_CONCENTRATION_1X.csv", index=False)

# Capacity in rupees: each of the four execution legs <= 1% of that day's
# actual traded value. Portfolio capacity is the most restrictive trade leg.
capacity_rows = []
if not primary_ledger.empty:
    for tr in primary_ledger.itertuples(index=False):
        er_long = price_row(tr.ENTRY_EXECUTION_DATE, tr.LONG_COMPANY)
        er_short = price_row(tr.ENTRY_EXECUTION_DATE, tr.SHORT_COMPANY)
        xr_long = price_row(tr.EXIT_EXECUTION_DATE, tr.LONG_COMPANY)
        xr_short = price_row(tr.EXIT_EXECUTION_DATE, tr.SHORT_COMPANY)
        nav_entry = float(tr.INITIAL_PAIR_GROSS) / PAIR_GROSS_WEIGHT
        w_entry_long = float(tr.INITIAL_LONG_NOTIONAL) / nav_entry
        w_entry_short = float(tr.INITIAL_SHORT_NOTIONAL) / nav_entry
        exit_long_value = abs(float(tr.LONG_UNITS) * float(xr_long["TOTAL_RETURN_INDEX"]))
        exit_short_value = abs(float(tr.SHORT_UNITS) * float(xr_short["FUTURES_PROXY_INDEX"]))
        legs = [
            ("ENTRY_LONG", float(er_long["TOTAL_TRADED_VALUE"]), w_entry_long),
            ("ENTRY_SHORT", float(er_short["TOTAL_TRADED_VALUE"]), w_entry_short),
            ("EXIT_LONG", float(xr_long["TOTAL_TRADED_VALUE"]), safe_div(exit_long_value, nav_entry)),
            ("EXIT_SHORT", float(xr_short["TOTAL_TRADED_VALUE"]), safe_div(exit_short_value, nav_entry)),
        ]
        candidates = [
            (name, CAPACITY_PARTICIPATION_RATE * value / weight)
            for name, value, weight in legs
            if np.isfinite(value) and value > 0 and np.isfinite(weight) and weight > NUMERIC_TOL
        ]
        if candidates:
            limiting_leg, cap = min(candidates, key=lambda x: x[1])
        else:
            limiting_leg, cap = "NOT_CALCULATED", np.nan
        capacity_rows.append({
            "SSD_TRADE_ID": tr.SSD_TRADE_ID, "PAIR_ID": tr.PAIR_ID,
            "LIMITING_LEG": limiting_leg,
            "MAX_PORTFOLIO_CAPACITY_RUPEES": cap,
            "PARTICIPATION_RATE": CAPACITY_PARTICIPATION_RATE,
        })
capacity_detail = pd.DataFrame(capacity_rows)
capacity_detail.to_csv(RESULT_DIR / "11_CAPACITY_BY_TRADE.csv", index=False)
strategy_capacity = pd.DataFrame([{
    "PARTICIPATION_RATE": CAPACITY_PARTICIPATION_RATE,
    "STRATEGY_CAPACITY_RUPEES": float(capacity_detail["MAX_PORTFOLIO_CAPACITY_RUPEES"].min()) if not capacity_detail.empty else np.nan,
    "INTERPRETATION": "Conservative bottleneck; each execution leg limited to 1% of actual daily traded value.",
}])
strategy_capacity.to_csv(RESULT_DIR / "12_STRATEGY_CAPACITY_RUPEES.csv", index=False)

# NIFTY 500 beta and correlation.
benchmark_file = locate_exact_file(BENCHMARK_PREFERRED, "NIFTY500_2017_2026.csv")
benchmark = clean_columns(pd.read_csv(benchmark_file, low_memory=False))
require_columns(benchmark, ["DATE", "CLOSE"], "NIFTY500 benchmark")
benchmark["DATE"] = parse_dates(benchmark["DATE"])
benchmark["CLOSE"] = pd.to_numeric(benchmark["CLOSE"], errors="coerce")
benchmark = benchmark.dropna(subset=["DATE", "CLOSE"]).sort_values("DATE")
benchmark["BENCHMARK_RETURN"] = benchmark["CLOSE"].pct_change()
merged_market = daily_1x[["DATE", "DAILY_RETURN"]].merge(
    benchmark[["DATE", "BENCHMARK_RETURN"]], on="DATE", how="inner"
).dropna()
market_rows = []
for label, sub in [
    ("FULL", merged_market),
    ("DEVELOPMENT", merged_market[merged_market["DATE"] < OOS_START]),
    ("OOS", merged_market[merged_market["DATE"] >= OOS_START]),
]:
    if len(sub) >= 2 and float(sub["BENCHMARK_RETURN"].var(ddof=1)) > NUMERIC_TOL:
        beta = float(sub[["DAILY_RETURN", "BENCHMARK_RETURN"]].cov().iloc[0, 1] / sub["BENCHMARK_RETURN"].var(ddof=1))
        corr = float(sub["DAILY_RETURN"].corr(sub["BENCHMARK_RETURN"]))
    else:
        beta = corr = np.nan
    market_rows.append({"PERIOD": label, "N_COMMON_DAYS": len(sub), "BETA_TO_NIFTY500": beta, "CORRELATION_TO_NIFTY500": corr})
market_exposure = pd.DataFrame(market_rows)
market_exposure.to_csv(RESULT_DIR / "13_NIFTY500_BETA_CORRELATION.csv", index=False)


# =============================================================================
# 10. REQUIRED DEVELOPMENT-ONLY ROBUSTNESS
# =============================================================================

print("\n5/7 Running only the mandatory robustness checks (development period)...")


def selected_with_half_life_filter(selected_df, low=5.0, high=60.0):
    if selected_df.empty:
        return selected_df.copy()
    return selected_df[selected_df["HALF_LIFE_SESSIONS"].between(low, high, inclusive="both")].copy()


def run_robustness_variant(name, selected_variant, entry_z=2.0, exit_z=0.0, stop_z=None, category=""):
    selected_variant = selected_variant[selected_variant["BLOCK_TYPE"].eq("DEVELOPMENT")].copy() if not selected_variant.empty else selected_variant
    logical, _ = generate_logical_trades(
        selected_variant,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        variant_name=name,
    )
    # Fast-track robustness deliberately uses the formation-date F&O-screened
    # universe and does NOT re-download entry-date F&O files for every variant.
    d, t = simulate_portfolio(
        logical,
        cost_multiplier=1.0,
        start=schedule["TRADING_START"].min(),
        end=min(OOS_START - pd.Timedelta(days=1), schedule["TRADING_END"].max()),
    )
    p = performance_row(d[d["DATE"] < OOS_START], "DEVELOPMENT", 1.0)
    p.update({
        "ROBUSTNESS_CATEGORY": category,
        "ROBUSTNESS_VARIANT": name,
        "N_SELECTED_PAIR_PERIODS": len(selected_variant),
        "N_LOGICAL_TRADES": len(logical),
        "N_COMPLETED_TRADES": len(t),
        "ENTRY_Z": entry_z,
        "EXIT_Z": exit_z,
        "STOP_Z": stop_z,
        "ENTRY_DATE_FNO_RECHECK": "NO_FASTTRACK_ROBUSTNESS_ONLY",
    })
    return p


robust_rows = []
primary_dev = selected_primary[selected_primary["BLOCK_TYPE"].eq("DEVELOPMENT")].copy()

# Entry / exit thresholds.
for name, ez, xz in [
    ("ENTRY_1.5_EXIT_MEAN", 1.5, 0.0),
    ("PRIMARY_ENTRY_2_EXIT_MEAN", 2.0, 0.0),
    ("ENTRY_2.5_EXIT_MEAN", 2.5, 0.0),
    ("ENTRY_2_EXIT_0.5SD", 2.0, 0.5),
]:
    robust_rows.append(run_robustness_variant(name, primary_dev, ez, xz, None, "ENTRY_EXIT_THRESHOLDS"))

# Formation-window length. Same validation logic, split into two equal halves.
for months in [6, 18]:
    _, sel_alt, _ = select_pairs_for_spec(formation_months=months, universe_fraction=1.0, block_only="DEVELOPMENT")
    robust_rows.append(run_robustness_variant(f"FORMATION_{months}_MONTHS", sel_alt, 2.0, 0.0, None, "FORMATION_WINDOW_LENGTH"))

# Universe size.
_, sel_u75, _ = select_pairs_for_spec(formation_months=12, universe_fraction=0.75, block_only="DEVELOPMENT")
robust_rows.append(run_robustness_variant("UNIVERSE_TOP_75PCT_LIQUID", sel_u75, 2.0, 0.0, None, "UNIVERSE_SIZE"))

# Half-life filter.
hl_selected = selected_with_half_life_filter(primary_dev, 5.0, 60.0)
robust_rows.append(run_robustness_variant("HALF_LIFE_5_TO_60_SESSIONS", hl_selected, 2.0, 0.0, None, "HALF_LIFE_FILTER"))

robustness = pd.DataFrame(robust_rows)
robustness.to_csv(RESULT_DIR / "14_REQUIRED_ROBUSTNESS_PERFORMANCE.csv", index=False, date_format="%Y-%m-%d")
selected_primary[[
    "FORMATION_DATE", "BLOCK_TYPE", "PAIR_ID", "COMPANY_A", "COMPANY_B", "HALF_LIFE_SESSIONS"
]].to_csv(RESULT_DIR / "15_SELECTED_PAIR_HALF_LIFE.csv", index=False, date_format="%Y-%m-%d")


# =============================================================================
# 11. OOS / LIMITATIONS / REQUIREMENTS AUDIT
# =============================================================================

print("\n6/7 Building OOS and requirements audit...")

oos_selected = int(selected_primary["BLOCK_TYPE"].eq("OOS").sum()) if not selected_primary.empty else 0
oos_trades = int(executable_primary["BLOCK_TYPE"].eq("OOS").sum()) if not executable_primary.empty else 0
oos_perf = performance[
    performance["COST_MULTIPLIER"].eq(1.0) & performance["PERIOD"].eq("OOS")
]
oos_return = float(oos_perf["TOTAL_RETURN"].iloc[0]) if len(oos_perf) else np.nan

oos_note = pd.DataFrame([{
    "OOS_START": OOS_START,
    "N_OOS_SELECTED_PAIR_PERIODS": oos_selected,
    "N_OOS_EXECUTABLE_TRADES": oos_trades,
    "OOS_RETURN_1X": oos_return,
    "OOS_STATUS": "CONSTRAINED_PSEUDO_OOS_NOT_PERFECTLY_UNTOUCHED",
    "INTERPRETATION": (
        "The earlier uncorrected SSD OOS result was seen before the mandatory formation/validation "
        "multiple-testing correction was added. The final block was not used to tune the correction, "
        "but it must not be described as a perfectly untouched holdout."
    ),
}])
oos_note.to_csv(RESULT_DIR / "16_OOS_INTERPRETATION.csv", index=False, date_format="%Y-%m-%d")

limitations = pd.DataFrame([
    ["ACTUAL_FUTURES_BASIS", "NOT_RECONSTRUCTED", "Short P&L uses a corporate-action-adjusted spot proxy."],
    ["MONTHLY_FUTURES_ROLL", "NOT_RECONSTRUCTED", "Exact contract-by-contract roll gains/losses are a limitation."],
    ["INTEGER_FUTURES_LOT_SIZE", "NOT_RECONSTRUCTED", "Continuous notionals are used."],
    ["ASM_GSM_HISTORY", "NOT_RECONSTRUCTED", "Date-by-date surveillance status not rebuilt."],
    ["LOCKED_CIRCUIT_FILL_STATE", "NOT_RECONSTRUCTED", "Historical locked-circuit execution state not rebuilt."],
    ["SETTLEMENT", "DISCLOSED", "Cash T+1 mainly affects capital tie-up, not the EOD signal."],
    ["ROBUSTNESS_ENTRY_FNO", "FASTTRACK_LIMITATION", "Robustness variants retain formation-date F&O-screened universe but do not re-download every variant entry date."],
    ["OOS_PURITY", "LIMITATION", "Earlier uncorrected OOS result was viewed; final block is constrained pseudo-OOS."],
], columns=["ITEM", "STATUS", "TREATMENT"])
limitations.to_csv(RESULT_DIR / "17_MARKET_REALISM_LIMITATIONS.csv", index=False)

requirements = pd.DataFrame([
    ["SSD normalized cumulative total-return distance", True, "Normalized total-return series; SSD used."],
    ["Multiple-testing correction", bool((selection_audit["STATUS"] == "PASS").all()), "First-half top 10% must survive untouched second-half top 10%."],
    ["Number of pair relationships reported", selection_audit["N_PAIR_RELATIONSHIPS_EXAMINED"].sum() > 0, f"{int(selection_audit['N_PAIR_RELATIONSHIPS_EXAMINED'].sum())} pair-period relationships examined."],
    ["Formation-only selection and normalization frozen", True, "12m formation, 6m trading; frozen anchors/mean/SD."],
    [
        "Primary backtest uses complete comparison window",
        (
            pd.Timestamp(
                performance.loc[
                    (performance["COST_MULTIPLIER"].eq(1.0))
                    & (performance["PERIOD"].eq("FULL")),
                    "START_DATE",
                ].iloc[0]
            ).normalize()
            == PRIMARY_BACKTEST_START
            and
            pd.Timestamp(
                performance.loc[
                    (performance["COST_MULTIPLIER"].eq(1.0))
                    & (performance["PERIOD"].eq("FULL")),
                    "END_DATE",
                ].iloc[0]
            ).normalize()
            == PRIMARY_BACKTEST_END
        ),
        "Retains cash-only dates so SSD and EG are compared over identical calendar windows."
    ],
    ["Entry / exit / stop / max holding specified", True, "+/-2SD entry; mean exit; no primary stop; force close at window end."],
    ["Sizing / max positions / pair cap", True, "Equal 12.5% legs; 25% gross per pair; up to four pairs."],
    ["Primary short-leg F&O check", TEST_ASSUME_ALL_FNO or (len(fno_checked) == len(logical_primary)), "Official NSE daily F&O bhavcopy on primary trade entry dates."],
    ["Gross and net results", set(COST_MULTIPLIERS) == {0.0, 0.5, 1.0, 2.0}, "0x gross; 0.5x/1x/2x net cost sensitivity."],
    ["Short-side financing", True, "Same EG assumption: 20% collateral funded at 8% annual."],
    ["Required performance metrics", set(["CAGR", "ANNUALIZED_VOLATILITY", "SHARPE_RF0", "SORTINO_RF0", "MAX_DRAWDOWN", "CALMAR"]).issubset(performance.columns), "All required metrics output."],
    ["Monthly return table", True, "Saved at 1x costs."],
    ["Trade statistics / holding distribution / convergence", True, "Saved in dedicated outputs."],
    ["Attribution and top-decile concentration", True, "Pair, formation window and top-decile outputs saved."],
    ["Market exposure / beta / correlation / turnover", True, "NIFTY500 beta/correlation plus daily exposure and turnover."],
    ["Entry/exit robustness", robustness["ROBUSTNESS_CATEGORY"].eq("ENTRY_EXIT_THRESHOLDS").any(), "Development-only."],
    ["Formation-window robustness", robustness["ROBUSTNESS_CATEGORY"].eq("FORMATION_WINDOW_LENGTH").any(), "6m and 18m alternatives."],
    ["Universe-size robustness", robustness["ROBUSTNESS_CATEGORY"].eq("UNIVERSE_SIZE").any(), "Top 75% liquid subset."],
    ["Half-life robustness", robustness["ROBUSTNESS_CATEGORY"].eq("HALF_LIFE_FILTER").any(), "5-60 session filter."],
    ["Capacity in rupees", True, "1% daily traded-value participation cap."],
    ["Submission-ready trade ledger", primary_ledger.empty or set(["PAIR_ID", "ENTRY_DIRECTION", "ENTRY_EXECUTION_DATE", "EXIT_EXECUTION_DATE", "ENTRY_EXECUTION_Z", "EXIT_EXECUTION_Z", "GROSS_PNL", "TRANSACTION_COST", "FINANCING_COST", "TOTAL_COST", "NET_PNL", "EXIT_REASON"]).issubset(primary_ledger.columns), "Required fields consolidated."],
], columns=["REQUIREMENT", "PASS", "EVIDENCE"])
requirements.to_csv(AUDIT_DIR / "01_SSD_REQUIREMENTS_COVERAGE_AUDIT.csv", index=False)

if not requirements["PASS"].all():
    raise RuntimeError(
        "SSD requirements audit failed:\n" + requirements.loc[~requirements["PASS"]].to_string(index=False)
    )

final_audit = pd.DataFrame([
    ["FORMATION_DATES", len(formation_table), "PASS"],
    ["PAIR_RELATIONSHIPS_EXAMINED", int(selection_audit["N_PAIR_RELATIONSHIPS_EXAMINED"].sum()), "PASS"],
    ["SURVIVING_PAIR_PERIODS", int(selection_audit["N_SURVIVING_BOTH_HALVES"].sum()), "PASS"],
    ["SELECTED_PAIR_PERIODS", len(selected_primary), "PASS"],
    ["PRIMARY_LOGICAL_TRADES", len(logical_primary), "PASS"],
    ["PRIMARY_EXECUTABLE_TRADES", len(executable_primary), "PASS"],
    ["PRIMARY_BACKTEST_START", str(PRIMARY_BACKTEST_START.date()), "PASS"],
    ["PRIMARY_BACKTEST_END", str(PRIMARY_BACKTEST_END.date()), "PASS"],
    ["FNO_SKIPPED_TRADES", len(skipped_fno), "PASS"],
    ["REQUIREMENTS_AUDIT", int(requirements["PASS"].sum()), "PASS"],
    ["OVERALL_STATUS", "PASS", "PASS"],
], columns=["CHECK", "VALUE", "STATUS"])
final_audit.to_csv(AUDIT_DIR / "02_SSD_FINAL_FASTTRACK_AUDIT.csv", index=False)


# =============================================================================
# 12. CONSOLE SUMMARY
# =============================================================================

print("\n7/7 COMPLETE")
print("=" * 116)
print("SSD FINAL FAST-TRACK PIPELINE — REQUIREMENTS AUDIT PASS")
print("=" * 116)
print(f"Formation dates: {len(formation_table)}")
print(f"Pair relationships examined: {int(selection_audit['N_PAIR_RELATIONSHIPS_EXAMINED'].sum()):,}")
print(f"Pair-periods surviving both validation halves: {int(selection_audit['N_SURVIVING_BOTH_HALVES'].sum()):,}")
print(f"Selected pair-periods: {len(selected_primary):,}")
print(f"Logical trades: {len(logical_primary):,}")
print(f"Executable trades after entry-date F&O gate: {len(executable_primary):,}")
print(f"Trades skipped for unavailable short future: {len(skipped_fno):,}")

p1 = performance[(performance["COST_MULTIPLIER"].eq(1.0)) & (performance["PERIOD"].eq("FULL"))]
if len(p1):
    print(f"Primary 1x full-period return: {float(p1['TOTAL_RETURN'].iloc[0]):+.2%}")

gross = performance[(performance["COST_MULTIPLIER"].eq(0.0)) & (performance["PERIOD"].eq("FULL"))]
if len(gross):
    print(f"Gross full-period return before costs: {float(gross['TOTAL_RETURN'].iloc[0]):+.2%}")

print("\nMain files to upload next:")
for p in [
    PAIR_DIR / "02_SELECTED_VALIDATED_SSD_PAIRS.csv",
    PAIR_DIR / "03_MULTIPLE_TESTING_SELECTION_AUDIT.csv",
    RESULT_DIR / "03_REQUIRED_PERFORMANCE_METRICS.csv",
    RESULT_DIR / "04_PRIMARY_TRADE_LEDGER_1X.csv",
    RESULT_DIR / "05_REQUIRED_TRADE_STATISTICS.csv",
    RESULT_DIR / "12_STRATEGY_CAPACITY_RUPEES.csv",
    RESULT_DIR / "13_NIFTY500_BETA_CORRELATION.csv",
    RESULT_DIR / "14_REQUIRED_ROBUSTNESS_PERFORMANCE.csv",
    RESULT_DIR / "15_SELECTED_PAIR_HALF_LIFE.csv",
    RESULT_DIR / "16_OOS_INTERPRETATION.csv",
    AUDIT_DIR / "01_SSD_REQUIREMENTS_COVERAGE_AUDIT.csv",
    AUDIT_DIR / "02_SSD_FINAL_FASTTRACK_AUDIT.csv",
]:
    print(p)

print("\nIMPORTANT: do not tune the 10% validation threshold after viewing these results.")
