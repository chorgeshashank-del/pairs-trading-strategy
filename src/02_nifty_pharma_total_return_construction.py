import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# NIFTY PHARMA: TOTAL-RETURN CONSTRUCTION
# ============================================================
#
# PURPOSE
# -------
# Build an auditable single-stock total-return series from:
#
#   1) the cleaned NSE EQ-only price/volume base; and
#   2) the finalized corporate-action treatment ledger.
#
# IMPORTANT DESIGN RULES
# ----------------------
# * Raw NSE fields are NEVER overwritten.
# * Cash dividends are included in shareholder return on ex-date.
# * Splits / consolidations / bonus issues neutralize the mechanical
#   change in share units.
# * Rights issues use the theoretical ex-rights price (TERP), using
#   the verified rights ratio and issue price.
# * Buybacks and administrative events do not create automatic
#   price/return adjustments.
# * Demergers are treated as STRUCTURAL BREAKS, not as ordinary
#   price adjustments. A statistical segment restarts at the first
#   available EQ observation on/after the event date.
# * No forward-filling of prices is performed.
#
# OUTPUT
# ------
# One row per original EQ observation, with all original columns
# preserved and new research columns appended.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

EQ_BASE_FILE = Path(
    r"C:\fin proj\nse_pharma_clean_base"
) / "NIFTY_PHARMA_EQ_BASE_2016_2026.csv"

LEDGER_FILE = Path(
    r"C:\fin proj\nse_pharma_corporate_action_ledger"
) / "CORPORATE_ACTION_TREATMENT_LEDGER_2016_2026.csv"

OUTPUT_DIR = Path(
    r"C:\fin proj\nse_pharma_total_return"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "NIFTY_PHARMA_TOTAL_RETURN_BASE_2016_2026.csv"
)


# ============================================================
# 2. FINALIZED MANUAL OVERRIDES
# ============================================================
#
# These 27 rows were flagged by the conservative parser because
# their PURPOSE text contains special + ordinary dividend amounts
# (or another parser ambiguity).
#
# For a raw single-stock close series, we include the FULL cash
# distribution once. We do not use an index divisor adjustment.
# ============================================================

MANUAL_DIVIDEND_TOTAL_OVERRIDES = {
    "CA0013": 47.0,    # SANOFI: 25 + 22
    "CA0068": 20.0,    # PFIZER: 15 + 5
    "CA0092": 55.0,    # ABBOTINDIA: 50 + 5
    "CA0146": 65.0,    # ABBOTINDIA: 50 + 15
    "CA0171": 10.0,    # APLLTD: 7 + 3
    "CA0172": 32.0,    # TORNTPHARM: 17 + 15
    "CA0173": 4.0,     # CIPLA: 3 + 1
    "CA0177": 320.0,   # PFIZER: special dividend 320
    "CA0179": 349.0,   # SANOFI: 106 + 243
    "CA0183": 40.0,    # GLAXO: 20 + 20
    "CA0197": 250.0,   # ABBOTINDIA: 107 + 143
    "CA0219": 365.0,   # SANOFI: 125 + 240
    "CA0225": 275.0,   # ABBOTINDIA: 120 + 155
    "CA0237": 35.0,    # PFIZER: 30 + 5
    "CA0263": 490.0,   # SANOFI: 181 + 309
    "CA0270": 90.0,    # GLAXO: 30 + 60
    "CA0278": 275.0,   # ABBOTINDIA: 145 + 130
    "CA0280": 193.0,   # SANOFI: special dividend 193
    "CA0292": 30.0,    # PFIZER: special dividend 30
    "CA0300": 40.0,    # ALKEM: 15 + 25
    "CA0305": 377.0,   # SANOFI: 194 + 183
    "CA0314": 325.0,   # ABBOTINDIA: 180 + 145
    "CA0322": 25.0,    # AJANTPHARM: 10 + 15
    "CA0327": 40.0,    # PFIZER: 35 + 5
    "CA0397": 16.0,    # CIPLA: 13 + 3
    "CA0400": 165.0,   # PFIZER: 35 + 100 + 30
    "CA0441": 656.0,   # ABBOTINDIA: 525 + 131
}


# Rights issue terms:
# A = new rights shares
# B = existing shares
# S = issue price per rights share
#
# The issue prices below are TOTAL issue prices, not merely premium.
# They were finalized from the ledger terms + face value / verified
# event terms:
#
# PEL 2018: 1:23 at Rs 2,380
# PEL 2019: 11:83 at Rs 1,300
# WOCKPHARMA 2022: 3:10 at Rs 225
# PPLPHARMA 2023: 5:46 at Rs 81
# ============================================================

RIGHTS_ISSUE_PRICE_OVERRIDES = {
    "CA0081": 2380.0,
    "CA0162": 1300.0,
    "CA0262": 225.0,
    "CA0319": 81.0,
}


# Final structural breaks.
#
# CA0087 was originally classified as SCHEME_OF_ARRANGEMENT but
# was finalized as a demerger-type structural break.
# ============================================================

STRUCTURAL_BREAK_OVERRIDES = {
    "CA0087": ("STAR",   "2018-04-06", "DEMERGER"),
    "CA0290": ("PEL",    "2022-08-30", "DEMERGER"),
    "CA0353": ("SANOFI", "2024-06-13", "DEMERGER"),
    "CA0386": ("STAR",   "2024-12-06", "DEMERGER"),
}


# Final harmless typo correction.
ADMIN_ONLY_OVERRIDES = {
    "CA0033",  # DIVISLAB "Annnual General Meeting"
}


# ============================================================
# 3. HELPERS
# ============================================================

def parse_dates(series):
    """Parse dates robustly across pandas versions."""
    try:
        return pd.to_datetime(
            series,
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
    except TypeError:
        return pd.to_datetime(
            series,
            dayfirst=True,
            errors="coerce",
        )


def require_columns(df, required, name):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing)}"
        )


def boolify(series):
    """
    Robust conversion of CSV boolean-like values to True/False.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    text = (
        series.astype("string")
        .str.strip()
        .str.upper()
    )

    mapping = {
        "TRUE": True,
        "FALSE": False,
        "1": True,
        "0": False,
        "YES": True,
        "NO": False,
    }

    return text.map(mapping).fillna(False).astype(bool)


# ============================================================
# 4. LOAD INPUTS
# ============================================================

print("=" * 100)
print("LOADING INPUTS")
print("=" * 100)

if not EQ_BASE_FILE.exists():
    raise FileNotFoundError(
        f"EQ base file not found:\n{EQ_BASE_FILE}"
    )

if not LEDGER_FILE.exists():
    raise FileNotFoundError(
        f"Corporate-action ledger not found:\n{LEDGER_FILE}"
    )


eq = pd.read_csv(
    EQ_BASE_FILE,
    low_memory=False,
)

ledger = pd.read_csv(
    LEDGER_FILE,
    low_memory=False,
)


# Standardize column names only.
eq.columns = (
    eq.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.upper()
)

ledger.columns = (
    ledger.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.upper()
)


require_columns(
    eq,
    [
        "DATE",
        "SYMBOL",
        "COMPANY_ID",
        "CLOSE",
    ],
    "EQ base",
)

require_columns(
    ledger,
    [
        "LEDGER_ID",
        "EX_DATE",
        "COMPANY_ID",
        "SYMBOL",
        "PRICE_SYMBOL",
        "PRIMARY_CLASS",
        "DIVIDEND_RUPEES_PER_SHARE",
        "BONUS_SHARE_MULTIPLIER",
        "SPLIT_SHARE_MULTIPLIER",
        "RIGHTS_NEW_SHARES",
        "RIGHTS_EXISTING_SHARES",
    ],
    "Corporate-action ledger",
)


# ============================================================
# 5. CLEAN / VALIDATE INPUT TYPES
# ============================================================

eq["DATE"] = parse_dates(eq["DATE"])
ledger["EX_DATE"] = parse_dates(ledger["EX_DATE"])

if eq["DATE"].isna().any():
    raise ValueError("EQ base contains invalid DATE values.")

if ledger["EX_DATE"].isna().any():
    raise ValueError("Ledger contains invalid EX_DATE values.")


for col in ["SYMBOL", "COMPANY_ID"]:
    eq[col] = (
        eq[col].astype("string")
        .str.strip()
        .str.upper()
    )

for col in [
    "LEDGER_ID",
    "SYMBOL",
    "PRICE_SYMBOL",
    "COMPANY_ID",
    "PRIMARY_CLASS",
]:
    ledger[col] = (
        ledger[col].astype("string")
        .str.strip()
        .str.upper()
    )


numeric_ledger_cols = [
    "DIVIDEND_RUPEES_PER_SHARE",
    "BONUS_SHARE_MULTIPLIER",
    "SPLIT_SHARE_MULTIPLIER",
    "RIGHTS_NEW_SHARES",
    "RIGHTS_EXISTING_SHARES",
]

for col in numeric_ledger_cols:
    ledger[col] = pd.to_numeric(
        ledger[col],
        errors="coerce",
    )

eq["CLOSE"] = pd.to_numeric(
    eq["CLOSE"],
    errors="coerce",
)

if eq["CLOSE"].isna().any():
    raise ValueError("EQ base contains missing/non-numeric CLOSE.")

if (eq["CLOSE"] <= 0).any():
    raise ValueError("EQ base contains non-positive CLOSE.")


# Preserve source row order for final output.
eq["_ORIGINAL_ROW_ORDER"] = np.arange(len(eq))


# No duplicate company/date observations should exist in EQ base.
dups = eq.duplicated(
    subset=["DATE", "COMPANY_ID"],
    keep=False,
)

if dups.any():
    sample = eq.loc[
        dups,
        ["DATE", "COMPANY_ID", "SYMBOL", "CLOSE"],
    ].head(20)

    raise ValueError(
        "Duplicate DATE + COMPANY_ID rows found in EQ base.\n"
        + sample.to_string(index=False)
    )


# Ledger should still be the 446-row audited file.
if len(ledger) != 446:
    raise ValueError(
        f"Expected 446 corporate-action ledger rows, "
        f"found {len(ledger)}."
    )

if ledger["LEDGER_ID"].duplicated().any():
    raise ValueError("Duplicate LEDGER_ID values found.")


# ============================================================
# 6. VERIFY FINALIZED OVERRIDES ARE PRESENT
# ============================================================

expected_override_ids = (
    set(MANUAL_DIVIDEND_TOTAL_OVERRIDES)
    | set(RIGHTS_ISSUE_PRICE_OVERRIDES)
    | set(STRUCTURAL_BREAK_OVERRIDES)
    | set(ADMIN_ONLY_OVERRIDES)
)

missing_override_ids = (
    expected_override_ids
    - set(ledger["LEDGER_ID"])
)

if missing_override_ids:
    raise ValueError(
        "Finalized override IDs missing from ledger: "
        f"{sorted(missing_override_ids)}"
    )


# Extra validation of the four structural break identities/dates.
for ledger_id, (symbol, date_text, _) in STRUCTURAL_BREAK_OVERRIDES.items():
    row = ledger.loc[
        ledger["LEDGER_ID"].eq(ledger_id)
    ].iloc[0]

    expected_date = pd.Timestamp(date_text)

    if row["SYMBOL"] != symbol:
        raise ValueError(
            f"{ledger_id}: expected symbol {symbol}, "
            f"found {row['SYMBOL']}."
        )

    if row["EX_DATE"] != expected_date:
        raise ValueError(
            f"{ledger_id}: expected ex-date {expected_date.date()}, "
            f"found {row['EX_DATE']}."
        )


# ============================================================
# 7. FINALIZE LEDGER TREATMENTS IN MEMORY
# ============================================================
#
# We do NOT overwrite the source ledger on disk.
# We create final treatment columns for this construction run.
# ============================================================

ledger["FINAL_PRIMARY_CLASS"] = ledger["PRIMARY_CLASS"].copy()

# DIVISLAB typo -> administrative only.
ledger.loc[
    ledger["LEDGER_ID"].isin(ADMIN_ONLY_OVERRIDES),
    "FINAL_PRIMARY_CLASS",
] = "ADMIN_ONLY"

# STAR 2018 scheme -> demerger structural break.
ledger.loc[
    ledger["LEDGER_ID"].isin(STRUCTURAL_BREAK_OVERRIDES),
    "FINAL_PRIMARY_CLASS",
] = "DEMERGER_SPINOFF"


# Final cash dividend per share.
ledger["FINAL_DIVIDEND_PER_SHARE"] = (
    ledger["DIVIDEND_RUPEES_PER_SHARE"]
)

for ledger_id, amount in MANUAL_DIVIDEND_TOTAL_OVERRIDES.items():
    ledger.loc[
        ledger["LEDGER_ID"].eq(ledger_id),
        "FINAL_DIVIDEND_PER_SHARE",
    ] = amount


# Final rights issue price.
ledger["FINAL_RIGHTS_ISSUE_PRICE"] = np.nan

for ledger_id, issue_price in RIGHTS_ISSUE_PRICE_OVERRIDES.items():
    ledger.loc[
        ledger["LEDGER_ID"].eq(ledger_id),
        "FINAL_RIGHTS_ISSUE_PRICE",
    ] = issue_price


# ============================================================
# 8. STRICT FINALIZED-TERMS VALIDATION
# ============================================================

div_rows = ledger[
    ledger["FINAL_PRIMARY_CLASS"].eq("CASH_DIVIDEND")
].copy()

if div_rows["FINAL_DIVIDEND_PER_SHARE"].isna().any():
    bad = div_rows.loc[
        div_rows["FINAL_DIVIDEND_PER_SHARE"].isna(),
        ["LEDGER_ID", "EX_DATE", "SYMBOL", "PRIMARY_CLASS"],
    ]

    raise ValueError(
        "Some cash dividends still lack a finalized amount.\n"
        + bad.to_string(index=False)
    )

if (div_rows["FINAL_DIVIDEND_PER_SHARE"] < 0).any():
    raise ValueError("Negative finalized dividend amount found.")


bonus_rows = ledger[
    ledger["FINAL_PRIMARY_CLASS"].eq("BONUS_ISSUE")
].copy()

if bonus_rows["BONUS_SHARE_MULTIPLIER"].isna().any():
    raise ValueError(
        "Some bonus issues lack BONUS_SHARE_MULTIPLIER."
    )

if (bonus_rows["BONUS_SHARE_MULTIPLIER"] <= 0).any():
    raise ValueError(
        "Invalid BONUS_SHARE_MULTIPLIER found."
    )


split_rows = ledger[
    ledger["FINAL_PRIMARY_CLASS"].eq(
        "SPLIT_CONSOLIDATION"
    )
].copy()

if split_rows["SPLIT_SHARE_MULTIPLIER"].isna().any():
    raise ValueError(
        "Some split/consolidation events lack "
        "SPLIT_SHARE_MULTIPLIER."
    )

if (split_rows["SPLIT_SHARE_MULTIPLIER"] <= 0).any():
    raise ValueError(
        "Invalid SPLIT_SHARE_MULTIPLIER found."
    )


rights_rows = ledger[
    ledger["FINAL_PRIMARY_CLASS"].eq("RIGHTS_ISSUE")
].copy()

if len(rights_rows) != 4:
    raise ValueError(
        f"Expected 4 rights issues, found {len(rights_rows)}."
    )

rights_required = [
    "RIGHTS_NEW_SHARES",
    "RIGHTS_EXISTING_SHARES",
    "FINAL_RIGHTS_ISSUE_PRICE",
]

if rights_rows[rights_required].isna().any().any():
    raise ValueError(
        "At least one rights issue lacks finalized terms."
    )

if (
    (rights_rows["RIGHTS_NEW_SHARES"] <= 0).any()
    or
    (rights_rows["RIGHTS_EXISTING_SHARES"] <= 0).any()
    or
    (rights_rows["FINAL_RIGHTS_ISSUE_PRICE"] < 0).any()
):
    raise ValueError("Invalid rights issue terms found.")


struct_rows = ledger[
    ledger["FINAL_PRIMARY_CLASS"].eq("DEMERGER_SPINOFF")
].copy()

if len(struct_rows) != 4:
    raise ValueError(
        f"Expected 4 finalized structural demergers, "
        f"found {len(struct_rows)}."
    )


# ============================================================
# 9. BUILD EVENT-LEVEL TABLE
# ============================================================
#
# We only carry events that can affect the research return series:
#
#   CASH_DIVIDEND
#   BONUS_ISSUE
#   SPLIT_CONSOLIDATION
#   RIGHTS_ISSUE
#   DEMERGER_SPINOFF
#
# BUYBACK and ADMIN_ONLY remain in the source ledger but do not
# alter returns.
# ============================================================

affecting_classes = {
    "CASH_DIVIDEND",
    "BONUS_ISSUE",
    "SPLIT_CONSOLIDATION",
    "RIGHTS_ISSUE",
    "DEMERGER_SPINOFF",
}

events = ledger[
    ledger["FINAL_PRIMARY_CLASS"].isin(affecting_classes)
].copy()


# ============================================================
# 10. MAP EACH EVENT TO AN EQ OBSERVATION DATE
# ============================================================
#
# Non-structural events MUST have an exact EQ observation on ex-date.
#
# Structural breaks may have no EQ row on the ex-date. In that case
# the break is applied to the first available EQ observation AFTER
# the event date, so no return is computed across the regime change.
# ============================================================

company_dates = {
    company_id: np.array(
        sorted(
            group["DATE"].dropna().unique()
        ),
        dtype="datetime64[ns]",
    )
    for company_id, group
    in eq.groupby("COMPANY_ID", sort=False)
}


apply_dates = []
apply_lags = []

for _, row in events.iterrows():

    company_id = row["COMPANY_ID"]
    ex_date = row["EX_DATE"]
    final_class = row["FINAL_PRIMARY_CLASS"]

    if company_id not in company_dates:
        raise ValueError(
            f"{row['LEDGER_ID']}: COMPANY_ID {company_id} "
            "not found in EQ base."
        )

    dates = company_dates[company_id]

    exact_match = (
        np.datetime64(ex_date)
        in dates
    )

    if final_class != "DEMERGER_SPINOFF":

        if not exact_match:
            raise ValueError(
                f"{row['LEDGER_ID']} {company_id} "
                f"{ex_date.date()}: non-structural corporate action "
                "has no exact EQ observation on ex-date. "
                "Do not silently move it to another date."
            )

        apply_date = ex_date

    else:

        eligible = dates[
            dates >= np.datetime64(ex_date)
        ]

        if len(eligible) == 0:
            raise ValueError(
                f"{row['LEDGER_ID']} {company_id}: "
                "no EQ observation exists on/after structural "
                f"break date {ex_date.date()}."
            )

        apply_date = pd.Timestamp(
            eligible[0]
        )

    apply_dates.append(apply_date)

    apply_lags.append(
        int(
            (
                pd.Timestamp(apply_date)
                - ex_date
            ).days
        )
    )


events["APPLY_DATE"] = apply_dates
events["APPLY_LAG_CALENDAR_DAYS"] = apply_lags


# ============================================================
# 11. COLLAPSE LEDGER ROWS TO COMPANY + APPLY_DATE
# ============================================================
#
# Multiple dividend rows on one day are allowed and summed.
# Multiple same-stock unit changes are multiplied.
#
# Complex combinations are NOT guessed.
# ============================================================

daily_event_rows = []

for (company_id, apply_date), group in events.groupby(
    ["COMPANY_ID", "APPLY_DATE"],
    sort=True,
):

    classes = set(
        group["FINAL_PRIMARY_CLASS"]
    )

    has_dividend = (
        "CASH_DIVIDEND" in classes
    )

    has_bonus = (
        "BONUS_ISSUE" in classes
    )

    has_split = (
        "SPLIT_CONSOLIDATION" in classes
    )

    has_rights = (
        "RIGHTS_ISSUE" in classes
    )

    has_structural = (
        "DEMERGER_SPINOFF" in classes
    )

    unit_classes = classes.intersection(
        {"BONUS_ISSUE", "SPLIT_CONSOLIDATION"}
    )


    # --------------------------------------------------------
    # Refuse ambiguous combinations rather than inventing a rule.
    # --------------------------------------------------------

    if has_structural and len(classes) > 1:
        raise ValueError(
            f"{company_id} {pd.Timestamp(apply_date).date()}: "
            f"structural break is combined with {sorted(classes)}. "
            "Manual treatment required."
        )

    if has_rights and len(classes) > 1:
        raise ValueError(
            f"{company_id} {pd.Timestamp(apply_date).date()}: "
            f"rights issue is combined with {sorted(classes)}. "
            "Manual treatment required."
        )

    if has_dividend and unit_classes:
        raise ValueError(
            f"{company_id} {pd.Timestamp(apply_date).date()}: "
            "dividend and split/bonus occur on the same apply date. "
            "Dividend share basis must be verified manually."
        )

    if has_bonus and has_split:
        raise ValueError(
            f"{company_id} {pd.Timestamp(apply_date).date()}: "
            "bonus and split/consolidation occur together. "
            "Manual sequencing required."
        )


    cash_dividend = (
        group.loc[
            group["FINAL_PRIMARY_CLASS"].eq("CASH_DIVIDEND"),
            "FINAL_DIVIDEND_PER_SHARE",
        ].sum()
        if has_dividend
        else 0.0
    )


    bonus_multiplier = (
        group.loc[
            group["FINAL_PRIMARY_CLASS"].eq("BONUS_ISSUE"),
            "BONUS_SHARE_MULTIPLIER",
        ].prod()
        if has_bonus
        else 1.0
    )


    split_multiplier = (
        group.loc[
            group["FINAL_PRIMARY_CLASS"].eq(
                "SPLIT_CONSOLIDATION"
            ),
            "SPLIT_SHARE_MULTIPLIER",
        ].prod()
        if has_split
        else 1.0
    )


    same_stock_multiplier = (
        bonus_multiplier
        *
        split_multiplier
    )


    if has_rights:

        if (
            group["FINAL_PRIMARY_CLASS"]
            .eq("RIGHTS_ISSUE")
            .sum()
            != 1
        ):
            raise ValueError(
                f"{company_id} {pd.Timestamp(apply_date).date()}: "
                "more than one rights issue on same date."
            )

        rights_row = group[
            group["FINAL_PRIMARY_CLASS"].eq("RIGHTS_ISSUE")
        ].iloc[0]

        rights_new = float(
            rights_row["RIGHTS_NEW_SHARES"]
        )

        rights_existing = float(
            rights_row["RIGHTS_EXISTING_SHARES"]
        )

        rights_issue_price = float(
            rights_row["FINAL_RIGHTS_ISSUE_PRICE"]
        )

    else:

        rights_new = np.nan
        rights_existing = np.nan
        rights_issue_price = np.nan


    if has_structural:
        event_type = "STRUCTURAL_BREAK"
    elif has_rights:
        event_type = "RIGHTS"
    elif has_bonus or has_split:
        event_type = "UNIT_CHANGE"
    elif has_dividend:
        event_type = "DIVIDEND"
    else:
        raise RuntimeError(
            "Unexpected empty affecting-event group."
        )


    break_event_dates = group.loc[
        group["FINAL_PRIMARY_CLASS"].eq("DEMERGER_SPINOFF"),
        "EX_DATE",
    ]

    structural_event_date = (
        break_event_dates.min()
        if len(break_event_dates) > 0
        else pd.NaT
    )


    daily_event_rows.append(
        {
            "COMPANY_ID": company_id,
            "DATE": pd.Timestamp(apply_date),

            "CA_EVENT_TYPE": event_type,

            "CA_LEDGER_IDS": ";".join(
                sorted(
                    group["LEDGER_ID"].astype(str)
                )
            ),

            "CA_ORIGINAL_EX_DATES": ";".join(
                sorted(
                    {
                        d.strftime("%Y-%m-%d")
                        for d in group["EX_DATE"]
                    }
                )
            ),

            "CASH_DIVIDEND_PER_SHARE": float(
                cash_dividend
            ),

            "SAME_STOCK_SHARE_MULTIPLIER": float(
                same_stock_multiplier
            ),

            "RIGHTS_NEW_SHARES": rights_new,
            "RIGHTS_EXISTING_SHARES": rights_existing,
            "RIGHTS_ISSUE_PRICE": rights_issue_price,

            "STRUCTURAL_BREAK_FLAG": bool(
                has_structural
            ),

            "STRUCTURAL_BREAK_EVENT_DATE":
                structural_event_date,

            "MAX_EVENT_APPLY_LAG_DAYS": int(
                group["APPLY_LAG_CALENDAR_DAYS"].max()
            ),
        }
    )


daily_events = pd.DataFrame(
    daily_event_rows
)


# ============================================================
# 12. MERGE EVENTS INTO EQ BASE
# ============================================================

new_columns = [
    "CA_EVENT_TYPE",
    "CA_LEDGER_IDS",
    "CA_ORIGINAL_EX_DATES",
    "CASH_DIVIDEND_PER_SHARE",
    "SAME_STOCK_SHARE_MULTIPLIER",
    "RIGHTS_NEW_SHARES",
    "RIGHTS_EXISTING_SHARES",
    "RIGHTS_ISSUE_PRICE",
    "STRUCTURAL_BREAK_FLAG",
    "STRUCTURAL_BREAK_EVENT_DATE",
    "MAX_EVENT_APPLY_LAG_DAYS",
    "RAW_PREV_DATE_UNSEGMENTED",
    "RAW_PREV_CLOSE_UNSEGMENTED",
    "RAW_PRICE_RETURN",
    "TR_PREV_DATE",
    "TR_PREV_CLOSE",
    "DAYS_SINCE_PREV_OBS",
    "CA_ADJUSTED_REFERENCE_PRICE",
    "RIGHTS_BENEFIT_PER_POST_SHARE",
    "RIGHTS_NSE_ADJUSTMENT_FACTOR",
    "TOTAL_RETURN",
    "SEGMENT_NUMBER",
    "SEGMENT_ID",
    "TOTAL_RETURN_INDEX",
]

conflicts = set(new_columns).intersection(eq.columns)

if conflicts:
    raise ValueError(
        "EQ base already contains output column names: "
        f"{sorted(conflicts)}"
    )


research = eq.merge(
    daily_events,
    on=["COMPANY_ID", "DATE"],
    how="left",
    validate="one_to_one",
)


# Fill no-event defaults.
research["CA_EVENT_TYPE"] = (
    research["CA_EVENT_TYPE"]
    .fillna("NONE")
)

research["CA_LEDGER_IDS"] = (
    research["CA_LEDGER_IDS"]
    .fillna("")
)

research["CA_ORIGINAL_EX_DATES"] = (
    research["CA_ORIGINAL_EX_DATES"]
    .fillna("")
)

research["CASH_DIVIDEND_PER_SHARE"] = (
    research["CASH_DIVIDEND_PER_SHARE"]
    .fillna(0.0)
)

research["SAME_STOCK_SHARE_MULTIPLIER"] = (
    research["SAME_STOCK_SHARE_MULTIPLIER"]
    .fillna(1.0)
)

research["STRUCTURAL_BREAK_FLAG"] = (
    research["STRUCTURAL_BREAK_FLAG"]
    .fillna(False)
    .astype(bool)
)

research["MAX_EVENT_APPLY_LAG_DAYS"] = (
    research["MAX_EVENT_APPLY_LAG_DAYS"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 13. SORT CHRONOLOGICALLY FOR RETURN CONSTRUCTION
# ============================================================

research = (
    research.sort_values(
        ["COMPANY_ID", "DATE"]
    )
    .reset_index(drop=True)
)

# Raw close-to-close comparison across ALL available observations,
# deliberately ignoring structural segmentation. This is retained
# only for audit, so the mechanical demerger move remains visible.
research["RAW_PREV_DATE_UNSEGMENTED"] = (
    research.groupby("COMPANY_ID", sort=False)["DATE"].shift(1)
)

research["RAW_PREV_CLOSE_UNSEGMENTED"] = (
    research.groupby("COMPANY_ID", sort=False)["CLOSE"].shift(1)
)

research["RAW_PRICE_RETURN"] = (
    research["CLOSE"]
    / research["RAW_PREV_CLOSE_UNSEGMENTED"]
    - 1.0
)


# ============================================================
# 14. CREATE STRUCTURAL SEGMENTS
# ============================================================
#
# Segment 1 begins with the company's first observation.
# Each structural break increments the segment number AT the
# first post-break observation.
#
# Therefore no return is ever computed across a demerger.
# ============================================================

research["SEGMENT_NUMBER"] = (
    research.groupby(
        "COMPANY_ID",
        sort=False,
    )["STRUCTURAL_BREAK_FLAG"]
    .cumsum()
    .astype(int)
    + 1
)

research["SEGMENT_ID"] = (
    research["COMPANY_ID"].astype(str)
    + "_SEG"
    + research["SEGMENT_NUMBER"].astype(str)
)


# ============================================================
# 15. PREVIOUS OBSERVATION WITHIN SAME SEGMENT
# ============================================================

group_keys = [
    "COMPANY_ID",
    "SEGMENT_NUMBER",
]

research["TR_PREV_DATE"] = (
    research.groupby(
        group_keys,
        sort=False,
    )["DATE"]
    .shift(1)
)

research["TR_PREV_CLOSE"] = (
    research.groupby(
        group_keys,
        sort=False,
    )["CLOSE"]
    .shift(1)
)

research["DAYS_SINCE_PREV_OBS"] = (
    research["DATE"]
    - research["TR_PREV_DATE"]
).dt.days


# ============================================================
# 16. TOTAL RETURN CALCULATION
# ============================================================
#
# Ordinary day:
#   R = P_t / P_{t-1} - 1
#
# Cash dividend:
#   R = (P_t + D_t) / P_{t-1} - 1
#
# Split / bonus / consolidation:
#   M = new share units per old share
#   R = (M * P_t) / P_{t-1} - 1
#
# Rights:
#   A = new rights shares
#   B = existing shares
#   S = issue price
#   P = previous cum-rights close
#
#   NSE-equivalent benefit per post-rights share:
#       E = ((P - S) * A) / (A + B)
#
#   TERP = P - E
#        = (B*P + A*S) / (A+B)
#
#   R = P_t / TERP - 1
#
# Structural break:
#   R = NaN and TRI restarts at 100.
# ============================================================

research["CA_ADJUSTED_REFERENCE_PRICE"] = (
    research["TR_PREV_CLOSE"]
)

research["RIGHTS_BENEFIT_PER_POST_SHARE"] = np.nan
research["RIGHTS_NSE_ADJUSTMENT_FACTOR"] = np.nan
research["TOTAL_RETURN"] = np.nan


has_prev = research["TR_PREV_CLOSE"].notna()


# ------------------------------------------------------------
# Ordinary days
# ------------------------------------------------------------

mask = (
    has_prev
    &
    research["CA_EVENT_TYPE"].eq("NONE")
)

research.loc[
    mask,
    "TOTAL_RETURN",
] = (
    research.loc[mask, "CLOSE"]
    /
    research.loc[mask, "TR_PREV_CLOSE"]
    - 1.0
)


# ------------------------------------------------------------
# Cash dividends
# ------------------------------------------------------------

mask = (
    has_prev
    &
    research["CA_EVENT_TYPE"].eq("DIVIDEND")
)

research.loc[
    mask,
    "TOTAL_RETURN",
] = (
    (
        research.loc[mask, "CLOSE"]
        +
        research.loc[
            mask,
            "CASH_DIVIDEND_PER_SHARE",
        ]
    )
    /
    research.loc[mask, "TR_PREV_CLOSE"]
    - 1.0
)


# ------------------------------------------------------------
# Splits / bonus / consolidation
# ------------------------------------------------------------

mask = (
    has_prev
    &
    research["CA_EVENT_TYPE"].eq("UNIT_CHANGE")
)

research.loc[
    mask,
    "CA_ADJUSTED_REFERENCE_PRICE",
] = (
    research.loc[
        mask,
        "TR_PREV_CLOSE",
    ]
    /
    research.loc[
        mask,
        "SAME_STOCK_SHARE_MULTIPLIER",
    ]
)

research.loc[
    mask,
    "TOTAL_RETURN",
] = (
    research.loc[mask, "CLOSE"]
    /
    research.loc[
        mask,
        "CA_ADJUSTED_REFERENCE_PRICE",
    ]
    - 1.0
)


# ------------------------------------------------------------
# Rights
# ------------------------------------------------------------

mask = (
    has_prev
    &
    research["CA_EVENT_TYPE"].eq("RIGHTS")
)

P = research.loc[
    mask,
    "TR_PREV_CLOSE",
]

A = research.loc[
    mask,
    "RIGHTS_NEW_SHARES",
]

B = research.loc[
    mask,
    "RIGHTS_EXISTING_SHARES",
]

S = research.loc[
    mask,
    "RIGHTS_ISSUE_PRICE",
]


benefit = (
    (P - S) * A
    /
    (A + B)
)

terp = (
    P
    -
    benefit
)


if (terp <= 0).any():
    bad = research.loc[
        mask,
        [
            "DATE",
            "COMPANY_ID",
            "TR_PREV_CLOSE",
            "RIGHTS_NEW_SHARES",
            "RIGHTS_EXISTING_SHARES",
            "RIGHTS_ISSUE_PRICE",
        ],
    ].copy()

    bad["TERP"] = terp

    raise ValueError(
        "Non-positive TERP encountered.\n"
        + bad.to_string(index=False)
    )


research.loc[
    mask,
    "RIGHTS_BENEFIT_PER_POST_SHARE",
] = benefit

research.loc[
    mask,
    "CA_ADJUSTED_REFERENCE_PRICE",
] = terp

research.loc[
    mask,
    "RIGHTS_NSE_ADJUSTMENT_FACTOR",
] = (
    terp
    /
    P
)

research.loc[
    mask,
    "TOTAL_RETURN",
] = (
    research.loc[mask, "CLOSE"]
    /
    terp
    - 1.0
)


# ------------------------------------------------------------
# Structural breaks
# ------------------------------------------------------------
#
# These rows deliberately keep TOTAL_RETURN = NaN.
# ------------------------------------------------------------

struct_mask = research[
    "CA_EVENT_TYPE"
].eq("STRUCTURAL_BREAK")

research.loc[
    struct_mask,
    "CA_ADJUSTED_REFERENCE_PRICE",
] = np.nan

research.loc[
    struct_mask,
    "TOTAL_RETURN",
] = np.nan


# ============================================================
# 17. VALIDATE RETURN COMPLETENESS
# ============================================================
#
# TOTAL_RETURN should be missing only at the first row of each
# segment (including structural-break restart rows).
# ============================================================

segment_first = (
    research.groupby(
        group_keys,
        sort=False,
    )
    .cumcount()
    .eq(0)
)

# A non-structural action on the first observation of a segment
# cannot be evaluated because there is no cum-action previous close.
uncomputable_first_event = (
    segment_first
    &
    research["CA_EVENT_TYPE"].isin(
        ["DIVIDEND", "UNIT_CHANGE", "RIGHTS"]
    )
)

if uncomputable_first_event.any():
    bad = research.loc[
        uncomputable_first_event,
        [
            "DATE", "COMPANY_ID", "SYMBOL",
            "SEGMENT_ID", "CA_EVENT_TYPE", "CA_LEDGER_IDS"
        ],
    ]
    raise ValueError(
        "A non-structural corporate action occurs on a segment's "
        "first available observation, so its return cannot be "
        "computed from this dataset.\n"
        + bad.to_string(index=False)
    )

unexpected_missing = (
    research["TOTAL_RETURN"].isna()
    &
    ~segment_first
)

if unexpected_missing.any():

    bad = research.loc[
        unexpected_missing,
        [
            "DATE",
            "COMPANY_ID",
            "SYMBOL",
            "SEGMENT_ID",
            "CA_EVENT_TYPE",
            "CLOSE",
            "TR_PREV_CLOSE",
        ],
    ]

    raise ValueError(
        "Unexpected missing TOTAL_RETURN values.\n"
        + bad.to_string(index=False)
    )


unexpected_first_return = (
    segment_first
    &
    research["TOTAL_RETURN"].notna()
)

if unexpected_first_return.any():
    raise RuntimeError(
        "A segment-first observation unexpectedly has a return."
    )


# ============================================================
# 18. BUILD TOTAL RETURN INDEX
# ============================================================
#
# Each segment starts at 100.
# No TRI is carried across a structural break.
# ============================================================

def build_tri(group):

    r = group["TOTAL_RETURN"].copy()

    if len(r) == 0:
        return pd.Series(
            dtype=float,
            index=group.index,
        )

    # Only first row should be NaN; treat the initial observation
    # as the base level 100.
    growth = (
        1.0
        +
        r.fillna(0.0)
    )

    tri = (
        100.0
        *
        growth.cumprod()
    )

    return pd.Series(
        tri.values,
        index=group.index,
    )


research["TOTAL_RETURN_INDEX"] = (
    research.groupby(
        group_keys,
        sort=False,
    )["TOTAL_RETURN"]
    .transform(
        lambda s: 100.0 * (1.0 + s.fillna(0.0)).cumprod()
    )
)


# ============================================================
# 19. SANITY CHECKS
# ============================================================

if (~np.isfinite(
    research["TOTAL_RETURN_INDEX"]
)).any():
    raise ValueError(
        "Non-finite TOTAL_RETURN_INDEX values found."
    )

if (research["TOTAL_RETURN_INDEX"] <= 0).any():
    raise ValueError(
        "Non-positive TOTAL_RETURN_INDEX values found."
    )

if (research["TOTAL_RETURN"].dropna() <= -1.0).any():
    bad = research.loc[
        research["TOTAL_RETURN"].le(-1.0),
        [
            "DATE", "COMPANY_ID", "SYMBOL", "CA_EVENT_TYPE",
            "TR_PREV_CLOSE", "CLOSE", "TOTAL_RETURN"
        ],
    ]
    raise ValueError(
        "TOTAL_RETURN <= -100% found; this indicates an adjustment "
        "or data problem.\n" + bad.to_string(index=False)
    )


# Corporate-action row counts actually applied.
applied_event_counts = (
    research.loc[
        research["CA_EVENT_TYPE"].ne("NONE"),
        "CA_EVENT_TYPE",
    ]
    .value_counts()
    .rename_axis("CA_EVENT_TYPE")
    .reset_index(name="ROWS")
)


# We expect four rights apply-dates and four structural-break rows.
rights_applied = int(
    research["CA_EVENT_TYPE"].eq("RIGHTS").sum()
)

breaks_applied = int(
    research["STRUCTURAL_BREAK_FLAG"].sum()
)

if rights_applied != 4:
    raise ValueError(
        f"Expected 4 rights apply rows, found {rights_applied}."
    )

if breaks_applied != 4:
    raise ValueError(
        f"Expected 4 structural-break rows, found {breaks_applied}."
    )


# ============================================================
# 20. EVENT APPLICATION AUDIT
# ============================================================

event_audit_cols = [
    "DATE",
    "COMPANY_ID",
    "SYMBOL",
    "CA_EVENT_TYPE",
    "CA_LEDGER_IDS",
    "CA_ORIGINAL_EX_DATES",
    "MAX_EVENT_APPLY_LAG_DAYS",
    "STRUCTURAL_BREAK_EVENT_DATE",
    "SEGMENT_ID",
    "RAW_PREV_DATE_UNSEGMENTED",
    "RAW_PREV_CLOSE_UNSEGMENTED",
    "RAW_PRICE_RETURN",
    "TR_PREV_DATE",
    "TR_PREV_CLOSE",
    "CLOSE",
    "CASH_DIVIDEND_PER_SHARE",
    "SAME_STOCK_SHARE_MULTIPLIER",
    "RIGHTS_NEW_SHARES",
    "RIGHTS_EXISTING_SHARES",
    "RIGHTS_ISSUE_PRICE",
    "RIGHTS_BENEFIT_PER_POST_SHARE",
    "RIGHTS_NSE_ADJUSTMENT_FACTOR",
    "CA_ADJUSTED_REFERENCE_PRICE",
    "TOTAL_RETURN",
    "TOTAL_RETURN_INDEX",
]

event_audit = research.loc[
    research["CA_EVENT_TYPE"].ne("NONE"),
    event_audit_cols,
].copy()

event_audit.to_csv(
    OUTPUT_DIR
    / "01_EVENT_APPLICATION_AUDIT.csv",
    index=False,
    date_format="%Y-%m-%d",
)


# Structural-break audit.
break_audit = research.loc[
    research["STRUCTURAL_BREAK_FLAG"],
    [
        "DATE",
        "COMPANY_ID",
        "SYMBOL",
        "CA_LEDGER_IDS",
        "CA_ORIGINAL_EX_DATES",
        "STRUCTURAL_BREAK_EVENT_DATE",
        "MAX_EVENT_APPLY_LAG_DAYS",
        "SEGMENT_ID",
        "CLOSE",
        "TOTAL_RETURN",
        "TOTAL_RETURN_INDEX",
    ],
].copy()

break_audit.to_csv(
    OUTPUT_DIR
    / "02_STRUCTURAL_BREAK_AUDIT.csv",
    index=False,
    date_format="%Y-%m-%d",
)


# Segment summary.
segment_summary = (
    research.groupby(
        [
            "COMPANY_ID",
            "SEGMENT_NUMBER",
            "SEGMENT_ID",
        ],
        as_index=False,
    )
    .agg(
        START_DATE=("DATE", "min"),
        END_DATE=("DATE", "max"),
        OBSERVATIONS=("DATE", "size"),
        START_TRI=("TOTAL_RETURN_INDEX", "first"),
        END_TRI=("TOTAL_RETURN_INDEX", "last"),
    )
)

segment_summary.to_csv(
    OUTPUT_DIR
    / "03_SEGMENT_SUMMARY.csv",
    index=False,
    date_format="%Y-%m-%d",
)


# Return diagnostics.
return_diagnostics = pd.DataFrame(
    [
        {
            "CHECK": "Input EQ rows",
            "VALUE": len(eq),
        },
        {
            "CHECK": "Output rows",
            "VALUE": len(research),
        },
        {
            "CHECK": "Unique company identities",
            "VALUE": research["COMPANY_ID"].nunique(),
        },
        {
            "CHECK": "Total segments",
            "VALUE": research["SEGMENT_ID"].nunique(),
        },
        {
            "CHECK": "Dividend apply rows",
            "VALUE": int(
                research["CA_EVENT_TYPE"]
                .eq("DIVIDEND")
                .sum()
            ),
        },
        {
            "CHECK": "Unit-change apply rows",
            "VALUE": int(
                research["CA_EVENT_TYPE"]
                .eq("UNIT_CHANGE")
                .sum()
            ),
        },
        {
            "CHECK": "Rights apply rows",
            "VALUE": rights_applied,
        },
        {
            "CHECK": "Structural-break apply rows",
            "VALUE": breaks_applied,
        },
        {
            "CHECK": "Unexpected missing total returns",
            "VALUE": int(
                unexpected_missing.sum()
            ),
        },
        {
            "CHECK": "Missing TRI values",
            "VALUE": int(
                research["TOTAL_RETURN_INDEX"]
                .isna()
                .sum()
            ),
        },
        {
            "CHECK": "Minimum TOTAL_RETURN",
            "VALUE": research[
                "TOTAL_RETURN"
            ].min(),
        },
        {
            "CHECK": "Maximum TOTAL_RETURN",
            "VALUE": research[
                "TOTAL_RETURN"
            ].max(),
        },
    ]
)

return_diagnostics.to_csv(
    OUTPUT_DIR
    / "00_TOTAL_RETURN_AUDIT_SUMMARY.csv",
    index=False,
)


applied_event_counts.to_csv(
    OUTPUT_DIR
    / "04_APPLIED_EVENT_COUNTS.csv",
    index=False,
)


# Extreme total returns are NOT automatically deleted or winsorized.
# This file is for human audit only.
extreme_returns = research.loc[
    research["TOTAL_RETURN"].abs().ge(0.20),
    [
        "DATE", "COMPANY_ID", "SYMBOL", "SEGMENT_ID",
        "CA_EVENT_TYPE", "CA_LEDGER_IDS",
        "RAW_PREV_DATE_UNSEGMENTED",
        "RAW_PREV_CLOSE_UNSEGMENTED",
        "RAW_PRICE_RETURN",
        "TR_PREV_DATE", "TR_PREV_CLOSE", "CLOSE",
        "TOTAL_RETURN",
        "CASH_DIVIDEND_PER_SHARE",
        "SAME_STOCK_SHARE_MULTIPLIER",
        "RIGHTS_ISSUE_PRICE",
    ],
].copy()

extreme_returns.to_csv(
    OUTPUT_DIR / "05_EXTREME_TOTAL_RETURNS_GE_20PCT.csv",
    index=False,
    date_format="%Y-%m-%d",
)


# Finalized ledger snapshot actually used by this run.
ledger.to_csv(
    OUTPUT_DIR / "06_FINALIZED_LEDGER_USED.csv",
    index=False,
    date_format="%Y-%m-%d",
)


# ============================================================
# 21. RESTORE ORIGINAL EQ ROW ORDER AND SAVE MAIN OUTPUT
# ============================================================

research = (
    research.sort_values(
        "_ORIGINAL_ROW_ORDER"
    )
    .drop(
        columns=["_ORIGINAL_ROW_ORDER"]
    )
    .reset_index(drop=True)
)


# Input/raw columns are still present and unchanged.
research.to_csv(
    OUTPUT_FILE,
    index=False,
    date_format="%Y-%m-%d",
)


# ============================================================
# 22. README / METHOD NOTE
# ============================================================

method_note = r"""
NIFTY PHARMA TOTAL-RETURN CONSTRUCTION

RAW DATA
--------
The original NSE EQ base columns are preserved. The program does not
overwrite CLOSE, TOTAL_TRADED_QTY, TOTAL_TRADED_VALUE, TOTAL_TRADES,
or any other raw NSE field.

RETURN RULES
------------

1. Ordinary day
   TOTAL_RETURN = CLOSE_t / CLOSE_previous_observation - 1

2. Cash dividend
   TOTAL_RETURN = (CLOSE_t + dividend_per_share) /
                  CLOSE_previous_observation - 1

   The full cash distribution is included once because the input is a
   raw single-stock close series. No index-divisor adjustment has
   already removed a special dividend from this raw stock price.

3. Split / consolidation / bonus
   Let M = post-action share units per pre-action share.

   TOTAL_RETURN = (M * CLOSE_t) /
                  CLOSE_previous_observation - 1

4. Rights issue
   A = rights shares
   B = existing shares
   S = issue price
   P = previous cum-rights close

   Benefit per post-rights share:
       E = ((P - S) * A) / (A + B)

   Theoretical ex-rights price:
       TERP = P - E
            = (B*P + A*S)/(A+B)

   TOTAL_RETURN = CLOSE_t / TERP - 1

5. Demerger / spin-off
   No return is computed across the event.
   A new SEGMENT_ID begins at the first available EQ observation
   on/after the structural-break date.
   TOTAL_RETURN_INDEX restarts at 100.

NO FORWARD FILL
---------------
Missing/suspended observations are not filled. Ordinary returns use
the previous AVAILABLE EQ observation within the same structural
segment.

BUYBACKS / ADMINISTRATIVE EVENTS
--------------------------------
No automatic return adjustment.

IMPORTANT FOR PAIRS TRADING
---------------------------
Formation windows must later be required to lie entirely inside one
SEGMENT_ID. This prevents estimation of a pair relationship across a
demerger structural break.
"""

(
    OUTPUT_DIR
    / "README_TOTAL_RETURN_METHOD.txt"
).write_text(
    method_note,
    encoding="utf-8",
)


# ============================================================
# 23. FINAL PRINT
# ============================================================

print("\n")
print("=" * 100)
print("TOTAL-RETURN CONSTRUCTION COMPLETE")
print("=" * 100)

print("\nAudit summary:")
print(
    return_diagnostics.to_string(
        index=False
    )
)

print("\nApplied event counts:")
print(
    applied_event_counts.to_string(
        index=False
    )
)

print("\nMain output:")
print(OUTPUT_FILE)

print("\nAudit files:")
print(
    OUTPUT_DIR
    / "00_TOTAL_RETURN_AUDIT_SUMMARY.csv"
)
print(
    OUTPUT_DIR
    / "01_EVENT_APPLICATION_AUDIT.csv"
)
print(
    OUTPUT_DIR
    / "02_STRUCTURAL_BREAK_AUDIT.csv"
)
print(
    OUTPUT_DIR
    / "03_SEGMENT_SUMMARY.csv"
)
print(
    OUTPUT_DIR
    / "04_APPLIED_EVENT_COUNTS.csv"
)
print(
    OUTPUT_DIR
    / "05_EXTREME_TOTAL_RETURNS_GE_20PCT.csv"
)
print(
    OUTPUT_DIR
    / "06_FINALIZED_LEDGER_USED.csv"
)

print("\nIMPORTANT:")
print(
    "Do not move to pair selection until the event audit "
    "and extreme total returns have been checked."
)
