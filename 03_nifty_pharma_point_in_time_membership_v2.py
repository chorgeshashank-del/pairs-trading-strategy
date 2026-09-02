import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# NIFTY PHARMA POINT-IN-TIME MEMBERSHIP CONSTRUCTION
# ============================================================

PROJECT_ROOT = Path(r"C:\fin proj")

TOTAL_RETURN_FILE = (
    PROJECT_ROOT
    / "nse_pharma_total_return"
    / "NIFTY_PHARMA_TOTAL_RETURN_BASE_2016_2026.csv"
)

MEMBERSHIP_FILENAME = "historical_nifty_pharma_FINAL_AUDITED.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "nse_pharma_point_in_time_universe"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PRICE_ROWS = None  # Dynamic after Muhurat repair
EXPECTED_MEMBERSHIP_ROWS = 170

EXPECTED_MEMBERSHIP_START = pd.Timestamp("2016-08-01")
EXPECTED_MEMBERSHIP_END   = pd.Timestamp("2026-07-31")


def locate_membership_file():
    direct = PROJECT_ROOT / MEMBERSHIP_FILENAME

    if direct.exists():
        return direct

    matches = list(PROJECT_ROOT.rglob(MEMBERSHIP_FILENAME))

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            "\nCould not find the finalized audited membership file:\n"
            f"{MEMBERSHIP_FILENAME}\n\n"
            f"Searched under:\n{PROJECT_ROOT}\n"
        )

    raise RuntimeError(
        "\nMore than one copy of the finalized membership file was found.\n"
        "Do not guess which one to use.\n\n"
        + "\n".join(str(x) for x in matches)
    )


MEMBERSHIP_FILE = locate_membership_file()


def parse_dates(series):
    try:
        return pd.to_datetime(
            series,
            format="mixed",
            dayfirst=False,
            errors="coerce"
        )
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def clean_symbol(series):
    return (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )


def normalize_membership_symbol(symbol):
    if pd.isna(symbol):
        return pd.NA

    s = str(symbol).strip().upper()

    known_symbol_fixes = {
        "AJANTAPHARM": "AJANTPHARM",
    }

    return known_symbol_fixes.get(s, s)


def stable_company_id(symbol):
    if pd.isna(symbol):
        return pd.NA

    s = str(symbol).strip().upper()

    if s in {"CADILAHC", "ZYDUSLIFE"}:
        return "ZYDUS"

    if s == "AJANTAPHARM":
        return "AJANTPHARM"

    return s


print("=" * 100)
print("LOADING TOTAL-RETURN DATA")
print("=" * 100)

if not TOTAL_RETURN_FILE.exists():
    raise FileNotFoundError(
        f"Total-return file not found:\n{TOTAL_RETURN_FILE}"
    )

prices = pd.read_csv(
    TOTAL_RETURN_FILE,
    low_memory=False
)

INPUT_PRICE_ROWS = len(prices)

prices.columns = (
    prices.columns
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.upper()
)

print(f"Total-return rows loaded: {len(prices):,}")
print(f"File: {TOTAL_RETURN_FILE}")

required_price_columns = {
    "DATE",
    "SYMBOL",
    "CLOSE",
    "TOTAL_RETURN",
    "TOTAL_RETURN_INDEX",
    "SEGMENT_ID",
    "STRUCTURAL_BREAK_FLAG",
}

missing_price_columns = required_price_columns - set(prices.columns)

if missing_price_columns:
    raise ValueError(
        "Total-return file is missing required columns:\n"
        f"{sorted(missing_price_columns)}"
    )

if EXPECTED_PRICE_ROWS is not None and len(prices) != EXPECTED_PRICE_ROWS:
    raise ValueError(
        f"\nExpected {EXPECTED_PRICE_ROWS:,} total-return rows "
        f"but found {len(prices):,}.\n"
        "Stop and verify that the finalized total-return file "
        "is being used."
    )

prices["DATE"] = parse_dates(prices["DATE"])

if prices["DATE"].isna().any():
    raise ValueError(
        "Invalid/missing DATE values exist in total-return data."
    )

prices["SYMBOL"] = clean_symbol(prices["SYMBOL"])

if "COMPANY_ID" in prices.columns:
    prices["COMPANY_ID"] = clean_symbol(prices["COMPANY_ID"])
else:
    prices["COMPANY_ID"] = (
        prices["SYMBOL"]
        .map(stable_company_id)
        .astype("string")
    )

prices.loc[
    prices["SYMBOL"].isin(["CADILAHC", "ZYDUSLIFE"]),
    "COMPANY_ID"
] = "ZYDUS"

prices.loc[
    prices["SYMBOL"].eq("AJANTAPHARM"),
    "COMPANY_ID"
] = "AJANTPHARM"

if prices[["DATE", "COMPANY_ID"]].duplicated().any():
    dupes = prices[
        prices[["DATE", "COMPANY_ID"]].duplicated(keep=False)
    ].copy()

    dupe_file = OUTPUT_DIR / "ERROR_DUPLICATE_DATE_COMPANY_ID.csv"

    dupes.to_csv(
        dupe_file,
        index=False,
        date_format="%Y-%m-%d"
    )

    raise RuntimeError(
        "Duplicate DATE + COMPANY_ID rows found in total-return data.\n"
        f"Saved for inspection:\n{dupe_file}"
    )

prices["_SOURCE_ROW_ORDER"] = np.arange(len(prices))


print("\n")
print("=" * 100)
print("LOADING FINAL AUDITED NIFTY PHARMA MEMBERSHIP HISTORY")
print("=" * 100)

membership = pd.read_csv(
    MEMBERSHIP_FILE,
    low_memory=False
)

membership.columns = (
    membership.columns
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.upper()
)

required_membership_columns = {
    "SYMBOL",
    "COMPANY",
    "START_DATE",
    "END_DATE",
}

missing_membership_columns = (
    required_membership_columns
    -
    set(membership.columns)
)

if missing_membership_columns:
    raise ValueError(
        "Membership file is missing required columns:\n"
        f"{sorted(missing_membership_columns)}"
    )

print(f"Membership rows loaded: {len(membership):,}")
print(f"File: {MEMBERSHIP_FILE}")

if len(membership) != EXPECTED_MEMBERSHIP_ROWS:
    raise ValueError(
        f"\nExpected {EXPECTED_MEMBERSHIP_ROWS} rows in the "
        "final audited membership file, but found "
        f"{len(membership)}.\n\n"
        "This may be an older membership-history version. "
        "Stop rather than silently continue."
    )

membership["START_DATE"] = parse_dates(membership["START_DATE"])
membership["END_DATE"]   = parse_dates(membership["END_DATE"])

if membership[["START_DATE", "END_DATE"]].isna().any().any():
    bad = membership[
        membership["START_DATE"].isna()
        |
        membership["END_DATE"].isna()
    ]

    raise ValueError(
        "Invalid membership dates found:\n"
        + bad.to_string(index=False)
    )

if (membership["START_DATE"] > membership["END_DATE"]).any():
    bad = membership[
        membership["START_DATE"]
        >
        membership["END_DATE"]
    ]

    raise ValueError(
        "Membership START_DATE is after END_DATE:\n"
        + bad.to_string(index=False)
    )

actual_membership_start = membership["START_DATE"].min()
actual_membership_end   = membership["END_DATE"].max()

if actual_membership_start != EXPECTED_MEMBERSHIP_START:
    raise ValueError(
        "Unexpected membership-history start date.\n"
        f"Expected: {EXPECTED_MEMBERSHIP_START.date()}\n"
        f"Found:    {actual_membership_start.date()}"
    )

if actual_membership_end != EXPECTED_MEMBERSHIP_END:
    raise ValueError(
        "Unexpected membership-history end date.\n"
        f"Expected: {EXPECTED_MEMBERSHIP_END.date()}\n"
        f"Found:    {actual_membership_end.date()}"
    )


membership["MEMBERSHIP_SYMBOL_SOURCE"] = clean_symbol(
    membership["SYMBOL"]
)

membership["MEMBERSHIP_SYMBOL"] = (
    membership["MEMBERSHIP_SYMBOL_SOURCE"]
    .map(normalize_membership_symbol)
    .astype("string")
)

membership["MEMBERSHIP_COMPANY_ID"] = (
    membership["MEMBERSHIP_SYMBOL"]
    .map(stable_company_id)
    .astype("string")
)

membership["SYMBOL_NORMALIZATION_NOTE"] = ""

membership.loc[
    membership["MEMBERSHIP_SYMBOL_SOURCE"].eq("AJANTAPHARM"),
    "SYMBOL_NORMALIZATION_NOTE"
] = (
    "Membership file symbol AJANTAPHARM normalized "
    "to NSE price symbol AJANTPHARM"
)

membership.loc[
    membership["MEMBERSHIP_SYMBOL_SOURCE"].isin(
        ["CADILAHC", "ZYDUSLIFE"]
    ),
    "SYMBOL_NORMALIZATION_NOTE"
] = (
    "CADILAHC and ZYDUSLIFE mapped to stable COMPANY_ID ZYDUS"
)

membership["MEMBERSHIP_PERIOD_ID"] = [
    f"NM{i:04d}"
    for i in range(1, len(membership) + 1)
]


membership_dupes = membership[
    membership.duplicated(
        subset=[
            "MEMBERSHIP_COMPANY_ID",
            "START_DATE",
            "END_DATE",
        ],
        keep=False
    )
].copy()

if not membership_dupes.empty:
    duplicate_file = (
        OUTPUT_DIR
        / "ERROR_DUPLICATE_MEMBERSHIP_INTERVALS.csv"
    )

    membership_dupes.to_csv(
        duplicate_file,
        index=False,
        date_format="%Y-%m-%d"
    )

    raise RuntimeError(
        "Duplicate normalized membership intervals found.\n"
        f"Saved to:\n{duplicate_file}"
    )


overlap_records = []

for company_id, group in membership.groupby(
    "MEMBERSHIP_COMPANY_ID"
):
    g = group.sort_values(
        ["START_DATE", "END_DATE"]
    ).reset_index(drop=True)

    for i in range(1, len(g)):
        previous_end = g.loc[i - 1, "END_DATE"]
        current_start = g.loc[i, "START_DATE"]

        if current_start <= previous_end:
            overlap_records.append(
                {
                    "COMPANY_ID": company_id,
                    "PREVIOUS_PERIOD_ID":
                        g.loc[i - 1, "MEMBERSHIP_PERIOD_ID"],
                    "PREVIOUS_START":
                        g.loc[i - 1, "START_DATE"],
                    "PREVIOUS_END":
                        previous_end,
                    "CURRENT_PERIOD_ID":
                        g.loc[i, "MEMBERSHIP_PERIOD_ID"],
                    "CURRENT_START":
                        current_start,
                    "CURRENT_END":
                        g.loc[i, "END_DATE"],
                }
            )

overlap_df = pd.DataFrame(overlap_records)

if not overlap_df.empty:
    overlap_file = (
        OUTPUT_DIR
        / "ERROR_OVERLAPPING_MEMBERSHIP_INTERVALS.csv"
    )

    overlap_df.to_csv(
        overlap_file,
        index=False,
        date_format="%Y-%m-%d"
    )

    raise RuntimeError(
        "Overlapping membership intervals detected after "
        "stable-company mapping.\n"
        f"Saved to:\n{overlap_file}"
    )


price_company_ids = set(
    prices["COMPANY_ID"]
    .dropna()
    .unique()
)

membership_company_ids = set(
    membership["MEMBERSHIP_COMPANY_ID"]
    .dropna()
    .unique()
)

membership_not_in_prices = sorted(
    membership_company_ids
    -
    price_company_ids
)

if membership_not_in_prices:
    raise RuntimeError(
        "Some historical NIFTY Pharma members have no company "
        "identity anywhere in the total-return dataset:\n"
        f"{membership_not_in_prices}"
    )


print("\n")
print("=" * 100)
print("APPLYING POINT-IN-TIME MEMBERSHIP")
print("=" * 100)

prices["MEMBERSHIP_KNOWN_PERIOD"] = (
    prices["DATE"].between(
        EXPECTED_MEMBERSHIP_START,
        EXPECTED_MEMBERSHIP_END,
        inclusive="both"
    )
)

prices["IN_NIFTY_PHARMA"] = False

prices["MEMBERSHIP_MATCH_COUNT"] = np.zeros(
    len(prices),
    dtype=np.int16
)

prices["MEMBERSHIP_PERIOD_ID"] = pd.Series(
    pd.NA,
    index=prices.index,
    dtype="string"
)

prices["MEMBERSHIP_SYMBOL"] = pd.Series(
    pd.NA,
    index=prices.index,
    dtype="string"
)

prices["MEMBERSHIP_SYMBOL_SOURCE"] = pd.Series(
    pd.NA,
    index=prices.index,
    dtype="string"
)

prices["MEMBERSHIP_COMPANY"] = pd.Series(
    pd.NA,
    index=prices.index,
    dtype="string"
)

prices["MEMBERSHIP_START_DATE"] = pd.NaT
prices["MEMBERSHIP_END_DATE"] = pd.NaT


for _, member_row in membership.iterrows():

    mask = (
        prices["COMPANY_ID"].eq(
            member_row["MEMBERSHIP_COMPANY_ID"]
        )
        &
        prices["DATE"].between(
            member_row["START_DATE"],
            member_row["END_DATE"],
            inclusive="both"
        )
    )

    prices.loc[
        mask,
        "MEMBERSHIP_MATCH_COUNT"
    ] += 1

    prices.loc[
        mask,
        "IN_NIFTY_PHARMA"
    ] = True

    prices.loc[
        mask,
        "MEMBERSHIP_PERIOD_ID"
    ] = member_row["MEMBERSHIP_PERIOD_ID"]

    prices.loc[
        mask,
        "MEMBERSHIP_SYMBOL"
    ] = member_row["MEMBERSHIP_SYMBOL"]

    prices.loc[
        mask,
        "MEMBERSHIP_SYMBOL_SOURCE"
    ] = member_row["MEMBERSHIP_SYMBOL_SOURCE"]

    prices.loc[
        mask,
        "MEMBERSHIP_COMPANY"
    ] = member_row["COMPANY"]

    prices.loc[
        mask,
        "MEMBERSHIP_START_DATE"
    ] = member_row["START_DATE"]

    prices.loc[
        mask,
        "MEMBERSHIP_END_DATE"
    ] = member_row["END_DATE"]


if (prices["MEMBERSHIP_MATCH_COUNT"] > 1).any():

    bad = prices[
        prices["MEMBERSHIP_MATCH_COUNT"] > 1
    ].copy()

    bad_file = (
        OUTPUT_DIR
        / "ERROR_MULTIPLE_MEMBERSHIP_MATCHES.csv"
    )

    bad.to_csv(
        bad_file,
        index=False,
        date_format="%Y-%m-%d"
    )

    raise RuntimeError(
        "At least one price row matched multiple membership "
        "intervals.\n"
        f"Saved to:\n{bad_file}"
    )


if (
    prices["IN_NIFTY_PHARMA"]
    &
    ~prices["MEMBERSHIP_KNOWN_PERIOD"]
).any():

    raise RuntimeError(
        "A stock was marked IN_NIFTY_PHARMA outside the known "
        "membership-history period."
    )


bad_flagged_metadata = prices[
    prices["IN_NIFTY_PHARMA"]
    &
    prices["MEMBERSHIP_PERIOD_ID"].isna()
]

if not bad_flagged_metadata.empty:
    raise RuntimeError(
        "Some flagged member rows have no membership metadata."
    )


trading_dates = pd.DataFrame(
    {
        "DATE":
            sorted(
                prices.loc[
                    prices["MEMBERSHIP_KNOWN_PERIOD"],
                    "DATE"
                ].unique()
            )
    }
)

expected_parts = []

for _, member_row in membership.iterrows():

    dates = trading_dates[
        trading_dates["DATE"].between(
            member_row["START_DATE"],
            member_row["END_DATE"],
            inclusive="both"
        )
    ].copy()

    if dates.empty:
        continue

    dates["MEMBERSHIP_PERIOD_ID"] = (
        member_row["MEMBERSHIP_PERIOD_ID"]
    )

    dates["COMPANY_ID"] = (
        member_row["MEMBERSHIP_COMPANY_ID"]
    )

    dates["MEMBERSHIP_SYMBOL"] = (
        member_row["MEMBERSHIP_SYMBOL"]
    )

    dates["MEMBERSHIP_COMPANY"] = (
        member_row["COMPANY"]
    )

    dates["MEMBERSHIP_START_DATE"] = (
        member_row["START_DATE"]
    )

    dates["MEMBERSHIP_END_DATE"] = (
        member_row["END_DATE"]
    )

    expected_parts.append(dates)

expected_panel = pd.concat(
    expected_parts,
    ignore_index=True
)

if expected_panel[
    ["DATE", "COMPANY_ID"]
].duplicated().any():

    bad = expected_panel[
        expected_panel[
            ["DATE", "COMPANY_ID"]
        ].duplicated(keep=False)
    ]

    bad_file = (
        OUTPUT_DIR
        / "ERROR_DUPLICATE_EXPECTED_MEMBER_DATES.csv"
    )

    bad.to_csv(
        bad_file,
        index=False,
        date_format="%Y-%m-%d"
    )

    raise RuntimeError(
        "Expected membership panel has duplicate DATE + "
        "COMPANY_ID entries.\n"
        f"Saved to:\n{bad_file}"
    )

actual_price_key = (
    prices[
        [
            "DATE",
            "COMPANY_ID",
            "SYMBOL",
            "CLOSE",
            "SEGMENT_ID",
            "STRUCTURAL_BREAK_FLAG",
        ]
    ]
    .rename(
        columns={
            "SYMBOL": "ACTUAL_PRICE_SYMBOL",
            "CLOSE": "ACTUAL_CLOSE",
            "SEGMENT_ID": "ACTUAL_SEGMENT_ID",
            "STRUCTURAL_BREAK_FLAG":
                "ACTUAL_STRUCTURAL_BREAK_FLAG",
        }
    )
)

expected_panel = expected_panel.merge(
    actual_price_key,
    on=["DATE", "COMPANY_ID"],
    how="left",
    validate="one_to_one"
)

expected_panel["PRICE_ROW_AVAILABLE"] = (
    expected_panel["ACTUAL_CLOSE"].notna()
)

missing_member_prices = expected_panel[
    ~expected_panel["PRICE_ROW_AVAILABLE"]
].copy()


expected_daily = (
    expected_panel
    .groupby(
        "DATE",
        as_index=False
    )
    .agg(
        EXPECTED_CONSTITUENTS=(
            "COMPANY_ID",
            "nunique"
        ),
        MEMBER_PRICE_ROWS_AVAILABLE=(
            "PRICE_ROW_AVAILABLE",
            "sum"
        )
    )
)

observed_daily = (
    prices[
        prices["IN_NIFTY_PHARMA"]
    ]
    .groupby(
        "DATE",
        as_index=False
    )
    .agg(
        FLAGGED_CONSTITUENTS_WITH_PRICE=(
            "COMPANY_ID",
            "nunique"
        )
    )
)

daily_counts = expected_daily.merge(
    observed_daily,
    on="DATE",
    how="left"
)

daily_counts[
    "FLAGGED_CONSTITUENTS_WITH_PRICE"
] = (
    daily_counts[
        "FLAGGED_CONSTITUENTS_WITH_PRICE"
    ]
    .fillna(0)
    .astype(int)
)

daily_counts[
    "MISSING_MEMBER_PRICE_ROWS"
] = (
    daily_counts["EXPECTED_CONSTITUENTS"]
    -
    daily_counts[
        "FLAGGED_CONSTITUENTS_WITH_PRICE"
    ]
)

daily_counts["INDEX_SIZE_CHECK"] = np.where(
    daily_counts["DATE"] < pd.Timestamp("2021-09-30"),
    daily_counts["EXPECTED_CONSTITUENTS"].eq(10),
    daily_counts["EXPECTED_CONSTITUENTS"].eq(20)
)

if not daily_counts["INDEX_SIZE_CHECK"].all():

    bad = daily_counts[
        ~daily_counts["INDEX_SIZE_CHECK"]
    ].copy()

    bad_file = (
        OUTPUT_DIR
        / "ERROR_UNEXPECTED_DAILY_INDEX_SIZE.csv"
    )

    bad.to_csv(
        bad_file,
        index=False,
        date_format="%Y-%m-%d"
    )

    raise RuntimeError(
        "Daily historical membership count is not 10 before "
        "30-Sep-2021 / 20 thereafter.\n"
        f"Saved to:\n{bad_file}"
    )


interval_coverage = (
    expected_panel
    .groupby(
        [
            "MEMBERSHIP_PERIOD_ID",
            "COMPANY_ID",
            "MEMBERSHIP_SYMBOL",
            "MEMBERSHIP_COMPANY",
            "MEMBERSHIP_START_DATE",
            "MEMBERSHIP_END_DATE",
        ],
        as_index=False,
        dropna=False
    )
    .agg(
        EXPECTED_TRADING_DATES=(
            "DATE",
            "size"
        ),
        PRICE_ROWS_AVAILABLE=(
            "PRICE_ROW_AVAILABLE",
            "sum"
        )
    )
)

interval_coverage["MISSING_PRICE_ROWS"] = (
    interval_coverage["EXPECTED_TRADING_DATES"]
    -
    interval_coverage["PRICE_ROWS_AVAILABLE"]
)

interval_coverage["PRICE_COVERAGE_PCT"] = np.where(
    interval_coverage["EXPECTED_TRADING_DATES"] > 0,
    100
    *
    interval_coverage["PRICE_ROWS_AVAILABLE"]
    /
    interval_coverage["EXPECTED_TRADING_DATES"],
    np.nan
)


non_member_in_coverage = prices[
    prices["MEMBERSHIP_KNOWN_PERIOD"]
    &
    ~prices["IN_NIFTY_PHARMA"]
].copy()

non_member_summary = (
    non_member_in_coverage
    .groupby(
        ["COMPANY_ID", "SYMBOL"],
        as_index=False
    )
    .agg(
        NON_MEMBER_PRICE_ROWS=(
            "DATE",
            "size"
        ),
        FIRST_NON_MEMBER_PRICE_DATE=(
            "DATE",
            "min"
        ),
        LAST_NON_MEMBER_PRICE_DATE=(
            "DATE",
            "max"
        )
    )
    .sort_values(
        [
            "NON_MEMBER_PRICE_ROWS",
            "COMPANY_ID"
        ],
        ascending=[
            False,
            True
        ]
    )
)


membership_period_counts = (
    membership
    .groupby(
        ["START_DATE", "END_DATE"],
        as_index=False
    )
    .agg(
        CONSTITUENTS=(
            "MEMBERSHIP_COMPANY_ID",
            "nunique"
        )
    )
    .sort_values("START_DATE")
)


symbol_mapping_audit = (
    membership[
        [
            "MEMBERSHIP_PERIOD_ID",
            "MEMBERSHIP_SYMBOL_SOURCE",
            "MEMBERSHIP_SYMBOL",
            "MEMBERSHIP_COMPANY_ID",
            "COMPANY",
            "START_DATE",
            "END_DATE",
            "SYMBOL_NORMALIZATION_NOTE",
        ]
    ]
    .copy()
)


member_only = prices[
    prices["IN_NIFTY_PHARMA"]
].copy()


# Membership marking must preserve the input row count exactly.
if len(prices) != INPUT_PRICE_ROWS:
    raise RuntimeError(
        "Row count changed unexpectedly during membership marking."
    )

if prices[
    ["DATE", "COMPANY_ID"]
].duplicated().any():
    raise RuntimeError(
        "DATE + COMPANY_ID duplicates appeared after membership marking."
    )

if len(member_only) != int(
    prices["IN_NIFTY_PHARMA"].sum()
):
    raise RuntimeError(
        "Member-only row count does not equal IN_NIFTY_PHARMA flag count."
    )


prices = (
    prices
    .sort_values("_SOURCE_ROW_ORDER")
    .drop(columns=["_SOURCE_ROW_ORDER"])
    .reset_index(drop=True)
)

member_only = (
    member_only
    .sort_values(
        ["DATE", "COMPANY_ID"]
    )
    .drop(
        columns=["_SOURCE_ROW_ORDER"],
        errors="ignore"
    )
    .reset_index(drop=True)
)


audit_summary = pd.DataFrame(
    [
        {
            "CHECK": "Input total-return rows",
            "VALUE": INPUT_PRICE_ROWS
        },
        {
            "CHECK": "Output flagged rows",
            "VALUE": len(prices)
        },
        {
            "CHECK": "Membership-history rows",
            "VALUE": len(membership)
        },
        {
            "CHECK": "Membership-history start",
            "VALUE": str(
                EXPECTED_MEMBERSHIP_START.date()
            )
        },
        {
            "CHECK": "Membership-history end",
            "VALUE": str(
                EXPECTED_MEMBERSHIP_END.date()
            )
        },
        {
            "CHECK": "Unique total-return trading dates",
            "VALUE": prices["DATE"].nunique()
        },
        {
            "CHECK": "Unique stable company identities",
            "VALUE": prices["COMPANY_ID"].nunique()
        },
        {
            "CHECK":
                "Rows inside known membership-history period",
            "VALUE":
                int(
                    prices[
                        "MEMBERSHIP_KNOWN_PERIOD"
                    ].sum()
                )
        },
        {
            "CHECK":
                "Point-in-time member price rows",
            "VALUE":
                len(member_only)
        },
        {
            "CHECK":
                "Non-member price rows inside coverage period",
            "VALUE":
                len(non_member_in_coverage)
        },
        {
            "CHECK":
                "Price rows before membership-history start",
            "VALUE":
                int(
                    (
                        prices["DATE"]
                        <
                        EXPECTED_MEMBERSHIP_START
                    ).sum()
                )
        },
        {
            "CHECK":
                "Expected member-date observations",
            "VALUE":
                len(expected_panel)
        },
        {
            "CHECK":
                "Missing expected member price rows",
            "VALUE":
                len(missing_member_prices)
        },
        {
            "CHECK":
                "Trading dates with >=1 missing member price",
            "VALUE":
                int(
                    daily_counts[
                        "MISSING_MEMBER_PRICE_ROWS"
                    ].gt(0).sum()
                )
        },
        {
            "CHECK":
                "Maximum missing member prices on one date",
            "VALUE":
                int(
                    daily_counts[
                        "MISSING_MEMBER_PRICE_ROWS"
                    ].max()
                )
        },
        {
            "CHECK":
                "Rows matching >1 membership interval",
            "VALUE":
                int(
                    prices[
                        "MEMBERSHIP_MATCH_COUNT"
                    ].gt(1).sum()
                )
        },
        {
            "CHECK":
                "Historical member company IDs absent from price universe",
            "VALUE":
                len(membership_not_in_prices)
        },
        {
            "CHECK":
                "Daily index-size validation failures",
            "VALUE":
                int(
                    (
                        ~daily_counts[
                            "INDEX_SIZE_CHECK"
                        ]
                    ).sum()
                )
        },
    ]
)


MAIN_FLAGGED_FILE = (
    OUTPUT_DIR
    / "NIFTY_PHARMA_POINT_IN_TIME_FLAGGED_2016_2026.csv"
)

MEMBER_ONLY_FILE = (
    OUTPUT_DIR
    / "NIFTY_PHARMA_POINT_IN_TIME_MEMBERS_ONLY_2016_2026.csv"
)

prices.to_csv(
    MAIN_FLAGGED_FILE,
    index=False,
    date_format="%Y-%m-%d"
)

member_only.to_csv(
    MEMBER_ONLY_FILE,
    index=False,
    date_format="%Y-%m-%d"
)

audit_summary.to_csv(
    OUTPUT_DIR
    / "00_POINT_IN_TIME_AUDIT_SUMMARY.csv",
    index=False
)

daily_counts.to_csv(
    OUTPUT_DIR
    / "01_DAILY_MEMBERSHIP_COUNTS.csv",
    index=False,
    date_format="%Y-%m-%d"
)

interval_coverage.to_csv(
    OUTPUT_DIR
    / "02_MEMBERSHIP_INTERVAL_COVERAGE.csv",
    index=False,
    date_format="%Y-%m-%d"
)

missing_member_prices.to_csv(
    OUTPUT_DIR
    / "03_MISSING_EXPECTED_MEMBER_PRICES.csv",
    index=False,
    date_format="%Y-%m-%d"
)

symbol_mapping_audit.to_csv(
    OUTPUT_DIR
    / "04_MEMBERSHIP_SYMBOL_MAPPING_AUDIT.csv",
    index=False,
    date_format="%Y-%m-%d"
)

non_member_summary.to_csv(
    OUTPUT_DIR
    / "05_NON_MEMBER_ROWS_BY_COMPANY.csv",
    index=False,
    date_format="%Y-%m-%d"
)

membership_period_counts.to_csv(
    OUTPUT_DIR
    / "06_MEMBERSHIP_PERIOD_COUNTS.csv",
    index=False,
    date_format="%Y-%m-%d"
)


readme = f"""
NIFTY PHARMA POINT-IN-TIME UNIVERSE

Input total-return file:
{TOTAL_RETURN_FILE}

Input audited membership file:
{MEMBERSHIP_FILE}

Membership coverage:
{EXPECTED_MEMBERSHIP_START.date()} through {EXPECTED_MEMBERSHIP_END.date()}

Main outputs:
- NIFTY_PHARMA_POINT_IN_TIME_FLAGGED_2016_2026.csv
  Keeps all total-return rows and adds point-in-time membership fields.

- NIFTY_PHARMA_POINT_IN_TIME_MEMBERS_ONLY_2016_2026.csv
  Contains only stock/date rows where the stock was actually a
  NIFTY Pharma constituent on that date.

Important rules:
1. Membership intervals are inclusive of START_DATE and END_DATE.
2. AJANTAPHARM in the membership file is normalized to NSE symbol AJANTPHARM.
3. CADILAHC and ZYDUSLIFE are matched through stable COMPANY_ID ZYDUS,
   while actual historical price SYMBOL is preserved.
4. No missing constituent price is forward-filled.
5. Raw NSE fields and total-return fields are not altered.
6. Membership before 2016-08-01 is unknown from this audited membership
   file and is not inferred.
7. This stage applies NO liquidity, price, shortability/F&O, observation-
   count, formation-window, pair-selection, or OOS filters.

Next stage:
Formation-only investability filters.
"""

(
    OUTPUT_DIR
    / "README_POINT_IN_TIME_UNIVERSE.txt"
).write_text(
    readme,
    encoding="utf-8"
)


print("\n")
print("=" * 100)
print("POINT-IN-TIME NIFTY PHARMA MEMBERSHIP COMPLETE")
print("=" * 100)

print("\nAUDIT SUMMARY")
print("-" * 100)
print(
    audit_summary.to_string(
        index=False
    )
)

print("\nMEMBERSHIP PERIOD COUNTS")
print("-" * 100)
print(
    membership_period_counts.to_string(
        index=False
    )
)

print("\nOutputs saved in:")
print(OUTPUT_DIR)

print("\nMain flagged dataset:")
print(MAIN_FLAGGED_FILE)

print("\nMember-only dataset:")
print(MEMBER_ONLY_FILE)

print("\nFiles to inspect next:")
print(
    OUTPUT_DIR
    / "00_POINT_IN_TIME_AUDIT_SUMMARY.csv"
)
print(
    OUTPUT_DIR
    / "01_DAILY_MEMBERSHIP_COUNTS.csv"
)
print(
    OUTPUT_DIR
    / "02_MEMBERSHIP_INTERVAL_COVERAGE.csv"
)
print(
    OUTPUT_DIR
    / "03_MISSING_EXPECTED_MEMBER_PRICES.csv"
)

print("\n")
print("=" * 100)
print(
    "NO FORWARD FILL, INVESTABILITY FILTER, PAIR SELECTION, "
    "OR OOS ANALYSIS HAS BEEN PERFORMED."
)
print("=" * 100)
