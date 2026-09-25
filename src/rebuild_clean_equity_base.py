"""Reconstructed cleaning stage; original interactive program was not saved.

Uses repaired yearly extracts plus four cached NSE Muhurat reports. This does
not recreate the earlier downloads or repairs of the yearly extracts. No price
adjustment, forward filling, or index membership filtering is performed here.
Run from any directory. PAIR_TRADING_PROJECT_ROOT optionally overrides the root.
"""
from pathlib import Path
import os
import zipfile
import pandas as pd

ROOT = Path(os.environ.get('PAIR_TRADING_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
ALIASES = {'CADILAHC': 'ZYDUS', 'ZYDUSLIFE': 'ZYDUS'}
NUMERIC = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREV_CLOSE',
           'TOTAL_TRADED_QTY', 'TOTAL_TRADED_VALUE', 'TOTAL_TRADES']
COLS = ['DATE', 'COMPANY_ID', 'SYMBOL', 'COMPANY', 'SERIES', 'ISIN',
        *NUMERIC, 'SOURCE_FORMAT', 'SOURCE_YEAR']
SESSIONS = [('2016-10-30', 'cm30OCT2016bhav.csv.zip'),
            ('2019-10-27', 'cm27OCT2019bhav.csv.zip'),
            ('2020-11-14', 'cm14NOV2020bhav.csv.zip'),
            ('2023-11-12', 'cm12NOV2023bhav.csv.zip')]


def main():
    frames = []
    for year in range(2016, 2027):
        path = ROOT / 'nse_pharma_2016_2026_REPAIRED' / str(year) / f'NIFTY_PHARMA_NSE_{year}_REPAIRED.csv'
        frames.append(pd.read_csv(path).assign(SOURCE_YEAR=year))
    base = pd.concat(frames, ignore_index=True)
    for col in ['SYMBOL', 'SERIES', 'ISIN']:
        base[col] = base[col].astype('string').str.strip().str.upper()
    base = base.loc[base.SERIES.eq('EQ')].copy()
    base['COMPANY_ID'] = base.SYMBOL.replace(ALIASES)
    base['DATE'] = pd.to_datetime(base.DATE, errors='raise')
    for col in NUMERIC:
        base[col] = pd.to_numeric(base[col], errors='raise')
    base = base[COLS]
    if base.duplicated(['DATE', 'COMPANY_ID']).any():
        raise ValueError('Duplicate company/date keys in yearly inputs')
    names = base.groupby('SYMBOL')['COMPANY'].agg(lambda x: x.dropna().mode().iloc[0])
    symbols = set(base.SYMBOL)
    rename = {'PREVCLOSE': 'PREV_CLOSE', 'TOTTRDQTY': 'TOTAL_TRADED_QTY',
              'TOTTRDVAL': 'TOTAL_TRADED_VALUE', 'TOTALTRADES': 'TOTAL_TRADES'}
    appended = 0
    for date, filename in SESSIONS:
        path = ROOT / 'nse_pharma_muhurat_repair' / 'raw_nse_zip' / filename
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith('.csv')]
            if len(members) != 1:
                raise ValueError(f'Expected one CSV in {path}')
            with archive.open(members[0]) as handle:
                raw = pd.read_csv(handle)
        raw.columns = raw.columns.str.strip().str.upper()
        for col in ['SYMBOL', 'SERIES', 'ISIN']:
            raw[col] = raw[col].astype('string').str.strip().str.upper()
        raw = raw.loc[raw.SYMBOL.isin(symbols) & raw.SERIES.eq('EQ')].copy()
        if raw.empty or not pd.to_datetime(raw.TIMESTAMP).eq(pd.Timestamp(date)).all():
            raise ValueError(f'Missing observations or unexpected session date: {path}')
        new = raw.rename(columns=rename)
        new['DATE'] = pd.Timestamp(date)
        new['COMPANY_ID'] = new.SYMBOL.replace(ALIASES)
        new['COMPANY'] = new.SYMBOL.map(names)
        new['SOURCE_FORMAT'] = 'LEGACY_NSE_MUHURAT_ARCHIVE_REPAIR'
        # Preserve the saved dataset convention: repair rows have no SOURCE_YEAR.
        new['SOURCE_YEAR'] = float('nan')
        for col in NUMERIC:
            new[col] = pd.to_numeric(new[col], errors='raise')
        new = new[COLS]
        existing = pd.MultiIndex.from_frame(base[['DATE', 'COMPANY_ID']])
        new = new.loc[~pd.MultiIndex.from_frame(new[['DATE', 'COMPANY_ID']]).isin(existing)]
        appended += len(new)
        base = pd.concat([base, new], ignore_index=True)
    if base.duplicated(['DATE', 'COMPANY_ID']).any():
        raise ValueError('Duplicate company/date keys after repairs')
    if base[NUMERIC].isna().any().any() or (base.CLOSE <= 0).any():
        raise ValueError('Missing numerical fields or nonpositive close')
    base = base.sort_values(['DATE', 'COMPANY_ID']).reset_index(drop=True)
    output = ROOT / 'nse_pharma_clean_base' / 'NIFTY_PHARMA_EQ_BASE_2016_2026.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output, index=False, date_format='%Y-%m-%d')
    print(f'Wrote {len(base):,} rows, {len(base.columns)} columns; appended {appended} Muhurat rows: {output}')


if __name__ == '__main__':
    main()
