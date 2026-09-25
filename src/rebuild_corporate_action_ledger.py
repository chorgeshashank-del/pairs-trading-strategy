"""Reconstructed parser for the frozen 2016-2026 corporate-action source.

Original interactive code was not saved. This reproduces its conservative
classification, including source-text typos; it is not a new economic review.
Ambiguous terms remain flagged. Final reviewed overrides remain in
02_nifty_pharma_total_return_construction.py. No network access is required.
"""
from pathlib import Path
import os
import re
import pandas as pd

ROOT = Path(os.environ.get('PAIR_TRADING_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
ALIASES = {'CADILAHC': 'ZYDUS', 'ZYDUSLIFE': 'ZYDUS'}
TREATMENTS = {
 'CASH_DIVIDEND': 'LATER_INCLUDE_CASH_DIVIDEND_IN_TOTAL_RETURN',
 'ADMIN_ONLY': 'NO_PRICE_OR_TOTAL_RETURN_ADJUSTMENT',
 'BUYBACK': 'NO_AUTOMATIC_ADJUSTMENT_REVIEW_TERMS',
 'BONUS_ISSUE': 'LATER_NEUTRALIZE_SAME_STOCK_UNIT_CHANGE',
 'SPLIT_CONSOLIDATION': 'LATER_NEUTRALIZE_SAME_STOCK_UNIT_CHANGE',
 'RIGHTS_ISSUE': 'MANUAL_RIGHTS_TERMS_AND_THEORETICAL_ADJUSTMENT',
 'DEMERGER_SPINOFF': 'STRUCTURAL_BREAK_EVENT_SPECIFIC_VALUE_TREATMENT',
 'SCHEME_OF_ARRANGEMENT': 'STRUCTURAL_BREAK_MANUAL_REVIEW',
 'OTHER_MANUAL_REVIEW': 'MANUAL_REVIEW_UNCLASSIFIED',
}
REASONS = {
 'BUYBACK': 'BUYBACK_IS_NOT_AUTOMATIC_TOTAL_RETURN_DISTRIBUTION',
 'RIGHTS_ISSUE': 'RIGHTS_REQUIRE_RATIO_SUBSCRIPTION_PRICE_AND_EVENT_REVIEW',
 'DEMERGER_SPINOFF': 'STRUCTURAL_BREAK_REQUIRES_ENTITLEMENT_AND_SPINOFF_VALUATION',
 'SCHEME_OF_ARRANGEMENT': 'SCHEME_TYPE_NOT_SAFE_TO_INFER_FROM_PURPOSE_TEXT',
 'OTHER_MANUAL_REVIEW': 'PURPOSE_NOT_CLASSIFIED_CONFIDENTLY',
}
TAG_RULES = {'AGM': r'ANNUAL GENERAL MEETING', 'BOOK_CLOSURE': r'BOOK CLOSURE',
 'DIVIDEND': r'DIVIDEND', 'BUYBACK': r'BUY\s*BACK', 'BONUS': r'BONUS',
 'SPLIT': r'SPLIT|CONSOLIDATION', 'RIGHTS': r'RIGHTS', 'DEMERGER': r'DEMERGER',
 'SCHEME_OF_ARRANGEMENT': r'SCHEME OF ARRANGEMENT'}
CLASSES = [('DEMERGER','DEMERGER_SPINOFF'),('SCHEME_OF_ARRANGEMENT','SCHEME_OF_ARRANGEMENT'),
 ('SPLIT','SPLIT_CONSOLIDATION'),('BONUS','BONUS_ISSUE'),('RIGHTS','RIGHTS_ISSUE'),
 ('DIVIDEND','CASH_DIVIDEND'),('BUYBACK','BUYBACK'),('AGM','ADMIN_ONLY')]
NUM_FIELDS = ['DIVIDEND_RUPEES_PER_SHARE','BONUS_NEW_SHARES','BONUS_EXISTING_SHARES',
 'BONUS_SHARE_MULTIPLIER','SPLIT_OLD_FACE_VALUE','SPLIT_NEW_FACE_VALUE',
 'SPLIT_SHARE_MULTIPLIER','RIGHTS_NEW_SHARES','RIGHTS_EXISTING_SHARES']


def main():
    folder = ROOT / 'nse_pharma_corporate_action_ledger'
    source = pd.read_csv(folder / 'NIFTY_PHARMA_CORPORATE_ACTIONS_2016_2026.csv')
    prices = pd.read_csv(ROOT / 'nse_pharma_clean_base' / 'NIFTY_PHARMA_EQ_BASE_2016_2026.csv')
    if prices.duplicated(['DATE','COMPANY_ID']).any():
        raise ValueError('Duplicate price keys')
    closes = prices.set_index(['DATE','COMPANY_ID']).CLOSE
    records = []
    for i, row in source.iterrows():
        if row.SERIES != 'EQ':
            raise ValueError('Reconstructed parser requires EQ-only source')
        r = row.to_dict()
        r['LEDGER_ID'] = f'CA{i+1:04d}'
        r['SOURCE_ROW_NUMBER'] = i+1
        r['COMPANY_ID'] = ALIASES.get(row.SYMBOL,row.SYMBOL)
        r['PRICE_SYMBOL'] = 'CADILAHC' if r['COMPANY_ID']=='ZYDUS' and row.EX_DATE<'2022-03-07' else row.SYMBOL
        p = re.sub(r'\s+',' ',str(row.PURPOSE).strip().upper())
        r['PURPOSE_NORMALIZED'] = p
        tags = sorted(tag for tag, pattern in TAG_RULES.items() if re.search(pattern,p))
        r['ACTION_TAGS'] = ';'.join(tags)
        cls = next((cls for tag,cls in CLASSES if tag in tags),'OTHER_MANUAL_REVIEW')
        r['PRIMARY_CLASS'] = cls
        r['TREATMENT_CODE'] = TREATMENTS[cls]
        r['RESEARCH_RELEVANCE'] = 'EQ_SERIES'
        r['STRUCTURAL_BREAK_FLAG'] = cls in ['DEMERGER_SPINOFF','SCHEME_OF_ARRANGEMENT']
        special = 'SPECIAL' in p and 'DIVIDEND' in tags
        r['SPECIAL_DIVIDEND_FLAG'] = special
        reason = REASONS.get(cls,'')
        if special: reason = 'SPECIAL_DIVIDEND_VERIFY_AND_AVOID_DOUBLE_COUNT'
        r['REVIEW_REASON'] = reason
        r['MANUAL_REVIEW_REQUIRED'] = bool(reason)
        r['AUTO_TERMS_READY'] = False
        for field in NUM_FIELDS: r[field] = float('nan')
        for field in ['DIVIDEND_PARSE_STATUS','BONUS_PARSE_STATUS','SPLIT_PARSE_STATUS','RIGHTS_RATIO_PARSE_STATUS']:
            r[field] = 'NOT_APPLICABLE'
        if cls == 'CASH_DIVIDEND':
            amounts = re.findall(r'\bR(?:S|E)\.?\s*(\d+(?:\.\d+)?)',p)
            # The historical parser counted distinct amounts. Final special-
            # dividend overrides in the total-return stage resolve combined payments.
            amounts = list(dict.fromkeys(amounts))
            if len(amounts)==1:
                r['DIVIDEND_RUPEES_PER_SHARE'] = float(amounts[0])
                r['DIVIDEND_PARSE_STATUS'] = 'PARSED'
                r['AUTO_TERMS_READY'] = True
            elif len(amounts)>1:
                r['DIVIDEND_PARSE_STATUS'] = 'AMBIGUOUS_MULTIPLE_AMOUNTS'
            else: raise ValueError(f'Unrecognized dividend terms: {p}')
        if cls in ['BONUS_ISSUE','RIGHTS_ISSUE']:
            match = re.search(r'(\d+)\s*:\s*(\d+)',p)
            if not match: raise ValueError(f'Unrecognized ratio: {p}')
            a,b = map(float,match.groups())
            prefix = 'BONUS' if cls=='BONUS_ISSUE' else 'RIGHTS'
            r[prefix+'_NEW_SHARES'],r[prefix+'_EXISTING_SHARES'] = a,b
            r['BONUS_PARSE_STATUS' if prefix=='BONUS' else 'RIGHTS_RATIO_PARSE_STATUS'] = 'PARSED'
            if prefix=='BONUS':
                r['BONUS_SHARE_MULTIPLIER'] = (a+b)/b
                r['AUTO_TERMS_READY'] = True
        if cls=='SPLIT_CONSOLIDATION':
            amounts = re.findall(r'\bR(?:S|E)\.?\s*(\d+(?:\.\d+)?)',p)
            if len(amounts)!=2: raise ValueError(f'Unrecognized split: {p}')
            a,b=map(float,amounts)
            r['SPLIT_OLD_FACE_VALUE'],r['SPLIT_NEW_FACE_VALUE'] = a,b
            r['SPLIT_SHARE_MULTIPLIER'] = a/b
            r['SPLIT_PARSE_STATUS'] = 'PARSED'
            r['AUTO_TERMS_READY'] = True
        key=(row.EX_DATE,r['COMPANY_ID'])
        r['EQ_PRICE_ON_EX_DATE'] = key in closes.index
        r['EQ_CLOSE_ON_EX_DATE'] = closes.get(key,float('nan'))
        records.append(r)
    result=pd.DataFrame(records).reindex(columns=COLUMNS)
    result.to_csv(folder/'CORPORATE_ACTION_TREATMENT_LEDGER_2016_2026.csv',index=False)
    print(f'Rebuilt {len(result)} corporate-action records.')


# Column order preserves compatibility with the published ledger.
COLUMNS = ['LEDGER_ID', 'SOURCE_ROW_NUMBER', 'EX_DATE', 'RECORD_DATE', 'COMPANY_ID', 'SYMBOL', 'PRICE_SYMBOL', 'SERIES', 'PURPOSE', 'PURPOSE_NORMALIZED', 'ACTION_TAGS', 'PRIMARY_CLASS', 'TREATMENT_CODE', 'RESEARCH_RELEVANCE', 'STRUCTURAL_BREAK_FLAG', 'SPECIAL_DIVIDEND_FLAG', 'AUTO_TERMS_READY', 'MANUAL_REVIEW_REQUIRED', 'REVIEW_REASON', 'DIVIDEND_RUPEES_PER_SHARE', 'DIVIDEND_PARSE_STATUS', 'BONUS_NEW_SHARES', 'BONUS_EXISTING_SHARES', 'BONUS_SHARE_MULTIPLIER', 'BONUS_PARSE_STATUS', 'SPLIT_OLD_FACE_VALUE', 'SPLIT_NEW_FACE_VALUE', 'SPLIT_SHARE_MULTIPLIER', 'SPLIT_PARSE_STATUS', 'RIGHTS_NEW_SHARES', 'RIGHTS_EXISTING_SHARES', 'RIGHTS_RATIO_PARSE_STATUS', 'EQ_PRICE_ON_EX_DATE', 'EQ_CLOSE_ON_EX_DATE', 'COMPANY', 'NSE_COMPANY_NAME', 'FACE_VALUE', 'BOOK_CLOSURE_START', 'BOOK_CLOSURE_END', 'PAYMENT_DATE', 'REMARKS']

if __name__ == "__main__":
    main()
