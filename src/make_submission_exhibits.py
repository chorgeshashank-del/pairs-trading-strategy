from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
EXHIBITS=ROOT/'exhibits'
EXHIBITS.mkdir(exist_ok=True)
np.random.seed(20260830)

cost=pd.read_csv(RESULTS/'FINAL_COST_SENSITIVITY.csv')
plt.figure(figsize=(6.5,3.2))
plt.plot(cost['COST_MULTIPLIER'],100*cost['SSD_TOTAL_RETURN'],marker='o',label='SSD')
plt.plot(cost['COST_MULTIPLIER'],100*cost['ENGLE_GRANGER_TOTAL_RETURN'],marker='o',label='Engle-Granger')
plt.axhline(0,linewidth=.8)
plt.xlabel('Cost multiplier'); plt.ylabel('Full-period total return (%)')
plt.xticks([0,.5,1,2]); plt.legend(frameon=False); plt.tight_layout()
plt.savefig(EXHIBITS/'figure_1_cost_sensitivity.png',dpi=220,bbox_inches='tight'); plt.close()

rob=pd.read_csv(RESULTS/'FINAL_ROBUSTNESS_COMPARISON.csv')
period=pd.read_csv(RESULTS/'FINAL_FULL_DEV_OOS_COMPARISON.csv')
r=rob[rob['ROBUSTNESS_CATEGORY'].eq('FORMATION_WINDOW_LENGTH')].copy()
r['MONTHS']=r['ROBUSTNESS_VARIANT'].map({'FORMATION_6_MONTHS':6,'FORMATION_18_MONTHS':18})
primary=pd.DataFrame([{'MONTHS':12,'SSD_TOTAL_RETURN':float(period.loc[period.PERIOD.eq('DEVELOPMENT'),'SSD_TOTAL_RETURN'].iloc[0]),'ENGLE_GRANGER_TOTAL_RETURN':float(period.loc[period.PERIOD.eq('DEVELOPMENT'),'ENGLE_GRANGER_TOTAL_RETURN'].iloc[0])}])
r2=pd.concat([r[['MONTHS','SSD_TOTAL_RETURN','ENGLE_GRANGER_TOTAL_RETURN']],primary],ignore_index=True).sort_values('MONTHS')
x=np.arange(len(r2)); w=.36
plt.figure(figsize=(6.5,3.2))
plt.bar(x-w/2,100*r2['SSD_TOTAL_RETURN'],width=w,label='SSD')
plt.bar(x+w/2,100*r2['ENGLE_GRANGER_TOTAL_RETURN'],width=w,label='Engle-Granger')
plt.axhline(0,linewidth=.8)
plt.xticks(x,[f'{int(m)}m' for m in r2['MONTHS']]); plt.xlabel('Formation-window length'); plt.ylabel('Development-period total return (%)')
plt.legend(frameon=False); plt.tight_layout()
plt.savefig(EXHIBITS/'figure_2_formation_window_sensitivity.png',dpi=220,bbox_inches='tight'); plt.close()
print('Reproduced memo figures in', EXHIBITS)
