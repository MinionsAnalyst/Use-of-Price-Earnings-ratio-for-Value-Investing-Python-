#!/usr/bin/env python
# coding: utf-8

# 1 Import libraries and Set path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as scs
from scipy.stats.mstats import winsorize
from scipy.stats.mstats import gmean
from tabulate import tabulate


# 2 Set path of my sub-directory
from pathlib import Path
# key in your own file path below
myfolder = Path('key in your own file path here')


# 3 Set up files to write output and charts
from matplotlib.backends.backend_pdf import PdfPages
outfile = open('output.txt', 'w')
chartfile = PdfPages('chart-retreg.pdf')


# Stock returns data

# 4 Read Compustat monthly stock returns data 
df1 = pd.read_csv(my-folder / 'stock-returns.csv', parse_dates = ['datadate'])
df1 = df1.sort_values(by=['gvkey','datadate'])
df1 = df1.dropna()


# 5 Create portfolio formation year (pfy) variable, where
#      pfy = current year for Jul-Dec dates and previous year for Jan-Jun dates.
#      This is to facilitate compounding returns over Jul-Jun by pfy later below.
df1['year'], df1['month'] = df1['datadate'].dt.year, df1['datadate'].dt.month
df1['pfy'] = np.where(df1.month > 6, df1.year, df1.year - 1)


# 6 Compute monthly return compounding factor (1+monthly return)
# trt1m is the monthly return, express as percentage, need to convert to % by / 100
df1['mretfactor'] = 1 + df1.trt1m/100


df1 = df1.sort_values(by=['gvkey','pfy'])
df2 = df1[['gvkey', 'conm', 'datadate', 'pfy', 'mretfactor']]


# 7 Compound monthly returns to get annual returns at end-June of each pfy,
#      ensuring only firm-years with 12 mths of return data from Jul-Jun are selected.
df2['yret'] = df2.groupby(['gvkey', 'pfy'])['mretfactor'].cumprod() - 1
df3 = df2.groupby(['gvkey', 'pfy']).nth(11)
df3['yret'] = winsorize(df3['yret'], limits=[0.025,0.025])


df3 = df3.drop(['mretfactor'], axis=1)    # "axis=1" means to drop column


# Accounting data

# 8 Read Compustat accounting data
df4 = pd.read_csv(myfolder / 'accounting-data2.csv', parse_dates = ['datadate'])
df4 = df4.sort_values(by=['gvkey','datadate'])


# 9 Create portfolio formation year (pfy) variable, portfolio formation in April where
#      pfy = current year for Jan-Mar year-end dates and next year for Apr-Dec year-end dates.
#      This is to facilitate compounding returns over July-June by pfy below.
#      dt.year is pandas method to extract year from 'datadate' variable
#      dt.month is pandas method to extract month from 'datadate' variable
df4['year'], df4['month'] = df4['datadate'].dt.year, df4['datadate'].dt.month
df4['pfy'] = np.where(df4.month < 4, df4.year, df4.year + 1)


# 10 Compute accounting variables from Compustat data, keep relevant variables, delete missing values
# Profitability
df4['ROA'] = df4['ni'] / df4['at']
df4['ROA_prev'] = df4.groupby('gvkey')['ROA'].shift(1)
# Leverage
df4['Leverage_ratio'] = df4['dltt'] / df4['seq']
df4['Leverage_ratio_prev'] = df4.groupby('gvkey')['Leverage_ratio'].shift(1)
df4['Current_ratio'] = df4['act'] / df4['lct']
df4['Current_ratio_prev'] = df4.groupby('gvkey')['Current_ratio'].shift(1)
df4['csho_prev'] = df4.groupby('gvkey')['csho'].shift(1)
df4['Shares_issued'] = df4['csho'] - df4['csho_prev']
# Operating
df4['GP_margin'] = df4['gp'] / df4['revt']
df4['GP_margin_prev'] = df4.groupby('gvkey')['GP_margin'].shift(1)
df4['at_prev'] = df4.groupby('gvkey')['at'].shift(1)
df4['at_average']= (df4['at'] + df4['at_prev'])/2
df4['Asset_TO'] = df4['revt'] / df4['at_average']
df4['Asset_TO_prev'] = df4.groupby('gvkey')['Asset_TO'].shift(1)

df4['GP_profitability'] = df4['gp']/df4['at']

df4 = df4[['ib', 'gvkey', 'pfy', 'ni', 'oancf', 'mkvalt', 'gsector', 'ROA', 'ROA_prev', 'Leverage_ratio', 'Leverage_ratio_prev', 'Current_ratio', 
           'Current_ratio_prev', 'csho_prev', 'Shares_issued', 'GP_margin', 'GP_margin_prev', 'at_prev', 
           'at_average', 'Asset_TO', 'Asset_TO_prev', 'GP_profitability' ]]
df4 = df4[np.isfinite(df4)]
df4 = df4.dropna()


# 11 EDA before winsorize
dfeda = df4[['ROA', 'ROA_prev', 'oancf', 'ib', 'Leverage_ratio', 'Current_ratio', 'Shares_issued',
            'GP_margin', 'Asset_TO', 'mkvalt', 'ni']]
dfeda['PE'] = dfeda['mkvalt'] / dfeda['ni']
dfeda['CROA'] = dfeda['ROA'] - dfeda['ROA_prev']
dfeda['Cquality'] = np.where(dfeda['oancf']> dfeda['ib'], 1, 0)
dfeda2 = dfeda[['ROA', 'oancf', 'CROA', 'Cquality', 'Leverage_ratio', 'Current_ratio', 'Shares_issued',
            'GP_margin', 'Asset_TO', 'PE']]
print('EDA before winsorize \n\n', dfeda2.describe(), '\n'*5, file=outfile)


# 12 Winsorize variables at 2.5% of left and right tails
for var in ['ib', 'ni', 'oancf', 'mkvalt', 'ROA', 'ROA_prev', 'Leverage_ratio', 'Leverage_ratio_prev', 'Current_ratio', 
            'Current_ratio_prev', 'csho_prev', 'Shares_issued', 'GP_margin', 'GP_margin_prev', 'at_prev', 
            'at_average', 'Asset_TO', 'Asset_TO_prev', 'GP_profitability']:
   df4[var] = winsorize(df4[var], limits=[0.025,0.025])


# 13 EDA after winsorize
dfeda3 = df4[['ROA', 'ROA_prev', 'oancf', 'ib', 'Leverage_ratio', 'Current_ratio', 'Shares_issued',
            'GP_margin', 'Asset_TO', 'mkvalt', 'ni']]
dfeda3['PE'] = dfeda3['mkvalt'] / dfeda3['ni']
dfeda3['CROA'] = dfeda3['ROA'] - dfeda3['ROA_prev']
dfeda3['Cquality'] = np.where(dfeda3['oancf']> dfeda3['ib'], 1, 0)
dfeda4 = dfeda3[['ROA', 'oancf', 'CROA', 'Cquality', 'Leverage_ratio', 'Current_ratio', 'Shares_issued',
            'GP_margin', 'Asset_TO', 'PE']]

print('EDA after winsorize \n\n', dfeda4.describe(), '\n'*5, file=outfile)


# Merge Stock returns data with Accounting data

# 14 Merge accounting dataset (df4) with returns dataset (df3)
#        "inner" means to merge only observations that have data in BOTH datasets
df5 = pd.merge(df3, df4, how='inner', on=['gvkey', 'pfy'])
df5 = df5[['ib', 'gvkey', 'conm', 'pfy', 'yret', 'ni', 'mkvalt', 'oancf', 'gsector', 'ROA', 'ROA_prev', 'Leverage_ratio', 'Leverage_ratio_prev', 'Current_ratio', 
           'Current_ratio_prev', 'csho_prev', 'Shares_issued', 'GP_margin', 'GP_margin_prev', 'at_prev', 
           'at_average', 'Asset_TO', 'Asset_TO_prev', 'GP_profitability']]


# Compute F-score

# 15 Compute 9 F-score ratios
# Profitability
df5['F_income'] = np.where(df5['ROA']> 0, 1, 0)
df5['F_opcash'] = np.where(df5['oancf']> 0, 1, 0)
df5['F_ROA'] = np.where(df5['ROA']>df5['ROA_prev'], 1, 0)
df5['F_quality'] = np.where(df5['oancf']> df5['ib'], 1, 0)
# Leverage
df5['F_leverage'] = np.where(df5['Leverage_ratio']< df5['Leverage_ratio_prev'], 1, 0)
df5['F_currentratio'] = np.where(df5['Current_ratio']> df5['Current_ratio_prev'], 1, 0)
df5['F_dilute'] = np.where(df5['Shares_issued']< 0 , 1, 0)
# Operating
df5['F_GPM'] = np.where(df5['GP_margin']< df5['GP_margin_prev'], 1, 0)
df5['F_ATO'] = np.where(df5['Asset_TO']< df5['Asset_TO_prev'], 1, 0)


# 16 Group F-score based on categories
df5['F-profitability'] = df5['F_income'] + df5['F_opcash'] + df5['F_ROA'] + df5['F_quality']
df5['F_leverage_liquidity'] = df5['F_leverage'] + df5['F_currentratio'] + df5['F_dilute']
df5['F_operating'] = df5['F_GPM'] + df5['F_ATO']

df5['F_score'] = df5['F-profitability'] + df5['F_leverage_liquidity'] + df5['F_operating'] 


# Long Portfolio

# 17 Filter out F_score more than 7
df6 = df5[df5.F_score > 7]


# 18 Average PE per pfy per gsector
df6['PE'] = df6['mkvalt'] / df6['ni']
df7 = df6.groupby(['pfy','gsector'], as_index=False)['PE'].mean()


# 19 Filter for stocks with PE lower than gsector average
df8 = df6.merge(df7, on = ['pfy','gsector'], how='left')
df8['y_x'] = df8['PE_y'] - df8['PE_x']
df11 = df8[df8['y_x'] > 0]


# 20 Finding the number of unique company/gvkey in our long portfolio
df12 = df11['gvkey'].unique()


# 21 Mean yret of each pfy
df23 = pd.DataFrame(df11.groupby(['pfy'], as_index=False)['yret'].mean())
df23.rename(columns={'yret':'pyret'}, inplace = True)

# 22 add pfy count number
df24 = df11.groupby(['pfy'], as_index=False)['yret'].count()

df25 = pd.merge(df23, df24, how='inner', on=['pfy'])
df25.rename(columns={'yret':'count'}, inplace = True)


# 23 Compute yearly return compounding factor (1+yearly return)
df25['ppyret'] = df25['pyret'] + 1


# Risk free rate

# 24 Calculate risk free rate using UStreasury 1month
import quandl
from datetime import datetime
# Key in your quandl api key below
QUANDL_API_KEY = 'key in your quandl api key here'
quandl.ApiConfig.api_key = QUANDL_API_KEY

start = datetime(2002, 1, 1)
end  = datetime(2020, 12, 31)

rf = quandl.get('USTREASURY/YIELD.1',start_date=start, end_date=end)
risk_free = rf['1 MO']
rfr = risk_free.mean()/100


# 25 Annualise the total return, based on average and total
Lportfolio_annualised_return_rut = scs.gmean(df25.loc[:,"ppyret"])-1

# 26 Calculate annualized volatility from the standard deviation
Lportfolio_vola_rut = np.std(df25['pyret'], ddof=1)

# 27 Calculate the Sharpe ratio 
Lportfolio_sharpe_rut = ((Lportfolio_annualised_return_rut - rfr)/ Lportfolio_vola_rut)

# 28 Define negative returns and compute standard deviation 
Lportfolio_negative_ret_rut = df25.loc[df25['pyret'] < 0]
Lportfolio_expected_ret_rut = np.mean(df25['pyret'])
Lportfolio_downside_std_rut = Lportfolio_negative_ret_rut['pyret'].std()

# 29 Compute Sortino Ratio
Lportfolio_sortino_rut = (Lportfolio_expected_ret_rut - rfr)/Lportfolio_downside_std_rut

# 30 Compute Worst and Best pfy return
Lpcolumn = df25["pyret"]
Lpmax_value = Lpcolumn.max()
Lpmin_value = Lpcolumn.min()

# 31 Compute % of profitable pfy
Lpprofitable_pfy = len(df25[df25['pyret']>0]['pyret'])/len(df25['pyret'])


# 32 Compute long portfolio monthly price
#Merge long portofio df11 with stock return to get monthly close price
col = ['pfy','gvkey']
df21 = df11[col]
df26 = pd.merge(df1, df21, how='inner', on=['gvkey', 'pfy'])
# Calculate long portfolio monthly price
df27 = df26.groupby(['pfy','month'], as_index=False)['prccm'].mean()


# 33 Compute max drawdown and duration
# Initialize variables: hwm (high watermark), drawdown, duration
lphwm = np.zeros(len(df27))
lpdrawdown = np.zeros(len(df27))
lpduration = 0


# 34 Determine maximum drawdown (maxDD)
for t in range(len(df27)):
    lphwm[t] = max(lphwm[t-1], df27['prccm'][t])
    lpdrawdown[t] = ((lphwm[t] - df27.prccm[t]) / lphwm[t]) * 100
lpmaxDD = lpdrawdown.max()


# 35 Determine maximum drawdown duration
#       numpy.allclose compares whether two floating values are equal to the absolute 
#        tolerance (atol) precision (1e-8 is 1x10^-8)
for j in range(len(df27)):
    if np.allclose(lpdrawdown[j], lpmaxDD, atol=1e-8):
        for k in range(j):
            if np.allclose(df27.prccm[k], lphwm[j], atol=1e-8):
                lpduration = j - k
            else: 
                continue
        else:
            continue


# Short portfolio

# 36 Filter out F_score less than 2
df28 = df5[df5.F_score < 2]


# 37 Average PE per pfy per gsector
df28['PE'] = df28['mkvalt'] / df28['ni']
df29 = df28.groupby(['pfy','gsector'], as_index=False)['PE'].mean()


# 38 Filter for stocks with PE lower than gsector average
df30 = df28.merge(df29, on = ['pfy','gsector'], how='left')
df30['y_x'] = df30['PE_y'] - df30['PE_x']
df33 = df30[df30['y_x'] > 0]


# 39 Finding the number of unique company/gvkey in our short portfolio
df34 = df33['gvkey'].unique()


# 40 Mean yret of each pfy
df37 = pd.DataFrame(df33.groupby(['pfy'], as_index=False)['yret'].mean())
df37.rename(columns={'yret':'pyret'}, inplace = True)

# 41 add pfy count number
df38 = df33.groupby(['pfy'], as_index=False)['yret'].count()

df39 = pd.merge(df37, df38, how='inner', on=['pfy'])
df39.rename(columns={'yret':'count'}, inplace = True)


# 42 Reverse return sign due to short portfolio
df39['spyret'] = df39['pyret'] * -1
# 43 Compute yearly return compounding factor (1+yearly return)
df39['sppyret'] = df39['spyret'] + 1


# 44 Annualise the total return, based on average and total
Sportfolio_annualised_return_rut = scs.gmean(df39.loc[:,"sppyret"])-1

# 45 Calculate annualized volatility from the standard deviation
Sportfolio_vola_rut = np.std(df39['spyret'], ddof=1)

# 46 Calculate the Sharpe ratio 
Sportfolio_sharpe_rut = ((Sportfolio_annualised_return_rut - rfr)/ Sportfolio_vola_rut)

# 47 Define negative returns and compute standard deviation 
Sportfolio_negative_ret_rut = df39.loc[df39['spyret'] < 0]
Sportfolio_expected_ret_rut = np.mean(df39['spyret'])
Sportfolio_downside_std_rut = Sportfolio_negative_ret_rut['spyret'].std()

# 48 Compute Sortino Ratio
Sportfolio_sortino_rut = (Sportfolio_expected_ret_rut - rfr)/Sportfolio_downside_std_rut

# 49 Compute Worst and Best pfy return
Spcolumn = df39["spyret"]
Spmax_value = Spcolumn.max()
Spmin_value = Spcolumn.min()

# 50 Compute % of profitable pfy
Spprofitable_pfy = len(df39[df39['spyret']>0]['spyret'])/len(df39['spyret'])


# 51 Compute short portfolio monthly price
# Prepare the short portofio df11 to merge with yahoo finance data
col = ['pfy','gvkey']
df40 = df33[col]
# Merge short portofio df33 with stock return to get monthly close price
df41 = pd.merge(df1, df40, how='inner', on=['gvkey', 'pfy'])
# Calculate short portfolio monthly price
df42 = df41.groupby(['pfy','month'], as_index=False)['prccm'].mean()


# 52 Compute max drawdown and duration
# Initialize variables: hwm (high watermark), drawdown, duration
sphwm = np.zeros(len(df42))
spdrawdown = np.zeros(len(df42))
spduration = 0


# 53 Determine maximum drawdown (maxDD)
for t in range(len(df42)):
    sphwm[t] = max(sphwm[t-1], df42['prccm'][t])
    spdrawdown[t] = ((sphwm[t] - df42.prccm[t]) / sphwm[t]) * 100
spmaxDD = spdrawdown.max()


# 54 Determine maximum drawdown duration
#       numpy.allclose compares whether two floating values are equal to the absolute 
#        tolerance (atol) precision (1e-8 is 1x10^-8)
for j in range(len(df42)):
    if np.allclose(spdrawdown[j], spmaxDD, atol=1e-8):
        for k in range(j):
            if np.allclose(df42.prccm[k], sphwm[j], atol=1e-8):
                spduration = j - k
            else: 
                continue
        else:
            continue


# Long & Short Portfolio

# 55 Merge long and short portofio
df43 = df25[['pfy','pyret']]
df44 = df39[['pfy','spyret']]
df45 = pd.merge(df43, df44, how='inner', on=['pfy'])


# 56 compute long short return
df45['lspyret'] = df45['pyret']/2 + df45['spyret']/2
# Compute yearly return compounding factor (1+yearly return)
df45['lsppyret'] = df45['lspyret'] + 1


# 57 Annualise the total return, based on average and total
LSportfolio_annualised_return_rut = scs.gmean(df45.loc[:,"lsppyret"])-1

# 58 Calculate annualized volatility from the standard deviation
LSportfolio_vola_rut = np.std(df45['lspyret'], ddof=1)

# 59 Calculate the Sharpe ratio 
LSportfolio_sharpe_rut = ((LSportfolio_annualised_return_rut - rfr)/ LSportfolio_vola_rut)

# 60 Define negative returns and compute standard deviation 
LSportfolio_negative_ret_rut = df45.loc[df45['lspyret'] < 0]
LSportfolio_expected_ret_rut = np.mean(df45['lspyret'])
LSportfolio_downside_std_rut = LSportfolio_negative_ret_rut['lspyret'].std()

# 61 Compute Sortino Ratio
LSportfolio_sortino_rut = (LSportfolio_expected_ret_rut - rfr)/LSportfolio_downside_std_rut

# 62 Compute Worst and Best pfy return
LSpcolumn = df45["lspyret"]
LSpmax_value = LSpcolumn.max()
LSpmin_value = LSpcolumn.min()

# 63 Compute % of profitable pfy
LSpprofitable_pfy = len(df45[df45['lspyret']>0]['lspyret'])/len(df45['lspyret'])


# 64 Merge long and short portofio monthly price
df46 = pd.merge(df27, df42, how='inner', on=['pfy', 'month'])
df46['lsprccm'] = df46['prccm_x']/2 + df46['prccm_y']/2


# 65 Compute max drawdown and duration
# Initialize variables: hwm (high watermark), drawdown, duration
lsphwm = np.zeros(len(df46))
lspdrawdown = np.zeros(len(df46))
lspduration = 0


# 66 Determine maximum drawdown (maxDD)
for t in range(len(df46)):
    lsphwm[t] = max(lsphwm[t-1], df46['lsprccm'][t])
    lspdrawdown[t] = ((lsphwm[t] - df46.lsprccm[t]) / lsphwm[t]) * 100
lspmaxDD = lspdrawdown.max()


# 67 Determine maximum drawdown duration
#       numpy.allclose compares whether two floating values are equal to the absolute 
#        tolerance (atol) precision (1e-8 is 1x10^-8)
for j in range(len(df46)):
    if np.allclose(lspdrawdown[j], lspmaxDD, atol=1e-8):
        for k in range(j):
            if np.allclose(df46.lsprccm[k], lsphwm[j], atol=1e-8):
                lspduration = j - k
            else: 
                continue
        else:
            continue


# Market return

# 68 Monthly return of Russell 3000
rut = pd.read_csv(myfolder / '^RUA.csv', parse_dates=['Date'])
rut['rutret'] = rut.sort_values(by='Date')['Adj Close'].pct_change()


# 69 Create portfolio formation year (pfy) variable, where
#      pfy = current year for Jul-Dec dates and previous year for Jan-Jun dates.
#      This is to facilitate compounding returns over Jul-Jun by pfy later below.
rut['year'], rut['month'] = rut['Date'].dt.year, rut['Date'].dt.month
rut['pfy'] = np.where(rut.month > 6, rut.year, rut.year - 1)
rut


# 70 Compute monthly return compounding factor (1+monthly return)
rut['mretfactor'] = 1 + rut.rutret


rut2 = rut[['Date','Adj Close','rutret', 'pfy', 'mretfactor']]


# 71 Compound monthly returns to get annual returns at end-June of each pfy,
#      ensuring only firm-years with 12 mths of return data from Jul-Jun are selected.
rut2['rutyret'] = rut2.groupby(['pfy'])['mretfactor'].cumprod() - 1
rut3 = rut2.groupby(['pfy']).nth(11)


# 72 Compute yearly return compounding factor (1+yearly return)
rut3['rrutyret'] = rut3['rutyret'] + 1


# 73 Compute Returns, Sharpe and Sortino ratio

# 74 Compute monthly stock returns from price data
rut4 = rut3[['Date', 'Adj Close','rutyret']]
rut4 = rut3.rename(columns = {'Adj Close': 'price'})

# 75 Annualise the total return, based on average and total
annualised_return_rut = scs.gmean(rut3.loc[:,"rrutyret"])-1

# 76 Calculate annualized volatility from the standard deviation
vola_rut = np.std(rut4['rutyret'], ddof=1)

# 77 Calculate the Sharpe ratio 
sharpe_rut = ((annualised_return_rut - rfr)/ vola_rut)

# 78 Define negative returns and compute standard deviation 
negative_ret_rut = rut4.loc[rut4['rutyret'] < 0]
expected_ret_rut = np.mean(rut4['rutyret'])
downside_std_rut = negative_ret_rut['rutyret'].std()

# 79 Compute Sortino Ratio
sortino_rut = (expected_ret_rut - rfr)/downside_std_rut

# 80 Compute Worst and Best pfy return
rcolumn = rut4["rutyret"]
rmax_value = rcolumn.max()
rmin_value = rcolumn.min()

# 81 Compute % of profitable pfy
rprofitable_pfy = len(rut4[rut4['rutyret']>0]['rutyret'])/len(rut4['rutyret'])


# Compute Max drawdown and duration

# 82 Rename to price
rut5 = rut2.rename(columns = {'Adj Close': 'price'})

# 83 Initialize variables: hwm (high watermark), drawdown, duration
rhwm = np.zeros(len(rut5))
rdrawdown = np.zeros(len(rut5))
rduration = 0

# 84 Determine maximum drawdown (maxDD)
for t in range(len(rut5)):
    rhwm[t] = max(rhwm[t-1], rut5['price'][t])
    rdrawdown[t] = ((rhwm[t] - rut5.price[t]) / rhwm[t]) * 100
rmaxDD = rdrawdown.max()


# 85 Determine maximum drawdown duration
#       numpy.allclose compares whether two floating values are equal to the absolute 
#        tolerance (atol) precision (1e-8 is 1x10^-8)
for j in range(len(rut5)):
    if np.allclose(rdrawdown[j], rmaxDD, atol=1e-8):
        for k in range(j):
            if np.allclose(rut5.price[k], rhwm[j], atol=1e-8):
                rduration = j - k
            else: 
                continue
        else:
            continue


# Investment peformance

# 86 Plot Portfolio and Russell 3000 Returns
rut6 = rut4.drop(['Date', 'price', 'rutret', 'mretfactor', 'rrutyret'], axis=1)    # "axis=1" means to drop column

df47 = df45.iloc[: , :-1]

df48 = pd.merge(df47, rut6, how='inner', on=['pfy'])

df48.rename(columns={'pyret':'Long Portfolio', 'spyret':'Short Portfolio', 'lspyret':'Long Short Portfolio','rutyret':'Market Index'}, inplace = True)

df48_plot = pd.melt(df48,id_vars='pfy', var_name='Returns',value_name='returns')

fig, ax = plt.subplots(figsize=(8,6)) 
ax = sns.lineplot(data=df48_plot, x='pfy', y='returns', hue='Returns')  
ax.set(xlabel = 'pfy', ylabel = 'Returns')
ax.set_title('Plot of Portfolio and Russell 3000 Returns')
plt.show()
chartfile.savefig(fig)


# 87 Calculate market Wealth Index
#rut7 = rut.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'rutret', 'year', 'Adj Close'], axis=1)
rut3['RUT_WI'] = (rut3['rrutyret']).cumprod()
rut3 = rut3.reset_index()
rut8 = rut3.drop(['Date', 'Adj Close', 'rutret', 'mretfactor', 'rutyret', 'rrutyret'], axis=1) 


# 88 Calculate long portfolio Wealth Index
df25['P_WI'] = (df25['ppyret']).cumprod()
df49 = df25.drop(['pyret', 'count', 'ppyret'], axis=1)


# 89 Calculate short portfolio Wealth Index
df39['S_WI'] = (df39['sppyret']).cumprod()
df50 = df39.drop(['pyret', 'count', 'spyret', 'sppyret'], axis=1)


# 90 Calculate long short portfolio Wealth Index
df45['LS_WI'] = (df45['lsppyret']).cumprod()
df52 = df45.drop(['pyret', 'spyret', 'lspyret', 'lsppyret'], axis=1) 


# 91 Plot Portfolio and Russell 3000 Wealth Index Line plot
df53 = pd.merge(df49, df50, how='right', on=['pfy'])
df54 = pd.merge(df53, df52, how='left', on=['pfy'])
df55 = pd.merge(df54, rut8, how='left', on=['pfy'])

df55.rename(columns={'P_WI':'Long Portfolio WI', 'S_WI':'Short Portfolio WI', 'LS_WI':'Long Short Portfolio WI','RUT_WI':'Market Index WI'}, inplace = True)

df55_plot = pd.melt(df55,id_vars='pfy', var_name='Wealth Index',value_name='wealth index')

fig2, ax2 = plt.subplots(figsize=(8,6)) 
ax2 = sns.lineplot(data=df55_plot, x='pfy', y='wealth index', hue='Wealth Index')  
ax2.set(xlabel = 'pfy', ylabel = 'Wealth Index')
ax2.set_title('Plot of Portfolio and Russell 3000 Wealth Index')
plt.show()
chartfile.savefig(fig2)

# 92 Print Investment Performance in a table
table = [['Performance Matrix', 'Long', 'Short', 'Long & Short', 'Russell 3000 Index)'], 
    ['Compounded annual return', Lportfolio_annualised_return_rut, Sportfolio_annualised_return_rut, LSportfolio_annualised_return_rut, annualised_return_rut], 
    ['Standard deviation of return', Lportfolio_vola_rut, Sportfolio_vola_rut, LSportfolio_vola_rut, vola_rut], 
    ['Downside deviation of return', Lportfolio_downside_std_rut, Sportfolio_downside_std_rut, LSportfolio_downside_std_rut, downside_std_rut],
    ['Sharpe ratio', Lportfolio_sharpe_rut, Sportfolio_sharpe_rut, LSportfolio_sharpe_rut, sharpe_rut], 
    ['Sortino ratio', Lportfolio_sortino_rut, Sportfolio_sortino_rut, LSportfolio_sortino_rut, sortino_rut],
    ['Maximum drawdown %', lpdrawdown.round(2).max(), spdrawdown.round(2).max(), lspdrawdown.round(2).max(), rdrawdown.round(2).max()], 
    ['Worst-pfy return', Lpmin_value, Spmin_value, LSpmin_value, rmin_value],
    ['Best-pfy return', Lpmax_value, Spmax_value, LSpmax_value, rmax_value], 
    ['% of profitable pfy', Lpprofitable_pfy, Spprofitable_pfy, LSpprofitable_pfy, rprofitable_pfy]]

print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'), '\n\n', file=outfile)


# 93 t-test between Long Portfolio and Russell 3000
a = df48['Market Index'][pd.isnull(df48['Market Index'])==False]
b = df48['Long Portfolio'][pd.isnull(df48['Long Portfolio'])==False]
model1 = scs.ttest_ind(a, b, equal_var=False, alternative="greater")
print('Long Portfolio returns vs. Russell 3000 returns using t-test \n\n', 
      model1, '\n'*5, file=outfile)


# 94 t-test between Short Portfolio and Russell 3000
a = df48['Market Index'][pd.isnull(df48['Market Index'])==False]
c = df48['Short Portfolio'][pd.isnull(df48['Short Portfolio'])==False]
model2 = scs.ttest_ind(a, c, equal_var=False, alternative="greater")
print('Short Portfolio returns vs. Russell 3000 returns using t-test \n\n', 
      model2, '\n'*5, file=outfile)


# 95 t-test between Long Short Portfolio and Russell 3000
a = df48['Market Index'][pd.isnull(df48['Market Index'])==False]
d = df48['Long Short Portfolio'][pd.isnull(df48['Long Short Portfolio'])==False]
model3 = scs.ttest_ind(a, d, equal_var=False, alternative="greater")
print('Long Short Portfolio returns vs. Russell 3000 returns using t-test \n\n', 
      model3, '\n'*5, file=outfile)


# 96 Close and save output files
outfile.close()
chartfile.close()

