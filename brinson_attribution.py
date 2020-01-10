# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:42:47 2019

@author: HP
"""

import os
import numpy as np
import pandas as pd
from itertools import dropwhile
import warnings
warnings.filterwarnings('ignore') 

wdir = os.path.dirname(__file__)

def get_ind_data(weight, ind_type='zx'):
    industry_dat = pd.read_csv(os.path.join(wdir, 'quote_data', f'industry_{ind_type}.csv'),
                               encoding='gbk', engine='python', index_col=[0])
    industry_dat.columns = pd.to_datetime(industry_dat.columns)
    industry_dat = industry_dat.loc[weight.index, weight.columns]
    industry_dat = industry_dat.where(pd.notnull(weight), np.nan)
    return industry_dat

def get_trade_date(date):
    month_map = pd.read_excel(wdir, 'quote_data', 'month_map.xlsx')
    month_map = month_map.applymap(pd.to_datetime)
    month_map = month_map.set_index(['calendar_date'])
    date = month_map.loc[date][0]
    return date

def get_panel_data(date, panel_dir):
    dat = pd.read_csv(os.path.join(panel_dir, f'{str(date)[:10]}.csv'), encoding='gbk',
                      engine='python', index_col=[0])
    return dat
    
def get_stocks_ret(weight, freq='M'):    
    wt = weight.copy()
    if freq.endswith('M'):
        fname = f'pct_chg_{freq}.csv'
    elif freq == 'd':
        fname = 'pct_chg.csv'
    else:
        raise RuntimeError("Unsupported return Type!")
        
    pct_chg = pd.read_csv(os.path.join(wdir, 'quote_data', fname),
                          engine='c', index_col=[0], encoding='gbk')
    pct_chg.columns = pd.to_datetime(pct_chg.columns)
    
    if freq.endswith('M'):
        dates = [get_cdate(date) for date in weight.columns]
        pct_chg = pct_chg.loc[weight.index, dates]
        pct_chg.columns = weight.columns
        pct_chg = pct_chg.where(pd.notnull(wt), np.nan)
        pct_chg = pct_chg.dropna(how='all', axis=1)
    elif freq == 'd':
        pct_chg = pct_chg.loc[weight.index, :]
        start_date = f'{weight.columns[0].year}-{weight.columns[0].month}'
        end_date = f'{weight.columns[-1].year}-{weight.columns[-1].month}'
        pct_chg = pct_chg.loc[:, start_date:end_date]
    return pct_chg

def cal_group_ret(datdf):
    return (datdf['return'] * datdf['weight']).sum() / datdf['weight'].sum()

def cal_ind_ret_weight(weight, freq='6M'):
    dates = weight.columns.tolist()
    ind_dat = get_ind_data(weight)
    ret_dat = get_stocks_ret(weight, freq)
        
    ind_return = []; ind_weight = []
    for date in dates: 
        if freq.endswith('M'):
            cur_stk = weight[date].dropna().index
            cur_ind = ind_dat.loc[cur_stk, date]
            cur_ret = ret_dat.loc[cur_stk, date]
            cur_weight = weight.loc[cur_stk, date]
                    
            cur_datdf = pd.concat([cur_ind, cur_ret, cur_weight], axis=1)
            cur_datdf.columns = ['industry', 'return', 'weight']
            
            cur_ind_ret = cur_datdf.groupby(['industry']).apply(cal_group_ret)
            cur_ind_weight = cur_datdf.groupby(['industry']).apply(lambda df: df['weight'].sum())
        cur_ind_ret.name = cur_ind_weight.name = date
        
        ind_return.append(cur_ind_ret)
        ind_weight.append(cur_ind_weight)
        
    ind_return = pd.DataFrame(ind_return).fillna(0)
    ind_weight = pd.DataFrame(ind_weight).T.fillna(0)
    ind_weight /= ind_weight.sum()
    return ind_return, ind_weight.T    

def brinson_attr_asset(stock_weight, asset_weight, fund_code, stock_bm='000300.SH', 
                       bond_bm='000012.SH', freq='6M', version=2, verbose=False):
    brinson_stock = brinson_attr_stock(stock_weight, stock_bm, freq, version, 
                                       verbose, fund_code)
    brinson_stock.index = [get_cdate(date) for date in brinson_stock.index]
    
    bond_bm_ret = get_index_ret(bond_bm, freq)
    bond_bm_ret = bond_bm_ret.loc[brinson_stock.index]
    
    stock_bm_ret = get_index_ret(stock_bm, freq)
    stock_bm_ret = stock_bm_ret.loc[brinson_stock.index]
    
#    bm_ret = brinson_stock['re_b'] * asset_weight['bm_stock'] + bond_bm_ret * asset_weight['bm_bond']
    bm_ret = stock_bm_ret * asset_weight['bm_stock'] + bond_bm_ret * asset_weight['bm_bond']
    
    fund_ret = get_index_ret(fund_code, freq)
    fund_ret = fund_ret.loc[brinson_stock.index]
    
    timing_ret = (asset_weight['pt_stock'] - asset_weight['bm_stock']) * (brinson_stock['re_b'] - bm_ret) + \
                 (asset_weight['pt_bond'] - asset_weight['bm_bond']) * (bond_bm_ret - bm_ret)
    ind_ret = brinson_stock['配置效应(AR)'] * asset_weight['pt_stock']
    select_ret = brinson_stock['选股效应(SR)'] * asset_weight['pt_stock']
    
    res_con_timing = pd.concat([fund_ret, bm_ret, timing_ret, ind_ret, select_ret], axis=1)
    res_con_timing.columns = ['基金收益', '基准实际收益', '大类资产择时收益(TR)', '配置效应(AR)', '选股效应(SR)']
    res_con_timing['估计误差'] = res_con_timing['基金收益'] - res_con_timing[['基准实际收益', '大类资产择时收益(TR)', '配置效应(AR)', '选股效应(SR)']].sum(axis=1)
    res_con_timing['是否调整'] = '调整后'
    
    res_wo_timing = pd.concat([fund_ret, bm_ret, brinson_stock[['配置效应(AR)', '选股效应(SR)']]], axis=1)
    res_wo_timing.columns = ['基金收益', '基准实际收益', '配置效应(AR)', '选股效应(SR)']
    res_wo_timing['大类资产择时收益(TR)'] = np.nan
    res_wo_timing['估计误差'] = res_wo_timing['基金收益'] - res_wo_timing[['基准实际收益', '配置效应(AR)', '选股效应(SR)']].sum(axis=1)
    res_wo_timing['是否调整'] = '调整前'

    res = pd.concat([res_con_timing, res_wo_timing], axis=0)
    res.index.name = 'date'
    res = res.reset_index()
    res = res.set_index(['date','是否调整'])
    res = res.sort_index()
    res = res[['基金收益', '基准实际收益', '大类资产择时收益(TR)',
               '选股效应(SR)', '配置效应(AR)', '估计误差']]
    return res

def brinson_attr_stock(weight, benchmark='000300.SH', freq='6M', version=2, 
                       verbose=False, fund_code=None):
    r1 = ['配置效应(AR)','选股效应(SR)','交互效应(IR)']
    r2 = ['配置效应(AR)','选股效应(SR)']

    bm_weight = pd.read_csv(os.path.join(wdir, 'index_weight', f'{benchmark.split(".")[0]}.csv'),
                            engine='python', encoding='gbk', index_col=[0])
    bm_weight.columns = pd.to_datetime(bm_weight.columns)
    bm_weight = bm_weight[weight.columns]
    bm_weight /= bm_weight.sum()
        
    pt_ind_ret, pt_ind_weight = cal_ind_ret_weight(weight)
    bm_ind_ret, bm_ind_weight = cal_ind_ret_weight(bm_weight)
    bm_ind_weight.iloc[0]

    mut_ind = pt_ind_weight.columns | bm_ind_weight.columns
    if len(mut_ind.difference(pt_ind_weight.columns)) > 0:
        cols = mut_ind.difference(pt_ind_weight.columns)
        for col in cols:
            pt_ind_weight.loc[:, col] = 0
            pt_ind_ret.loc[:, col] = 0
            
    if len(mut_ind.difference(bm_ind_weight.columns)) > 0:
        cols = mut_ind.difference(bm_ind_weight.columns)
        for col in cols:
            bm_ind_weight.loc[:, col] = 0
            bm_ind_ret.loc[:, col] = 0
    
    brinson_single = brinson_attr_single_period(pt_ind_ret, pt_ind_weight, 
                                    bm_ind_ret, bm_ind_weight, version)
    if verbose:
        if fund_code is None:
            raise RuntimeError('Need to pass in "fund_code" to save attribution result file!')
        brinson_single.to_excel(os.path.join(wdir, 'brinson_result',
                                f'{fund_code}_res.xlsx'), encoding='gbk')
    brinson_returns = r1 if version == 1 else r2
    single_ret = pd.DataFrame()
    for r in brinson_returns:
        dat_panel = brinson_single.minor_xs(r)
        if '总计' in dat_panel.index:
            dat_panel.drop(['总计'], inplace=True)
        single_ret[r] = dat_panel.sum(axis=0)
    
    port_ret = (pt_ind_ret * pt_ind_weight).sum(axis=1)
    bm_ret = (bm_ind_ret * bm_ind_weight).sum(axis=1)
    
    single_ret.index = pd.to_datetime(single_ret.index)
    single_ret = pd.concat([single_ret, port_ret, bm_ret], axis=1)
    single_ret.columns = brinson_returns + ['re_p', 're_b']
    return single_ret

def brinson_attr_single_period(pt_ret, pt_weight, bm_ret, bm_weight, version=2):
    result = {}
    bm_total_ret = (bm_ret * bm_weight).sum(axis=1)
    for date in pt_ret.index: 
        res = pd.DataFrame(index=bm_weight.columns)
        res['组合权重'] = pt_weight.loc[date]
        res['基准权重'] = bm_weight.loc[date]
        res['组合收益'] = pt_ret.loc[date]
        res['基准收益'] = bm_ret.loc[date]
        if version == 1: #BHB
            res['配置效应(AR)'] = (res['组合权重'] - res['基准权重']) * res['基准收益']
            res['选股效应(SR)'] = res['基准权重'] * (res['组合收益'] - res['基准收益'])
            res['交互效应(IR)'] = (res['组合权重'] - res['基准权重']) * (res['组合收益'] - res['基准收益'])
        elif version == 2: #BF
            res['配置效应(AR)'] = (res['组合权重'] - res['基准权重']) * (res['基准收益'] - bm_total_ret.loc[date])
            res['选股效应(SR)'] = res['组合权重'] * (res['组合收益'] - res['基准收益'])
        res['超额收益'] = res['组合权重'] * res['组合收益'] - res['基准收益'] * res['基准权重']
        res.loc['总计'] = res.sum()
        res.loc['总计', ['组合收益', '基准收益']] = np.nan
        result[str(date)[:10]] = res
    result = pd.Panel(result)
    return result

def get_cdate(date):
    nextdate = date + pd.tseries.offsets.MonthEnd(1)
    if nextdate.month > date.month:
        cdate = date
    else:
        cdate = nextdate
    return cdate

def get_index_ret(code, freq='6M'):
	"""
		将日收益率转换为设定频率的收益率。
		例如，freq默认为6个月时，将日收益率转换为半年度收益，且默认计算时间范围为
		1-6月及7-12月，计算起始日期选择基金或者指数成立后的首个完整的半年度的首个
		月份第1个交易日（1月或者7月）
	"""
    ret = pd.read_csv(os.path.join(wdir, 'quote_data', f'{code}.csv'), parse_dates=True,
                           engine='python', encoding='gbk', index_col=[0])
    ret = ret.dropna(how='any', axis=0)['pct_change']
    if freq.endswith('M') and freq != 'M':
        num_months = int(freq[:-1])
        freq = 'M'
    else:
        num_months = 0
    ret = ret.groupby(pd.Grouper(freq=freq)).apply(lambda df: ((1+df).cumprod()-1).iloc[-1]).iloc[1:]
    if num_months > 0:
        if ret.index[0].month % num_months != 0:
            startdate = list(dropwhile(lambda d: d.month % num_months != 0, ret.index))[0]
            ret = ret.loc[startdate:]
        if ret.index[-1].month % num_months != 0:
            enddate = list(dropwhile(lambda d: d.month % num_months != 0, ret.index[::-1]))[0]
            ret = ret.loc[:enddate]
        ret = ret.groupby(pd.Grouper(freq=f'{num_months}M')).apply(lambda df: ((1+df).cumprod()-1).iloc[-1]).iloc[1:]
    return ret

def clean_index_quote(save_cols=('close',), save_ori=False):
    """
       input: 
	   从wind终端下载的基金/指数日频行情数据文件，文件名格式：'基金代码'.xls
       output:
           根据close_price计算日收益率, 根据save_cols参数决定所要原始的数据列；
	   结果存储为csv，通过save_ori关键字参数决定是否保留原始xls文件，默认值为False
	   存储结果见quote_data文件夹
    """		    
    quote_dir = os.path.join(wdir, 'quote_data')
    files = [f for f in os.listdir(quote_dir) if f.endswith('xls')]
    col_map = {
            'open': '开盘价(元)',
            'close': '收盘价(元)',
            'high': '最高价(元)',
            'low': '最低价(元)',
            'name': '名称',
            'code': '代码',
            'amount': '成交额(百万)',
            'volume': '成交量(股)',
            }
    
    save_map = {col_map[col.lower()]:col.lower() for col in save_cols 
                if col.lower() in col_map.keys()}
    
    for f in files: 
        dat = pd.read_excel(os.path.join(quote_dir, f), encoding='gbk')
        dat = dat.dropna(how='any', axis=0)
        dat['pct_change'] = dat[['收盘价(元)']].pct_change()
        dat = dat.rename(columns=save_map)
        dat = dat.set_index(['日期'])
        dat.index.name = 'date'
        dat = dat[list(save_map.values()) + ['pct_change']]
        dat.iloc[1:].to_csv(os.path.join(quote_dir, f[:-4]+'.csv'))
        if not save_ori:
            os.remove(os.path.join(quote_dir, f))

def clean_fund_holding(save_ori=True):
    """
       input: 
	   从wind终端下载的基金持仓明细文件，文件名格式：'基金代码'持股.csv
       output:
	   结果存储为xlsx，每个sheet名为对应持仓报告期日期，
	   通过save_ori关键字参数决定是否保留原始csv文件，默认值为True
	   存储结果见fund_holding文件夹
    """
    fund_dir = os.path.join(wdir, 'fund_holding')
    files = [f for f in os.listdir(fund_dir) if '持股' in f]
    for f in files: 
        try:
            dat = pd.read_csv(os.path.join(fund_dir, f), engine='c', 
                              encoding='utf-8', index_col=[0], parse_dates=True)
        except UnicodeDecodeError:
            dat = pd.read_csv(os.path.join(fund_dir, f), engine='c', 
                              encoding='gbk', index_col=[0], parse_dates=True)
  
        dat = dat.reset_index()
        dat['报告期'] = dat['报告期'].map(lambda d: str(d)[:10])
        del dat['序号']
        dat = dat.set_index(['品种代码', '报告期'])
        dat = dat.to_panel()
        dat = dat.swapaxes(0, 2)
        dat.to_excel(os.path.join(fund_dir, f.split('持股')[0]+'.xlsx'),
                     encoding='gbk')
        if not save_ori:
            os.remove(os.path.join(fund_dir, f))

def read_fund_holding(code, index=None, bm_stock_wt=0.80):
    fund_dir = os.path.join(wdir, 'fund_holding')
    dat = pd.read_excel(os.path.join(fund_dir, code+'.xlsx'), encoding='gbk', 
                        sheet_name=None)
    if index:
        index_weight = pd.read_csv(os.path.join(wdir, 'index_weight', f'{index}.csv'),
                         engine='python', encoding='gbk', index_col=[0])
        index_weight.columns = pd.to_datetime(index_weight.columns)
        
    stock_weight = pd.DataFrame(); asset_weight = pd.DataFrame(columns=['stock', 'bond'])
    for date in dat.keys(): 
        panel = dat[date]
        date = pd.to_datetime(date)
        panel = panel[panel['所属行业名称'].notnull()]
        panel = panel.rename(columns={'占股票市值比(%)': 'weight',
                                      '品种代码':'code'})
        if index:
            tdate = get_trade_date(date)
            panel = panel[panel['code'].isin(index_weight[tdate].dropna().index)]
        panel = panel.set_index(['code'])
        stock_weight = pd.concat([stock_weight, panel['weight']], axis=1)
        asset_weight.loc[date] = [panel['占基金净值比(%)'].sum()/100, 1 - panel['占基金净值比(%)'].sum()/100]
    
    asset_weight.columns = ['pt_stock', 'pt_bond']
    asset_weight['bm_stock'] = bm_stock_wt
    asset_weight['bm_bond'] = 1 - bm_stock_wt
    stock_weight.columns = [get_trade_date(pd.to_datetime(date)) for date in dat.keys()]
    return stock_weight, asset_weight

def brinson_attribution():
    fund_code = '161810.OF'        #基金代码
    bond_benchmark = '000012.SH'   #基金对应基准债券指数代码
    stock_benchmark = '000300.SH'  #基金对应基准股票指数基准代码
    bm_stock_wt = 0.80             #基准股票指数比例
    version = 2               #brinson归因模型版本 1--BHB; 2--BF
    freq = '6M'               #归因频率，与所选基金持仓频率对应，默认选择基金的半年报和年报
    verbose = True        #是否存储单层brinson归因结果（股票行业配置和选股效应）
    
    clean_index_quote()    #清洗并计算基金/基准指数日收益率
    clean_fund_holding()   #清洗基金持仓文件
    stock_weight, asset_weight = read_fund_holding(fund_code, None, bm_stock_wt) 
    res = brinson_attr_asset(stock_weight, asset_weight, fund_code, stock_benchmark, 
                             bond_benchmark, freq, version, verbose)
			    
    if not os.path.exists(os.path.join(wdir, 'brinson_result')):
	os.mkdir(os.path.join(wdir, 'brinson_result'))
    res.to_csv(os.path.join(wdir, 'brinson_result', f'{fund_code}.csv'), 
               encoding='gbk')
    print(f'Finish for {fund_code}.')
    
if __name__ == '__main__':
    brinson_attribution()
