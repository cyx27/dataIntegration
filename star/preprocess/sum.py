import pandas as pd

# 读入交易数据csv文件
df_dsf = pd.read_csv('../csv/tr_mx/dsf.csv', skiprows=[0], names=['uid', 'tran_date', 'dsf_amt']).fillna(0)
df_etc = pd.read_csv('../csv/tr_mx/etc.csv', skiprows=[0], names=['uid', 'tran_date', 'etc_amt']).fillna(0)
df_grwy = pd.read_csv('../csv/tr_mx/grwy.csv', skiprows=[0], names=['uid', 'tran_date', 'grwy_amt']).fillna(0)
df_gzdf = pd.read_csv('../csv/tr_mx/gzdf.csv', skiprows=[0], names=['uid', 'tran_date', 'gzdf_amt']).fillna(0)
df_sa = pd.read_csv('../csv/tr_mx/sa.csv', skiprows=[0], names=['uid', 'tran_date', 'sa_amt']).fillna(0)
df_sbyb = pd.read_csv('../csv/tr_mx/sbyb.csv', skiprows=[0], names=['uid', 'tran_date', 'sbyb_amt']).fillna(0)
df_sdrq = pd.read_csv('../csv/tr_mx/sdrq.csv', skiprows=[0], names=['uid', 'tran_date', 'sdrq_amt']).fillna(0)
df_shop = pd.read_csv('../csv/tr_mx/shop.csv', skiprows=[0], names=['uid', 'tran_date', 'shop_amt']).fillna(0)
df_sjyh = pd.read_csv('../csv/tr_mx/sjyh.csv', skiprows=[0], names=['uid', 'tran_date', 'sjyh_amt']).fillna(0)

# 将交易数据逐个转化为以uid为行索引、以交易类型为列名的形式
dfs = [df_dsf, df_etc, df_grwy, df_gzdf, df_sa, df_sbyb, df_sdrq, df_shop, df_sjyh]
trades = pd.DataFrame({'uid': df_dsf['uid']})
for df in dfs:
    trade_type = df.columns[-1][:-4]  # 提取交易类型名
    df[trade_type + '_amt'] = df[trade_type + '_amt'].astype(float)  # 将金额列转换为浮点型
    trade = df.groupby('uid', as_index=False).agg({trade_type + '_amt': ['sum', 'count']}).fillna(0)
    trade.columns = ['uid', trade_type + '_amt', trade_type + '_count']
    trades = trades.merge(trade, on='uid', how='outer')

# 读入星级表并合并
df_star_info = pd.read_csv('../csv/pri_star_info_unmarked.csv')
result = trades.merge(df_star_info, on='uid', how='left').fillna(0)
result.to_csv('../csv/tr_mx/sum_unmarked.csv', index=False)