import pandas as pd

# 读取 tr_mx.csv 文件
tr_mx = pd.read_csv('../csv/tr_mx.csv')

# 按照 uid 列分组并计算 tran_amt 总和
total_tran_amt = tr_mx.groupby('uid', as_index=False)['tran_amt'].sum()

# 选取 ‘uid’ 和 'tran_amt' 两列，并将结果保存到一个新的csv文件中
total_tran_amt[['uid', 'tran_amt']].to_csv('../csv/total_tran_amt.csv', index=False)