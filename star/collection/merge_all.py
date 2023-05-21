import pandas as pd

# 读取 merged_data_marked.csv 文件和 total_tran_amt.csv 文件
merged_data = pd.read_csv('../csv/merged_data_unmarked.csv')
total_tran_amt = pd.read_csv('../csv/total_tran_amt.csv')

# 按照 uid 列将两个表格合并
merged_data_with_tran_amt = pd.merge(merged_data, total_tran_amt, on='uid')

# 将 tran_amt 列重命名为 total_tran_amt
merged_data_with_tran_amt = merged_data_with_tran_amt.rename(columns={'tran_amt': 'total_tran_amt'})

# 调整列的顺序，将 star_level 移动到最后一列
cols = merged_data_with_tran_amt.columns.tolist()
cols = [col for col in cols if col != 'star_level'] + ['star_level']
merged_data_with_tran_amt = merged_data_with_tran_amt[cols]

# 将合并后的数据写回文件
merged_data_with_tran_amt.to_csv('../csv/all_merged_data_unmarked.csv', index=False)