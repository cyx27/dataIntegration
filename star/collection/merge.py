import pandas as pd

# 读取 prim_cust_asset_info.csv 文件
prim_cust_asset_info = pd.read_csv('../csv/prim_cust_asset_info.csv')

# 读取 pri_star_info_marked.csv 文件
pri_star_info_marked = pd.read_csv('../csv/pri_star_info_unmarked.csv')

# 按照 uid 列将两个表格合并
merged_data = pd.merge(prim_cust_asset_info, pri_star_info_marked, on='uid')

# 将合并后的数据写回文件
merged_data.to_csv('../csv/merged_data_unmarked.csv', index=False)