import pandas as pd

# 读取 all_merged_data_tran_cnt_marked.csv 文件并设置 uid 列为索引（主键）
df1 = pd.read_csv("../csv/all_merged_data_tran_cnt_unmarked.csv", index_col="uid")

# 读取 pri_cust_base_info.csv 文件并设置 uid 列为索引（主键）
df2 = pd.read_csv("../csv/pri_cust_base_info.csv", index_col="uid")

# 融合两张表格，使用 outer join 操作
merged_df = pd.merge(df1, df2, how="inner", left_index=True, right_index=True)

# 将 is_shareholder、is_black、is_contact 和 is_mgr_dep 列的 null 替换为 N
for col in ["is_shareholder", "is_black", "is_contact", "is_mgr_dep"]:
    merged_df[col].fillna("N", inplace=True)

# 新增一列 'uid'，值为行索引的值
merged_df.insert(0, 'uid', merged_df.index)

# 将新数据写入一个 csv 文件
merged_df.to_csv("../csv/all_merged_data_tran_cnt_info_pk_unmarked.csv", index=False)