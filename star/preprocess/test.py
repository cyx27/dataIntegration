import pandas as pd

# # 导入 all_merged_data_unmarked.csv 和 tr_mx.csv
# all_data = pd.read_csv("../csv/all_merged_data_marked.csv")
# tr_data = pd.read_csv("../csv/tr_mx.csv")
#
# # 统计每个UID在 tr_data 中出现的次数
# tran_cnt = tr_data.groupby("uid").size().reset_index(name="tran_cnt")
#
# # 将 tran_cnt 与 all_data 合并，新增一列 tran_cnt
# all_data = pd.merge(all_data, tran_cnt, on="uid")
# cols = list(all_data.columns)
# tran_cnt_col = cols.pop(-2)  # 移除并获取 tran_cnt 列
# cols.append(tran_cnt_col)  # 将 tran_cnt 列放到倒数第二列
# all_data = all_data[cols]  # 所有列按照新顺序重新排序
#
# # 将新数据写入一个 csv 文件
# all_data.to_csv("../csv/all_merged_data_tran_cnt_marked.csv", index=False)

df = pd.read_csv("../csv/all_merged_data_tran_cnt_info_pk_unmarked.csv")
# 将 star_level 列保存为一个 Series，并从原始表格中删除
star_lvl_col = df["star_level"]
df.dropna(subset=["star_level"])

# 将 star_level 列添加回 DataFrame 中的最后一列
df["star_level"] = star_lvl_col
df.to_csv("../csv/all_merged_data_tran_cnt_info_pk_unmarked.csv", index=False)
