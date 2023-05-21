import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('../csv/all_merged_data_tran_cnt_info_pk_unmarked.csv')
# # 取出某一列数据
# col_name = 'avg_mth'
# target_col = df[col_name]
#
# # 计算基本统计量
# mean_val = target_col.mean()
# q1_val = target_col.quantile(0.25)
# median_val = target_col.median()
# q3_val = target_col.quantile(0.75)
# max_val = target_col.max()
# min_val = target_col.min()
# count_val = target_col.count()
#
# # 输出结果
# print("列 '{}' 的统计量如下：".format(col_name))
# print("平均值：{:.2f}".format(mean_val))
# print("四分之一位数：{:.2f}".format(q1_val))
# print("中位数：{:.2f}".format(median_val))
# print("四分之三位数：{:.2f}".format(q3_val))
# print("最大值：{:.2f}".format(max_val))
# print("最小值：{:.2f}".format(min_val))
# print("个数：{}".format(count_val))

# 统计 Y 和 N 的数量
y_count = df['is_mgr_dep'].value_counts()['Y']
n_count = df['is_mgr_dep'].value_counts()['N']

# 打印结果
print("Y 的数量：{}".format(y_count))
print("N 的数量：{}".format(n_count))