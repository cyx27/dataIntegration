from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
import numpy as np


def replace_by_zscore(df, num_cols, k=5):
    for col in num_cols:
        z_scores = zscore(df[col])
        outliers = df[np.abs(z_scores) > k]
        # 替换策略：使用均值
        replacement_value = df[col].mean()
        # 替换异常值
        df.loc[outliers.index, col] = replacement_value
    return df


def replace_data(data):
    for feature in data.columns:
        if type(feature) == str:
            continue
        lower_bound = data[feature].quantile(0.05)  # 计算下百分位数
        upper_bound = data[feature].quantile(0.95)  # 计算上百分位数
        # 替换小于下百分位数的值
        data.loc[data[feature] < lower_bound, feature] = lower_bound
        # 替换大于上百分位数的值
        data.loc[data[feature] > upper_bound, feature] = upper_bound
    return data


def label_transform(df, cols):
    le = LabelEncoder()
    for col in cols:
        # print(col)
        data_not_null = df.loc[df[col].notnull(), col]
        encoded_values = le.fit_transform(data_not_null)
        df.loc[df[col].notnull(), col] = encoded_values
    return df
