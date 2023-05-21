import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso


def delete_cols(dataFrame, target_data, num):
    result_cols = dataFrame.columns
    for col in dataFrame.columns:
        correlation = dataFrame[col].corr(target_data)
        if np.abs(correlation) < num:
            result_cols.drop(col)
    return result_cols


def lasso_selection(df, colist):
    # 将DataFrame转换为numpy数组
    X = df[colist].values
    y = df['credit_level'].values
    # 创建Lasso对象，alpha值越大，剩余特征数目越少
    lasso = Lasso(alpha=0.1)
    # 使用Lasso进行特征选择
    lasso.fit(X, y)
    # 将特征系数和列名合并成字典
    coef_dict = dict(zip(df[colist].columns, lasso.coef_))
    # 将特征系数按照数值大小排序，并选择系数非零的特征
    keep_features = [feat for feat, coef in coef_dict.items() if abs(coef) > 0]
    return keep_features


def forward_delete_corr(df, colist, threshold):
    # 计算相关系数矩阵
    corr_matrix = df[colist].corr().abs()
    # 提取相关系数大于阈值的变量对
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # 删除相关系数大于阈值的变量并返回结果
    colist_new = [col for col in colist if col not in to_drop]
    return colist_new


def get_low_vif_cols(data):
    # 计算各个特征的方差膨胀因子（VIF）
    vif = pd.DataFrame()
    vif["Feature"] = data.columns
    vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    # 剔除VIF大于10的特征
    low_vif_cols = vif[vif["VIF"] <= 10]["Feature"].tolist()
    return low_vif_cols
