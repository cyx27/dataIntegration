# XGBoost
import pandas as pd
import xgboost as xgb  # 修改1：导入XGBoost
from sklearn.covariance import EllipticEnvelope
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# 读取数据文件
df_train = pd.read_csv('../../csv/all_merged_data_tran_cnt_info_pk_marked.csv')
df_test = pd.read_csv('../../csv/all_merged_data_tran_cnt_info_pk_unmarked.csv')

# 观察数据缺失情况
print('数据缺失情况：\n', df_train.isnull().sum())

# 填补缺失值
imputer = KNNImputer(n_neighbors=2)
df_train[['avg_mth', 'avg_year', 'total_tran_amt', 'tran_cnt']] = imputer.fit_transform(df_train[['avg_mth', 'avg_year', 'total_tran_amt', 'tran_cnt']])
df_test[['avg_mth', 'avg_year', 'total_tran_amt', 'tran_cnt']] = imputer.transform(df_test[['avg_mth', 'avg_year', 'total_tran_amt', 'tran_cnt']])

# 异常值处理
X_train = df_train[['avg_mth', 'avg_year', 'total_tran_amt', 'tran_cnt']].values
X_test = df_test[['avg_mth', 'avg_year', 'total_tran_amt', 'tran_cnt']].values

outlier_detector = EllipticEnvelope()
outlier_detector.fit(X_train)
outliers = outlier_detector.predict(X_train) == -1
X_train = X_train[~outliers, :]


# 将特征与标签拆分为X和y
X_train = df_train[['avg_mth', 'avg_year', 'total_tran_amt',
                    'tran_cnt', 'is_shareholder', 'is_black', 'is_contact', 'is_mgr_dep']].replace(
    {'Y': True, 'N': False}).values
y_train = df_train['star_level'].values
# 将所有值减1，针对报错：ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4 5 6 7 8], got [1 2 3 4 5 6 7 8 9]
y_train = [y - 1 for y in y_train]
# 将测试集中的star_level设置为-1以进行预测
X_test = df_test[['avg_mth', 'avg_year', 'total_tran_amt',
                  'tran_cnt', 'is_shareholder', 'is_black', 'is_contact', 'is_mgr_dep']].replace(
    {'Y': True, 'N': False}).values
y_test = [-1] * len(df_test)
uid_test = df_test['uid'].values  # 修改处：获取测试集的 uid 列

# 数据预处理：标准化特征值
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将数据集分为训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# 定义并训练XGBoost模型（默认使用GBtree booster）
xgbc = xgb.XGBClassifier()  # 修改2：创建XGBoost分类器，这里使用默认超参数设置
xgbc.fit(X_train, y_train)

# 在验证集上评估性能
y_pred_val = xgbc.predict(X_valid)

acc = accuracy_score(y_valid, y_pred_val)
prec = precision_score(y_valid, y_pred_val, average='macro')
rec = recall_score(y_valid, y_pred_val, average='macro')
f1 = f1_score(y_valid, y_pred_val, average='macro')
kappa = cohen_kappa_score(y_valid, y_pred_val)

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1 score: {f1}")
print(f"Cohen's Kappa: {kappa}")

# 使用训练好的模型对测试集进行预测，并输出结果
y_pred = xgbc.predict(X_test)  # 修改3：使用XGBoost模型进行预测
y_pred = [y + 1 for y in y_pred]
df_test = pd.DataFrame({'uid': uid_test, 'star_level': y_pred})
df_test.to_csv('predicted_star_level_info_XGBoost.csv', index=False)
