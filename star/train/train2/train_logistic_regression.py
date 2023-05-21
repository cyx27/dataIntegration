# 逻辑回归
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取数据文件
df_train = pd.read_csv('../../csv/all_merged_data_tran_cnt_marked.csv')
df_test = pd.read_csv('../../csv/all_merged_data_tran_cnt_unmarked.csv')

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
X_train = df_train[['avg_mth', 'avg_year', 'total_tran_amt', 'tran_cnt']].values
y_train = df_train['star_level'].values
# 将测试集中的star_level设置为-1以进行预测
X_test = df_test[['avg_mth', 'avg_year', 'total_tran_amt', 'tran_cnt']].values
y_test = [-1] * len(df_test)

# 数据预处理：标准化特征值
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建一个特征名称列表
feature_names = ['avg_mth', 'avg_year', 'total_tran_amt', 'tran_cnt']

# 计算特征之间的相关矩阵
corr_matrix = pd.DataFrame(X_train, columns=feature_names).corr()
corr_matrix.columns, corr_matrix.index = feature_names, feature_names
print("特征间相关矩阵：\n", corr_matrix)

# 计算特征之间的多重共线性
vif_series = pd.Series([variance_inflation_factor(
    X_train, i) for i in range(X_train.shape[1])], index=df_train.columns[1:-1])
print("多重共线性：\n", vif_series)

# 将数据集分为训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 定义并训练逻辑回归模型
lr = LogisticRegression()
lr.fit(X_train, y_train)

# 在验证集上评估性能
y_pred_val = lr.predict(X_valid)

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
y_pred = lr.predict(X_test)
df_test['star_level'] = y_pred
df_test.to_csv('predicted_star_level_logistic_regression.csv', index=False)
