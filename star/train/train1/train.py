import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 数据预处理
train_df = pd.read_csv('../../csv/tr_mx/sum_marked.csv')
test_df = pd.read_csv('../../csv/tr_mx/sum_unmarked.csv')

# 去重
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

# 缺失值填充，这里不需要填充

# 异常值处理，这里不需要处理

# 2. 特征工程
feature_cols = ['dsf_amt', 'dsf_count', 'etc_amt', 'etc_count', 'grwy_amt', 'grwy_count', 'gzdf_amt', 'gzdf_count', 'sa_amt', 'sa_count', 'sbyb_amt', 'sbyb_count', 'sdrq_amt', 'sdrq_count', 'shop_amt', 'shop_count', 'sjyh_amt', 'sjyh_count']
label_col = 'star_level'

X_train = train_df[feature_cols]
y_train = train_df[label_col]
X_test = test_df[feature_cols]

# 3. 特征选择，这里不需要手动选择

# 4. 模型选择
clf = LogisticRegression()

# 5. 模型训练
clf.fit(X_train, y_train)

# 6. 模型评估
y_pred = clf.predict(X_test)

# 将预测结果写入测试集
test_df[label_col] = y_pred
test_df.to_csv('sum_predicted.csv', index=False)

# 计算模型评价指标并输出
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
clf.fit(X_train_val, y_train_val)
y_pred_val = clf.predict(X_test_val)

acc = accuracy_score(y_test_val, y_pred_val)
prec = precision_score(y_test_val, y_pred_val, average='macro')
rec = recall_score(y_test_val, y_pred_val, average='macro')
f1 = f1_score(y_test_val, y_pred_val, average='macro')

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1 score: {f1}")