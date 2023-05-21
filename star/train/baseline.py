import pandas as pd
from sklearn.preprocessing import StandardScaler
import filter
import prepocess as pp

'''
filter和preporcess的代码需要大家自己写，这里只是调用
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import model_evaluation as me

# 读取 star_train.csv 文件
star_train = pd.read_csv('star_train.csv')
# 1.2 数据盘点
star_train.describe().to_csv('star_train_describe.csv')
# 2.1 缺失值处理
# 计算文件每一列的缺失值比例并保存至 star_train_missing.csv 文件
star_train.isnull().mean().to_csv('star_train_missing.csv')
# 去除缺失值比例大于 0.7 的列
star_train = star_train.loc[:, star_train.isnull().mean() < 0.7]
# 对于数值型变量，用中位数填充缺失值
star_train.fillna(star_train.median(), inplace=True)
# 对于类别型变量，用众数填充缺失值
star_train.fillna(star_train.mode().iloc[0], inplace=True)
# 输出处理后的列名
print(star_train.columns)
# 输出处理后的数据类型
print(star_train.dtypes)
catelist = []
colist = []
for i in star_train.columns:
    colist.append(i)
    if star_train[i].dtype == 'object':
        catelist.append(i)
colist.remove('star_level')
colist.remove('uid')
catelist.remove('uid')
numlist = [i for i in colist if i not in catelist]
# 2.2 异常值处理
# <5%和>95%的使用5%和95%的数据替换
star_train = pp.replace_data(star_train)
# 2.3 数据转换
le = LabelEncoder()
star_train[catelist] = star_train[catelist].apply(le.fit_transform)
star_train.to_csv('star_train_trans.csv', index=False)
# 2.4 数据标准化
scaler = StandardScaler()
star_train[colist] = scaler.fit_transform(star_train[colist])
# 保存处理后的数据至 star_train_clean.csv 文件
star_train.to_csv('star_train_std.csv', index=False)
# 3 特征工程
# 3.1 计算变量相关性
# 计算变量相关性并保存至 star_train_corr.csv 文件
star_train[colist].corr().to_csv('star_train_corr.csv')
# 剔除相关性大于 0.7 的变量
colist = filter.forward_delete_corr(star_train[colist], colist, 0.7)
catelist = [i for i in catelist if i in colist]
numlist = [i for i in numlist if i in colist]

# 3.2 计算多重共线性
# 计算多重共线性并剔除相关性大于 0.7 的变量
colist = filter.get_low_vif_cols(star_train[colist])
catelist = [i for i in catelist if i in colist]
numlist = [i for i in numlist if i in colist]
# 4 模型预测
# 4.1 逻辑回归模型
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(star_train[colist], star_train['star_level'], test_size=0.3,
                                                    random_state=0)
# 训练模型
lr = LogisticRegression()
lr.fit(X_train, y_train)
# 预测
y_pred = lr.predict(X_test)

# 5 模型评估
# 5.1 计算准确率
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('逻辑回归模型的准确率为：', accuracy)
# 5.2 混淆矩阵
# 计算混淆矩阵并保存为图片
# 假设 y_true, y_pred, class_names 已经定义
save_path = 'confusion_matrix_lr.png'
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
me.plot_confusion_matrix(y_test, y_pred, class_names, save_path)
# 5.3 计算精确率和召回率
precision = precision_score(y_test, y_pred, average='macro')  # 计算宏平均精确率
recall = recall_score(y_test, y_pred, average='macro')  # 计算宏平均召回率
# 打印出来
print("精确率为: ", precision)
print("召回率为: ", recall)
# 5.4 计算F1分数
f1 = f1_score(y_test, y_pred, average='macro')  # 计算宏平均F1分数
print("F1分数为: ", f1)
# 计算Cohen's Kappa系数
kappa = cohen_kappa_score(y_test, y_pred)
print("Cohen's Kappa系数为: ", kappa)

# 6 模型应用
star_test = pd.read_csv('star_test.csv')
# 6.1 数据处理
# 对于数值型变量，用中位数填充缺失值
star_test[numlist].fillna(star_test[numlist].median(), inplace=True)
# 对于类别型变量，用众数填充缺失值
star_test.fillna(star_test.mode().iloc[0], inplace=True)
# 6.1.2 数据转换
le = LabelEncoder()
star_test[catelist] = star_test[catelist].apply(le.fit_transform)
# 6.1.3 数据标准化
scaler = StandardScaler()
star_test[colist] = scaler.fit_transform(star_test[colist])
# 6.2 模型预测
# 6.2.1 逻辑回归模型
# 预测
y_pred = lr.predict(star_test[colist])
# 对预测结果进行处理
y_pred = y_pred.astype(int)
# 保存预测结果至 star_test_lr.csv 文件
star_test['star_level'] = y_pred
star_test.to_csv('star_test_lr.csv', index=False)
