from pandas.errors import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import filter
import prepocess as pp
import warnings
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    cohen_kappa_score, log_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


# import model_evaluation as me

def add_args(parser):
    parser.add_argument("--train_file_path", type=str, default="csv4/train.csv")
    parser.add_argument("--test_file_path", type=str, default="csv4/test.csv")
    parser.add_argument("--model_type", type=str, default="LogisticRegression",
                        help="All chooses: LogisticRegression, DecisionTree, RandomForest, XGBoost")
    args = parser.parse_args()
    return args


def fillna_KNN(df, cols, k):
    # 创建KNNImputer对象，设置要使用的邻居数
    uids = df['uid']
    imputer = KNNImputer(n_neighbors=k)
    # 填补缺失值
    filled_data = imputer.fit_transform(df[cols])
    # 将填补后的数据替换原始DataFrame的值
    df_filled = pd.DataFrame(filled_data, columns=cols)
    df_filled['uid'] = uids
    return df_filled


# 2.1 缺失值处理
def handleMissingValue(df):
    # 计算文件每一列的缺失值比例并保存至 credit_train_missing.csv 文件
    df.isnull().mean().to_csv('credit_train_missing.csv')
    # 去除缺失值比例大于 0.7 的列
    colists = df.select_dtypes(include='object').columns.drop('uid')
    df = pp.label_transform(df, colists)
    df = df.loc[:, df.isnull().mean() < 0.7]
    df = fillna_KNN(df, df.columns.drop('uid'), 6)
    # df.to_csv('credit_knn.csv')
    # # 对于数值型变量，用中位数填充缺失值
    # df.fillna(df.median(), inplace=True)
    # # 对于类别型变量，用众数填充缺失值
    # df.fillna(df.mode().iloc[0], inplace=True)
    return df


def ord_encoder(df, col_name, cate_cols):
    ordinal_encoder = OrdinalEncoder(categories=[cate_cols])
    df[col_name] = ordinal_encoder.fit_transform(df[[col_name]])
    return df


# 2.3 数据转换
def dataTransform(df, catelist):
    le = LabelEncoder()
    df[catelist] = df[catelist].apply(le.fit_transform)
    df.to_csv('credit_train_trans.csv', index=False)
    return df


# 2.4 数据标准化
def dataStandardization(df, colist):
    scaler = StandardScaler()
    df[colist] = scaler.fit_transform(df[colist])
    # 保存处理后的数据至 credit_train_clean.csv 文件
    df.to_csv('credit_train_std.csv', index=False)
    return df


# 2 数据预处理
def preprocess(df):
    # 2.1 缺失值处理
    df = handleMissingValue(df)
    catelist = []  # 非数值列
    colist = []  # 所有列
    for i in df.columns:
        colist.append(i)
        if df[i].dtype == 'object':
            catelist.append(i)
    colist.remove('credit_level')
    colist.remove('uid')
    catelist.remove('uid')
    numlist = [i for i in colist if i not in catelist]

    # 2.2 异常值处理
    df = pp.replace_by_zscore(df, numlist)
    # 2.3 数据转换
    df = dataTransform(df, catelist)
    # 2.4 数据标准化
    df = dataStandardization(df, colist)
    return df, catelist, colist, numlist


# 3 特征工程
def featureEngineering(df, catelist, colist, numlist):
    # 计算变量相关性并保存至 credit_train_corr.csv 文件
    df[colist].corr().to_csv('credit_train_corr.csv')
    # 剔除相关性大于 0.7 的变量
    colist = filter.forward_delete_corr(df[colist], colist, 0.7)
    catelist = [i for i in catelist if i in colist]
    numlist = [i for i in numlist if i in colist]
    # 计算多重共线性并剔除相关性大于 10 的变量
    colist = filter.get_low_vif_cols(df[colist])  # 所有列
    catelist = [i for i in catelist if i in colist]  # 非数值列
    numlist = [i for i in numlist if i in colist]  # 数值列
    # # lasso
    # colist = filter.lasso_selection(df, colist)
    # catelist = [i for i in catelist if i in colist]
    # numlist = [i for i in numlist if i in colist]
    # print(colist)
    return df, catelist, colist, numlist


# 4 训练及预测
def train_and_predict(args, X_train, X_test, y_train):
    lr = LogisticRegression()
    if args.model_type == "DecisionTree":
        lr = DecisionTreeClassifier()
    elif args.model_type == "RandomForest":
        lr = RandomForestClassifier()
    elif args.model_type == "XGBoost":
        lr = XGBClassifier()
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
    elif args.model_type != "LogisticRegression":
        print("No such model!")
    # 训练模型
    lr.fit(X_train, y_train)
    # 预测
    y_pred = lr.predict(X_test)
    if args.model_type == "XGBoost":
        y_pred = le.inverse_transform(y_pred)
    return lr, y_pred


def draw_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels,
           title='Confusion matrix', ylabel='True label', xlabel='Predicted label')
    fig.tight_layout()
    plt.show()


# 5 模型评估
def model_evaluation(args, model, y_test, y_pred, X_test):
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    model_name = '逻辑回归'
    if args.model_type == 'DecisionTree':
        model_name = '决策树'
    elif args.model_type == 'RandomForest':
        model_name = '随机森林'
    elif args.model_type == 'XGBoost':
        model_name = 'XGBoost'
    print('{}模型的准确率为：'.format(model_name), accuracy)

    # 计算混淆矩阵并保存为图片
    # 假设 y_true, y_pred, class_names 已经定义
    class_names = ['35', '50', '60', '70', '85']
    # draw_confusion_matrix(y_test, y_pred, class_names)
    # 计算精确率和召回率
    precision = precision_score(y_test, y_pred, average='macro')  # 计算宏平均精确率
    recall = recall_score(y_test, y_pred, average='macro')  # 计算宏平均召回率
    print("精确率为: ", precision)
    print("召回率为: ", recall)
    # 计算F1分数
    f1 = f1_score(y_test, y_pred, average='macro')  # 计算宏平均F1分数
    print("F1分数为: ", f1)
    # 计算Cohen's Kappa系数
    kappa = cohen_kappa_score(y_test, y_pred)
    print("Cohen's Kappa系数为: ", kappa)
    y_pred_probs = model.predict_proba(X_test)
    logloss = log_loss(y_test, y_pred_probs)
    print("对数损失：", logloss)


# 6 模型应用
def model_application(args, lr):
    credit_test = pd.read_csv(args.test_file_path)
    credit_test.describe().to_csv('credit_test_describe.csv')
    # 处理缺失值
    credit_test = handleMissingValue(credit_test)
    # # 对于数值型变量，用中位数填充缺失值
    # credit_test[numlist].fillna(credit_test[numlist].median(), inplace=True)
    # # 对于类别型变量，用众数填充缺失值
    # credit_test.fillna(credit_test.mode().iloc[0], inplace=True)
    # 数据转换
    le = LabelEncoder()
    credit_test[catelist] = credit_test[catelist].apply(le.fit_transform)
    # 数据标准化
    scaler = StandardScaler()
    credit_test[colist] = scaler.fit_transform(credit_test[colist])

    # 预测
    y_pred = lr.predict(credit_test[colist])
    # 对预测结果进行处理
    y_pred = y_pred.astype(int)
    if args.model_type == 'XGBoost':
        labels = [35, 50, 60, 70, 85]
        for i in range(0, len(y_pred)):
            y_pred[i] = labels[y_pred[i]]
    # 保存预测结果至 credit_test_lr.csv文件
    credit_test['credit_level'] = y_pred
    credit_test.to_csv('credit_test_lr.csv', index=False)


# 添加参数
parser = argparse.ArgumentParser()
args = add_args(parser)
# 读取 credit_train.csv 文件
credit_train = pd.read_csv(args.train_file_path)
# 1 数据盘点
credit_train.describe().to_csv('credit_train_describe.csv')
# 2 数据预处理
credit_train, catelist, colist, numlist = preprocess(credit_train)
# 3 特征工程
credit_train, catelist, colist, numlist = featureEngineering(credit_train, catelist, colist, numlist)
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(credit_train[colist], credit_train['credit_level'], test_size=0.3,
                                                    random_state=0)
# 4 模型预测
lr, y_pred = train_and_predict(args, X_train, X_test, y_train)
# 5 模型评估
model_evaluation(args, lr, y_test, y_pred, X_test)
# 6 模型应用
model_application(args, lr)
