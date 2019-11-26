# 导包

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import tree

import warnings
warnings.filterwarnings('ignore')

"""
随机森林模型

数据导入和预处理: 加载数据(已被整理好的训练数据和测试数据), 进行数据的预处理
划分数据集: 将训练数据分为训练数据和验证数据, 验证模型并进一步优化模型
处理缺失值: 使用Imputer处理数据, 将缺失值用均值替代
建立模型: 使用训练集中的训练数据建立随机森林模型
          处理不平衡数据分布, 赋予正负样本不同的惩罚权重
          使用交叉验证进行参数调整
          输出最佳模型, 使用测试数据进行预测

"""


# 创建字典函数
def createDictKV(keys, vals):
    lookup = {}
    if len(keys) == len(vals):
        for i in range(len(keys)):
            key = keys[i]
            val = vals[i]
            lookup[key] = val
    return lookup


# 创建包含AUC的函数, 传入参数并打印AUC的值
def computeAUC(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    print("auc=", auc)
    return auc


def main():
    # 导入数据与数据的预处理
    colnames = ['ID', 'label', 'RUUnsecuredL', 'age', 'NOTime30-59',
                'DebtRatio', 'Income', 'NOCredit', 'NOTimes90',
                'NORealEstate', 'NOTime60-89', 'NODependents']
    col_nas = ['', 'NA', 'NA', 0, [98, 96], 'NA', 'NA', 'NA', [98, 96], 'NA', [98, 96], 'NA']
    col_na_values = createDictKV(colnames, col_nas)
    dftrain = pd.read_csv('data/cs-training.csv', names=colnames, na_values=col_na_values, skiprows=[0])

    train_id = [int(x) for x in dftrain.pop('ID')]
    y_train = np.asarray([int(x) for x in dftrain.pop('label')])
    x_train = dftrain.as_matrix()

    dftest = pd.read_csv('data/cs-test.csv', names=colnames, na_values=col_na_values, skiprows=[0])
    test_id = [int(x) for x in dftest.pop('ID')]
    y_test = np.asarray(dftest.pop('label'))
    x_test = dftest.as_matrix()

    # 将训练数据集拆分为训练数据和验证数据, 用于交叉验证
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33333, random_state=0)
    for train_index, test_index in sss.split(x_train, y_train):
        print('TRAIN:', train_index, 'TEST:', test_index)
        x_train_new, x_test_new = x_train[train_index], x_train[test_index]
        y_train_new, y_test_new = y_train[train_index], y_train[test_index]

    x_train = x_train_new
    y_train = y_train_new

    # 将缺失值替换为平均值
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(x_train)
    x_train = imp.transform(x_train)
    x_test_new = imp.transform(x_test_new)
    x_test = imp.transform(x_test)

    # 使用训练数据建立随机森林模型

    rf = RandomForestClassifier(n_estimators=100,
                                oob_score=True,
                                min_samples_split=2,
                                min_samples_leaf=50,
                                n_jobs=-1,
                                class_weight='balanced_subsample',
                                bootstrap=True)

    # 建立逻辑回归模型, 查看模型的效果
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # 查看训练集模型效果, 使用AUC
    pre_probs_train = lr.predict_proba(x_train)
    pre_probs_train = [x[1] for x in pre_probs_train]
    computeAUC(y_train, pre_probs_train)
    # 查看验证集模型效果, 使用AUC, 即ROC曲线下的面积衡量模型预测效果
    pre_probs_test_new = lr.predict_proba(x_test_new)
    pre_probs_test_new = [x[1] for x in pre_probs_test_new]
    computeAUC(y_test_new, pre_probs_test_new)

    # 建立决策树模型, 并查看模型效果
    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)
    # 查看训练集模型效果
    pre_probs_train = model.predict_proba(x_train)
    pre_probs_train = [x[1] for x in pre_probs_train]
    computeAUC(y_train, pre_probs_train)
    # 查看验证集模型效果
    pre_probs_test_new = lr.predict_proba(x_test_new)
    pre_probs_test_new = [x[1] for x in pre_probs_test_new]
    computeAUC(y_test_new, pre_probs_test_new)

    # 输出特征重要性评估
    rf.fit(x_train, y_train)
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), dftrain.columns), reverse=True))

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = dftrain.columns
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))

    # 交叉验证
    param_grid = {'max_features': [2, 3, 4], 'min_samples_leaf': [50]}
    grid_search = GridSearchCV(rf, cv=10, scoring='roc_auc', param_grid=param_grid, iid=False)

    # 输出最佳模型, 使用测试数据进行预测
    # 使用最优参数和training_new数据构建模型
    grid_search.fit(x_train, y_train)
    print('the best parameter:', grid_search.best_estimator_)
    print('the best score:', grid_search.best_score_)

    # 使用上述训练模型预测train_new数据
    pre_probs_train = grid_search.predict_proba(x_train)
    pre_probs_train = [x[1] for x in pre_probs_train]
    computeAUC(y_train, pre_probs_train)

    # 使用训练模型预测test_new数据, 即交叉验证
    pre_probs_test_new = grid_search.predict_proba(x_test_new)
    pre_probs_test_new = [x[1] for x in pre_probs_test_new]
    computeAUC(y_test_new, pre_probs_test_new)

    # 使用上述模型预测test data
    pre_probs_test = grid_search.predict_proba(x_test)
    pre_probs_test = ['%.9f' % x[1] for x in pre_probs_test]
    submission = pd.DataFrame({'ID': test_id, 'Probability': pre_probs_test})
    submission.to_csv('rf_benchmark.csv', index=False)


if __name__ == '__main__':
    main()
