# -*- coding: UTF-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")
train_df.head()

def show_heat_map(corr_largest_df):
    data = np.corrcoef(train_df[corr_largest_df].values.T)
    #print(data.shape)
    #print(train_df[largest_n].dtypes)
    sns.set(font_scale=1.25)
    sns.heatmap(data, yticklabels=corr_largest_df.values, xticklabels=corr_largest_df.values)
    plt.show()
#show_heat_map(largest_n)

train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

# todo 效果有待观察
train_df.loc[train_df.SalePrice > 600000, "SalePrice"] = 600000

#print(train_df.shape)

all_df = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'], test_df.loc[:,'MSSubClass':'SaleCondition']), axis=0,ignore_index=True)
print(all_df.head())
print(all_df.shape)
#all_df[largest_n].head()

#sns.distplot(train_df['SalePrice'])
#sns.distplot(np.log(train_df['SalePrice']))
#sns.distplot(np.log(train_df['SalePrice']), kde=True, hist=True, fit=norm);
#todo 校验正则化，与不正则化的差异
id_vars = np.log(train_df['SalePrice'])


NoBmstIndex = (pd.isnull(all_df["BsmtCond"])==True)&(pd.isnull(all_df["BsmtQual"])==True)
all_df.loc[NoBmstIndex, "BsmtCond"] = all_df.loc[NoBmstIndex, "BsmtCond"].fillna("NA")
all_df.loc[NoBmstIndex, "BsmtQual"] = all_df.loc[NoBmstIndex, "BsmtQual"].fillna("NA")
all_df.loc[NoBmstIndex, "BsmtExposure"] = all_df.loc[NoBmstIndex, "BsmtExposure"].fillna("NA")
all_df.BsmtCond.fillna(all_df.BsmtCond.mode()[0], inplace=True)
all_df.BsmtQual.fillna(all_df.BsmtQual.mode()[0], inplace=True)
all_df.BsmtExposure.fillna(all_df.BsmtExposure.mode()[0], inplace=True)


#处理缺失值
missing_data = all_df.isnull().sum()
missing_data = missing_data[missing_data > 0]
percent = missing_data / all_df.shape[0]
#print(all_df.count())
#print(percent.sort_values(ascending=False)[:100])
#all_df["MiscFeature"].value_counts()
all_df["PoolQC"] = all_df["PoolQC"].fillna("None")
all_df["MiscFeature"] = all_df["MiscFeature"].fillna("None")
all_df["FireplaceQu"] = all_df["FireplaceQu"].fillna("None")
all_df["GarageFinish"] = all_df["GarageFinish"].fillna("None")
all_df["GarageQual"] = all_df["GarageQual"].fillna("None")
all_df["Fence"] = all_df["Fence"].fillna("None")
all_df["KitchenQual"] = all_df["KitchenQual"].fillna("None")
all_df["GarageCond"] = all_df["GarageCond"].fillna("None")
all_df["ExterQual"] = all_df["ExterQual"].fillna("None")
all_df.GarageYrBlt.fillna(all_df.GarageYrBlt.mode()[0], inplace=True)
all_df.GarageArea.fillna(all_df.GarageArea.mode()[0], inplace=True)
all_df.TotalBsmtSF.fillna(all_df.TotalBsmtSF.mode()[0], inplace=True)
all_df.YearBuilt.fillna(all_df.YearBuilt.mode()[0], inplace=True)
#all_df["PoolQC"].value_counts()


from sklearn.cluster import KMeans
data = train_df.groupby(by='Neighborhood').SalePrice.mean()
location = np.array(data.index)
price_lo = np.array(data.values).reshape(-1,1)
km = KMeans(n_clusters=6)
label = km.fit_predict(price_lo)#计算簇中心及为簇分配序号
expense = np.sum(km.cluster_centers_,axis=1)
CityCluster = [[],[],[], [], [], []]
dict_map = {}
for i in range(len(location)):
    #print(label[i])
    #print(location[i])
    CityCluster[label[i]].append(location[i])
    dict_map[location[i]] = label[i]


nei_dict = dict()
new1 = dict()
new2 = dict()
nei_dict = nei_dict.fromkeys(['NoRidge', 'NridgHt', 'StoneBr'],2)
new1 = new1.fromkeys(['Blueste', 'BrDale', 'BrkSide', 'Edwards',
                              'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill',
                              'OldTown', 'SWISU', 'Sawyer'],1)
new2 = new2.fromkeys(['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert',
                              'NWAmes', 'SawyerW', 'Somerst', 'Timber', 'Veenker'],0)
nei_dict.update(new1)
nei_dict.update(new2)
#print(dict_map)

#all_df['Neighborhood'] = all_df['Neighborhood'].map(nei_dict).astype(int)
all_df['Neighborhood'] = all_df['Neighborhood'].map(dict_map).astype(int)

qual_dict = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
all_df["ExterQual"] = all_df["ExterQual"].map(qual_dict).astype(int)


#print(all_df["Neighborhood"].value_counts())


#all_df['Remolded'] = (all_df["YearRemodAdd"] != all_df["YearBuilt"]) * 1


dumies = ["FullBath", "GarageCars", "OverallQual", "TotRmsAbvGrd"]
#train_df["TotRmsAbvGrd"][train_df["TotRmsAbvGrd"] > 600000] = 600000



#sns.distplot(all_df["TotRmsAbvGrd"], fit=norm)
#sns.regplot(x=train_df["TotRmsAbvGrd"],y=train_df["SalePrice"])
#print(pd.get_dummies(train_df["FullBath"], "value").head())
def get_dumps(train_data, value):
    dumps = pd.get_dummies(train_data[value], value)
    return dumps

def list_to_get_dumps(to_code_list):
    in_all_dumps = pd.DataFrame()
    for code in to_code_list:
        in_all_dumps = pd.concat([in_all_dumps, get_dumps(all_df, code)], axis=1)
    return in_all_dumps

one_hot_list = ["FullBath", "OverallQual", "GarageCars", "TotRmsAbvGrd", "FireplaceQu", "GarageFinish",
                "KitchenQual", "GarageCond", "ExterQual", "Neighborhood", "MSSubClass"]
all_dumps = list_to_get_dumps(one_hot_list)
#enc.transform(train_df["FullBath"])
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "FullBath")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "OverallQual")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "GarageCars")], axis=1)
##all_dumps = pd.concat([all_dumps, get_dumps(all_df, "FullBath")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "TotRmsAbvGrd")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "FireplaceQu")], axis=1)
##all_dumps = pd.concat([all_dumps, get_dumps(all_df, "Fence")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "GarageFinish")], axis=1)
##all_dumps = pd.concat([all_dumps, get_dumps(all_df, "GarageYrBlt")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "KitchenQual")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "GarageCond")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "ExterQual")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "Neighborhood")], axis=1)




#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "GarageQual")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "MiscFeature")], axis=1)
#all_dumps = pd.concat([all_dumps, get_dumps(all_df, "PoolQC")], axis=1)


# 准备num类型的数据
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
N = StandardScaler()
nums = ["GrLivArea", "GarageArea", "TotalBsmtSF", "1stFlrSF", "YearBuilt"]
num_features = [f for f in all_df.columns if all_df.dtypes[f] != 'object']

scal_all_data = N.fit_transform(np.log1p(all_df[nums]))
#year_data = pd.DataFrame({"YearBuilt" : (2012 - all_df["YearBuilt"])})
#year_data = pd.DataFrame({"YearBuilt" : (all_df["YearBuilt"])})

#scal_all_data = N.fit_transform(all_df[nums])




#sns.distplot(scal_all_data[:, 3], fit=norm)
#print(all_df[nums].head())
#print(scal_all_data.shape)
#print(type(scal_all_data))
#print(scal_all_data[:10])
#sns.distplot(scal_all_data[3], fit=norm);

#all_dumps = pd.concat([all_dumps, scal_all_data[:, 3]], axis=1)

all_dumps["1stFlrSF"] = scal_all_data[:, 3]
all_dumps["GrLivArea"] = scal_all_data[:, 0]
all_dumps["GarageArea"] = scal_all_data[:, 1]
all_dumps["TotalBsmtSF"] = scal_all_data[:, 2]
all_dumps["YearBuilt"] = scal_all_data[:, 4]

#all_dumps["YearBuilt"] = N.fit_transform(year_data)
#all_dumps["YearBuilt"] = N.fit_transform(np.log1p(year_data))[:, 0]
#sns.distplot(all_dumps["YearBuilt"], fit=norm)
#all_dumps.head()

# 准备测试，训练数据
x_train = all_dumps[:train_df.shape[0]]
x_test = all_dumps[train_df.shape[0]:]
#y_train = np.log1p(train_df["SalePrice"])
y_train = train_df["SalePrice"]
x_train.head()

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
def rmse_cv(model):
    #print(x_train.head())
    diff = -cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv=5)
    rmse= np.sqrt(diff)
    return(rmse)


def kernel_regressor():
    from sklearn.kernel_ridge import KernelRidge
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    KRR.fit(x_train, y_train)
    score = rmse_cv(KRR)
    #y_final_kernel = KRR.predict(x_test)
    mean = score.mean()
    score_std = score.std()
    print("Kernel Ridge score: %f (%f)\n" % (mean, score_std))
    #submission_df = pd.DataFrame(data={'Id': test_df.index, 'SalePrice': y_final_kernel})
    #submission_df.to_csv('./input/submission_kernel.csv', columns=['Id', 'SalePrice'], index=False)
    return KRR


def ridge_regressor():
    #score_test = rmse_cv_test(KRR)

    #print("Kernel Ridge score: mean : {:.4f} std : ({:.4f})".format(score.mean(), score.std()))
    ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
    ridge.fit(x_train, y_train)
    #print(dir(ridge))
    y_final_kernel = ridge.predict(x_test)

    alpha = ridge.alpha_
    print("Best alpha :", alpha)
    print("Try again for more precision with alphas centered around " + str(alpha))
    ridge = RidgeCV(alphas=[alpha * .5,alpha * .55,alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                              alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                              alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],
                    cv = 10)
    #ridge.fit(x_train, y_train)
    #alpha = ridge.alpha_


    #y_final = ridge.predict(x_test)

    #submission_df = pd.DataFrame(data={'Id': test_df.index, 'SalePrice': y_final})
    #submission_df.to_csv('./input/submission_ridge.csv', columns=['Id', 'SalePrice'], index=False)

    print("Best alpha :", alpha)
    print("Ridge RMSE on Training set :", rmse_cv(ridge).mean())
    return ridge

def lass_regressor():
    lasso = LassoCV(alphas = [117,117.5,  118, 118.5, 119])
    lasso.fit(x_train, y_train)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)
    print("Try again for more precision with alphas centered around " + str(alpha))
    lasso = LassoCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8,
                            alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05,
                            alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35,
                            alpha * 1.4],
                    max_iter=50000, cv=10)
    lasso.fit(x_train, y_train)
    print("alpha : ", lasso.alphas)
    print("Lasso RMSE on Training set :", rmse_cv(lasso).mean())
    return lasso


def xgb_regressor():
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV

    cv_params = {'subsample': [0.6, 0.63, 0.64, 0.65,  0.66, 0.67, 0.7, 0.72, 0.75]}
    other_params = {'learning_rate': 0.07, 'n_estimators': 80, 'max_depth': 6, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    model = XGBRegressor(**other_params)
    print('Xgboodt RMSE on training data: %f' % (rmse_cv(model).mean()))
    #optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    return model
    #optimized_GBM.fit(x_train, y_train)
    ##print(dir(optimized_GBM))
    #evalute_result = optimized_GBM.cv_results_
    #print('每轮迭代运行结果:{0}'.format(evalute_result))
    #print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    #print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    #print('Xgboodt RMSE on training data: %f estormator : %d' % (rmse_cv(xgb).mean(), estimators))
    #for estimators in range(1, 10):
    #    estimators *= 100
    #    xgb = XGBRegressor(n_estimators=estimators, learning_rate=0.07, subsample=0.9, colsample_bytree=0.7)
    #    print('Xgboodt RMSE on training data: %f estormator : %d' % (rmse_cv(xgb).mean(), estimators))

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,clone

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights


    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.sum(self.weights * predictions, axis=1)


def ensemele_models():
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
    from sklearn.model_selection import KFold

if __name__ == "__main__":
    lass_model = lass_regressor()
    ridge_model = ridge_regressor()
    kernel_model = kernel_regressor()
    xgb_model = xgb_regressor()

    avger_model = AveragingModels(models=(ridge_model, kernel_model, xgb_model, lass_model), weights=[0.2, 0.2, 0.3, 0.3])
    print('aver model on train:', rmse_cv(avger_model).mean())

    avger_model.fit(x_train, y_train)
    y_final = avger_model.predict(x_test)
    submission_df = pd.DataFrame(data={'Id': test_df.Id, 'SalePrice': y_final})
    submission_df.to_csv('./input/submission_avger.csv', columns=['Id', 'SalePrice'], index=False)


#print(submission_df.head(10))
