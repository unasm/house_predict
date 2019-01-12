# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#matplotlib inline
plt.style.use('ggplot')


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer

from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

full = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']), axis=0, ignore_index=True)


def add_neig():
    data = train.groupby(by='Neighborhood').SalePrice.mean()
    location = np.array(data.index)
    price_lo = np.array(data.values).reshape(-1, 1)
    km = KMeans(n_clusters=6)
    label = km.fit_predict(price_lo)  # 计算簇中心及为簇分配序号
    expense = np.sum(km.cluster_centers_, axis=1)
    CityCluster = [[], [], [], [], [], []]
    dict_map = {}
    for i in range(len(location)):
        # print(label[i])
        # print(location[i])
        CityCluster[label[i]].append(location[i])
        dict_map[location[i]] = label[i]
    print(dict_map)
    nei_dict = dict()
    new1 = dict()
    new2 = dict()
    nei_dict = nei_dict.fromkeys(['NoRidge', 'NridgHt', 'StoneBr'], 2)
    new1 = new1.fromkeys(['Blueste', 'BrDale', 'BrkSide', 'Edwards',
                          'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill',
                          'OldTown', 'SWISU', 'Sawyer'], 1)
    new2 = new2.fromkeys(['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert',
                          'NWAmes', 'SawyerW', 'Somerst', 'Timber', 'Veenker'], 0)
    nei_dict.update(new1)
    nei_dict.update(new2)
    # print(dict_map)

    # print(full["Neighborhood"].value_counts())
    # print(full["Neighborhood"].isna().count())
    # print(full.shape)
    full['Neighborhood_2'] = full.Neighborhood.map(dict_map).astype(int)
    # print(full["Neighborhood_2"].value_counts())


def rmse_cv(model, x, y):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


def grid_get(to_run_model, x, y, param_grid):
    grid_search = GridSearchCV(to_run_model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(x, y)
    print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
    grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
    print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])


# plt.figure(figsize=(15,8))
# sns.boxplot(train.YearBuilt, train.SalePrice)
# plt.show()
# full = pd.concat([train, test], axis=0, ignore_index=True)

# full.BsmtCond.fillna(full.BsmtCond.mode()[0], inplace=True)

# all_df = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'], test_df.loc[:,'MSSubClass':'SaleCondition']), axis=0,ignore_index=True)
# full.drop(['Id'], axis=1, inplace=True)


class fix_na(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ##print(full.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count']))
        X["LotAreaCut"] = pd.qcut(X.LotArea, 10)
        X['LotFrontage'] = X.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        ##print(full.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count']))
        # print(full["LotFrontage"].value_counts())
        ##full['LotFrontage']=full.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        # full['LotFrontage']=full.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        # print(full[["LotArea", "SalePrice"]].head())
        # print(full[["LotFrontage", "LotAreaCut"]].head())

        cols = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
        for col in cols:
            X[col].fillna(0, inplace=True)

        cols1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish",
                 "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1",
                 "MasVnrType"]
        for col in cols1:
            X[col].fillna("None", inplace=True)
        # fill in with mode
        cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual",
                 "SaleType", "Exterior1st", "Exterior2nd"]
        for col in cols2:
            X[col].fillna(X[col].mode()[0], inplace=True)
        X.drop("LotAreaCut", axis=1, inplace=True)
        return X


# def map_values():
class map_values(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        todo    尝试下别的dumpy
        :return:
        """
        numStr = ["MSSubClass", "BsmtFullBath", "BsmtHalfBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
                  "MoSold", "YrSold", "YearBuilt", "YearRemodAdd", "LowQualFinSF", "GarageYrBlt"]
        for col in numStr:
            X[col] = X[col].astype(str)
        X["oMSSubClass"] = X.MSSubClass.map({'180': 1,
                                             '30': 2, '45': 2,
                                             '190': 3, '50': 3, '90': 3,
                                             '85': 4, '40': 4, '160': 4,
                                             '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                             '120': 6, '60': 6})

        X["oMSZoning"] = X.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

        X["oNeighborhood"] = X.Neighborhood.map({'MeadowV': 1,
                                                 'IDOTRR': 2, 'BrDale': 2,
                                                 'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                 'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                 'NPkVill': 5, 'Mitchel': 5,
                                                 'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                 'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                 'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                 'StoneBr': 9,
                                                 'NoRidge': 10, 'NridgHt': 10})

        X["oCondition1"] = X.Condition1.map({'Artery': 1,
                                             'Feedr': 2, 'RRAe': 2,
                                             'Norm': 3, 'RRAn': 3,
                                             'PosN': 4, 'RRNe': 4,
                                             'PosA': 5, 'RRNn': 5})

        X["oBldgType"] = X.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

        X["oHouseStyle"] = X.HouseStyle.map({'1.5Unf': 1,
                                             '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                             '1Story': 3, 'SLvl': 3,
                                             '2Story': 4, '2.5Fin': 4})

        X["oExterior1st"] = X.Exterior1st.map({'BrkComm': 1,
                                               'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                               'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3,
                                               'HdBoard': 3,
                                               'BrkFace': 4, 'Plywood': 4,
                                               'VinylSd': 5,
                                               'CemntBd': 6,
                                               'Stone': 7, 'ImStucc': 7})

        X["oMasVnrType"] = X.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

        X["oExterQual"] = X.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

        X["oFoundation"] = X.Foundation.map({'Slab': 1,
                                             'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                             'Wood': 3, 'PConc': 4})

        X["oBsmtQual"] = X.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

        X["oBsmtExposure"] = X.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

        X["oHeating"] = X.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

        X["oHeatingQC"] = X.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        X["oKitchenQual"] = X.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

        X["oFunctional"] = X.Functional.map(
            {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

        X["oFireplaceQu"] = X.FireplaceQu.map({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        X["oGarageType"] = X.GarageType.map({'CarPort': 1, 'None': 1,
                                             'Detchd': 2,
                                             '2Types': 3, 'Basment': 3,
                                             'Attchd': 4, 'BuiltIn': 5})

        X["oGarageFinish"] = X.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

        X["oPavedDrive"] = X.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

        X["oSaleType"] = X.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                         'CWD': 2, 'Con': 3, 'New': 3})

        X["oSaleCondition"] = X.SaleCondition.map(
            {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})

        return X


class labelenc(BaseEstimator, TransformerMixin):
    """
    标签化处理
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lab = LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X


class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self, additional=1):
        self.additional = additional

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.additional == 1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]

            X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
            X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
            X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]

            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"] + X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"] + X[
                "EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
            return X


class skew_dummies(BaseEstimator, TransformerMixin):
    """
    处理偏态
    """

    def __init__(self, skew=0.5):
        self.skew = skew

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X_numeric = X.select_dtypes(["float64", "int64"])
        X_numeric = X.select_dtypes(exclude=["object"])
        # print(X_numeric.dtypes)
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X


class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mod, meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))

        for i, model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X, y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index, i] = renew_model.predict(X[val_index])

        self.meta_model.fit(oof_train, y)
        return self

    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)

    #def get_oof(self, X, y, test_X):
    #    oof = np.zeros((X.shape[0], len(self.mod)))
    #    test_single = np.zeros((test_X.shape[0], 5))
    #    test_mean = np.zeros((test_X.shape[0], len(self.mod)))
    #    for i, model in enumerate(self.mod):
    #        for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):
    #            clone_model = clone(model)
    #            clone_model.fit(X[train_index], y[train_index])
    #            oof[val_index, i] = clone_model.predict(X[val_index])
    #            test_single[:, j] = clone_model.predict(test_X)
    #        test_mean[:, i] = test_single.mean(axis=1)
    #    return oof, test_mean



# pipe = Pipeline([
#    ('labenc', labelenc()),
#    ('skew_dummies', skew_dummies(skew=1))
# ])





def get_outliner(fulldata, n_train, model, y_log):
    data_pipe = pipe.fit_transform(fulldata)
    #data_pipe.head()
    scaler = RobustScaler()
    data_scaled = scaler.fit(data_pipe).transform(data_pipe)
    #n_train = train.shape[0]
    x_train = data_pipe[:n_train]
    #x_train = data_scaled[:n_train]
    #x_test = data_scaled[n_train:]


    score = rmse_cv(model, x_train, y_log)
    model.fit(x_train, y_log)
    y_pred = model.predict(x_train)
    #diff = abs(y_pred - y_log)
    #print(diff.sort_values()[-20:])
    #print(diff.sort_values()[:10])
    print("{}: {:.6f}, {:.4f}".format("ela", score.mean(), score.std()))
    outliner = x_train[abs(y_pred - y_log) >= 0.294232]
    return outliner


def run_model(fulldata, n_train, model, y_log, is_output=False, is_stacking=False):
    data_pipe = pipe.fit_transform(fulldata)
    #data_pipe.head()
    scaler = RobustScaler()
    data_scaled = scaler.fit(data_pipe).transform(data_pipe)
    #n_train = train.shape[0]
    x_train = data_scaled[:n_train]
    x_test = data_scaled[n_train:]
    #y_log = np.log(train.SalePrice)

    if is_stacking:
        x_train = Imputer().fit_transform(x_train)
        x_test = Imputer().fit_transform(x_test)
        y_log = Imputer().fit_transform(y_log.values.reshape(-1, 1)).ravel()
        #print(test[(test['GrLivArea'] > 3500)].Id)

    if is_output:
        model.fit(x_train, y_log)
        y_pred = model.predict(x_test)
        submission_df = pd.DataFrame(data={'Id': test.Id, 'SalePrice': np.exp(y_pred)})
        submission_df.to_csv('./input/submission_stacking_outline_droped.csv', columns=['Id', 'SalePrice'], index=False)
    else:
        score = rmse_cv(model, x_train, y_log)
        #diff = abs(y_pred - y_log)
        #print(diff.sort_values()[-20:])
        #print(diff.sort_values()[:10])
        print("{}: {:.6f}, {:.4f}".format("ela", score.mean(), score.std()))

# 标准的full 数据处理流程
pipe = Pipeline([
    ("fix_na", fix_na()),
    ("map_values", map_values()),
    ('labenc', labelenc()),
    ('skew_dummies', skew_dummies(skew=1))
])

lasso = Lasso(alpha=0.0002, max_iter=10000)
ridge = Ridge(alpha=25)
svr = SVR(gamma=0.00001, kernel='rbf', C=17, epsilon=0.009)
ker = KernelRidge(alpha=0.09, kernel='polynomial', degree=1, coef0=1.1)
ela = ElasticNet(alpha=0.006, l1_ratio=0.11, max_iter=10000)
bay = BayesianRidge()
xgb = XGBRegressor(silent=1, n_estimators=260, learning_rate=0.085, max_depth=4, min_child_weight=3)
rfr = RandomForestRegressor(max_depth=12, random_state=0, n_estimators=400)

#print(xgb.get_xgb_params())
#print(dir(xgb))

#model_instance_list = [lasso, ridge, svr, ela, xgb]
model_instance_list = [lasso, ridge, svr, ela, bay, xgb, ker, rfr]
model_instance_arr = {"lasso": lasso, "ridge": ridge, "svr": svr, "ela": ela, "bay": bay, "ker": ker, "xgb": xgb}

#score = rmse_cv(lasso, x_train, y_log)
#print("{}: {:.6f}, {:.4f}".format("lasso", score.mean(), score.std()))

lasso = Lasso(alpha=0.0002, max_iter=10000)
full2 = full.copy()
outliner = get_outliner(full2, train.shape[0], lasso, np.log(train.SalePrice))


to_drop_train = train.copy()
train_droped = to_drop_train.drop(outliner.index)
full3 = pd.concat((train_droped.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']), axis=0, ignore_index=True)
print(train.shape)
run_model(full3.copy(), train_droped.shape[0], lasso, False, np.log(train_droped.SalePrice))

stackingModel = stacking(mod=model_instance_list, meta_model=ker)

run_model(full3.copy(), train_droped.shape[0], stackingModel, np.log(train_droped.SalePrice), is_output=False, is_stacking=True)
