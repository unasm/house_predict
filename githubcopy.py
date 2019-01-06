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

#print(test.head())

#train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# todo 效果有待观察
#train.loc[train.SalePrice > 600000, "SalePrice"] = 600000
#plt.figure(figsize=(15,8))
#sns.boxplot(train.YearBuilt, train.SalePrice)
#plt.show()
#full = pd.concat([train, test], axis=0, ignore_index=True)
full = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']), axis=0, ignore_index=True)
#all_df = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'], test_df.loc[:,'MSSubClass':'SaleCondition']), axis=0,ignore_index=True)
#full.drop(['Id'], axis=1, inplace=True)
def fix_na():
    ##print(full.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count']))
    full["LotAreaCut"] = pd.qcut(full.LotArea, 10)
    full['LotFrontage'] = full.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    ##print(full.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count']))
    #print(full["LotFrontage"].value_counts())
    ##full['LotFrontage']=full.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    #full['LotFrontage']=full.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    #print(full[["LotArea", "SalePrice"]].head())
    #print(full[["LotFrontage", "LotAreaCut"]].head())

    cols = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
    for col in cols:
        full[col].fillna(0, inplace=True)

    cols1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
    for col in cols1:
        full[col].fillna("None", inplace=True)
    # fill in with mode
    cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]

    for col in cols2:
        full[col].fillna(full[col].mode()[0], inplace=True)

def map_values():
    """
    todo    尝试下别的dumpy
    :return:
    """
    numStr = ["MSSubClass", "BsmtFullBath", "BsmtHalfBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "MoSold",
              "YrSold", "YearBuilt", "YearRemodAdd", "LowQualFinSF", "GarageYrBlt"]
    for col in numStr:
        full[col] = full[col].astype(str)
    full["oMSSubClass"] = full.MSSubClass.map({'180': 1,
                                               '30': 2, '45': 2,
                                               '190': 3, '50': 3, '90': 3,
                                               '85': 4, '40': 4, '160': 4,
                                               '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                               '120': 6, '60': 6})

    full["oMSZoning"] = full.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

    full["oNeighborhood"] = full.Neighborhood.map({'MeadowV': 1,
                                                   'IDOTRR': 2, 'BrDale': 2,
                                                   'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                   'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                   'NPkVill': 5, 'Mitchel': 5,
                                                   'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                   'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                   'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                   'StoneBr': 9,
                                                   'NoRidge': 10, 'NridgHt': 10})

    full["oCondition1"] = full.Condition1.map({'Artery': 1,
                                               'Feedr': 2, 'RRAe': 2,
                                               'Norm': 3, 'RRAn': 3,
                                               'PosN': 4, 'RRNe': 4,
                                               'PosA': 5, 'RRNn': 5})

    full["oBldgType"] = full.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

    full["oHouseStyle"] = full.HouseStyle.map({'1.5Unf': 1,
                                               '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                               '1Story': 3, 'SLvl': 3,
                                               '2Story': 4, '2.5Fin': 4})

    full["oExterior1st"] = full.Exterior1st.map({'BrkComm': 1,
                                                 'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                 'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3,
                                                 'HdBoard': 3,
                                                 'BrkFace': 4, 'Plywood': 4,
                                                 'VinylSd': 5,
                                                 'CemntBd': 6,
                                                 'Stone': 7, 'ImStucc': 7})

    full["oMasVnrType"] = full.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

    full["oExterQual"] = full.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    full["oFoundation"] = full.Foundation.map({'Slab': 1,
                                               'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                               'Wood': 3, 'PConc': 4})

    full["oBsmtQual"] = full.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oBsmtExposure"] = full.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

    full["oHeating"] = full.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

    full["oHeatingQC"] = full.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oKitchenQual"] = full.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    full["oFunctional"] = full.Functional.map(
        {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

    full["oFireplaceQu"] = full.FireplaceQu.map({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oGarageType"] = full.GarageType.map({'CarPort': 1, 'None': 1,
                                               'Detchd': 2,
                                               '2Types': 3, 'Basment': 3,
                                               'Attchd': 4, 'BuiltIn': 5})

    full["oGarageFinish"] = full.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

    full["oPavedDrive"] = full.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

    full["oSaleType"] = full.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                           'CWD': 2, 'Con': 3, 'New': 3})

    full["oSaleCondition"] = full.SaleCondition.map(
        {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})

    return "Done!"


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
        #X_numeric = X.select_dtypes(["float64", "int64"])
        X_numeric=X.select_dtypes(exclude=["object"])
        #print(X_numeric.dtypes)
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X


def rmse_cv(model, x, y):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


def grid_get(to_run_model, x, y, param_grid):
    grid_search = GridSearchCV(to_run_model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(x, y)
    print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
    grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
    print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])


#pipe = Pipeline([
#    ('labenc', labelenc()),
#    ('skew_dummies', skew_dummies(skew=1))
#])




pipe = Pipeline([
    ('labenc', labelenc()),
    ('skew_dummies', skew_dummies(skew=1))
])

#('add_feature', add_feature(additional=2)),

fix_na()
map_values()

full.drop("LotAreaCut", axis=1, inplace=True)
#full.drop(['SalePrice'], axis=1, inplace=True)

full2 = full.copy()
data_pipe = pipe.fit_transform(full2)
scaler = RobustScaler()
data_scaled = scaler.fit(data_pipe).transform(data_pipe)

n_train = train.shape[0]
x_train = data_pipe[:n_train]
x_test = data_pipe[n_train:]
y_log = np.log(train.SalePrice)

print(x_train.shape)

#pca = PCA(n_components=400)
#x_train = pca.fit_transform(x_train)
#x_test = pca.transform(x_test)
#print(x_train.shape)
#print(x_test.shape)

#print(data_pipe.shape)
#print(data_pipe.head())

models_arr = [
    LinearRegression(),
    Ridge(),
    Lasso(alpha=0.01, max_iter=10000),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),
    LinearSVR(),
    ElasticNet(alpha=0.001, max_iter=10000),
    SGDRegressor(max_iter=1000, tol=1e-3),
    BayesianRidge(),
    KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
    ExtraTreesRegressor(),
    XGBRegressor()]


class AvagerModel(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, model, weight):
        self.model = model
        self.weight = weight

    def fit(self, x, y):
        self.models_ = [clone(sub_model) for sub_model in self.model]
        for model in self.models_:
            model.fit(x, y)
        return self

    def predict(self, x):
        predictions = np.column_stack([model.predict(x) for model in self.models_])
        return np.sum(self.weight * predictions, axis=1)

class StackingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, meta_model):
        self.models = models
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)


    def fit(self, X, y):
        #self.models_ = [clone(x) for x in self.models]
        self.cloned_models = []
        oof_train = []
        for i, sub_model in enumerate(self.models):
            cloned_model = clone(sub_model)
            self.cloned_models.append(cloned_model)
            cloned_model.fit(X, y)
            oof_train.append(cloned_model.predict(X))
        oof_train_data = np.column_stack(oof_train)
        #model.fit(X, y)
        return self.meta_model.fit(oof_train_data, y)

    def predict(self, X):
        whole_data = np.column_stack([new_model.predict(X) for new_model in self.cloned_models])
        return self.meta_model.predict(whole_data)


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


names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD", "Bay", "Ker","Extra","Xgb"]

#print(full.shape)
#for name, model in zip(names, models_arr):
#    score = rmse_cv(model, x_train, y_log)
#    print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))

lasso = Lasso(alpha=0.0002, max_iter=10000)
ridge = Ridge(alpha=25)
svr = SVR(gamma=0.00001, kernel='rbf', C=17, epsilon=0.009)
ker = KernelRidge(alpha=0.09, kernel='polynomial', degree=1, coef0=1.1)
ela = ElasticNet(alpha=0.006, l1_ratio=0.11, max_iter=10000)
bay = BayesianRidge()
xgb = XGBRegressor()

print(xgb.get_xgb_params())
print(dir(xgb))

model_instance_list = [lasso, ridge, svr, ela, bay, xgb]
model_instance_arr = {"lasso": lasso, "ridge": ridge, "svr": svr, "ela": ela, "bay": bay, "ker": ker, "xgb": xgb}

grid_get(xgb, x_train, y_log, {'booster': [15, 16, 17, 18, 19, 20, 21]})

#for name, model in model_instance_arr.iteritems():
#    score = rmse_cv(model, x_train, y_log)
#    print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))

##weight_avg = AveragingModels(models=[lasso, ridge, svr, ela, bay], weights=[0.3, 0.2, 0.1, 0.2, 0.2])
##weight_avg = AveragingModels(models=[lasso, ridge, svr, ela, bay], weights=[0.02, 0.2, 0.25, 0.2, 0.3])
#


#weight_avg = AvagerModel(model=model_instance_list, weight=[0.25, 0.25, 0.05, 0.25, 0.2])
#score = rmse_cv(weight_avg, x_train, y_log)
#print("{}: {:.6f}, {:.4f}".format("avger_model", score.mean(), score.std()))
#weight_avg.fit(x_train, y_log)
#y_final = weight_avg.predict(x_test)

#stackModel = StackingModels(models=model_instance_list, meta_model=xgb)
#score = rmse_cv(stackModel, x_train, y_log)
##print(dir(stackModel.meta_model))
##print(stackModel.meta_model.kernel_params)
##print(stackModel.meta_model.alpha)
##print(stackModel.meta_model.coef0)
##print(stackModel.meta_model.degree)
#print("{}: {:.6f}, {:.4f}".format("stack_model", score.mean(), score.std()))


a_train = Imputer().fit_transform(x_train)
a_test = Imputer().fit_transform(x_test)
b = Imputer().fit_transform(y_log.values.reshape(-1, 1)).ravel()
#
#print(a.shape)
#print(b.shape)
#print(x_train.shape)
#print(y_log.shape)
#
#stackingModel = stacking(mod=model_instance_list, meta_model=ker)
#score = rmse_cv(stackingModel, a_train, b)
##score = rmse_cv(stackingModel, x_train, y_log)
#print("{}: {:.6f}, {:.4f}".format("stack_model", score.mean(), score.std()))

#stackingModel = stacking(mod=model_instance_list, meta_model=ker)
#score = rmse_cv(stackingModel, a_train, b)
##score = rmse_cv(stackingModel, x_train, y_log)
#print("{}: {:.6f}, {:.4f}".format("stack_model_xgb", score.mean(), score.std()))

#stackingModel.fit(a_train, b)
#y_final = stackingModel.predict(a_test)

#submission_df = pd.DataFrame(data={'Id': test.Id, 'SalePrice': np.exp(y_final)})
#submission_df.to_csv('./input/submission_stacking_2.csv', columns=['Id', 'SalePrice'], index=False)
#print(x_train.shape)
#print(x_test.shape)
#print(data_pipe.shape)

#print(data_pipe.head())
#print(data_pipe.shape)
#print(full.isnull().sum()[full.isnull().sum()>0])
