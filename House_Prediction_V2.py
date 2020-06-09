
import copy 
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy import stats
from scipy.stats import pearsonr
import warnings
import scipy.stats as stats
import pylab 
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')
%matplotlib inline

# train data
df_train = pd.read_csv('/Users/amanprasad/Documents/Kaggle/House Prices/House_File_V2/df_train_Splitted_V1.csv')
df_train_1=copy.deepcopy(df_train)

# test data
df_validation = pd.read_csv('/Users/amanprasad/Documents/Kaggle/House Prices/House_File_V2/df_validation_FeatEngg_9.csv')
df_validation_1=copy.deepcopy(df_validation)

#-------------------------------------------------------------------------------------------------------------------------------------------
# data preparation

Y_train=df_train_1['SalePrice']
df_train_1.drop('SalePrice',axis=1,inplace=True)
X_train=df_train_1

Y_validation=df_validation_1['SalePrice']
df_validation_1.drop('SalePrice',axis=1,inplace=True)
X_validation=df_validation_1

def display_scores(scores):
    print("Scores RMSE:", scores)
    print("Mean RMSE:", scores.mean())
    print("Standard deviation of RMSE:", scores.std())


#-------------------------------------------------------------------------------------------------------------------------------------------
# Prediction
#-------------------------------------------------------------------------------------------------------------------------------------------
# Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train, Y_train)

# find error with cross validation
from sklearn.model_selection import cross_val_score
lin_scores=cross_val_score(lin_reg,X_train,Y_train,scoring='neg_mean_squared_error',cv=5)
lin_rmse_scores=np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)

#Scores RMSE: [1.37505028e+09 5.39345126e+08 1.50375973e+09 8.46562861e+07
# 8.89934240e+08]
#Mean RMSE: 878549133.2543052
#Standard deviation of RMSE: 525900468.9223266

# Prediction
prediction_lin=lin_reg.predict(X_validation)

lin_test_mse=((Y_validation-prediction_lin)**2).sum()/len(Y_validation)
lin_test_rmse=np.sqrt(lin_test_mse)
lin_test_rmse
# 23736807855.154545

#-----------------------------------------------------------------------------
# Linear Regression using Stochastic Gradient Descent
# penalty=l1
from sklearn.linear_model import SGDRegressor
sgd_reg=SGDRegressor(penalty='None',max_iter=1000,eta0=0.001,random_state=60616)
sgd_reg.fit(X_train,Y_train.ravel())
predict_lin_Stochastic=sgd_reg.predict(X_validation)

# Prediction
lin_test_mse=((Y_validation-predict_lin_Stochastic)**2).sum()/len(Y_validation)
lin_test_rmse=np.sqrt(lin_test_mse)
lin_test_rmse
# 0.903577214121022


#-----------------------------------------------------------------------------
# Linear Regression using Mini Batch Gradient Descent






#-------------------------------------------------------------------------------------------------------------------------------------------
# Ridge Regression

from sklearn.linear_model import Ridge


# applying grid search to find alpha
# prepare a range of alpha values to test
alphas = np.array([100,50,29,27,25,23,21,20,15,10,1,0.1,0.01,0.001,0.0001,0])
# create and fit a ridge regression model, testing each alpha
model = Ridge()
grid = GridSearchCV(cv=5,estimator=model, param_grid=dict(alpha=alphas),scoring='neg_mean_squared_error')
grid.fit(X_train, Y_train)
print(grid)

# summarize the results of the grid search
ridge_train_rmse=np.sqrt(-grid.best_score_)
print(ridge_train_rmse)
print(grid.best_estimator_.alpha)

#0.11382586579804059
#27.0

cvres=grid.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)

'''
0.11575067684382939 {'alpha': 100.0}
0.11422010464905469 {'alpha': 50.0}
0.11383342245853126 {'alpha': 29.0}
0.11382586579804059 {'alpha': 27.0}
0.11382784580670703 {'alpha': 25.0}
0.11384146167159667 {'alpha': 23.0}
0.11386946465922634 {'alpha': 21.0}
0.11388996503861677 {'alpha': 20.0}
0.11408416368334312 {'alpha': 15.0}
0.11454845214036802 {'alpha': 10.0}
0.11979849913249614 {'alpha': 1.0}
0.12573083787573025 {'alpha': 0.1}
0.12756382315042417 {'alpha': 0.01}
0.12780002787051198 {'alpha': 0.001}
0.127824426699984 {'alpha': 0.0001}
1886658606165.024 {'alpha': 0.0}
'''

# Prediction
ridge_reg = Ridge(alpha=27, solver="cholesky")
ridge_reg.fit(X_train, Y_train)

prediction_ridge=ridge_reg.predict(X_validation)

ridge_test_mse=((Y_validation-prediction_ridge)**2).sum()/len(Y_validation)
ridge_test_rmse=np.sqrt(ridge_test_mse)
ridge_test_rmse
# 0.15502651269939982

#-----------------------------------------------------------------------------
# using Stochastic Gradient Descent
# penalty=l2
sgd_reg=SGDRegressor(penalty='l2',max_iter=1000,eta0=0.01,alpha=27,shuffle=True)
sgd_reg.fit(X_train,Y_train.ravel())

# find error with cross validation
ridge_scores_stochastic=cross_val_score(sgd_reg,X_train,Y_train,scoring='neg_mean_squared_error',cv=5)
ridge_rmse_scores_stochastic=np.sqrt(-ridge_scores_stochastic)

display_scores(ridge_rmse_scores_stochastic)
#Scores RMSE: [0.23861149 0.33331774 0.25138662 0.22804513 0.22653649]
#Mean RMSE: 0.2555794950868803
#Standard deviation of RMSE: 0.03987572074230565

# Prediction
ridge_predict_lin_Stochastic=sgd_reg.predict(X_validation)
ridge_test_mse_stochastic=((Y_validation-ridge_predict_lin_Stochastic)**2).sum()/len(Y_validation)
ridge_test_rmse_stochastic=np.sqrt(ridge_test_mse_stochastic)
ridge_test_rmse_stochastic
# 0.28638961850775196



#-------------------------------------------------------------------------------------------------------------------------------------------
# Lasso Regression

from sklearn.linear_model import Lasso

# applying grid search to find alpha
# prepare a range of alpha values to test
alphas = np.array([100,50,29,27,25,23,21,20,15,10,1,0.1,0.01,0.001,0.0008,0.0005,0.0003,0.0001,0])
# create and fit a ridge regression model, testing each alpha
model = Lasso()
grid = GridSearchCV(cv=5,estimator=model, param_grid=dict(alpha=alphas),scoring='neg_mean_squared_error')
grid.fit(X_train, Y_train)
print(grid)

# summarize the results of the grid search
lasso_train_rmse=np.sqrt(-grid.best_score_)
print(lasso_train_rmse)
print(grid.best_estimator_.alpha)

#0.11345006473735222
#0.0008

cvres=grid.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)

'''
0.3968279147427862 {'alpha': 100.0}
0.3968279147427862 {'alpha': 50.0}
0.3968279147427862 {'alpha': 29.0}
0.3968279147427862 {'alpha': 27.0}
0.3968279147427862 {'alpha': 25.0}
0.3968279147427862 {'alpha': 23.0}
0.3968279147427862 {'alpha': 21.0}
0.3968279147427862 {'alpha': 20.0}
0.3968279147427862 {'alpha': 15.0}
0.3968279147427862 {'alpha': 10.0}
0.3968279147427862 {'alpha': 1.0}
0.1901350471630296 {'alpha': 0.1}
0.1291513923772325 {'alpha': 0.01}
0.11366644862233372 {'alpha': 0.001}
0.11345006473735222 {'alpha': 0.0008}
0.1141149856479724 {'alpha': 0.0005}
0.11540933012820477 {'alpha': 0.0003}
0.11968014166848127 {'alpha': 0.0001}
0.1276080882299927 {'alpha': 0.0}
'''


# Prediction
lasso_reg = Lasso(alpha=0.0008)
lasso_reg.fit(X_train, Y_train)

prediction_lasso=lasso_reg.predict(X_validation)

lasso_test_mse=((Y_validation-prediction_lasso)**2).sum()/len(Y_validation)
lasso_test_rmse=np.sqrt(lasso_test_mse)
lasso_test_rmse
# 0.14954863164942334

#-----------------------------------------------------------------------------
# using Stochastic Gradient Descent
# penalty=l1
sgd_reg=SGDRegressor(penalty='l1',max_iter=10000,eta0=0.0001,alpha=0.0008,shuffle=True)
sgd_reg.fit(X_train,Y_train.ravel())

# find error with cross validation
lasso_scores_stochastic=cross_val_score(sgd_reg,X_train,Y_train,scoring='neg_mean_squared_error',cv=5)
lasso_rmse_scores_stochastic=np.sqrt(-lasso_scores_stochastic)

display_scores(lasso_rmse_scores_stochastic)
#Scores RMSE: [0.5647835  0.45462606 0.53828197 0.43122615 0.46760726]
#Mean RMSE: 0.49130498580778337
#Standard deviation of RMSE: 0.051229527108556015

# Prediction
lasso_predict_lin_Stochastic=sgd_reg.predict(X_validation)
lasso_test_mse_stochastic=((Y_validation-lasso_predict_lin_Stochastic)**2).sum()/len(Y_validation)
lasso_test_rmse_stochastic=np.sqrt(lasso_test_mse_stochastic)
lasso_test_rmse_stochastic
# 0.8093628575728181



#-------------------------------------------------------------------------------------------------------------------------------------------
# Elastic Net

from sklearn.linear_model import ElasticNet
parametersGrid = {"alpha": [100,50,29,27,25,23,21,20,15,10,1,0.1,0.01,0.001,0.0008,0.0005,0.0003,0.0001,0],"l1_ratio": np.arange(0.0, 1.0, 0.1)}

model = ElasticNet()
grid = GridSearchCV(cv=5,estimator=model, param_grid=parametersGrid ,scoring='neg_mean_squared_error')
grid.fit(X_train, Y_train)
print(grid)

# summarize the results of the grid search
elastic_train_rmse=np.sqrt(-grid.best_score_)
print(elastic_train_rmse)
print(grid.best_estimator_.alpha)
print(grid.best_estimator_.l1_ratio)
#0.11347339693559097
#0.001
#0.8

cvres=grid.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)


# Prediction
elastic_reg = ElasticNet(alpha=0.001,l1_ratio=0.8)
elastic_reg.fit(X_train, Y_train)

prediction_elastic=elastic_reg.predict(X_validation)

elastic_test_mse=((Y_validation-prediction_elastic)**2).sum()/len(Y_validation)
elastic_test_rmse=np.sqrt(elastic_test_mse)
elastic_test_rmse
# 0.14888627346166744

'''
#-------------------------------------------------------------------------------------------------------------------------------------------
# SVM Regression - taking long time

from sklearn.svm import SVR

parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5,5, 10,50,100],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
model = SVR()
grid = GridSearchCV(estimator=model, param_grid=parameters,scoring='neg_mean_squared_error')
grid.fit(X_train,Y_train)
grid.best_params_

# summarize the results of the grid search
elastic_train_rmse=np.sqrt(-grid.best_score_)
print(elastic_train_rmse)
print(grid.best_estimator_.alpha)
print(grid.best_estimator_.l1_ratio)
'''


#-------------------------------------------------------------------------------------------------------------------------------------------
# Random Forest

from sklearn.ensemble import RandomForestRegressor
# Create the parameter grid based on the results of random search 

param_grid = {
    'bootstrap': [True],
    'max_depth': [70,75,77],
    'max_features': [40,50,75,100],
    'min_samples_leaf': [1,2],
    'min_samples_split': [2,3],
    'n_estimators': [350,400,500]
}

# Create a based model
model = RandomForestRegressor()
# Instantiate the grid search model
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2,scoring='neg_mean_squared_error')
grid.fit(X_train,Y_train)
grid.best_params_

'''
{'bootstrap': True,
 'max_depth': 77,
 'max_features': 50,
 'min_samples_leaf': 1,
 'min_samples_split': 3,
 'n_estimators': 500}
'''

randomForest_train_rmse=np.sqrt(-grid.best_score_)
randomForest_train_rmse
#0.12903180245135284

# Prediction
ranfomForest_reg = RandomForestRegressor(bootstrap=True,max_depth=77,max_features=50,min_samples_leaf=1,min_samples_split=3,n_estimators=500)
ranfomForest_reg.fit(X_train, Y_train)

prediction_randomForest=ranfomForest_reg.predict(X_validation)

ranfomForest_mse=((Y_validation-prediction_randomForest)**2).sum()/len(Y_validation)
randomForest_test_rmse=np.sqrt(ranfomForest_mse)
randomForest_test_rmse
# 0.14272532051839343

feature_importances = pd.DataFrame(ranfomForest_reg.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance',ascending=False)

#-----------------------------------------------------------------------------
# exporting the feature importance generated by Random Forest
feature_importances.to_csv(r'/Users/amanprasad/Documents/Kaggle/House Prices/Feature_Importances_RandomForest.csv', index = True)


#-------------------------------------------------------------------------------------------------------------------------------------------
# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

param_grid = {
    'learning_rate': [0.01,0.1],
    'n_estimators': [400,500,600],
    'min_samples_split': [2,3,4],
    'max_depth': [60,70,80],
    'max_features': [20,35],
    'min_samples_leaf' : [2,3,4]
}

# Create a based model
model = GradientBoostingRegressor()
# Instantiate the grid search model
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2,scoring='neg_mean_squared_error')
grid.fit(X_train,Y_train)
grid.best_params_

'''
{'learning_rate': 0.01,
 'max_depth': 60,
 'max_features': 20,
 'min_samples_leaf': 4,
 'min_samples_split': 4,
 'n_estimators': 600}
'''

gradBoost_train_rmse=np.sqrt(-grid.best_score_)
gradBoost_train_rmse
# 0.11996227721676053

display_scores(gradBoost_train_rmse)
#Scores RMSE: 0.11996227721676053
#Mean RMSE: 0.11996227721676053
#Standard deviation of RMSE: 0.0

# Prediction
gradBoost_reg = GradientBoostingRegressor(max_depth=60,max_features=20,min_samples_leaf=4,min_samples_split=4,n_estimators=600,learning_rate= 0.01)
gradBoost_reg.fit(X_train, Y_train)

prediction_gradBoost=gradBoost_reg.predict(X_validation)

gradBoost_mse=((Y_validation-prediction_gradBoost)**2).sum()/len(Y_validation)
gradBoost_test_rmse=np.sqrt(gradBoost_mse)
gradBoost_test_rmse
# 0.14045639753683603

















