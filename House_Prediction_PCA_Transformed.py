
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
from sklearn.model_selection import cross_val_score
from numpy.linalg import inv
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
X_train_noPCA=df_train_1

Y_validation=df_validation_1['SalePrice']
df_validation_1.drop('SalePrice',axis=1,inplace=True)
X_validation_noPCA=df_validation_1

def display_scores(scores):
    print("Scores RMSE:", scores)
    print("Mean RMSE:", scores.mean())
    print("Standard deviation of RMSE:", scores.std())

#-------------------------------------------------------------------------------------------------------------------------------------------
# PCA
from sklearn.decomposition import PCA

X_train=X_train_noPCA
pca=PCA(n_components=0.95)
X_train=pd.DataFrame(pca.fit_transform(X_train_noPCA))
X_train

X_validation=X_validation_noPCA
pca=PCA(n_components=53)
X_validation=pd.DataFrame(pca.fit_transform(X_validation_noPCA))
X_validation

#-------------------------------------------------------------------------------------------------------------------------------------------
# Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train, Y_train)

# find error with cross validation
lin_scores=cross_val_score(lin_reg,X_train,Y_train,scoring='neg_mean_squared_error',cv=5)
lin_rmse_scores=np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)

#Scores RMSE: [0.11424355 0.11362777 0.11824368 0.11472412 0.12468623]
#Mean RMSE: 0.11710506977706689
#Standard deviation of RMSE: 0.004116290624815727

# Prediction
prediction_lin=lin_reg.predict(X_validation)

lin_test_mse=((Y_validation-prediction_lin)**2).sum()/len(Y_validation)
lin_test_rmse=np.sqrt(lin_test_mse)
lin_test_rmse
# 0.21866147149548182

#-----------------------------------------------------------------------------
# Prediction Interval

X_validation_arr=np.array(X_validation)

from scipy.stats import t
tt = t.ppf(0.75, 284)

Y_max=[]
Y_min=[]

for i in range(X_validation_arr.shape[0]):
    a=inv(np.matmul(X_validation_arr.transpose(), X_validation_arr))
    b=np.matmul(X_validation_arr[i].transpose(),a)
    c=(1+np.matmul(b,X_validation_arr[i]))
    s_pred=np.sqrt(lin_test_mse*c)
    Y_max.append(prediction_lin[i]+(tt*s_pred))
    Y_min.append(prediction_lin[i]-(tt*s_pred))


Y_pred_lin_PCA=pd.DataFrame([[Y_min,prediction_lin,Y_max]],columns=['Y_max','Y_predicted','Y_min'])

Y_max_df=pd.DataFrame(Y_max,columns=['Y_max'])
Y_min_df=pd.DataFrame(Y_min,columns=['Y_min'])
Y_pred_df=pd.DataFrame(prediction_lin,columns=['Y_pred'])

df_pred_linear_PCA=pd.concat([Y_min_df, Y_pred_df,Y_max_df], axis=1)

# taking antilog
df_pred_linear_PCA_Actual=np.exp(df_pred_linear_PCA)

# Exporting df_pred_linear_PCA_Actual to csv
df_pred_linear_PCA_Actual.to_csv(r'/Users/amanprasad/Documents/Kaggle/House Prices/df_predictionInterval_linear_PCA_Actual.csv', index = False)



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
# 0.23171273738088263


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

# summarize the results of the grid search
ridge_train_rmse=np.sqrt(-grid.best_score_)
print(ridge_train_rmse)
print(grid.best_estimator_.alpha)
grid.best_params_

# 0.11673464401746662
# {'alpha': 29.0}

cvres=grid.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)

'''
0.1176913377450759 {'alpha': 100.0}
0.1168494812374999 {'alpha': 50.0}
0.11673464401746662 {'alpha': 29.0}
0.11673732688978865 {'alpha': 27.0}
0.11674303743728581 {'alpha': 25.0}
0.11675197777267281 {'alpha': 23.0}
0.11676436687293221 {'alpha': 21.0}
0.11677192817632381 {'alpha': 20.0}
0.11682470766452244 {'alpha': 15.0}
0.11690588679125587 {'alpha': 10.0}
0.1171424418997758 {'alpha': 1.0}
0.11717380852656187 {'alpha': 0.1}
0.11717703283246005 {'alpha': 0.01}
0.11717735615136705 {'alpha': 0.001}
0.11717738849215295 {'alpha': 0.0001}
0.1171773920856735 {'alpha': 0.0}
'''

# Prediction
ridge_reg = Ridge(alpha=29, solver="cholesky")
ridge_reg.fit(X_train, Y_train)

prediction_ridge=ridge_reg.predict(X_validation)

ridge_test_mse=((Y_validation-prediction_ridge)**2).sum()/len(Y_validation)
ridge_test_rmse=np.sqrt(ridge_test_mse)
ridge_test_rmse
# 0.2159260272536825

#-----------------------------------------------------------------------------
# using Stochastic Gradient Descent
# penalty=l2
sgd_reg=SGDRegressor(penalty='l2',max_iter=1000,eta0=0.01,alpha=29,shuffle=True,random_state=60616)
sgd_reg.fit(X_train,Y_train.ravel())

# find error with cross validation
ridge_scores_stochastic=cross_val_score(sgd_reg,X_train,Y_train,scoring='neg_mean_squared_error',cv=5)
ridge_rmse_scores_stochastic=np.sqrt(-ridge_scores_stochastic)

display_scores(ridge_rmse_scores_stochastic)
#Scores RMSE: [0.27969832 0.28017722 0.30795146 0.29460924 0.33245309]
#Mean RMSE: 0.2989778654057608
#Standard deviation of RMSE: 0.0197249400024865

# Prediction
ridge_predict_lin_Stochastic=sgd_reg.predict(X_validation)
ridge_test_mse_stochastic=((Y_validation-ridge_predict_lin_Stochastic)**2).sum()/len(Y_validation)
ridge_test_rmse_stochastic=np.sqrt(ridge_test_mse_stochastic)
ridge_test_rmse_stochastic
# 0.3184118909709933

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

# summarize the results of the grid search
lasso_train_rmse=np.sqrt(-grid.best_score_)
print(lasso_train_rmse)
print(grid.best_estimator_.alpha)
grid.best_params_

#0.11676191996399303
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
0.3325932688475053 {'alpha': 1.0}
0.1684149570398996 {'alpha': 0.1}
0.12825297098902405 {'alpha': 0.01}
0.11678708364101473 {'alpha': 0.001}
0.11676191996399303 {'alpha': 0.0008}
0.11678546834895803 {'alpha': 0.0005}
0.11689213853283041 {'alpha': 0.0003}
0.117065351889815 {'alpha': 0.0001}
0.1171773920856735 {'alpha': 0.0}
'''


# Prediction
lasso_reg = Lasso(alpha=0.0008)
lasso_reg.fit(X_train, Y_train)

prediction_lasso=lasso_reg.predict(X_validation)

lasso_test_mse=((Y_validation-prediction_lasso)**2).sum()/len(Y_validation)
lasso_test_rmse=np.sqrt(lasso_test_mse)
lasso_test_rmse
# 0.21583738578178066

#-----------------------------------------------------------------------------
# using Stochastic Gradient Descent
# penalty=l1
sgd_reg=SGDRegressor(penalty='l1',max_iter=10000,eta0=0.01,alpha=0.0008,shuffle=True)
sgd_reg.fit(X_train,Y_train.ravel())

# find error with cross validation
lasso_scores_stochastic=cross_val_score(sgd_reg,X_train,Y_train,scoring='neg_mean_squared_error',cv=5)
lasso_rmse_scores_stochastic=np.sqrt(-lasso_scores_stochastic)

display_scores(lasso_rmse_scores_stochastic)
#Scores RMSE: [0.11550815 0.11248361 0.13262272 0.12082946 0.12929685]
#Mean RMSE: 0.12214815903906791
#Standard deviation of RMSE: 0.007746646496122993

# Prediction
lasso_predict_lin_Stochastic=sgd_reg.predict(X_validation)
lasso_test_mse_stochastic=((Y_validation-lasso_predict_lin_Stochastic)**2).sum()/len(Y_validation)
lasso_test_rmse_stochastic=np.sqrt(lasso_test_mse_stochastic)
lasso_test_rmse_stochastic
# 0.22022969724341554



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
grid.best_params_
#0.11675505949345945
# {'alpha': 0.001, 'l1_ratio': 0.7000000000000001}

cvres=grid.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)


# Prediction
elastic_reg = ElasticNet(alpha=0.001,l1_ratio=0.7)
elastic_reg.fit(X_train, Y_train)

prediction_elastic=elastic_reg.predict(X_validation)

elastic_test_mse=((Y_validation-prediction_elastic)**2).sum()/len(Y_validation)
elastic_test_rmse=np.sqrt(elastic_test_mse)
elastic_test_rmse
# 0.21613504090408497



#-------------------------------------------------------------------------------------------------------------------------------------------
# Random Forest

from sklearn.ensemble import RandomForestRegressor
# Create the parameter grid based on the results of random search 

param_grid = {
    'bootstrap': [True],
    'max_depth': [50,60,70],
    'max_features': [35,45,50],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [2,3,4],
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
 'max_depth': 50,
 'max_features': 50,
 'min_samples_leaf': 2,
 'min_samples_split': 4,
 'n_estimators': 350}
'''

randomForest_train_rmse=np.sqrt(-grid.best_score_)
randomForest_train_rmse
display_scores(randomForest_train_rmse)
#0.1418293940827534

#Scores RMSE: 0.1418293940827534
#Mean RMSE: 0.1418293940827534
#Standard deviation of RMSE: 0.0

# Prediction
ranfomForest_reg = RandomForestRegressor(bootstrap=True,max_depth=50,max_features=50,min_samples_leaf=2,min_samples_split=4,n_estimators=350)
ranfomForest_reg.fit(X_train, Y_train)

prediction_randomForest=ranfomForest_reg.predict(X_validation)

ranfomForest_mse=((Y_validation-prediction_randomForest)**2).sum()/len(Y_validation)
randomForest_test_rmse=np.sqrt(ranfomForest_mse)
randomForest_test_rmse
# 0.1890364900292125

feature_importances = pd.DataFrame(ranfomForest_reg.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance',ascending=False)



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
 'max_features': 35,
 'min_samples_leaf': 4,
 'min_samples_split': 3,
 'n_estimators': 600}
'''

gradBoost_train_rmse=np.sqrt(-grid.best_score_)
gradBoost_train_rmse
# 0.14648460664021323

display_scores(gradBoost_train_rmse)
#Scores RMSE: 0.14163909473526046
#Mean RMSE: 0.14163909473526046
#Standard deviation of RMSE: 0.0

# Prediction
gradBoost_reg = GradientBoostingRegressor(max_depth=60,max_features=35,min_samples_leaf=4,min_samples_split=3,n_estimators=600,learning_rate= 0.01)
gradBoost_reg.fit(X_train, Y_train)

prediction_gradBoost=gradBoost_reg.predict(X_validation)

gradBoost_mse=((Y_validation-prediction_gradBoost)**2).sum()/len(Y_validation)
gradBoost_test_rmse=np.sqrt(gradBoost_mse)
gradBoost_test_rmse
# 0.19501973484749527









