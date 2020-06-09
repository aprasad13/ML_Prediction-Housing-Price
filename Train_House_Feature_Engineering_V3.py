
#Feature engineering
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
warnings.filterwarnings('ignore')
%matplotlib inline

df_train_original_FeatEngg = pd.read_csv('/Users/amanprasad/Documents/Kaggle/House Prices/df_train_noNull21_V2.csv')
df_FeatEngg_1=copy.deepcopy(df_train_original_FeatEngg)

df_train_original = pd.read_csv('/Users/amanprasad/Documents/Kaggle/House Prices/house-prices-advanced-regression-techniques/House_price_train.csv')
df_train_1=copy.deepcopy(df_train_original)

df_train_1=copy.deepcopy(df_train_original)
df_train_1.head()

# removing id column
id=df_train_1['Id']
df_train_1=df_train_1.drop(['Id'],axis=1)


#----------------------------------------------------------------------------------------------------------------------------------
#Total number of Bathrooms
#There are 4 bathroom variables. Individually, these variables are not very important. However, 
#we add them up into one predictor, this predictor is likely to become a strong one.

#“A half-bath, also known as a powder room or guest bath, has only two of the four main bathroom 
#components-typically a toilet and sink.” Consequently, we will also count the half bathrooms as half.
df_FeatEngg_2=copy.deepcopy(df_FeatEngg_1)
df_FeatEngg_2['TotBathrooms']=df_FeatEngg_2['FullBath'] + (df_FeatEngg_2['HalfBath']*0.5) + df_FeatEngg_2['BsmtFullBath'] + (df_FeatEngg_2['BsmtHalfBath']*0.5)
df_FeatEngg_2['TotBathrooms']=df_FeatEngg_2['TotBathrooms'].astype('category')
df_FeatEngg_2.dtypes['TotBathrooms']
# category

sns.stripplot(x='TotBathrooms',y='SalePrice',data=df_FeatEngg_2)

sns.countplot(df_FeatEngg_2['TotBathrooms'])
plt.title('CountPlot_TotBathrooms')
plt.xlabel('TotBathrooms')
plt.ylabel('Count')

df_FeatEngg_2['TotBathrooms']=df_FeatEngg_2['TotBathrooms'].astype('float')
# float

# correlation between them
df_FeatEngg_2.loc[:,['TotBathrooms','SalePrice']].corr()
c,_=pearsonr(df_FeatEngg_2['TotBathrooms'], df_FeatEngg_2['SalePrice'])
c
#0.631
#As we can see in the first graph, there now seems to be a clear correlation (it’s 0.63). 
#The frequency distribution of Bathrooms in all data is shown in the second graph.

# removing FullBath HalfBath BsmtFullBath BsmtHalfBath
drop_col=['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
df_FeatEngg_2.drop(drop_col, axis=1,inplace=True)

#----------------------------------------------------------------------------------------------------------------------------------
#Adding ‘House Age’, ‘Remodeled (Yes/No)’, and IsNew variables

# lets consider following 3 variables and do some feature engineering
#YearBuilt: Original construction date
#YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
#YrSold: Year Sold (YYYY)

# we decide to make an age column age=YrSold-YearRemodAdd and we will keep YrSold as it is, to capture the effect of that year
# when it is being sold

# YearBuilt

df_FeatEngg_2['YearBuilt'].dtype
# int64

df_FeatEngg_2['YearBuilt'].isnull().sum()
#0

plt.scatter(df_FeatEngg_2['YearBuilt'],df_FeatEngg_2['SalePrice'])

#-----------------------------------------------------------------------------
# YearRemodAdd
df_FeatEngg_2['YearRemodAdd'].dtype
# int64

df_FeatEngg_2['YearRemodAdd'].isnull().sum()
#0

plt.scatter(df_FeatEngg_2['YearRemodAdd'],df_FeatEngg_2['SalePrice'])

df_FeatEngg_2[df_FeatEngg_2['YearRemodAdd']==0]['YearRemodAdd'].sum()
#0
# this means that no 0 is present and we also observe some same year present in YearRemodAdd and YearBuilt. So it's the
# indication that we can compute House_age = YrSold - YearRemodAdd

df_FeatEngg_2['House_Age']=df_FeatEngg_2['YrSold']-df_FeatEngg_2['YearRemodAdd']

plt.scatter(df_FeatEngg_2['House_Age'],df_FeatEngg_2['SalePrice'])

#However, as parts of old constructions will always remain and only parts of the house might have been renovated, 
#will also introduce a column=Remodeled Yes/No variable. This should be seen as some sort of penalty parameter that 
#indicates that if the Age is based on a remodeling date, it is probably worth less than houses that were built from 
#scratch in that same year.

Remodeled=list()
for i in range(df_FeatEngg_2.shape[0]):
    Remodeled.append(1)

len(Remodeled)
#1460

for i in range(df_FeatEngg_2.shape[0]):
    if (df_FeatEngg_2['YearRemodAdd'][i]==df_FeatEngg_2['YearBuilt'][i]):
        Remodeled[i]=0

df_FeatEngg_2['YearRemodAdd'][2]==df_FeatEngg_2['YearBuilt'][2]

len(Remodeled)

# print unique value in list
def unique(list1): 
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    # print list 
    return (unique_list)
        
u=unique(Remodeled)
len(u)

# merging Remodeled with df_train_noNull21
Remodeled=pd.DataFrame(Remodeled,columns=['Remodeled'])

df_FeatEngg_2 = pd.concat([df_FeatEngg_2, Remodeled], axis=1)

df_FeatEngg_2['Remodeled'].unique()

#-----------------------------------------------------------------------------
# IsNew
IsNew=list()
for i in range(df_FeatEngg_2.shape[0]):
    IsNew.append(0)

len(IsNew)
#1460

for i in range(df_FeatEngg_2.shape[0]):
    if (df_FeatEngg_2['YrSold'][i]==df_FeatEngg_2['YearBuilt'][i]):
        IsNew[i]=1

len(IsNew)

unique(IsNew)

# merging Remodeled with df_train_noNull21
IsNew=pd.DataFrame(IsNew,columns=['IsNew'])

df_FeatEngg_2 = pd.concat([df_FeatEngg_2, IsNew], axis=1)

df_FeatEngg_2['IsNew'].unique()

#-----------------------------------------------------------------------------
# converting YearBuilt into categorical
df_FeatEngg_2['YrSold']=df_FeatEngg_2['YrSold'].astype('category')


#-----------------------------------------------------------------------------
# removing YearBuilt and YearRemodAdd
drop_col=['YearBuilt', 'YearRemodAdd']
df_FeatEngg_2.drop(drop_col, axis=1,inplace=True)

#--------------------------------------------------------------------------------------------------------------------------------
#Binning Neighborhood

# plotting neighbour with Salesprice median 
Neigh_median=df_FeatEngg_2.groupby('Neighborhood')['SalePrice'].median()
Neigh_median=Neigh_median.sort_values(ascending=True)
Neigh_median=pd.DataFrame(Neigh_median)
Neigh_median = Neigh_median.reset_index()
Neigh_median.columns = ['Neighborhood', 'SalePrice_median']

chart=sns.barplot(x='Neighborhood',y='SalePrice_median',data=Neigh_median,estimator=np.median)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart=plt.axhline(y=Neigh_median['SalePrice_median'].median(), color='red')
plt.tight_layout()


# plotting neighbour with Salesprice mean
 
Neigh_median=df_FeatEngg_2.groupby('Neighborhood')['SalePrice'].mean()
Neigh_median=Neigh_median.sort_values(ascending=True)
Neigh_median=pd.DataFrame(Neigh_median)
Neigh_median = Neigh_median.reset_index()
Neigh_median.columns = ['Neighborhood', 'SalePrice_mean']

chart=sns.barplot(x='Neighborhood',y='SalePrice_mean',data=Neigh_median,estimator=np.median)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart=plt.axhline(y=Neigh_median['SalePrice_mean'].median(), color='red')
plt.tight_layout()

#Both the median and mean Saleprices agree on 3 neighborhoods with substantially higher saleprices. 
#The separation of the 3 relatively poor neighborhoods is less clear, but at least both graphs agree on the 
#same 3 poor neighborhoods. Since I do not want to ‘overbin’, I am only creating categories for those ‘extremes’.

df_FeatEngg_3=copy.deepcopy(df_FeatEngg_2)

low= ['MeadowV','IDOTRR','BrDale']
high=['StoneBr', 'NridgHt', 'NoRidge']
for i in range(df_FeatEngg_3.shape[0]):
    if (df_FeatEngg_3['Neighborhood'][i] in low):
        df_FeatEngg_3['Neighborhood'][i]=0
    elif (df_FeatEngg_3['Neighborhood'][i] in high):
        df_FeatEngg_3['Neighborhood'][i]=2
    else:
        df_FeatEngg_3['Neighborhood'][i]=1

df_FeatEngg_3['Neighborhood'].unique()

#--------------------------------------------------------------------------------------------------------------------------------
#Total Square Feet (numeric)
#As the total living space generally is very important when people buy houses, 
#we add a predictors that adds up the living space above and below ground.

df_FeatEngg_3['TotalSqFeet'] = df_FeatEngg_3['GrLivArea'] + df_FeatEngg_3['TotalBsmtSF']

# scatter plot between TotalSqFeet and SalePrice
sns.lmplot(x='TotalSqFeet',y='SalePrice',data=df_FeatEngg_3)

#As expected, the correlation with SalePrice is very strong indeed (0.78).
c,_=pearsonr(df_FeatEngg_3['TotalSqFeet'], df_FeatEngg_3['SalePrice'])
c
#0.7789588289942257

# we see some outlier where GrLivArea>4000 and SalePrice<250000. we find its index
df_FeatEngg_3[(df_FeatEngg_3['GrLivArea']>4000) & (df_FeatEngg_3['SalePrice']<250000)][['SalePrice','GrLivArea','OverallQual']]

#      SalePrice  GrLivArea OverallQual
#303     160000       5642           10

outlier_index=df_FeatEngg_3[(df_FeatEngg_3['GrLivArea']>4000) & (df_FeatEngg_3['SalePrice']<250000)].index
#Int64Index([303],

# removing the outliers
df_FeatEngg_3.drop(df_FeatEngg_3.index[[outlier_index[0],outlier_index[0]]],inplace=True)

c,_=pearsonr(df_FeatEngg_3['TotalSqFeet'], df_FeatEngg_3['SalePrice'])
c
# 0.8290419781065508
# So, by taking out these two outliers, the correlation increases by 5%.

# removing TotalBsmtSF to avaoid multicollinearity as GrLivArea has less correlation as compared to TotalBsmtSF
drop_col=['TotalBsmtSF']
df_FeatEngg_3.drop(drop_col,axis=1,inplace=True)

#--------------------------------------------------------------------------------------------------------------------------------
#Consolidating Porch variables
#Below, is the listed the variables that seem related regarding porches.

#WoodDeckSF: Wood deck area in square feet
#OpenPorchSF: Open porch area in square feet
#EnclosedPorch: Enclosed porch area in square feet
#3SsnPorch: Three season porch area in square feet
#ScreenPorch: Screen porch area in square feet

# porches are sheltered areas outside of the house, and a wooden deck is unsheltered. 
# Therefore, I am leaving WoodDeckSF alone, and are only consolidating the 4 porch variables.
df_FeatEngg_4=copy.deepcopy(df_FeatEngg_3)
df_FeatEngg_4['TotalPorchSF'] = df_FeatEngg_4['OpenPorchSF'] + df_FeatEngg_4['EnclosedPorch'] + df_FeatEngg_4['3SsnPorch'] + df_FeatEngg_4['ScreenPorch']

#Although adding up these Porch areas makes sense (there should not be any overlap between areas), 
#the correlation with SalePrice is not very strong.
c,_=pearsonr(df_FeatEngg_4['TotalPorchSF'], df_FeatEngg_4['SalePrice'])
c
#0.19687503944832505

# scatter plot between TotalSqFeet and SalePrice
sns.lmplot(x='TotalPorchSF',y='SalePrice',data=df_FeatEngg_4)

# removing OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch
drop_col=['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
df_FeatEngg_4.drop(drop_col,axis=1,inplace=True)

#--------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------
#Preparing data for modeling
#Dropping highly correlated variables

#First of all, I am dropping a variable if two variables are highly correlated. To find these correlated pairs, 
#I have used the correlations matrix. For instance: GarageCars and GarageArea have a 
#correlation of 0.89. Of those two, I am dropping the variable with the lowest correlation with 
#SalePrice (which is GarageArea with a SalePrice correlation of 0.62. GarageCars has a SalePrice correlation of 0.64).
df_FeatEngg_5=copy.deepcopy(df_FeatEngg_4)
# finding coorelation between numeric variables
correlation_num_variable(df_FeatEngg_5,0.5,'SalePrice')

# print numeric column names
#num_col.columns

# removing GarageArea
drop_col=['GarageArea']
df_FeatEngg_5.drop(drop_col,axis=1,inplace=True)


correlation_num_variable(df_FeatEngg_5,0.5,'SalePrice')

# finding coorelation between categorical variables - ChiSquare test
df_FeatEngg_5.dtypes

cols=df_FeatEngg_5.columns
# check datatype of columns
for i in range(len(cols)):
    print('{i}  {n}      {d}'.format (i=i,n=cols[i], d=df_FeatEngg_5.dtypes[cols[i]]))


#--------------------------------------------------------------------------------------------------------------------------------
#PreProcessing predictor variables

df_FeatEngg_6=copy.deepcopy(df_FeatEngg_5)    

num_ordinal_col=df_FeatEngg_6._get_numeric_data()
num_ordinal_col_name=num_ordinal_col.columns

# columns to remove from numeric category

# separating ordinal and numeric columns
num_col_name=separate_ordinal_numric_var(df_FeatEngg_6,30)[0]
ordinal_col_name=separate_ordinal_numric_var(df_FeatEngg_6,30)[1]
len(num_col_name)
#14

# adding MiscVal, PoolArea, LowQualFinSF to num_col_name and deleting from ordinal_col_name
dict=['MiscVal', 'PoolArea', 'LowQualFinSF']

for i in range(len(dict)):
    num_col_name.append(dict[i])

len(num_col_name)
#17

len(ordinal_col_name)
#39

for i in range(len(dict)):
    ordinal_col_name.remove(dict[i])

len(ordinal_col_name)
#36

# There are 17 numeric variables including SalePrice, and 36 ordinal variables


#--------------------------------------------------------------------------------------------------------------------------------
#Skewness and normalizing of the numeric predictors

# Skewness is a measure of the symmetry in a distribution. A symmetrical dataset will have a skewness equal to 0. 
#So, a normal distribution will have a skewness of 0. Skewness essentially measures the relative size of the two tails. 
#As a rule of thumb, skewness should be between -1 and 1. In this range, data are considered fairly symmetrical. 
#In order to fix the skewness, I am taking the log for all numeric predictors with an absolute skew greater 
#than 0.8 (actually: log+1, to avoid division by zero issues).

df_FeatEngg_num_var=copy.deepcopy(df_FeatEngg_6.loc[:,num_col_name])

for i in range(df_FeatEngg_num_var.shape[1]):
    if (abs(df_FeatEngg_num_var[num_col_name[i]].skew())>0.8):
        df_FeatEngg_num_var[num_col_name[i]]=np.log(df_FeatEngg_num_var.iloc[:,i]+1)

#-----------------------------------------------------------------------------
#Data Normalization
#Normalization refers to rescaling real valued numeric attributes into the range 0 and 1.
#It is useful to scale the input attributes for a model that relies on the magnitude of values, such as distance measures 
#used in k-nearest neighbors and in the preparation of coefficients in regression.

#Data Standardization
#Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation 
#of one (unit variance).
#It is useful to standardize attributes for a model that relies on the distribution of attributes such as Gaussian processes.

# we decide to go with Standardization

# removing target variable (SalePrice) from df_FeatEngg_num_var as we do not need to standardise it
target_var=df_FeatEngg_num_var['SalePrice']
df_FeatEngg_num_var.drop('SalePrice',axis=1,inplace=True)


for i in (range(df_FeatEngg_num_var.shape[1])):
    df_FeatEngg_num_var[df_FeatEngg_num_var.columns[i]]=preprocessing.scale(df_FeatEngg_num_var.iloc[:,i])



#-----------------------------------------------------------------------------
# Dealing with response variable
   

# we can also check this with Q-Q plot  
stats.probplot(target_var, dist="norm", plot=pylab)
pylab.show()
# we see this shows a good result - target variable aligns with the line (normal dist)

sns.distplot(target_var)
# so target varible has a normal bell shape

#-----------------------------------------------------------------------------
# removing columns (in df_FeatEngg_num_var) from df_FeatEngg_7 and adding target and columns (in df_FeatEngg_num_var) to df_FeatEngg_7 

df_FeatEngg_7=copy.deepcopy(df_FeatEngg_6)

df_FeatEngg_7.drop(df_FeatEngg_num_var.columns,axis=1,inplace=True)

df_FeatEngg_7 = pd.concat([df_FeatEngg_7, df_FeatEngg_num_var], axis=1, sort=False)

df_FeatEngg_7['SalePrice']=target_var

#-----------------------------------------------------------------------------
#One hot encoding the categorical variables

df_FeatEngg_8=copy.deepcopy(df_FeatEngg_7)
numeric_ordinal_col=df_FeatEngg_8._get_numeric_data()
numeric_ordinal_col_name=numeric_ordinal_col.columns

cat_col=df_FeatEngg_8.loc[:,(set(df_FeatEngg_8) - set(numeric_ordinal_col_name))].columns
len(cat_col)
#21
# there are 21 categorical columns and total of 74 columns including nominal

#One hot encoding
for i in range(len(cat_col)):
    df_FeatEngg_8=pd.get_dummies(df_FeatEngg_8,prefix=[cat_col[i]],columns=[cat_col[i]])



#-------------------------------------------------------------------------------------------------------------------------------
#Removing levels with few or no observations in train  
#checking all columns in df_FeatEngg_8 sum()!=0

for i in range(len(df_FeatEngg_8.columns)):
    if (df_FeatEngg_8.loc[:,df_FeatEngg_8.columns[i]].sum()==0):
        print(df_FeatEngg_8.columns[i])

# no result - hence no column in df_FeatEngg_8 sum()!=0 - so all good


#-------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------
# exporting df_train_noNull21 into csv
df_FeatEngg_8.to_csv(r'/Users/amanprasad/Documents/Kaggle/House Prices/df_train_FeatEngg_8_V2.csv', index = False)







