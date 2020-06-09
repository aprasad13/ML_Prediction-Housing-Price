from sklearn.model_selection import train_test_split
import copy 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

df_train_original = pd.read_csv('/Users/amanprasad/Documents/Kaggle/House Prices/house-prices-advanced-regression-techniques/House_price_train.csv')

df_train=copy.deepcopy(df_train_original)
df_train.head()

# removing id column
id=df_train['Id']
df_train=df_train.drop(['Id'],axis=1)

df_train.info()
#Without the Id’s, the dataframe consists of 79 predictors and our response variable SalePrice.

cols=df_train.columns
cols

'''
Index(['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']
'''

len(cols)
#80

# check datatype of columns
for i in range(len(cols)):
    print('{i}  {n}      {d}'.format (i=i,n=cols[i], d=df_train.dtypes[cols[i]]))

# separate numeric valraible
num_col=df_train._get_numeric_data()
type(num_col)
num_col.shape
# 1460,37

#--------------------------------------------------------------------------------------------------------------------------------
#Exploring some of the most important variables
#The response variable; SalePrice

df_train.hist('SalePrice')
#As you can see, the sale prices are right skewed. This was expected as few people can afford very expensive houses. 
#We will keep this in mind, and take measures before modeling.

df_train['SalePrice'].describe()

'''
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: SalePrice, dtype: float64
'''
#-----------------------------------------------------------------------------
#The most important numeric predictors
#The character variables need some work before We can use them. To get a feel for the dataset, 
#We decided to first see which numeric variables have a high correlation with the SalePrice.

#Correlations with SalePrice

corr = num_col.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(num_col.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(num_col.columns)
ax.set_yticklabels(num_col.columns)
plt.show()

# the corr metrix
sorted_cor = np.sort(corr)
sorted_cor = sorted_cor[::-1]

corr_nan=corr[corr>0.5]
eq1_c = corr_nan[~np.isnan(corr_nan)]

c = df_train.corr().abs()

s = c.unstack()
so = s.sort_values(kind="quicksort")
type(so)
soo=so.to_frame()
soo.columns
type(soo)

soo = pd.DataFrame(s).reset_index()
soo.columns = ['var1', 'var2','correlation']
soo=soo[(soo['correlation']>0.5) & (soo['correlation']!=1)]
soo=soo.sort_values(by='correlation', ascending=False)

soo[soo['var1']=='SalePrice']

# this shows that many varaibles are highly coorelated with target variable and there is also multicollinearity issue bewteen varibale 
# lets visualize the highly correlated varaibles with SalesPrice
#OverallQual: has the highest correlation with SalePrice among the numeric variables (0.79). 
#It rates the overall material and finish of the house on a scale from 1 (very poor) to 10 (very excellent).

# Hence we decide to do stratified sampling based on OverallQual

# converting this to category variable
df_train['OverallQual']=df_train['OverallQual'].astype('category')

train_distribution_column('OverallQual', df_train, 'SalePrice')

df_train_11, df_validation = train_test_split(df_train, test_size = 0.3, random_state = 60616, stratify = df_train['OverallQual'])
df_validation=df_validation.reset_index(drop=True)
#The positive correlation is certainly there indeed, and seems to be a slightly upward curve. 
#Regarding outliers, I do not see any extreme values. If there is a candidate to take out as an outlier later on, 
#it seems to be the expensive house with grade 4.

#-----------------------------------------------------------------------------
# Above Grade (Ground) Living Area (square feet)
#The numeric variable with the second highest correlation with SalesPrice is the Above Grade Living Area. 
#This make a lot of sense; big houses are generally more expensive.
# GrLivArea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_validation[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

m, b = np.polyfit(df_validation[var],df_validation['SalePrice'], 1)
plt.plot(df_validation[var], m*df_validation[var] + b,color='red')

# we see some outlier where GrLivArea>4000 and SalePrice<250000. we find its index
df_validation[(df_validation['GrLivArea']>4000) & (df_validation['SalePrice']<250000)][['SalePrice','GrLivArea','OverallQual']]

#      SalePrice  GrLivArea OverallQual
#385     184750       4676          10

df_validation[(df_validation['GrLivArea']>4000) & (df_validation['SalePrice']<250000)].index
#Int64Index([1298]

#the two houses with really big living areas and low SalePrices seem outliers (houses 524 and 1298). 
#I will not take them out yet, as taking outliers can be dangerous. For instance, a low score on the Overall Quality could 
#explain a low price. However, as you can see below, these two houses actually also score maximum points on Overall Quality. 
#Therefore, I will keep houses 1298 and 523 in mind as prime candidates to take out as outliers.

#--------------------------------------------------------------------------------------------------------------------------------
#Missing data, label encoding, and factorizing variables
#Completeness of the data
#which variables contain missing values.
null_all=df_validation.isnull().sum()
null_all.size
type(null_all)
null_all['PoolQC']
null_all_desc=null_all[null_all>0].sort_values(ascending = False) 

type(null_all_desc)
null_all_desc=pd.DataFrame(null_all_desc)
null_all_desc = null_all_desc.reset_index()
null_all_desc.columns = ['column', 'missing_count']
null_all_desc.columns

#          column  missing_count
#0         PoolQC            436
#1    MiscFeature            423
#2          Alley            417
#3          Fence            362
#4    FireplaceQu            197
#5    LotFrontage             89
#6    GarageYrBlt             21
#7     GarageType             21
#8   GarageFinish             21
#9     GarageQual             21
#10    GarageCond             21
#11  BsmtFinType1             13
#12  BsmtExposure             13
#13      BsmtCond             13
#14      BsmtQual             13
#15  BsmtFinType2             13
#16    MasVnrArea              2
#17    MasVnrType              2

#--------------------------------------------------------------------------------------------------------------------------------
#Imputing missing data
#--------------------------------------------------------------------------------------------------------------------------------
#Pool Quality and the PoolArea variable
#The PoolQC is the variable with most NAs - 1453. The description is as follows:
#PoolQC (categorical): Pool quality
# So, lets check for every NA in PoolQC there is 0 in PoolArea, if yes then it is obvious that I need to just assign ‘No Pool’ to the NAs. 
#Also, the high number of NAs makes sense as normally only a small proportion of houses have a pool.
df_train_noNull = copy.deepcopy(df_validation) 

df_train_noNull[df_train_noNull['PoolQC'].isnull()==True][['PoolArea']].sum()
#0 - so no PoolArea value

df_train_noNull['PoolQC']=df_train_noNull['PoolQC'].fillna('None')
df_train_noNull['PoolQC'].isnull().sum()
#0

# unique values
df_train_noNull['PoolQC'].unique()
#array(['None', 'Ex', 'Fa', 'Gd']

# distribution within Functional
train_distribution_column('PoolQC',df_train_noNull,'SalePrice')

#None : Count = 1453 and Percentage = 99.52054794520548
#Ex : Count = 2 and Percentage = 0.136986301369863
#Fa : Count = 2 and Percentage = 0.136986301369863
#Gd : Count = 3 and Percentage = 0.2054794520547945

#It is also clear that I can label encode this variable as the values are ordinal. 
#As there a multiple variables that use the same quality levels
PoolQC_dict= {'None':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}
df_train_noNull['PoolQC_Ordinal']=df_train_noNull.PoolQC.map(PoolQC_dict)
df_train_noNull['PoolQC'].unique()
df_train_noNull['PoolQC_Ordinal'].unique()
#array([0, 4, 1, 3])

# droping PoolQC as we can use PoolQC_Ordinal
df_train_noNull=df_train_noNull.drop(['PoolQC'],axis=1)
df_train_noNull.columns[df_train_noNull.columns=='PoolQC']

#--------------------------------------------------------------------------------------------------------------------------------
# MiscFeature (categorical) - Miscellaneous feature not covered in other categories
# missing values= 1406
df_train_noNull['MiscFeature'].unique()
# array([nan, 'Shed', 'Gar2', 'Othr', 'TenC']

# imputing the missing value in MiscFeature with None
df_train_noNull['MiscFeature']=df_train_noNull['MiscFeature'].fillna('None')
df_train_noNull['MiscFeature'].isnull().sum()    

# distribution within MiscFeature
train_distribution_column('MiscFeature',df_train_noNull,'SalePrice')

#None : Count = 1406 and Percentage = 96.3013698630137
#Shed : Count = 49 and Percentage = 3.356164383561644
#Gar2 : Count = 2 and Percentage = 0.136986301369863
#Othr : Count = 2 and Percentage = 0.136986301369863
#TenC : Count = 1 and Percentage = 0.0684931506849315

# we donot see Elev - Elevator so we keep this in other
# we see a relation bewteen lebels of this feature and salesprice of house: TenC>Gar2>Shed>Other>None
#It is also clear that I can label encode this variable as the values are ordinal. 
#As there a multiple variables that use the same quality levels, I am going to create a dictonary that I can reuse later on.
df_train_noNull2 = copy.deepcopy(df_train_noNull) 
mics_dict= {'None':0,'Othr':1,'Elev':1,'Shed':2,'Gar2':3,'TenC':4}
df_train_noNull2['MiscFeature_Ordinal']=df_train_noNull2.MiscFeature.map(mics_dict)
df_train_noNull2['MiscFeature'].unique()
df_train_noNull2['MiscFeature_Ordinal'].unique()

# droping MiscFeature as we can use MiscFeature_Ordinal
df_train_noNull2=df_train_noNull2.drop(['MiscFeature'],axis=1)
df_train_noNull2.columns[df_train_noNull2.columns=='MiscFeature']

# -------------------------------------------------------------------------------------------------------------------------------
# Alley (categorical) : Type of alley access to property
# missing value=1369

df_train_noNull3 = copy.deepcopy(df_train_noNull2) 

df_train_noNull3['Alley'].unique()
# array([nan, 'Grvl', 'Pave']

# imputing the missing value in Alley with None
df_train_noNull3['Alley']=df_train_noNull3['Alley'].fillna('None')
df_train_noNull3['Alley'].isnull().sum()    

# distribution within Alley
train_distribution_column('Alley',df_train_noNull3,'SalePrice')

#None : Count = 952 and Percentage = 93.15068493150685
#Grvl : Count = 37 and Percentage = 3.620352250489237
#Pave : Count = 33 and Percentage = 3.228962818003914

# we do not see any relation bewteen lebels of this feature and salesprice of house
# hence convert this to categorical

df_train_noNull3['Alley']=df_train_noNull3['Alley'].astype('category')

# -------------------------------------------------------------------------------------------------------------------------------
# Fence (categorical) : Fence quality
# missing value = 1179
df_train_noNull4=copy.deepcopy(df_train_noNull3)

df_train_noNull4['Fence'].unique()
# array([nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw']

# imputing the missing value in Alley with None
df_train_noNull4['Fence']=df_train_noNull4['Alley'].fillna('None')
df_train_noNull4['Fence'].isnull().sum()    

# distribution within Alley
train_distribution_column('Fence',df_train_noNull4,'SalePrice')

#None : Count = 952 and Percentage = 93.15068493150685
#Grvl : Count = 37 and Percentage = 3.620352250489237
#Pave : Count = 33 and Percentage = 3.228962818003914

# we do not see any relation bewteen lebels of this feature and salesprice of house
# hence convert this to categorical

df_train_noNull4['Fence']=df_train_noNull4['Fence'].astype('category')

# -------------------------------------------------------------------------------------------------------------------------------
#Fireplace quality, and Number of fireplaces
# FireplaceQu (categorical) - missing = 690 and Fireplaces (numeric) - missing = 0
df_train_noNull5=copy.deepcopy(df_train_noNull4)

df_train_noNull5['FireplaceQu'].unique()
# array([nan, 'TA', 'Gd', 'Fa', 'Ex', 'Po']

# lets check if there is any value in Fireplaces where FireplaceQu has null value
df_train_noNull5[df_train_noNull5['FireplaceQu'].isnull()==True][['Fireplaces']].sum()
#0 - so 0 everywhere, so now we can replace NA with None

# imputing the missing value in Alley with None
df_train_noNull5['FireplaceQu']=df_train_noNull5['FireplaceQu'].fillna('None')
df_train_noNull5['FireplaceQu'].isnull().sum()  

# distribution within Alley
train_distribution_column('FireplaceQu',df_train_noNull5,'SalePrice')
#TA : Count = 218 and Percentage = 21.3307240704501
#None : Count = 493 and Percentage = 48.23874755381605
#Gd : Count = 261 and Percentage = 25.538160469667318
#Fa : Count = 21 and Percentage = 2.0547945205479454
#Po : Count = 13 and Percentage = 1.2720156555772995
#Ex : Count = 16 and Percentage = 1.5655577299412915

# we see a relation with SalesPrice: 
fire_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_train_noNull5['Fire_Ordinal']=df_train_noNull5.FireplaceQu.map(fire_dict)
df_train_noNull5['FireplaceQu'].unique()
df_train_noNull5['Fire_Ordinal'].unique()

# droping MiscFeature as we can use MiscFeature_Ordinal
df_train_noNull5=df_train_noNull5.drop(['FireplaceQu'],axis=1)
df_train_noNull5.columns[df_train_noNull5.columns=='FireplaceQu']

# -------------------------------------------------------------------------------------------------------------------------------
# Lot Variables
# LotFrontage (Numeric) : Linear feet of street connected to property
df_train_noNull6=copy.deepcopy(df_train_noNull5)

df_train_noNull6['LotFrontage'].dtype
null_all_desc[null_all_desc['column']=='LotFrontage']
# missing value = 170

# distribution within LotFrontage
fig=plt.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(df_train_noNull6['LotFrontage'],df_train_noNull6['SalePrice'],'b')
axes.set_xlabel('LotFrontage')
axes.set_ylabel('SalePrice')
axes.set_title('Scatter_Plot')

# distribution with Neighborhood 
sns.barplot(x='LotFrontage',y='Neighborhood',data=df_train_noNull6)

#The most reasonable imputation seems to take the median per neigborhood
LotFrontage_Neighborhood_median=df_train_noNull6.groupby('Neighborhood')['LotFrontage'].median()
type(LotFrontage_Neighborhood_median)
LotFrontage_Neighborhood_median=pd.DataFrame(LotFrontage_Neighborhood_median)
LotFrontage_Neighborhood_median = LotFrontage_Neighborhood_median.reset_index()
LotFrontage_Neighborhood_median.columns = ['Neighborhood', 'LotFrontage_median']



for i in range(df_train_noNull6.shape[0]):
    for j in range(LotFrontage_Neighborhood_median.shape[0]):
        if df_train_noNull6['LotFrontage'].isnull()[i]==True:
            if (df_train_noNull6['Neighborhood'][i]== LotFrontage_Neighborhood_median['Neighborhood'][j]):
                df_train_noNull6['LotFrontage'][i]=LotFrontage_Neighborhood_median['LotFrontage_median'][j]

df_train_noNull6['LotFrontage'].isnull().sum()
# 0

# ----------------------------------------------------------------------------
# LotShape (categorical): General shape of property
df_train_noNull6['LotShape'].isnull().sum()
# no missing value

df_train_noNull6['LotShape'].unique()
# array(['Reg', 'IR1', 'IR2', 'IR3']

# distribution within LotShape
train_distribution_column('LotShape',df_train_noNull6,'SalePrice')

#Reg : Count = 925 and Percentage = 63.35616438356164
#IR1 : Count = 484 and Percentage = 33.15068493150685
#IR2 : Count = 41 and Percentage = 2.808219178082192
#IR3 : Count = 10 and Percentage = 0.684931506849315

# Values seem ordinal (Regular=best)
lot_dict= {'IR3':0,'IR1':2,'IR2':1,'Reg':3}
df_train_noNull6['Lot_Ordinal']=df_train_noNull6.LotShape.map(lot_dict)
df_train_noNull6['LotShape'].unique()
df_train_noNull6['Lot_Ordinal'].unique()
#array([4, 2, 1, 0])

# droping MiscFeature as we can use MiscFeature_Ordinal
df_train_noNull6=df_train_noNull6.drop(['LotShape'],axis=1)
df_train_noNull6.columns[df_train_noNull6.columns=='LotShape']

# ----------------------------------------------------------------------------
# LotConfig (categorical): Lot configuration
df_train_noNull6['LotConfig'].isnull().sum()
# no missing value

df_train_noNull6['LotConfig'].unique()
# array(['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3']

# distribution within LotConfig
train_distribution_column('LotConfig',df_train_noNull6,'SalePrice')
#Inside : Count = 1052 and Percentage = 72.05479452054794
#FR2 : Count = 47 and Percentage = 3.219178082191781
#Corner : Count = 263 and Percentage = 18.013698630136986
#CulDSac : Count = 94 and Percentage = 6.438356164383562
#FR3 : Count = 4 and Percentage = 0.273972602739726

#The values seemed possibly ordinal, but the visualization does not show this. 
#Therefore, convert this into categorical datatype

df_train_noNull6['LotConfig']=df_train_noNull6['LotConfig'].astype('category')

# -------------------------------------------------------------------------------------------------------------------------------
# Altogether, there are 7 variables related to garages
# GarageYrBlt (year) : Year garage was built
df_train_noNull7=copy.deepcopy(df_train_noNull6)
df_train_noNull7['GarageYrBlt'].isnull().sum()
# 81

#check if NA is present in all 7 variables GarageYrBlt, GarageType, GarageArea, GarageFinish, GarageCars, GarageQual, GarageCond
gar=['GarageYrBlt','GarageType','GarageArea','GarageFinish','GarageCars','GarageQual','GarageCond']
for i in range(len(gar)):
    print('column = {c} and number/sum_value = {v}'.format
          (c=gar[i],v=df_train_noNull7[df_train_noNull7['GarageYrBlt'].isnull()==True][gar[i]].sum()))

#column = GarageYrBlt and number/sum_value = 0.0
#column = GarageType and number/sum_value = 0
#column = GarageArea and number/sum_value = 0
#column = GarageFinish and number/sum_value = 0
#column = GarageCars and number/sum_value = 0
#column = GarageQual and number/sum_value = 0
#column = GarageCond and number/sum_value = 0

# so this os the evedence that values are missing in all 7 variables simultaneously. So we can replace NA with None

# checking relation of GarageYrBlt with salesprice
df_train_noNull7['GarageYrBlt'].dtype

plt.scatter(df_train_noNull7['GarageYrBlt'],df_train_noNull7['SalePrice'])
# we see that as the year increases saleprice increases but this relation is not clear so cannot say anything.
# Hence we decided to bucket the year in the interval of 5 years and imputing the null value with None

for i in range(df_train_noNull7.shape[0]):
    if df_train_noNull7['GarageYrBlt'].isnull()[i]==True:
        df_train_noNull7['GarageYrBlt'][i]=0

df_train_noNull7['GarageYrBlt'].isnull().sum()
#0

# convert float value to int
df_train_noNull7['GarageYrBlt']=df_train_noNull7['GarageYrBlt'].astype(int)
        
year_range=list(range(1901,2010,1))

# imputing missing with None
GarageYrBlt_interval=list()
for i in range(df_train_noNull7.shape[0]):
    GarageYrBlt_interval.append('None')

len(GarageYrBlt_interval)

for j in range(0,len(year_range)-2,3):
    for i in range(df_train_noNull7.shape[0]):
        if (df_train_noNull7['GarageYrBlt'][i]>0) & (df_train_noNull7['GarageYrBlt'][i]<1901):
            GarageYrBlt_interval[i]='before_1901'
        if (df_train_noNull7['GarageYrBlt'][i]>2009):
            GarageYrBlt_interval[i]='after_2009'
        if (df_train_noNull7['GarageYrBlt'][i]>=year_range[j]) & (df_train_noNull7['GarageYrBlt'][i]<=year_range[j+2]):
            p=str(year_range[j]) +'_to_'+ str(year_range[j+2])
            GarageYrBlt_interval[i]=p

len(GarageYrBlt_interval)

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
        
u=unique(GarageYrBlt_interval)
len(u)
# merging GarageYrBlt_interval with df_train_noNull7
GarageYrBlt_interval=pd.DataFrame(GarageYrBlt_interval,columns=['GarageYrBlt_interval'])

df_train_noNull8=copy.deepcopy(df_train_noNull7)
df_train_noNull8 = pd.concat([df_train_noNull7, GarageYrBlt_interval], axis=1)

df_train_noNull8['GarageYrBlt_interval'].unique()

#array(['2003_to_2005', '1976_to_1978', '2000_to_2002', '1997_to_1999',
#       '1991_to_1993', '1973_to_1975', '1931_to_1933', '1937_to_1939',
#       '1964_to_1966', '1961_to_1963', '2006_to_2008', '1958_to_1960',
#       '1970_to_1972', '1967_to_1969', '1928_to_1930', '1955_to_1957',
#       '1919_to_1921', '1994_to_1996', '1952_to_1954', 'None',
#       '1982_to_1984', '1985_to_1987', '1979_to_1981', '1934_to_1936',
#       '1988_to_1990', '1943_to_1945', '1913_to_1915', '1946_to_1948',
#       '1949_to_1951', 'before_1901', '1922_to_1924', '1925_to_1927',
#       '1916_to_1918', '1940_to_1942', '1910_to_1912', 'after_2009',
#       '1904_to_1906', '1907_to_1909']

# dropping GarageYrBlt from df_train_noNull8
df_train_noNull8=df_train_noNull8.drop(['GarageYrBlt'],axis=1)
df_train_noNull8.columns[df_train_noNull8.columns=='GarageYrBlt']

# changing the datatype of GarageYrBlt_interval to category
df_train_noNull8['GarageYrBlt_interval']=df_train_noNull8['GarageYrBlt_interval'].astype('category')

# ----------------------------------------------------------------------
# GarageType (categorical): Garage location
df_train_noNull10=copy.deepcopy(df_train_noNull8)
df_train_noNull10['GarageType'].isnull().sum()
#81

# check if values are present in GarageArea, GarageFinish, GarageCars, GarageQual, GarageCond
# print unique values present in GarageArea, GarageFinish, GarageCars, GarageQual, GarageCond
gar=['GarageArea', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond']
for i in range(len(gar)):
    print('column= {n} and unique_value= {u}'.format(n=gar[i],u=df_train_noNull10[gar[i]].unique()))

#column=GarageArea - no missing
#column= GarageFinish and unique_value= ['RFn' 'Unf' 'Fin' nan] - yes missing
#column= GarageCars and unique_value= [2 3 1 0 4] - no missing
#column= GarageQual and unique_value= ['TA' 'Fa' 'Gd' nan 'Ex' 'Po'] - yes missing
#column= GarageCond and unique_value= ['TA' 'Fa' nan 'Gd' 'Po' 'Ex'] - yes missing

df_train_noNull10['GarageType'].unique()
# array(['Attchd', 'Detchd', 'BuiltIn', 'CarPort', nan, 'Basment', '2Types']

#imputing values for GarageType with None according to document
df_train_noNull10['GarageType']=df_train_noNull10['GarageType'].fillna('None')
df_train_noNull10['GarageType'].isnull().sum()

# distribution within GarageType
train_distribution_column('GarageType',df_train_noNull10,'SalePrice')

#Attchd : Count = 870 and Percentage = 59.58904109589041
#Detchd : Count = 387 and Percentage = 26.506849315068493
#BuiltIn : Count = 88 and Percentage = 6.027397260273973
#CarPort : Count = 9 and Percentage = 0.6164383561643836
#None : Count = 81 and Percentage = 5.5479452054794525
#Basment : Count = 19 and Percentage = 1.3013698630136987
#2Types : Count = 6 and Percentage = 0.410958904109589

# we see a pattern None<CarPort<Detchd<Basment<2Types<Attchd<BuiltIn
Gtype_dict= {'None':0,'CarPort':1,'Detchd':2,'Basment':3,'2Types':4,'Attchd':5,'BuiltIn':6}
df_train_noNull10['Gtype_Ordinal']=df_train_noNull10.GarageType.map(Gtype_dict)
df_train_noNull10['GarageType'].unique()
df_train_noNull10['Gtype_Ordinal'].unique()
#array([5, 2, 6, 1, 0, 3, 4])

# dropping GarageType from df_train_noNull10
df_train_noNull10=df_train_noNull10.drop(['GarageType'],axis=1)
df_train_noNull10.columns[df_train_noNull10.columns=='GarageType']

# ----------------------------------------------------------------------------
# GarageFinish (categorical) : Interior finish of the garage
# imputing values for GarageFinish with None according to document
df_train_noNull10['GarageFinish']=df_train_noNull10['GarageFinish'].fillna('None')
df_train_noNull10['GarageFinish'].isnull().sum()

# unique values
df_train_noNull10['GarageFinish'].unique()
# array(['RFn', 'Unf', 'Fin', 'None']

# distribution within GarageType
train_distribution_column('GarageFinish',df_train_noNull10,'SalePrice')

#RFn : Count = 422 and Percentage = 28.904109589041095
#Unf : Count = 605 and Percentage = 41.43835616438356
#Fin : Count = 352 and Percentage = 24.10958904109589
#None : Count = 81 and Percentage = 5.5479452054794525

# we see a pattern of this varaible variable with SalePrice: None<Unf<RFn<Fin
GFin_dict= {'None':0,'Unf':1,'RFn':2,'Fin':3}
df_train_noNull10['GarageFinish_Ordinal']=df_train_noNull10.GarageFinish.map(GFin_dict)
df_train_noNull10['GarageFinish'].unique()
df_train_noNull10['GarageFinish_Ordinal'].unique()
#aarray([2, 1, 3, 0])

# dropping GarageFinish from df_train_noNull10
df_train_noNull10=df_train_noNull10.drop(['GarageFinish'],axis=1)
df_train_noNull10.columns[df_train_noNull10.columns=='GarageFinish']

# ----------------------------------------------------------------------------
# GarageQual (categorical) : Garage quality
# imputing values for GarageQual with None according to document
df_train_noNull10['GarageQual']=df_train_noNull10['GarageQual'].fillna('None')
df_train_noNull10['GarageQual'].isnull().sum()

# unique values
df_train_noNull10['GarageQual'].unique()
# array(['TA', 'Fa', 'Gd', 'None', 'Ex', 'Po']

# distribution within GarageQual
train_distribution_column('GarageQual',df_train_noNull10,'SalePrice')

#TA : Count = 1311 and Percentage = 89.79452054794521
#Fa : Count = 48 and Percentage = 3.287671232876712
#Gd : Count = 14 and Percentage = 0.958904109589041
#None : Count = 81 and Percentage = 5.5479452054794525
#Ex : Count = 3 and Percentage = 0.2054794520547945
#Po : Count = 3 and Percentage = 0.2054794520547945

# we see a pattern of this varaible variable with SalePrice: None<Unf<RFn<Fin
GarageQual_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_train_noNull10['GarageQual_Ordinal']=df_train_noNull10.GarageQual.map(GarageQual_dict)
df_train_noNull10['GarageQual'].unique()
df_train_noNull10['GarageQual_Ordinal'].unique()
#array([4, 2, 5, 0, 6, 1])

# dropping GarageQual from df_train_noNull10
df_train_noNull10=df_train_noNull10.drop(['GarageQual'],axis=1)
df_train_noNull10.columns[df_train_noNull10.columns=='GarageQual']


# ----------------------------------------------------------------------
# GarageCond (categorical): Garage condition
# imputing values for GarageCond with None according to document
df_train_noNull10['GarageCond']=df_train_noNull10['GarageCond'].fillna('None')
df_train_noNull10['GarageCond'].isnull().sum()

# unique values
df_train_noNull10['GarageCond'].unique()
# array(['TA', 'Fa', 'Gd', 'None', 'Ex', 'Po']

# distribution within GarageQual
train_distribution_column('GarageCond',df_train_noNull10,'SalePrice')

#TA : Count = 1326 and Percentage = 90.82191780821918
#Fa : Count = 35 and Percentage = 2.3972602739726026
#None : Count = 81 and Percentage = 5.5479452054794525
#Gd : Count = 9 and Percentage = 0.6164383561643836
#Po : Count = 7 and Percentage = 0.4794520547945205
#Ex : Count = 2 and Percentage = 0.136986301369863

# we see a pattern of this varaible variable with SalePrice: None<Unf<RFn<Fin
GarageCond_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_train_noNull10['GarageCond_Ordinal']=df_train_noNull10.GarageCond.map(GarageCond_dict)
df_train_noNull10['GarageCond'].unique()
df_train_noNull10['GarageCond_Ordinal'].unique()
#array([4, 2, 5, 0, 6, 1])

# dropping GarageCond from df_train_noNull10
df_train_noNull10=df_train_noNull10.drop(['GarageCond'],axis=1)
df_train_noNull10.columns[df_train_noNull10.columns=='GarageCond']

# -------------------------------------------------------------------------------------------------------------------------------
# Altogether, there are 11 variables that relate to the Basement of a house
df_train_noNull12=copy.deepcopy(df_train_noNull10)

# print null values for all varaibles
ll=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF',
    'TotalBsmtSF','BsmtFullBath','BsmtHalfBath']

for i in range(len(ll)):
    print('column = {c} and missing_value = {m}'.format(c=ll[i],m=df_train_noNull12[ll[i]].isnull().sum()))

#column = BsmtQual and missing_value = 13
#column = BsmtCond and missing_value = 13
#column = BsmtExposure and missing_value = 13
#column = BsmtFinType1 and missing_value = 13
#column = BsmtFinSF1 and missing_value = 0
#column = BsmtFinType2 and missing_value = 13
#column = BsmtFinSF2 and missing_value = 0
#column = BsmtUnfSF and missing_value = 0
#column = TotalBsmtSF and missing_value = 0
#column = BsmtFullBath and missing_value = 0
#column = BsmtHalfBath and missing_value = 0

BsmtQual_null=pd.DataFrame(df_train_noNull12[df_train_noNull12['BsmtQual'].isnull()][ll])
BsmtQual_null.dtypes

cols=BsmtQual_null.columns

for i in range(len(cols)):
    if (BsmtQual_null.dtypes[cols[i]]==int):
        print('column = {c} and null_value = {n}'.format(c=cols[i],n=BsmtQual_null[cols[i]].sum()))
    else:
        print('column = {cc} and null_value = {nn}'.format(cc=cols[i],nn=BsmtQual_null[cols[i]].isnull().sum()))

#column = BsmtQual and null_value = 37
#column = BsmtCond and null_value = 37
#column = df_train_noNull12 and null_value = 37
#column = BsmtFinType1 and null_value = 37
#column = BsmtFinSF1 and null_value = 0
#column = BsmtFinType2 and null_value = 37
#column = BsmtFinSF2 and null_value = 0
#column = BsmtUnfSF and null_value = 0
#column = TotalBsmtSF and null_value = 0
#column = BsmtFullBath and null_value = 0
#column = BsmtHalfBath and null_value = 0

'''
# so comparing the above result we see that BsmtExposure and BsmtFinType2 each one has 1 extra null value. We need to investigate those values

BsmtExposure_null=copy.deepcopy(df_train_noNull12[df_train_noNull12['BsmtExposure'].isnull()==True][ll])
BsmtExposure_null[BsmtExposure_null['BsmtQual'].isnull()==False]


BsmtFinType2_null=copy.deepcopy(df_train_noNull12[df_train_noNull12['BsmtFinType2'].isnull()==True][ll])
BsmtFinType2_null[BsmtFinType2_null['BsmtQual'].isnull()==False]

# we see that other variables which are related to basement has value. So there should be a value for this missing element. 
# we decide to predict these 2 values using RandomForest using the other values in the basement related column  
# code is in Predict_BsmtFinType2_BsmtExposure_RandomForest.py file

# so after running the algo we predicted following:
# for BsmtExposure_Ordinal : array([1]) - no exposer
# for BsmtFinType2_Ordinal : array([2]) - ALQ : Average Living Quarters

# substituting the above values
df_train_noNull12.loc[test,'BsmtExposure'] ='No'

df_train_noNull12.loc[test_BsmtFinType2,'BsmtFinType2'] ='ALQ'
'''

# ----------------------------------------------------------------------
#BsmtQual (categorical):Evaluates the height of the basement
df_train_noNull12['BsmtQual'].isnull().sum()
#13
# imputing values for GarageCond with None according to document
df_train_noNull12['BsmtQual']=df_train_noNull12['BsmtQual'].fillna('None')
df_train_noNull12['BsmtQual'].isnull().sum()
#0

# unique values
df_train_noNull12['BsmtQual'].unique()
#array(['Gd', 'TA', 'Ex', 'None', 'Fa']

# distribution within GarageQual
train_distribution_column('BsmtQual',df_train_noNull12,'SalePrice')
#Gd : Count = 618 and Percentage = 42.32876712328767
#TA : Count = 649 and Percentage = 44.45205479452055
#Ex : Count = 121 and Percentage = 8.287671232876713
#None : Count = 37 and Percentage = 2.5342465753424657
#Fa : Count = 35 and Percentage = 2.3972602739726026

# we see a pattern of this varaible variable with SalePrice: None<Po<Fa<TA<Gd<Ex
BsmtQual_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_train_noNull12['BsmtQual_Ordinal']=df_train_noNull12.BsmtQual.map(BsmtQual_dict)
df_train_noNull12['BsmtQual'].unique()
df_train_noNull12['BsmtQual_Ordinal'].unique()
#array([4, 3, 5, 0, 2])

# dropping GarageCond from df_train_noNull12
df_train_noNull12=df_train_noNull12.drop(['BsmtQual'],axis=1)
df_train_noNull12.columns[df_train_noNull12.columns=='BsmtQual']

# ----------------------------------------------------------------------
# BsmtCond (categorical): Evaluates the general condition of the basement
df_train_noNull12['BsmtCond'].isnull().sum()
#37
# imputing values for GarageCond with None according to document
df_train_noNull12['BsmtCond']=df_train_noNull12['BsmtCond'].fillna('None')
df_train_noNull12['BsmtCond'].isnull().sum()
#0

# unique values
df_train_noNull12['BsmtCond'].unique()
#array(['TA', 'Gd', 'None', 'Fa', 'Po']


# distribution within BsmtCond
train_distribution_column('BsmtCond',df_train_noNull12,'SalePrice')
#TA : Count = 1311 and Percentage = 89.79452054794521
#Gd : Count = 65 and Percentage = 4.4520547945205475
#None : Count = 37 and Percentage = 2.5342465753424657
#Fa : Count = 45 and Percentage = 3.0821917808219177
#Po : Count = 2 and Percentage = 0.136986301369863

# we see a pattern of this varaible variable with SalePrice: None<Po<Fa<TA<Gd<Ex
BsmtCond_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_train_noNull12['BsmtCond_Ordinal']=df_train_noNull12.BsmtCond.map(BsmtCond_dict)
df_train_noNull12['BsmtCond'].unique()
df_train_noNull12['BsmtCond_Ordinal'].unique()
#array([3, 4, 0, 2, 1])

# dropping BsmtCond from df_train_noNull12
df_train_noNull12=df_train_noNull12.drop(['BsmtCond'],axis=1)
df_train_noNull12.columns[df_train_noNull12.columns=='BsmtCond']

# ----------------------------------------------------------------------
# BsmtExposure (categorical): Refers to walkout or garden level walls
df_train_noNull12['BsmtExposure'].isnull().sum()
#37
# imputing values for BsmtExposure with None according to document
df_train_noNull12['BsmtExposure']=df_train_noNull12['BsmtExposure'].fillna('None')
df_train_noNull12['BsmtExposure'].isnull().sum()
#0

# unique values
df_train_noNull12['BsmtExposure'].unique()
#array(['No', 'Gd', 'Mn', 'Av', 'None']

# distribution within BsmtExposure
train_distribution_column('BsmtExposure',df_train_noNull12,'SalePrice')
#No : Count = 954 and Percentage = 65.34246575342466
#Gd : Count = 134 and Percentage = 9.178082191780822
#Mn : Count = 114 and Percentage = 7.808219178082192
#Av : Count = 221 and Percentage = 15.136986301369863
#None : Count = 37 and Percentage = 2.5342465753424657

# we see a pattern of this varaible variable with SalePrice: None<No<Mn<Av<Gd
BsmtExposure_dict= {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4}
df_train_noNull12['BsmtExposure_Ordinal']=df_train_noNull12.BsmtExposure.map(BsmtExposure_dict)
df_train_noNull12['BsmtExposure'].unique()
df_train_noNull12['BsmtExposure_Ordinal'].unique()
#array([1, 4, 2, 3, 0])

# dropping BsmtCond from df_train_noNull12
df_train_noNull12=df_train_noNull12.drop(['BsmtExposure'],axis=1)
df_train_noNull12.columns[df_train_noNull12.columns=='BsmtExposure']

# ----------------------------------------------------------------------
# BsmtFinType1 (categorical):Rating of basement finished area
df_train_noNull12['BsmtFinType1'].isnull().sum()
#37
# imputing values for BsmtExposure with None according to document
df_train_noNull12['BsmtFinType1']=df_train_noNull12['BsmtFinType1'].fillna('None')
df_train_noNull12['BsmtFinType1'].isnull().sum()
#0

# unique values
df_train_noNull12['BsmtFinType1'].unique()
#array(['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'None', 'LwQ']


# distribution within BsmtFinType1
train_distribution_column('BsmtFinType1',df_train_noNull12,'SalePrice')

#GLQ : Count = 418 and Percentage = 28.63013698630137
#ALQ : Count = 220 and Percentage = 15.068493150684931
#Unf : Count = 430 and Percentage = 29.45205479452055
#Rec : Count = 133 and Percentage = 9.10958904109589
#BLQ : Count = 148 and Percentage = 10.136986301369863
#None : Count = 37 and Percentage = 2.5342465753424657
#LwQ : Count = 74 and Percentage = 5.068493150684931

# we see a pattern of this varaible variable with SalePrice: None<No<Mn<Av<Gd
BsmtFinType1_dict= {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
df_train_noNull12['BsmtFinType1_Ordinal']=df_train_noNull12.BsmtFinType1.map(BsmtFinType1_dict)
df_train_noNull12['BsmtFinType1'].unique()
df_train_noNull12['BsmtFinType1_Ordinal'].unique()
#array([6, 5, 1, 3, 4, 0, 2])

# dropping BsmtCond from df_train_noNull12
df_train_noNull12=df_train_noNull12.drop(['BsmtFinType1'],axis=1)
df_train_noNull12.columns[df_train_noNull12.columns=='BsmtFinType1']

# ----------------------------------------------------------------------
# BsmtFinType2 (categorical):Rating of basement finished area (if multiple types)
df_train_noNull12['BsmtFinType2'].isnull().sum()
#37
# imputing values for BsmtFinType2 with None according to document
df_train_noNull12['BsmtFinType2']=df_train_noNull12['BsmtFinType2'].fillna('None')
df_train_noNull12['BsmtFinType2'].isnull().sum()
#0

# unique values
df_train_noNull12['BsmtFinType2'].unique()
#array(['Unf', 'BLQ', 'None', 'ALQ', 'Rec', 'LwQ', 'GLQ']


# distribution within BsmtFinType1
train_distribution_column('BsmtFinType2',df_train_noNull12,'SalePrice')

#Unf : Count = 1256 and Percentage = 86.02739726027397
#BLQ : Count = 33 and Percentage = 2.26027397260274
#None : Count = 37 and Percentage = 2.5342465753424657
#ALQ : Count = 20 and Percentage = 1.36986301369863
#Rec : Count = 54 and Percentage = 3.6986301369863015
#LwQ : Count = 46 and Percentage = 3.1506849315068495
#GLQ : Count = 14 and Percentage = 0.958904109589041

# we see a pattern of this varaible variable with SalePrice: None<No<Mn<Av<Gd
BsmtFinType2_dict= {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
df_train_noNull12['BsmtFinType2_Ordinal']=df_train_noNull12.BsmtFinType2.map(BsmtFinType2_dict)
df_train_noNull12['BsmtFinType2'].unique()
df_train_noNull12['BsmtFinType2_Ordinal'].unique()
#array([1, 4, 0, 5, 3, 2, 6])

# dropping BsmtCond from df_train_noNull12
df_train_noNull12=df_train_noNull12.drop(['BsmtFinType2'],axis=1)
df_train_noNull12.columns[df_train_noNull12.columns=='BsmtFinType2']

# ----------------------------------------------------------------------------------------------------------------------------------------
# confirming if all the features related to basement doesnot have any missing value

base_cols=['BsmtQual_Ordinal','BsmtCond_Ordinal','BsmtExposure_Ordinal','BsmtFinType1_Ordinal',
           'BsmtFinSF1','BsmtFinType2_Ordinal','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']

for i in range(len(base_cols)):
    print('column = {c} and missing_value = {m}'.format(c=base_cols[i],m=df_train_noNull12[base_cols[i]].isnull().sum()))

'''
column = BsmtQual_Ordinal and missing_value = 0
column = BsmtCond_Ordinal and missing_value = 0
column = BsmtExposure_Ordinal and missing_value = 0
column = BsmtFinType1_Ordinal and missing_value = 0
column = BsmtFinSF1 and missing_value = 0
column = BsmtFinType2_Ordinal and missing_value = 0
column = BsmtFinSF2 and missing_value = 0
column = BsmtUnfSF and missing_value = 0
column = TotalBsmtSF and missing_value = 0
column = BsmtFullBath and missing_value = 0
column = BsmtHalfBath and missing_value = 0
'''
# So no missing value

# ----------------------------------------------------------------------------------------------------------------------------------------
# Masonry Variables (Masonry veneer type, and masonry veneer area)
# Masonry veneer type
#MasVnrType (categorical):Masonry veneer type
df_train_noNull13=copy.deepcopy(df_train_noNull12)

df_train_noNull13['MasVnrType'].isnull().sum()
#8

# print all the 8 rows which has nan
MasVnrType_null_index=df_train_noNull13[df_train_noNull13['MasVnrType'].isnull()].index

df_train_noNull13[df_train_noNull13['MasVnrType'].isnull()][['MasVnrType','MasVnrArea']]
'''
    MasVnrType  MasVnrArea
264        NaN         NaN
295        NaN         NaN
'''
# this shows values are missing simultaneously in both the variables related to Masonry. 
# So we can put None in MasVnrType and 0 in MasVnrArea
df_train_noNull13['MasVnrType']=df_train_noNull13['MasVnrType'].fillna('None')
df_train_noNull13['MasVnrType'].isnull().sum()
#0

# unique values
df_train_noNull13['MasVnrType'].unique()
#array(['BrkFace', 'None', 'Stone', 'BrkCmn']

# distribution within MasVnrType
train_distribution_column('MasVnrType',df_train_noNull13,'SalePrice')

#BrkFace : Count = 445 and Percentage = 30.47945205479452
#None : Count = 872 and Percentage = 59.726027397260275
#Stone : Count = 128 and Percentage = 8.767123287671232
#BrkCmn : Count = 15 and Percentage = 1.0273972602739727

# we see a pattern of this varaible variable with SalePrice: None<BrkCmn<BrkFace<Stone
MasVnrType_dict= {'None':0,'BrkCmn':1,'BrkFace':2,'Stone':3}
df_train_noNull13['MasVnrType_Ordinal']=df_train_noNull13.MasVnrType.map(MasVnrType_dict)
df_train_noNull13['MasVnrType'].unique()
df_train_noNull13['MasVnrType_Ordinal'].unique()
#array([2, 0, 3, 1])

# dropping BsmtCond from df_train_noNull12
df_train_noNull13=df_train_noNull13.drop(['MasVnrType'],axis=1)
df_train_noNull13.columns[df_train_noNull13.columns=='MasVnrType']

#--------------------------------------------------------------------
#MasVnrArea (numeric): Masonry veneer area in square feet
df_train_noNull13['MasVnrArea'].isnull().sum()
#8

df_train_noNull13['MasVnrArea']=df_train_noNull13['MasVnrArea'].fillna(0)
df_train_noNull13['MasVnrArea'].isnull().sum()
#0

# ----------------------------------------------------------------------------------------------------------------------------------------
#MSZoning (categorical): Identifies the general zoning classification of the sale
#MSZoning
df_train_noNull14=copy.deepcopy(df_train_noNull13)

df_train_noNull14['MSZoning'].isnull().sum()
#0

# unique values
df_train_noNull14['MSZoning'].unique()
#array(['RL', 'RM', 'C (all)', 'FV', 'RH']

# distribution within MSZoning
train_distribution_column('MSZoning',df_train_noNull14,'SalePrice')

#RL : Count = 1151 and Percentage = 78.83561643835617
#RM : Count = 218 and Percentage = 14.931506849315069
#C (all) : Count = 10 and Percentage = 0.684931506849315
#FV : Count = 65 and Percentage = 4.4520547945205475
#RH : Count = 16 and Percentage = 1.095890410958904

# as all the elements of this features are not present in the train data, we decide to keep internal distribution of the feature 
# in mind and keep C (all) and RH in other
df_train_noNull14['MSZoning']=df_train_noNull14['MSZoning'].replace({'C (all)': 'other','RH':'other'})

# converting into categorical variable
df_train_noNull14['MSZoning']=df_train_noNull14['MSZoning'].astype('category')

# ----------------------------------------------------------------------------------------------------------------------------------------
# Kitchen Variables - Kitchen quality and numer of Kitchens above grade
# KitchenQual (category) :Kitchen quality
df_train_noNull14['KitchenQual'].isnull().sum()
#0

# unique values
df_train_noNull14['KitchenQual'].unique()
# array(['Gd', 'TA', 'Ex', 'Fa']

# distribution within BsmtFinType1
train_distribution_column('KitchenQual',df_train_noNull14,'SalePrice')

#Gd : Count = 586 and Percentage = 40.136986301369866
#TA : Count = 735 and Percentage = 50.342465753424655
#Ex : Count = 100 and Percentage = 6.8493150684931505
#Fa : Count = 39 and Percentage = 2.671232876712329

# we see a pattern of this varaible variable with SalePrice: 
KitchenQual_dict= {'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}
df_train_noNull14['KitchenQual_Ordinal']=df_train_noNull14.KitchenQual.map(KitchenQual_dict)
df_train_noNull14['KitchenQual'].unique()
df_train_noNull14['KitchenQual_Ordinal'].unique()
#array([3, 2, 4, 1])

# dropping BsmtCond from df_train_noNull14
df_train_noNull14=df_train_noNull14.drop(['KitchenQual'],axis=1)
df_train_noNull14.columns[df_train_noNull14.columns=='KitchenQual']

#-----------------------------------------------------------------------------
# KitchenAbvGr (numeric)
df_train_noNull14['KitchenAbvGr'].isnull().sum()
#0
# hence no operation is needed for this varaible

# ----------------------------------------------------------------------------------------------------------------------------------------
#Utilities (categorical): Type of utilities available
# Utilities

df_train_noNull14['Utilities'].isnull().sum()
#0

# unique values
df_train_noNull14['Utilities'].unique()
# array(['AllPub', 'NoSeWa']

# distribution within Utilities
train_distribution_column('Utilities',df_train_noNull14,'SalePrice')

#AllPub : Count = 1459 and Percentage = 99.93150684931507
#NoSeWa : Count = 1 and Percentage = 0.0684931506849315

# we see a pattern of this varaible variable with SalePrice: 
Utilities_dict= {'ELO':1,'NoSeWa':2,'NoSewr':3,'AllPub':4}
df_train_noNull14['Utilities_Ordinal']=df_train_noNull14.Utilities.map(Utilities_dict)
df_train_noNull14['Utilities'].unique()
df_train_noNull14['Utilities_Ordinal'].unique()
#array([4, 2])

# dropping BsmtCond from df_train_noNull14
df_train_noNull14=df_train_noNull14.drop(['Utilities'],axis=1)
df_train_noNull14.columns[df_train_noNull14.columns=='Utilities']

# ----------------------------------------------------------------------------------------------------------------------------------------
# Functional: Home functionality
# Functional
df_train_noNull15=copy.deepcopy(df_train_noNull14)

df_train_noNull15['Functional'].isnull().sum()
#0

# unique values
df_train_noNull15['Functional'].unique()
# array(['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev']

# distribution within Functional
train_distribution_column('Functional',df_train_noNull15,'SalePrice')

#Typ : Count = 1360 and Percentage = 93.15068493150685
#Min1 : Count = 31 and Percentage = 2.1232876712328768
#Maj1 : Count = 14 and Percentage = 0.958904109589041
#Min2 : Count = 34 and Percentage = 2.328767123287671
#Mod : Count = 15 and Percentage = 1.0273972602739727
#Maj2 : Count = 5 and Percentage = 0.3424657534246575
#Sev : Count = 1 and Percentage = 0.0684931506849315

# we see a pattern of this varaible variable with SalePrice: 
Functional_dict= {'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7}
df_train_noNull15['Functional_Ordinal']=df_train_noNull15.Functional.map(Functional_dict)
df_train_noNull15['Functional'].unique()
df_train_noNull15['Functional_Ordinal'].unique()
#array([7, 6, 3, 5, 4, 2, 1])

# dropping Functional from df_train_noNull15
df_train_noNull15=df_train_noNull15.drop(['Functional'],axis=1)
df_train_noNull15.columns[df_train_noNull15.columns=='Functional']


# ----------------------------------------------------------------------------------------------------------------------------------------
#There are 4 exterior variables
# Exterior1st (categorical): Exterior covering on house

df_train_noNull16=copy.deepcopy(df_train_noNull15)

df_train_noNull16['Exterior1st'].isnull().sum()
#0

# unique values
df_train_noNull16['Exterior1st'].unique()
# array(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing',
#       'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn',
#       'Stone', 'ImStucc', 'CBlock']

# distribution within Functional
train_distribution_column('Exterior1st',df_train_noNull16,'SalePrice')

'''
VinylSd : Count = 361 and Percentage = 35.32289628180039
HdBoard : Count = 162 and Percentage = 15.851272015655578
MetalSd : Count = 161 and Percentage = 15.753424657534246
BrkFace : Count = 38 and Percentage = 3.7181996086105675
Wd Sdng : Count = 144 and Percentage = 14.090019569471623
Stucco : Count = 15 and Percentage = 1.467710371819961
Plywood : Count = 68 and Percentage = 6.653620352250489
CemntBd : Count = 36 and Percentage = 3.522504892367906
Stone : Count = 2 and Percentage = 0.19569471624266144
WdShing : Count = 17 and Percentage = 1.6634050880626223
AsbShng : Count = 15 and Percentage = 1.467710371819961
CBlock : Count = 1 and Percentage = 0.09784735812133072
ImStucc : Count = 1 and Percentage = 0.09784735812133072
AsphShn : Count = 1 and Percentage = 0.09784735812133072
'''

# according to document all the 17 categories are as follows:
'''
       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
'''
# we donot see all the categories in train so we decide put CBlock,ImStucc,AsphShn to other category as their contribution is least
df_train_noNull16['Exterior1st']=df_train_noNull16['Exterior1st'].replace({'CBlock':'Other','ImStucc':'Other','AsphShn':'Other'})

# create categorical since there is no considerable relation between them
df_train_noNull16['Exterior1st']=df_train_noNull16['Exterior1st'].astype('category')


# -----------------------------------------------------------------------------
# Exterior2nd (categorical): Exterior covering on house (if more than one material)
# Exterior2nd
df_train_noNull16['Exterior2nd'].isnull().sum()
#0

# unique values
df_train_noNull16['Exterior2nd'].unique()
#array(['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng',
#       'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc',
#       'AsphShn', 'Stone', 'Other', 'CBlock']

# distribution within Functional
train_distribution_column('Exterior2nd',df_train_noNull16,'SalePrice')

'''
VinylSd : Count = 504 and Percentage = 34.52054794520548
MetalSd : Count = 214 and Percentage = 14.657534246575343
Wd Shng : Count = 38 and Percentage = 2.6027397260273974
HdBoard : Count = 207 and Percentage = 14.178082191780822
Plywood : Count = 142 and Percentage = 9.726027397260275
Wd Sdng : Count = 197 and Percentage = 13.493150684931507
CmentBd : Count = 60 and Percentage = 4.109589041095891
BrkFace : Count = 25 and Percentage = 1.7123287671232876
Stucco : Count = 26 and Percentage = 1.7808219178082192
AsbShng : Count = 20 and Percentage = 1.36986301369863
Brk Cmn : Count = 7 and Percentage = 0.4794520547945205
ImStucc : Count = 10 and Percentage = 0.684931506849315
AsphShn : Count = 3 and Percentage = 0.2054794520547945
Stone : Count = 5 and Percentage = 0.3424657534246575
Other : Count = 1 and Percentage = 0.0684931506849315
CBlock : Count = 1 and Percentage = 0.0684931506849315
'''

# according to document all the 17 categories are as follows:
'''
       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
'''

# we donot see all the categories in train so we decide put CBlock,ImStucc,AsphShn to other category as their contribution is least
df_train_noNull16['Exterior2nd']=df_train_noNull16['Exterior2nd'].replace({'CBlock':'Other','ImStucc':'Other','AsphShn':'Other'})

# create categorical since there is no considerable relation between them
df_train_noNull16['Exterior2nd']=df_train_noNull16['Exterior2nd'].astype('category')


# -----------------------------------------------------------------------------
# ExterQual(categorical): Evaluates the quality of the material on the exterior 
# ExterQual

df_train_noNull16['ExterQual'].isnull().sum()
#0

# unique values
df_train_noNull16['ExterQual'].unique()
# array(['Gd', 'TA', 'Ex', 'Fa']

# distribution within ExterQual
train_distribution_column('ExterQual',df_train_noNull16,'SalePrice')

#Gd : Count = 488 and Percentage = 33.42465753424658
#TA : Count = 906 and Percentage = 62.054794520547944
#Ex : Count = 52 and Percentage = 3.5616438356164384
#Fa : Count = 14 and Percentage = 0.958904109589041

# we see a pattern of this varaible variable with SalePrice: 
ExterQual_dict= {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
df_train_noNull16['ExterQual_Ordinal']=df_train_noNull16.ExterQual.map(ExterQual_dict)
df_train_noNull16['ExterQual'].unique()
df_train_noNull16['ExterQual_Ordinal'].unique()
#array([4, 3, 5, 2])

# dropping Functional from df_train_noNull15
df_train_noNull16=df_train_noNull16.drop(['ExterQual'],axis=1)
df_train_noNull16.columns[df_train_noNull16.columns=='ExterQual']

# -----------------------------------------------------------------------------
# ExterCond (categorical): Evaluates the present condition of the material on the exterior
# ExterCond

df_train_noNull16['ExterCond'].isnull().sum()
#0

# unique values
df_train_noNull16['ExterCond'].unique()
# array(['TA', 'Gd', 'Fa', 'Po', 'Ex']

# distribution within ExterCond
train_distribution_column('ExterCond',df_train_noNull16,'SalePrice')

#TA : Count = 1282 and Percentage = 87.8082191780822
#Gd : Count = 146 and Percentage = 10.0
#Fa : Count = 28 and Percentage = 1.917808219178082
#Po : Count = 1 and Percentage = 0.0684931506849315
#Ex : Count = 3 and Percentage = 0.2054794520547945

# we see a pattern of this varaible variable with SalePrice: 
ExterCond_dict= {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
df_train_noNull16['ExterCond_Ordinal']=df_train_noNull16.ExterCond.map(ExterCond_dict)
df_train_noNull16['ExterCond'].unique()
df_train_noNull16['ExterCond_Ordinal'].unique()
#array([4, 3, 5, 2])

# dropping Functional from df_train_noNull15
df_train_noNull16=df_train_noNull16.drop(['ExterCond'],axis=1)
df_train_noNull16.columns[df_train_noNull16.columns=='ExterCond']

# ------------------------------------------------------------------------------------------------------------------------------
# Electrical (categorical): Electrical system
# Electrical
df_train_noNull17=copy.deepcopy(df_train_noNull16)

df_train_noNull17['Electrical'].isnull().sum()
#1

# unique values
df_train_noNull17['Electrical'].unique()
# array(['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix', nan]

# distribution within ExterCond
train_distribution_column('Electrical',df_train_noNull17,'SalePrice')

#SBrkr : Count = 1334 and Percentage = 91.36986301369863
#FuseF : Count = 27 and Percentage = 1.8493150684931507
#FuseA : Count = 94 and Percentage = 6.438356164383562
#FuseP : Count = 3 and Percentage = 0.2054794520547945
#Mix : Count = 1 and Percentage = 0.0684931506849315
#nan : Count = 0 and Percentage = 0.0

# from above distribution see that we can impute the missing value with the mode (most common category) = SBrkr
df_train_noNull17['Electrical']=df_train_noNull17['Electrical'].fillna('SBrkr')

# we see a pattern of this varaible variable with SalePrice: 
Electrical_dict= {'Mix':1, 'FuseP':2, 'FuseF':3, 'FuseA':4, 'SBrkr':5}
df_train_noNull17['Electrical_Ordinal']=df_train_noNull17.Electrical.map(Electrical_dict)
df_train_noNull17['Electrical'].unique()
df_train_noNull17['Electrical_Ordinal'].unique()
#array([5, 3, 4, 2, 1])

# dropping Electrical from df_train_noNull15
df_train_noNull17=df_train_noNull17.drop(['Electrical'],axis=1)
df_train_noNull17.columns[df_train_noNull17.columns=='Electrical']


# --------------------------------------------------------------------------------------------------------------------------------------
# SaleType (categorical): Type of sale

df_train_noNull17['SaleType'].isnull().sum()
#0

# unique values
df_train_noNull17['SaleType'].unique()
# array(['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth']

# distribution within ExterCond
train_distribution_column('SaleType',df_train_noNull17,'SalePrice')

'''
WD : Count = 1267 and Percentage = 86.78082191780823
New : Count = 122 and Percentage = 8.356164383561644
COD : Count = 43 and Percentage = 2.9452054794520546
ConLD : Count = 9 and Percentage = 0.6164383561643836
ConLI : Count = 5 and Percentage = 0.3424657534246575
CWD : Count = 4 and Percentage = 0.273972602739726
ConLw : Count = 5 and Percentage = 0.3424657534246575
Con : Count = 2 and Percentage = 0.136986301369863
Oth : Count = 3 and Percentage = 0.2054794520547945
'''

# from above distribution we see that all categories are not present - VWD, so we put VWD in other (Oth) 
df_train_noNull17['SaleType']=df_train_noNull17['SaleType'].replace({'VWD':'Oth'})

# create categorical since there is no considerable relation between them
df_train_noNull17['SaleType']=df_train_noNull17['SaleType'].astype('category')

# ----------------------------------------------------------------------------
# SaleCondition (categorical): Condition of sale

df_train_noNull17['SaleCondition'].isnull().sum()
#0

# unique values
df_train_noNull17['SaleCondition'].unique()
# array(['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family']

# distribution within SaleCondition
train_distribution_column('SaleCondition',df_train_noNull17,'SalePrice')

#Normal : Count = 1198 and Percentage = 82.05479452054794
#Abnorml : Count = 101 and Percentage = 6.917808219178082
#Partial : Count = 125 and Percentage = 8.561643835616438
#AdjLand : Count = 4 and Percentage = 0.273972602739726
#Alloca : Count = 12 and Percentage = 0.821917808219178
#Family : Count = 20 and Percentage = 1.36986301369863

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_train_noNull17['SaleCondition']=df_train_noNull17['SaleCondition'].astype('category')

# --------------------------------------------------------------------------------------------------------------------------------------
# Foundation (categorical): Type of foundation
df_train_noNull18=copy.deepcopy(df_train_noNull17)

df_train_noNull18['Foundation'].isnull().sum()
#0

# unique values
df_train_noNull18['Foundation'].unique()
# array(['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone']

# distribution within SaleCondition
train_distribution_column('Foundation',df_train_noNull18,'SalePrice')

#PConc : Count = 647 and Percentage = 44.31506849315068
#CBlock : Count = 634 and Percentage = 43.42465753424658
#BrkTil : Count = 146 and Percentage = 10.0
#Wood : Count = 3 and Percentage = 0.2054794520547945
#Slab : Count = 24 and Percentage = 1.643835616438356
#Stone : Count = 6 and Percentage = 0.410958904109589

# here all the categories are present in the train data

# create catgorical since there is no considerable relation between them
df_train_noNull18['Foundation']=df_train_noNull18['Foundation'].astype('category')

# --------------------------------------------------------------------------------------------------------------------------------------
# Heating and Air condition
# Heating (categorical): Type of heating

df_train_noNull18['Heating'].isnull().sum()
#0

# unique values
df_train_noNull18['Heating'].unique()
# array(['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor']

# distribution within Heating
train_distribution_column('Heating',df_train_noNull18,'SalePrice')

#GasA : Count = 1428 and Percentage = 97.8082191780822
#GasW : Count = 18 and Percentage = 1.2328767123287672
#Grav : Count = 7 and Percentage = 0.4794520547945205
#Wall : Count = 4 and Percentage = 0.273972602739726
#OthW : Count = 2 and Percentage = 0.136986301369863
#Floor : Count = 1 and Percentage = 0.0684931506849315

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_train_noNull18['Heating']=df_train_noNull18['Heating'].astype('category')

# ----------------------------------------------------------------------------
# HeatingQC (categorical): Heating quality and condition

df_train_noNull18['HeatingQC'].isnull().sum()
#0

# unique values
df_train_noNull18['HeatingQC'].unique()
# array(['Ex', 'Gd', 'TA', 'Fa', 'Po']

# distribution within Heating
train_distribution_column('HeatingQC',df_train_noNull18,'SalePrice')

#Ex : Count = 741 and Percentage = 50.75342465753425
#Gd : Count = 241 and Percentage = 16.506849315068493
#TA : Count = 428 and Percentage = 29.315068493150687
#Fa : Count = 49 and Percentage = 3.356164383561644
#Po : Count = 1 and Percentage = 0.0684931506849315

# here all the categories are present in the train data

# we see a pattern of this varaible variable with SalePrice: 
HeatingQC_dict= {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
df_train_noNull18['HeatingQC_Ordinal']=df_train_noNull18.HeatingQC.map(HeatingQC_dict)
df_train_noNull18['HeatingQC'].unique()
df_train_noNull18['HeatingQC_Ordinal'].unique()
#array([4, 3, 5, 2])

# dropping Functional from df_train_noNull15
df_train_noNull18=df_train_noNull18.drop(['HeatingQC'],axis=1)
df_train_noNull18.columns[df_train_noNull18.columns=='HeatingQC']

# ----------------------------------------------------------------------------
# CentralAir: Central air conditioning
df_train_noNull18['CentralAir'].isnull().sum()
#0

# unique values
df_train_noNull18['CentralAir'].unique()
# array(['Y', 'N']

# distribution within Heating
train_distribution_column('CentralAir',df_train_noNull18,'SalePrice')

#Y : Count = 1365 and Percentage = 93.4931506849315
#N : Count = 95 and Percentage = 6.506849315068493

# here all the categories are present in the train data

df_train_noNull18['CentralAir']=df_train_noNull18['CentralAir'].replace({'N':0,'Y':1})

# --------------------------------------------------------------------------------------------------------------------------------------
# There are 2 variables that deal with the roof of houses.
# RoofStyle (categorical): Type of roof
df_train_noNull19=copy.deepcopy(df_train_noNull18)

df_train_noNull19['RoofStyle'].isnull().sum()
#0

# unique values
df_train_noNull19['RoofStyle'].unique()
# array(['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed']

# distribution within Heating
train_distribution_column('RoofStyle',df_train_noNull19,'SalePrice')

#Gable : Count = 1141 and Percentage = 78.15068493150685
#Hip : Count = 286 and Percentage = 19.589041095890412
#Gambrel : Count = 11 and Percentage = 0.7534246575342466
#Mansard : Count = 7 and Percentage = 0.4794520547945205
#Flat : Count = 13 and Percentage = 0.8904109589041096
#Shed : Count = 2 and Percentage = 0.136986301369863

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_train_noNull19['RoofStyle']=df_train_noNull19['RoofStyle'].astype('category')

# ----------------------------------------------------------------------------
# RoofMatl (categorical): Roof material
df_train_noNull19['RoofMatl'].isnull().sum()
#0

# unique values
df_train_noNull19['RoofMatl'].unique()
# array(['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv','Roll', 'ClyTile']

# distribution within Heating
train_distribution_column('RoofMatl',df_train_noNull19,'SalePrice')

#CompShg : Count = 1434 and Percentage = 98.21917808219177
#WdShngl : Count = 6 and Percentage = 0.410958904109589
#Metal : Count = 1 and Percentage = 0.0684931506849315
#WdShake : Count = 5 and Percentage = 0.3424657534246575
#Membran : Count = 1 and Percentage = 0.0684931506849315
#Tar&Grv : Count = 11 and Percentage = 0.7534246575342466
#Roll : Count = 1 and Percentage = 0.0684931506849315
#ClyTile : Count = 1 and Percentage = 0.0684931506849315

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_train_noNull19['RoofMatl']=df_train_noNull19['RoofMatl'].astype('category')


# --------------------------------------------------------------------------------------------------------------------------------------
# 2 variables that specify the flatness and slope of the propoerty.
# LandContour (categorical): Flatness of the property
df_train_noNull19['LandContour'].isnull().sum()
#0

# unique values
df_train_noNull19['LandContour'].unique()
# array(['Lvl', 'Bnk', 'Low', 'HLS']

# distribution within Heating
train_distribution_column('LandContour',df_train_noNull19,'SalePrice')

#Lvl : Count = 1311 and Percentage = 89.79452054794521
#Bnk : Count = 63 and Percentage = 4.315068493150685
#Low : Count = 36 and Percentage = 2.4657534246575343
#HLS : Count = 50 and Percentage = 3.4246575342465753

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_train_noNull19['LandContour']=df_train_noNull19['LandContour'].astype('category')

# ----------------------------------------------------------------------------
# LandSlope (categorical): Slope of property
df_train_noNull19['LandSlope'].isnull().sum()
#0

# unique values
df_train_noNull19['LandSlope'].unique()
# array(['Gtl', 'Mod', 'Sev']

# distribution within LandSlope
train_distribution_column('LandSlope',df_train_noNull19,'SalePrice')

#Gtl : Count = 1382 and Percentage = 94.65753424657534
#Mod : Count = 65 and Percentage = 4.4520547945205475
#Sev : Count = 13 and Percentage = 0.8904109589041096

# here all the categories are present in the train data

df_train_noNull19['LandSlope']=df_train_noNull19['LandSlope'].replace({'Sev':0,'Mod':1,'Gtl':2})


# --------------------------------------------------------------------------------------------------------------------------------------
#2 variables that specify the type and style of dwelling.
#BldgType: Type of dwelling
df_train_noNull20=copy.deepcopy(df_train_noNull19)
df_train_noNull20['BldgType'].isnull().sum()
#0

# unique values
df_train_noNull20['BldgType'].unique()
# array(['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs']

# distribution within BldgType
train_distribution_column('BldgType',df_train_noNull20,'SalePrice')

#1Fam : Count = 1220 and Percentage = 83.56164383561644
#2fmCon : Count = 31 and Percentage = 2.1232876712328768
#Duplex : Count = 52 and Percentage = 3.5616438356164384
#TwnhsE : Count = 114 and Percentage = 7.808219178082192
#Twnhs : Count = 43 and Percentage = 2.9452054794520546

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_train_noNull20['BldgType']=df_train_noNull20['BldgType'].astype('category')

# ----------------------------------------------------------------------------
#HouseStyle: Style of dwelling
df_train_noNull20['HouseStyle'].isnull().sum()
#0

# unique values
df_train_noNull20['HouseStyle'].unique()
# array(['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf','2.5Fin']

# distribution within BldgType
train_distribution_column('HouseStyle',df_train_noNull20,'SalePrice')

#2Story : Count = 445 and Percentage = 30.47945205479452
#1Story : Count = 726 and Percentage = 49.726027397260275
#1.5Fin : Count = 154 and Percentage = 10.547945205479452
#1.5Unf : Count = 14 and Percentage = 0.958904109589041
#SFoyer : Count = 37 and Percentage = 2.5342465753424657
#SLvl : Count = 65 and Percentage = 4.4520547945205475
#2.5Unf : Count = 11 and Percentage = 0.7534246575342466
#2.5Fin : Count = 8 and Percentage = 0.547945205479452

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_train_noNull20['HouseStyle']=df_train_noNull20['HouseStyle'].astype('category')


# --------------------------------------------------------------------------------------------------------------------------------------
#3 variables that specify the physical location, and the proximity of ‘conditions’.
#Neighborhood (categorical): Physical locations within Ames city limits
df_train_noNull20['Neighborhood'].isnull().sum()
#0

# unique values
df_train_noNull20['Neighborhood'].unique()
# array(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
#       'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
#       'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
#       'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
#       'Blueste']

# distribution within BldgType
train_distribution_column('Neighborhood',df_train_noNull20,'SalePrice')
'''
CollgCr : Count = 150 and Percentage = 10.273972602739725
Veenker : Count = 11 and Percentage = 0.7534246575342466
Crawfor : Count = 51 and Percentage = 3.493150684931507
NoRidge : Count = 41 and Percentage = 2.808219178082192
Mitchel : Count = 49 and Percentage = 3.356164383561644
Somerst : Count = 86 and Percentage = 5.890410958904109
NWAmes : Count = 73 and Percentage = 5.0
OldTown : Count = 113 and Percentage = 7.739726027397261
BrkSide : Count = 58 and Percentage = 3.9726027397260273
Sawyer : Count = 74 and Percentage = 5.068493150684931
NridgHt : Count = 77 and Percentage = 5.273972602739726
NAmes : Count = 225 and Percentage = 15.41095890410959
SawyerW : Count = 59 and Percentage = 4.041095890410959
IDOTRR : Count = 37 and Percentage = 2.5342465753424657
MeadowV : Count = 17 and Percentage = 1.1643835616438356
Edwards : Count = 100 and Percentage = 6.8493150684931505
Timber : Count = 38 and Percentage = 2.6027397260273974
Gilbert : Count = 79 and Percentage = 5.410958904109589
StoneBr : Count = 25 and Percentage = 1.7123287671232876
ClearCr : Count = 28 and Percentage = 1.917808219178082
NPkVill : Count = 9 and Percentage = 0.6164383561643836
Blmngtn : Count = 17 and Percentage = 1.1643835616438356
BrDale : Count = 16 and Percentage = 1.095890410958904
SWISU : Count = 25 and Percentage = 1.7123287671232876
Blueste : Count = 2 and Percentage = 0.136986301369863
'''
# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_train_noNull20['Neighborhood']=df_train_noNull20['Neighborhood'].astype('category')

# ----------------------------------------------------------------------------
#Condition1 (categorical): Proximity to various conditions
df_train_noNull20['Condition1'].isnull().sum()
#0

# unique values
df_train_noNull20['Condition1'].unique()
# array(['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA','RRNe']

# distribution within BldgType
train_distribution_column('Condition1',df_train_noNull20,'SalePrice')

#Norm : Count = 1260 and Percentage = 86.3013698630137
#Feedr : Count = 81 and Percentage = 5.5479452054794525
#PosN : Count = 19 and Percentage = 1.3013698630136987
#Artery : Count = 48 and Percentage = 3.287671232876712
#RRAe : Count = 11 and Percentage = 0.7534246575342466
#RRNn : Count = 5 and Percentage = 0.3424657534246575
#RRAn : Count = 26 and Percentage = 1.7808219178082192
#PosA : Count = 8 and Percentage = 0.547945205479452
#RRNe : Count = 2 and Percentage = 0.136986301369863

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_train_noNull20['Condition1']=df_train_noNull20['Condition1'].astype('category')

# ----------------------------------------------------------------------------
#Condition2 (categorical): Proximity to various conditions (if more than one is present)
df_train_noNull20['Condition2'].isnull().sum()
#0

# unique values
df_train_noNull20['Condition2'].unique()
# array(['Norm', 'Artery', 'RRNn', 'Feedr', 'PosN', 'PosA', 'RRAn', 'RRAe']

# distribution within BldgType
train_distribution_column('Condition2',df_train_noNull20,'SalePrice')

#Norm : Count = 1445 and Percentage = 98.97260273972603
#Artery : Count = 2 and Percentage = 0.136986301369863
#RRNn : Count = 2 and Percentage = 0.136986301369863
#Feedr : Count = 6 and Percentage = 0.410958904109589
#PosN : Count = 2 and Percentage = 0.136986301369863
#PosA : Count = 1 and Percentage = 0.0684931506849315
#RRAn : Count = 1 and Percentage = 0.0684931506849315
#RRAe : Count = 1 and Percentage = 0.0684931506849315

# here we see RRNe is not present, so we keep PasA,RRAn,RRAe in Other as their contribution is least

df_train_noNull20['Condition2']=df_train_noNull20['Condition2'].replace({'PasA':'Other','RRAn':'Other','RRAe':'Other'})

# create categorical since there is no considerable relation between them
df_train_noNull20['Condition2']=df_train_noNull20['Condition2'].astype('category')


# --------------------------------------------------------------------------------------------------------------------------------------
# Pavement of streets
#Street: Type of road access to property
df_train_noNull20['Street'].isnull().sum()
#0

# unique values
df_train_noNull20['Street'].unique()
# array(['Pave', 'Grvl']

# distribution within BldgType
train_distribution_column('Street',df_train_noNull20,'SalePrice')

#Pave : Count = 1454 and Percentage = 99.58904109589041
#Grvl : Count = 6 and Percentage = 0.410958904109589

# here all the categories are present in the train data

df_train_noNull20['Street']=df_train_noNull20['Street'].replace({'Grvl':0,'Pave':1})

# ----------------------------------------------------------------------------
#PavedDrive (categorical): Paved driveway
df_train_noNull20['PavedDrive'].isnull().sum()
#0

# unique values
df_train_noNull20['PavedDrive'].unique()
# array(['Y', 'N', 'P']

# distribution within BldgType
train_distribution_column('PavedDrive',df_train_noNull20,'SalePrice')

#Y : Count = 1340 and Percentage = 91.78082191780823
#N : Count = 90 and Percentage = 6.164383561643835
#P : Count = 30 and Percentage = 2.0547945205479454

# here all the categories are present in the train data

df_train_noNull20['PavedDrive']=df_train_noNull20['PavedDrive'].replace({'N':0,'P':1,'Y':2})


# --------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------

#Changing some numeric variables into factors
#At this point, all variables are complete (No NAs), and all character variables are converted into either numeric labels 
#of into factors. However, there are 3 variables that are recorded numeric but should actually be categorical.

# lets consider following 3 variables and do some feature engineering
#YearBuilt: Original construction date
#YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
#YrSold: Year Sold (YYYY)

# we decide to make an age column age=YrSold-YearRemodAdd and we will keep YrSold as it is, to capture the effect of that year
# when it is being sold

#MoSold

df_train_noNull21=copy.deepcopy(df_train_noNull20)

df_train_noNull21['MoSold']=df_train_noNull21['MoSold'].astype('category')

# distribution within BldgType
train_distribution_column('MoSold',df_train_noNull21,'SalePrice')

#2 : Count = 52 and Percentage = 3.5616438356164384
#5 : Count = 204 and Percentage = 13.972602739726028
#9 : Count = 63 and Percentage = 4.315068493150685
#12 : Count = 59 and Percentage = 4.041095890410959
#10 : Count = 89 and Percentage = 6.095890410958904
#8 : Count = 122 and Percentage = 8.356164383561644
#11 : Count = 79 and Percentage = 5.410958904109589
#4 : Count = 141 and Percentage = 9.657534246575343
#1 : Count = 58 and Percentage = 3.9726027397260273
#7 : Count = 234 and Percentage = 16.027397260273972
#3 : Count = 106 and Percentage = 7.260273972602739
#6 : Count = 253 and Percentage = 17.328767123287673

df_train_noNull21.shape
#(438, 80)

# --------------------------------------------------------------------------------------------------------------------------------------
#MSSubClass
df_train_noNull21['MSSubClass'].isnull().sum()
#0

df_train_noNull21['MSSubClass'].dtype
# int64

# converting to categorical variable
df_train_noNull21['MSSubClass']=df_train_noNull21['MSSubClass'].astype('category')

df_train_noNull21['MSSubClass'].dtype
#CategoricalDtype(categories=[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190], ordered=False)

# unique values
df_train_noNull21['MSSubClass'].unique()
# array([ 60,  20,  70,  50, 190,  45,  90, 120,  30,  85,  80, 160,  75, 180,  40])

# distribution within MSSubClass
train_distribution_column('MSSubClass',df_train_noNull21,'SalePrice')

#60 : Count = 299 and Percentage = 20.47945205479452
#20 : Count = 536 and Percentage = 36.71232876712329
#70 : Count = 60 and Percentage = 4.109589041095891
#50 : Count = 144 and Percentage = 9.863013698630137
#190 : Count = 30 and Percentage = 2.0547945205479454
#45 : Count = 12 and Percentage = 0.821917808219178
#90 : Count = 52 and Percentage = 3.5616438356164384
#120 : Count = 87 and Percentage = 5.958904109589041
#30 : Count = 69 and Percentage = 4.726027397260274
#85 : Count = 20 and Percentage = 1.36986301369863
#80 : Count = 58 and Percentage = 3.9726027397260273
#160 : Count = 63 and Percentage = 4.315068493150685
#75 : Count = 16 and Percentage = 1.095890410958904
#180 : Count = 10 and Percentage = 0.684931506849315
#40 : Count = 4 and Percentage = 0.273972602739726

# converting into categorical
df_train_noNull21['MSSubClass']=df_train_noNull21['MSSubClass'].astype('category')

# we donot see all the categories (150). So we introduce Other category which includes: 40, 180, 45,150
df_train_noNull21['MSSubClass']=df_train_noNull21['MSSubClass'].replace({40:'Other',180:'Other',45:'Other',150:'Other'})

# decoding the values for better understanding
df_train_noNull21['MSSubClass']=df_train_noNull21['MSSubClass'].replace({20:'1 story 1946+', 30:'1 story 1945-', 
                                                                            50:'1,5 story fin', 60:'2 story 1946+', 
                                                                           70:'2 story 1945-', 75:'2,5 story all ages', 80:'split/multi level', 
                                                                           85:'split foyer', 90:'duplex all style/age', 120:'1 story PUD 1946+', 
                                                                           150:'1,5 story PUD all', 160:'2 story PUD 1946+', 
                                                                           190:'2 family conversion'})

df_train_noNull21['MSSubClass'].unique()
'''
array(['2 story 1946+', '1 story 1946+', '2 story 1945-', '1,5 story fin',
       '2 family conversion', 'Other', 'duplex all style/age',
       '1 story PUD 1946+', '1 story 1945-', 'split foyer',
       'split/multi level', '2 story PUD 1946+', '2,5 story all ages']
'''
df_train_noNull21['MSSubClass']=df_train_noNull21['MSSubClass'].astype('category')

#-------------------------------------------------------------------------------------------------------------------------------
# exporting df_train_noNull21 into csv
df_train_noNull21.to_csv(r'/Users/amanprasad/Documents/Kaggle/House Prices/df_validation_noNull21.csv', index = False)


df_train_noNull21.isnull().sum().sum()












