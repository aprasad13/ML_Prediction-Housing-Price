
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

df_test_original = pd.read_csv('/Users/amanprasad/Documents/Kaggle/House Prices/house-prices-advanced-regression-techniques/House_price_test.csv')

df_test=copy.deepcopy(df_test_original)
df_test.head()

# removing id column
id=df_test['Id']
df_test=df_test.drop(['Id'],axis=1)

df_test.info()
#Without the Id’s, the dataframe consists of 79 predictors and our response variable SalePrice.

cols=df_test.columns
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
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
'''

len(cols)
#79

# check datatype of columns
for i in range(len(cols)):
    print('{i}  {n}      {d}'.format (i=i,n=cols[i], d=df_test.dtypes[cols[i]]))

# separate numeric valraible
num_col=df_test._get_numeric_data()
type(num_col)
num_col.shape
# 1459,36


# converting this to category variable
df_test['OverallQual']=df_test['OverallQual'].astype('category')

#--------------------------------------------------------------------------------------------------------------------------------
#Missing data, label encoding, and factorizing variables
#Completeness of the data
#which variables contain missing values.

missing_val_df_test=missing_value(df_test)

#          column  missing_count
#0         PoolQC           1456
#1    MiscFeature           1408
#2          Alley           1352
#3          Fence           1169
#4    FireplaceQu            730
#5    LotFrontage            227
#6    GarageYrBlt             78
#7     GarageCond             78
#8     GarageQual             78
#9   GarageFinish             78
#10    GarageType             76
#11      BsmtCond             45
#12  BsmtExposure             44
#13      BsmtQual             44
#14  BsmtFinType1             42
#15  BsmtFinType2             42
#16    MasVnrType             16
#17    MasVnrArea             15
#18      MSZoning              4
#19  BsmtFullBath              2
#20  BsmtHalfBath              2
#21     Utilities              2
#22    Functional              2
#23   Exterior2nd              1
#24   Exterior1st              1
#25      SaleType              1
#26    BsmtFinSF1              1
#27    BsmtFinSF2              1
#28     BsmtUnfSF              1
#29   KitchenQual              1
#30    GarageCars              1
#31    GarageArea              1
#32   TotalBsmtSF              1


#--------------------------------------------------------------------------------------------------------------------------------
#Imputing missing data
#--------------------------------------------------------------------------------------------------------------------------------
#Pool Quality and the PoolArea variable
#The PoolQC is the variable with most NAs - 1456. The description is as follows:
#PoolQC (categorical): Pool quality
# So, lets check for every NA in PoolQC there is 0 in PoolArea, if yes then it is obvious that I need to just assign ‘No Pool’ to the NAs. 
#Also, the high number of NAs makes sense as normally only a small proportion of houses have a pool.
df_test_noNull = copy.deepcopy(df_test) 

df_test_noNull[df_test_noNull['PoolQC'].isnull()==True][['PoolArea']].sum()
#1373

no_PoolQC_yes_PoolArea=df_test_noNull[df_test_noNull['PoolQC'].isnull()==True][['PoolArea','PoolQC']]

no_PoolQC_yes_PoolArea[no_PoolQC_yes_PoolArea['PoolArea']>0]
#      PoolArea PoolQC
#960        368    NaN
#1043       444    NaN
#1139       561    NaN

no_PoolQC_yes_PoolArea_index=no_PoolQC_yes_PoolArea[no_PoolQC_yes_PoolArea['PoolArea']>0].index

# impute None to all the index except above
len(no_PoolQC_yes_PoolArea.index.unique())
set(no_PoolQC_yes_PoolArea_index)

index_none_PoolQC=set(no_PoolQC_yes_PoolArea.index.unique())-set(no_PoolQC_yes_PoolArea_index)

df_test_noNull.shape


df_test_noNull.loc[index_none_PoolQC,'PoolQC']=df_test_noNull.loc[index_none_PoolQC,'PoolQC'].fillna('None')
df_test_noNull['PoolQC'].isnull().sum()

# distribution within Functional
train_distribution_column('PoolQC',df_test_noNull,'PoolArea')
# since most of the data are none in PoolQC. But we can estimate these 3 values from boxplot. It seems we can will with Ex

df_test_noNull.loc[no_PoolQC_yes_PoolArea_index,'PoolQC']=df_test_noNull.loc[no_PoolQC_yes_PoolArea_index,'PoolQC'].fillna('Ex')

# unique values
df_test_noNull['PoolQC'].unique()

# distribution within Functional
train_distribution_column('PoolQC',df_test_noNull,'PoolArea')

#None : Count = 1453 and Percentage = 99.58875942426319
#Ex : Count = 5 and Percentage = 0.3427004797806717
#Gd : Count = 1 and Percentage = 0.06854009595613433

#It is also clear that I can label encode this variable as the values are ordinal. 
#As there a multiple variables that use the same quality levels
PoolQC_dict= {'None':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}
df_test_noNull['PoolQC_Ordinal']=df_test_noNull.PoolQC.map(PoolQC_dict)
df_test_noNull['PoolQC'].unique()
df_test_noNull['PoolQC_Ordinal'].unique()
#array([0, 4, 1, 3])

# droping PoolQC as we can use PoolQC_Ordinal
df_test_noNull=df_test_noNull.drop(['PoolQC'],axis=1)
df_test_noNull.columns[df_test_noNull.columns=='PoolQC']


#--------------------------------------------------------------------------------------------------------------------------------
# MiscFeature (categorical) - Miscellaneous feature not covered in other categories
# missing values= 1408
df_test_noNull['MiscFeature'].unique()
# array([nan, 'Shed', 'Gar2', 'Othr', 'TenC']

# imputing the missing value in MiscFeature with None
df_test_noNull['MiscFeature']=df_test_noNull['MiscFeature'].fillna('None')
df_test_noNull['MiscFeature'].isnull().sum()    

# distribution within MiscFeature
test_distribution_column('MiscFeature',df_test_noNull)

#None : Count = 1408 and Percentage = 96.50445510623715
#Gar2 : Count = 3 and Percentage = 0.20562028786840303
#Shed : Count = 46 and Percentage = 3.1528444139821796
#Othr : Count = 2 and Percentage = 0.13708019191226867

# we donot see Elev - Elevator so we keep this in other
# we see a relation bewteen lebels of this feature and salesprice of house: TenC>Gar2>Shed>Other>None
#It is also clear that I can label encode this variable as the values are ordinal. 
#As there a multiple variables that use the same quality levels, I am going to create a dictonary that I can reuse later on.
df_test_noNull2 = copy.deepcopy(df_test_noNull) 
mics_dict= {'None':0,'Othr':1,'Elev':1,'Shed':2,'Gar2':3,'TenC':4}
df_test_noNull2['MiscFeature_Ordinal']=df_test_noNull2.MiscFeature.map(mics_dict)
df_test_noNull2['MiscFeature'].unique()
df_test_noNull2['MiscFeature_Ordinal'].unique()

# droping MiscFeature as we can use MiscFeature_Ordinal
df_test_noNull2=df_test_noNull2.drop(['MiscFeature'],axis=1)
df_test_noNull2.columns[df_test_noNull2.columns=='MiscFeature']


# -------------------------------------------------------------------------------------------------------------------------------
# Alley (categorical) : Type of alley access to property
# missing value=1352

df_test_noNull3 = copy.deepcopy(df_test_noNull2) 

df_test_noNull3['Alley'].unique()
# array([nan, 'Grvl', 'Pave']

# imputing the missing value in Alley with None
df_test_noNull3['Alley']=df_test_noNull3['Alley'].fillna('None')
df_test_noNull3['Alley'].isnull().sum()    

# distribution within Alley
test_distribution_column('Alley',df_test_noNull3)

#None : Count = 1352 and Percentage = 92.66620973269363
#Pave : Count = 37 and Percentage = 2.5359835503769705
#Grvl : Count = 70 and Percentage = 4.797806716929403

# we do not see any relation bewteen lebels of this feature and salesprice of house
# hence convert this to categorical

df_test_noNull3['Alley']=df_test_noNull3['Alley'].astype('category')



# -------------------------------------------------------------------------------------------------------------------------------
# Fence (categorical) : Fence quality
# missing value = 1169
df_test_noNull4=copy.deepcopy(df_test_noNull3)

df_test_noNull4['Fence'].unique()
# array([nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw']

# imputing the missing value in Alley with None
df_test_noNull4['Fence']=df_test_noNull4['Alley'].fillna('None')
df_test_noNull4['Fence'].isnull().sum()    

# distribution within Alley
test_distribution_column('Fence',df_test_noNull4)

#None : Count = 1352 and Percentage = 92.66620973269363
#Pave : Count = 37 and Percentage = 2.5359835503769705
#Grvl : Count = 70 and Percentage = 4.797806716929403

# we do not see any relation bewteen lebels of this feature and salesprice of house
# hence convert this to categorical

df_test_noNull4['Fence']=df_test_noNull4['Fence'].astype('category')


# -------------------------------------------------------------------------------------------------------------------------------
#Fireplace quality, and Number of fireplaces
# FireplaceQu (categorical) - missing = 730 and Fireplaces (numeric) - missing = 0
df_test_noNull5=copy.deepcopy(df_test_noNull4)

df_test_noNull5['FireplaceQu'].unique()
# array([nan, 'TA', 'Gd', 'Fa', 'Ex', 'Po']

# lets check if there is any value in Fireplaces where FireplaceQu has null value
df_test_noNull5[df_test_noNull5['FireplaceQu'].isnull()==True][['Fireplaces']].sum()
#0 - so 0 everywhere, so now we can replace NA with None

# imputing the missing value in Alley with None
df_test_noNull5['FireplaceQu']=df_test_noNull5['FireplaceQu'].fillna('None')
df_test_noNull5['FireplaceQu'].isnull().sum()  

# distribution within Alley
test_distribution_column('FireplaceQu',df_test_noNull5)

#None : Count = 730 and Percentage = 50.03427004797807
#TA : Count = 279 and Percentage = 19.12268677176148
#Gd : Count = 364 and Percentage = 24.9485949280329
#Po : Count = 26 and Percentage = 1.7820424948594928
#Fa : Count = 41 and Percentage = 2.8101439342015078
#Ex : Count = 19 and Percentage = 1.3022618231665524

# we see a relation with SalesPrice: 
fire_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_test_noNull5['Fire_Ordinal']=df_test_noNull5.FireplaceQu.map(fire_dict)
df_test_noNull5['FireplaceQu'].unique()
df_test_noNull5['Fire_Ordinal'].unique()

# droping MiscFeature as we can use MiscFeature_Ordinal
df_test_noNull5=df_test_noNull5.drop(['FireplaceQu'],axis=1)
df_test_noNull5.columns[df_test_noNull5.columns=='FireplaceQu']


# -------------------------------------------------------------------------------------------------------------------------------
# Lot Variables
# LotFrontage (Numeric) : Linear feet of street connected to property
df_test_noNull6=copy.deepcopy(df_test_noNull5)
df_test_noNull6['LotFrontage'].dtype
# missing value = 227

#The most reasonable imputation seems to take the median per neigborhood
LotFrontage_Neighborhood_median=df_test_noNull6.groupby('Neighborhood')['LotFrontage'].median()
type(LotFrontage_Neighborhood_median)
LotFrontage_Neighborhood_median=pd.DataFrame(LotFrontage_Neighborhood_median)
LotFrontage_Neighborhood_median = LotFrontage_Neighborhood_median.reset_index()
LotFrontage_Neighborhood_median.columns = ['Neighborhood', 'LotFrontage_median']

for i in range(df_test_noNull6.shape[0]):
    for j in range(LotFrontage_Neighborhood_median.shape[0]):
        if df_test_noNull6['LotFrontage'].isnull()[i]==True:
            if df_test_noNull6['Neighborhood'][i]== LotFrontage_Neighborhood_median['Neighborhood'][j]:
                df_test_noNull6['LotFrontage'][i]=LotFrontage_Neighborhood_median['LotFrontage_median'][j]

df_test_noNull6['LotFrontage'].isnull().sum()
# 0


# ----------------------------------------------------------------------------
# LotShape (categorical): General shape of property
df_test_noNull6['LotShape'].isnull().sum()
# no missing value

df_test_noNull6['LotShape'].unique()
# array(['Reg', 'IR1', 'IR2', 'IR3']

# distribution within LotShape
test_distribution_column('LotShape',df_test_noNull6)

#Reg : Count = 934 and Percentage = 64.01644962302947
#IR1 : Count = 484 and Percentage = 33.17340644276902
#IR2 : Count = 35 and Percentage = 2.3989033584647017
#IR3 : Count = 6 and Percentage = 0.41124057573680606

# Values seem ordinal (Regular=best)
lot_dict= {'IR3':0,'IR1':2,'IR2':1,'Reg':3}
df_test_noNull6['Lot_Ordinal']=df_test_noNull6.LotShape.map(lot_dict)
df_test_noNull6['LotShape'].unique()
df_test_noNull6['Lot_Ordinal'].unique()
#array([4, 2, 1, 0])

# droping MiscFeature as we can use MiscFeature_Ordinal
df_test_noNull6=df_test_noNull6.drop(['LotShape'],axis=1)
df_test_noNull6.columns[df_test_noNull6.columns=='LotShape']




# ----------------------------------------------------------------------------
# LotConfig (categorical): Lot configuration
df_test_noNull6['LotConfig'].isnull().sum()
# no missing value

df_test_noNull6['LotConfig'].unique()
# array(['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3']

# distribution within LotConfig
test_distribution_column('LotConfig',df_test_noNull6)
#Inside : Count = 1052 and Percentage = 72.05479452054794
#FR2 : Count = 47 and Percentage = 3.219178082191781
#Corner : Count = 263 and Percentage = 18.013698630136986
#CulDSac : Count = 94 and Percentage = 6.438356164383562
#FR3 : Count = 4 and Percentage = 0.273972602739726

#The values seemed possibly ordinal, but the visualization does not show this. 
#Therefore, convert this into categorical datatype

df_test_noNull6['LotConfig']=df_test_noNull6['LotConfig'].astype('category')



# -------------------------------------------------------------------------------------------------------------------------------
# Altogether, there are 7 variables related to garages
# GarageYrBlt (year) : Year garage was built
df_test_noNull7=copy.deepcopy(df_test_noNull6)
df_test_noNull7['GarageYrBlt'].isnull().sum()
# 78

#check if NA is present in all 7 variables GarageYrBlt, GarageType, GarageArea, GarageFinish, GarageCars, GarageQual, GarageCond
gar=['GarageYrBlt','GarageType','GarageArea','GarageFinish','GarageCars','GarageQual','GarageCond']
for i in range(len(gar)):
    print('column = {c} and number_nullValue = {v}'.format(c=gar[i],v=df_test_noNull7[df_test_noNull7['GarageYrBlt'].isnull()==True][gar[i]].isnull().sum()))

#column = GarageYrBlt and number_nullValue = 78
#column = GarageType and number_nullValue = 76
#column = GarageArea and number_nullValue = 1
#column = GarageFinish and number_nullValue = 78
#column = GarageCars and number_nullValue = 1
#column = GarageQual and number_nullValue = 78
#column = GarageCond and number_nullValue = 78

# grab index where GarageYrBlt is null
GarageYrBlt_null_index=df_test_noNull7[df_test_noNull7['GarageYrBlt'].isnull()].index
GarageType_null_index=df_test_noNull7[df_test_noNull7['GarageType'].isnull()].index
GarageArea_null_index=df_test_noNull7[df_test_noNull7['GarageArea'].isnull()].index
GarageFinish_null_index=df_test_noNull7[df_test_noNull7['GarageFinish'].isnull()].index
GarageCars_null_index=df_test_noNull7[df_test_noNull7['GarageCars'].isnull()].index
GarageQual_null_index=df_test_noNull7[df_test_noNull7['GarageQual'].isnull()].index
GarageCond_null_index=df_test_noNull7[df_test_noNull7['GarageCond'].isnull()].index

set(GarageYrBlt_null_index)-set(GarageType_null_index)
#{666, 1116}

# grab 'GarageYrBlt','GarageType','GarageArea','GarageFinish','GarageCars','GarageQual','GarageCond' with same index
df=df_test_noNull7.loc[GarageYrBlt_null_index,gar]

for i in range(len(df.columns)):
    if ((df.dtypes[df.columns[i]]==int) | (df.dtypes[df.columns[i]]==float)):
        print('column = {c} and number_null/zero_Value = {v}'.format(c=df.columns[i],v=df[df.columns[i]].isnull().sum() + df[df[df.columns[i]]==0][df.columns[i]].count()))
    else:
         print('column = {c} and number_null_Value = {v}'.format(c=df.columns[i],v=df[df.columns[i]].isnull().sum()))

#column = GarageYrBlt and number_null/zero_Value = 78
#column = GarageType and number_null_Value = 76
#column = GarageArea and number_null/zero_Value = 77
#column = GarageFinish and number_null_Value = 78
#column = GarageCars and number_null/zero_Value = 77
#column = GarageQual and number_null_Value = 78
#column = GarageCond and number_null_Value = 78                                 

# so only GarageType has 2 values extra, otherwise GarageArea, GarageCars all other have 78 null. Hence we cannot impute as we have very
# less information

# done in train data
# checking relation of GarageYrBlt with salesprice
#df_test_noNull7['GarageYrBlt'].dtype

#plt.scatter(df_test_noNull7['GarageYrBlt'],df_test_noNull7['SalePrice'])
# we see that as the year increases saleprice increases but this relation is not clear so cannot say anything.
# Hence we decided to bucket the year in the interval of 5 years and imputing the null value with None

for i in range(df_test_noNull7.shape[0]):
    if df_test_noNull7['GarageYrBlt'].isnull()[i]==True:
        df_test_noNull7['GarageYrBlt'][i]=0

df_test_noNull7['GarageYrBlt'].isnull().sum()
#0

# convert float value to int
df_test_noNull7['GarageYrBlt']=df_test_noNull7['GarageYrBlt'].astype(int)
        
year_range=list(range(1901,2010,1))

# imputing missing with None
GarageYrBlt_interval=list()
for i in range(df_test_noNull7.shape[0]):
    GarageYrBlt_interval.append('None')

len(GarageYrBlt_interval)

for j in range(0,len(year_range)-2,3):
    for i in range(df_test_noNull7.shape[0]):
        if (df_test_noNull7['GarageYrBlt'][i]>0) & (df_test_noNull7['GarageYrBlt'][i]<1901):
            GarageYrBlt_interval[i]='before_1901'
        if (df_test_noNull7['GarageYrBlt'][i]>2009):
            GarageYrBlt_interval[i]='after_2009'
        if (df_test_noNull7['GarageYrBlt'][i]>=year_range[j]) & (df_test_noNull7['GarageYrBlt'][i]<=year_range[j+2]):
            p=str(year_range[j]) +'_to_'+ str(year_range[j+2])
            GarageYrBlt_interval[i]=p

len(GarageYrBlt_interval)

# print unique value in list        
u=unique(GarageYrBlt_interval)
len(u)
# merging GarageYrBlt_interval with df_test_noNull7
GarageYrBlt_interval=pd.DataFrame(GarageYrBlt_interval,columns=['GarageYrBlt_interval'])

df_test_noNull8=copy.deepcopy(df_test_noNull7)
df_test_noNull8 = pd.concat([df_test_noNull7, GarageYrBlt_interval], axis=1)

df_test_noNull8['GarageYrBlt_interval'].unique()
df_test_noNull8['GarageYrBlt_interval'].isnull().sum()
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

# dropping GarageYrBlt from df_test_noNull8
df_test_noNull8=df_test_noNull8.drop(['GarageYrBlt'],axis=1)
df_test_noNull8.columns[df_test_noNull8.columns=='GarageYrBlt']

# changing the datatype of GarageYrBlt_interval to category
df_test_noNull8['GarageYrBlt_interval']=df_test_noNull8['GarageYrBlt_interval'].astype('category')



# ----------------------------------------------------------------------
# GarageType (categorical): Garage location
df_test_noNull10=copy.deepcopy(df_test_noNull8)
df_test_noNull10['GarageType'].isnull().sum()
#76

# check if values are present in GarageArea, GarageFinish, GarageCars, GarageQual, GarageCond
# print unique values present in GarageArea, GarageFinish, GarageCars, GarageQual, GarageCond
gar=['GarageArea', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond']
for i in range(len(gar)):
    print('column= {n} and unique_value= {u}'.format(n=gar[i],u=df_test_noNull10[gar[i]].unique()))

#column=GarageArea - yes missing
#column= GarageFinish and unique_value= ['Unf' 'Fin' 'RFn' nan]  - yes missing
#column= GarageCars and unique_value= [ 1.  2.  3.  0.  4.  5. nan]  -  yes missing
#column= GarageQual and unique_value= ['TA' nan 'Fa' 'Gd' 'Po']  - yes missing
#column= GarageCond and unique_value= ['TA' nan 'Fa' 'Gd' 'Po' 'Ex']  - yes missing

df_test_noNull10['GarageType'].unique()
# array(['Attchd', 'Detchd', 'BuiltIn', nan, 'Basment', '2Types', 'CarPort']

#imputing values for GarageType with None according to document
df_test_noNull10['GarageType']=df_test_noNull10['GarageType'].fillna('None')
df_test_noNull10['GarageType'].isnull().sum()

# distribution within GarageType
test_distribution_column('GarageType',df_test_noNull10)

#Attchd : Count = 853 and Percentage = 58.46470185058259
#Detchd : Count = 392 and Percentage = 26.86771761480466
#BuiltIn : Count = 98 and Percentage = 6.716929403701165
#None : Count = 76 and Percentage = 5.2090472926662095
#Basment : Count = 17 and Percentage = 1.1651816312542838
#2Types : Count = 17 and Percentage = 1.1651816312542838
#CarPort : Count = 6 and Percentage = 0.41124057573680606

# we see a pattern None<CarPort<Detchd<Basment<2Types<Attchd<BuiltIn
Gtype_dict= {'None':0,'CarPort':1,'Detchd':2,'Basment':3,'2Types':4,'Attchd':5,'BuiltIn':6}
df_test_noNull10['Gtype_Ordinal']=df_test_noNull10.GarageType.map(Gtype_dict)
df_test_noNull10['GarageType'].unique()
df_test_noNull10['Gtype_Ordinal'].unique()
#array([5, 2, 6, 1, 0, 3, 4])

# dropping GarageType from df_test_noNull10
df_test_noNull10=df_test_noNull10.drop(['GarageType'],axis=1)
df_test_noNull10.columns[df_test_noNull10.columns=='GarageType']


# ----------------------------------------------------------------------------
# GarageFinish (categorical) : Interior finish of the garage
# imputing values for GarageFinish with None according to document
df_test_noNull10['GarageFinish']=df_test_noNull10['GarageFinish'].fillna('None')
df_test_noNull10['GarageFinish'].isnull().sum()

# unique values
df_test_noNull10['GarageFinish'].unique()
# array(['RFn', 'Unf', 'Fin', 'None']

# distribution within GarageType
test_distribution_column('GarageFinish',df_test_noNull10)

#Unf : Count = 625 and Percentage = 42.83755997258396
#Fin : Count = 367 and Percentage = 25.154215215901303
#RFn : Count = 389 and Percentage = 26.66209732693626
#None : Count = 78 and Percentage = 5.346127484578479

# we see a pattern of this varaible variable with SalePrice: None<Unf<RFn<Fin
GFin_dict= {'None':0,'Unf':1,'RFn':2,'Fin':3}
df_test_noNull10['GarageFinish_Ordinal']=df_test_noNull10.GarageFinish.map(GFin_dict)
df_test_noNull10['GarageFinish'].unique()
df_test_noNull10['GarageFinish_Ordinal'].unique()
#aarray([2, 1, 3, 0])

# dropping GarageFinish from df_test_noNull10
df_test_noNull10=df_test_noNull10.drop(['GarageFinish'],axis=1)
df_test_noNull10.columns[df_test_noNull10.columns=='GarageFinish']

# ----------------------------------------------------------------------------
# GarageQual (categorical) : Garage quality
# imputing values for GarageQual with None according to document
df_test_noNull10['GarageQual']=df_test_noNull10['GarageQual'].fillna('None')
df_test_noNull10['GarageQual'].isnull().sum()

# unique values
df_test_noNull10['GarageQual'].unique()
# array(['TA', 'Fa', 'Gd', 'None', 'Ex', 'Po']

# distribution within GarageQual
test_distribution_column('GarageQual',df_test_noNull10)

#TA : Count = 1293 and Percentage = 88.6223440712817
#None : Count = 78 and Percentage = 5.346127484578479
#Fa : Count = 76 and Percentage = 5.2090472926662095
#Gd : Count = 10 and Percentage = 0.6854009595613434
#Po : Count = 2 and Percentage = 0.13708019191226867

# we see a pattern of this varaible variable with SalePrice: None<Unf<RFn<Fin
GarageQual_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_test_noNull10['GarageQual_Ordinal']=df_test_noNull10.GarageQual.map(GarageQual_dict)
df_test_noNull10['GarageQual'].unique()
df_test_noNull10['GarageQual_Ordinal'].unique()
#array([4, 2, 5, 0, 6, 1])

# dropping GarageQual from df_test_noNull10
df_test_noNull10=df_test_noNull10.drop(['GarageQual'],axis=1)
df_test_noNull10.columns[df_test_noNull10.columns=='GarageQual']


# ----------------------------------------------------------------------
# GarageCond (categorical): Garage condition
# imputing values for GarageCond with None according to document
df_test_noNull10['GarageCond']=df_test_noNull10['GarageCond'].fillna('None')
df_test_noNull10['GarageCond'].isnull().sum()

# unique values
df_test_noNull10['GarageCond'].unique()
# array(['TA', 'Fa', 'Gd', 'None', 'Ex', 'Po']

# distribution within GarageQual
test_distribution_column('GarageCond',df_test_noNull10)

#TA : Count = 1328 and Percentage = 91.0212474297464
#None : Count = 78 and Percentage = 5.346127484578479
#Fa : Count = 39 and Percentage = 2.6730637422892394
#Gd : Count = 6 and Percentage = 0.41124057573680606
#Po : Count = 7 and Percentage = 0.47978067169294036
#Ex : Count = 1 and Percentage = 0.06854009595613433

# we see a pattern of this varaible variable with SalePrice: None<Unf<RFn<Fin
GarageCond_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_test_noNull10['GarageCond_Ordinal']=df_test_noNull10.GarageCond.map(GarageCond_dict)
df_test_noNull10['GarageCond'].unique()
df_test_noNull10['GarageCond_Ordinal'].unique()
#array([4, 2, 5, 0, 6, 1])

# dropping GarageCond from df_test_noNull10
df_test_noNull10=df_test_noNull10.drop(['GarageCond'],axis=1)
df_test_noNull10.columns[df_test_noNull10.columns=='GarageCond']

# ----------------------------------------------------------------------
# GarageArea (numerical): 
df_test_noNull10['GarageArea']=df_test_noNull10['GarageArea'].fillna(0)
df_test_noNull10['GarageArea'].isnull().sum()


# ----------------------------------------------------------------------
#GarageCars (numerical)
df_test_noNull10['GarageCars']=df_test_noNull10['GarageCars'].fillna(0)
df_test_noNull10['GarageCars'].isnull().sum()




# -------------------------------------------------------------------------------------------------------------------------------
# Altogether, there are 11 variables that relate to the Basement of a house
df_test_noNull12=copy.deepcopy(df_test_noNull10)

# print null values for all varaibles
ll=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF',
    'TotalBsmtSF','BsmtFullBath','BsmtHalfBath']

for i in range(len(ll)):
    print('column = {c} and missing_value = {m}'.format(c=ll[i],m=df_test_noNull12[ll[i]].isnull().sum()))

#column = BsmtQual and missing_value = 44
#column = BsmtCond and missing_value = 45
#column = BsmtExposure and missing_value = 44
#column = BsmtFinType1 and missing_value = 42
#column = BsmtFinSF1 and missing_value = 1
#column = BsmtFinType2 and missing_value = 42
#column = BsmtFinSF2 and missing_value = 1
#column = BsmtUnfSF and missing_value = 1
#column = TotalBsmtSF and missing_value = 1
#column = BsmtFullBath and missing_value = 2
#column = BsmtHalfBath and missing_value = 2


BsmtQual_null=pd.DataFrame(df_test_noNull12[df_test_noNull12['BsmtQual'].isnull()][ll])
BsmtQual_null.dtypes

BsmtQual_null=pd.DataFrame(df_test_noNull12[df_test_noNull12['BsmtFinType2'].isnull()][ll])

BsmtQual_null=pd.DataFrame(df_test_noNull12[df_test_noNull12['BsmtCond'].isnull()][ll])

cols=BsmtQual_null.columns

for i in range(len(cols)):
    if ((BsmtQual_null.dtypes[cols[i]]==int) | (BsmtQual_null.dtypes[cols[i]]==float)):
        print('column = {c} and null_value = {n}'.format(c=cols[i],n=BsmtQual_null[cols[i]].sum()))
    else:
        print('column = {cc} and null_value = {nn}'.format(cc=cols[i],nn=BsmtQual_null[cols[i]].isnull().sum()))


# so after running the algo we predicted following:
    
# BsmtExposure : # 27 - 1 - no, 888 - 1 - no

# BsmtQual
# array([2, 4])
# 757 - 2 - Fa
# 758 - 4 - Gd

#BsmtCond
#array([3, 3, 3])
# 580 - 3 - TA
# 725 - 3 - TA
# 1064 - 3 - TA

# substituting the above values
df_test_noNull12.loc[[27,888],'BsmtExposure']='No'

df_test_noNull12.loc[757,'BsmtQual']='Fa'
df_test_noNull12.loc[758,'BsmtQual']='Gd'

df_test_noNull12.loc[[580,725,1064],'BsmtCond']='TA'


# ----------------------------------------------------------------------
#BsmtQual (categorical):Evaluates the height of the basement
df_test_noNull12['BsmtQual'].isnull().sum()
#42
# imputing values for GarageCond with None according to document
df_test_noNull12['BsmtQual']=df_test_noNull12['BsmtQual'].fillna('None')
df_test_noNull12['BsmtQual'].isnull().sum()
#0

# unique values
df_test_noNull12['BsmtQual'].unique()
#array(['Gd', 'TA', 'Ex', 'None', 'Fa']

# distribution within GarageQual
test_distribution_column('BsmtQual',df_test_noNull12)
#TA : Count = 634 and Percentage = 43.45442083618917
#Gd : Count = 592 and Percentage = 40.57573680603153
#Ex : Count = 137 and Percentage = 9.389993145990404
#Fa : Count = 54 and Percentage = 3.701165181631254
#None : Count = 42 and Percentage = 2.8786840301576424

# we see a pattern of this varaible variable with SalePrice: None<Po<Fa<TA<Gd<Ex
BsmtQual_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_test_noNull12['BsmtQual_Ordinal']=df_test_noNull12.BsmtQual.map(BsmtQual_dict)
df_test_noNull12['BsmtQual'].unique()
df_test_noNull12['BsmtQual_Ordinal'].unique()
#array([4, 3, 5, 0, 2])

# dropping GarageCond from df_test_noNull12
df_test_noNull12=df_test_noNull12.drop(['BsmtQual'],axis=1)
df_test_noNull12.columns[df_test_noNull12.columns=='BsmtQual']

# ----------------------------------------------------------------------
# BsmtCond (categorical): Evaluates the general condition of the basement
df_test_noNull12['BsmtCond'].isnull().sum()

# imputing values for GarageCond with None according to document
df_test_noNull12['BsmtCond']=df_test_noNull12['BsmtCond'].fillna('None')
df_test_noNull12['BsmtCond'].isnull().sum()
#0

# unique values
df_test_noNull12['BsmtCond'].unique()
#array(['TA', 'Gd', 'None', 'Fa', 'Po']


# distribution within BsmtCond
test_distribution_column('BsmtCond',df_test_noNull12)
#TA : Count = 1298 and Percentage = 88.96504455106238
#Po : Count = 3 and Percentage = 0.20562028786840303
#Fa : Count = 59 and Percentage = 4.043865661411926
#Gd : Count = 57 and Percentage = 3.906785469499657
#None : Count = 42 and Percentage = 2.8786840301576424

# we see a pattern of this varaible variable with SalePrice: None<Po<Fa<TA<Gd<Ex
BsmtCond_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_test_noNull12['BsmtCond_Ordinal']=df_test_noNull12.BsmtCond.map(BsmtCond_dict)
df_test_noNull12['BsmtCond'].unique()
df_test_noNull12['BsmtCond_Ordinal'].unique()
#array([3, 4, 0, 2, 1])

# dropping BsmtCond from df_test_noNull12
df_test_noNull12=df_test_noNull12.drop(['BsmtCond'],axis=1)
df_test_noNull12.columns[df_test_noNull12.columns=='BsmtCond']

# ----------------------------------------------------------------------
# BsmtExposure (categorical): Refers to walkout or garden level walls
df_test_noNull12['BsmtExposure'].isnull().sum()
#42
# imputing values for BsmtExposure with None according to document
df_test_noNull12['BsmtExposure']=df_test_noNull12['BsmtExposure'].fillna('None')
df_test_noNull12['BsmtExposure'].isnull().sum()
#0

# unique values
df_test_noNull12['BsmtExposure'].unique()
#array(['No', 'Gd', 'Mn', 'Av', 'None']

# distribution within BsmtExposure
test_distribution_column('BsmtExposure',df_test_noNull12)
#No : Count = 953 and Percentage = 65.31871144619602
#Gd : Count = 142 and Percentage = 9.732693625771075
#Mn : Count = 125 and Percentage = 8.567511994516792
#Av : Count = 197 and Percentage = 13.502398903358465
#None : Count = 42 and Percentage = 2.8786840301576424

# we see a pattern of this varaible variable with SalePrice: None<No<Mn<Av<Gd
BsmtExposure_dict= {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4}
df_test_noNull12['BsmtExposure_Ordinal']=df_test_noNull12.BsmtExposure.map(BsmtExposure_dict)
df_test_noNull12['BsmtExposure'].unique()
df_test_noNull12['BsmtExposure_Ordinal'].unique()
#array([1, 4, 2, 3, 0])

# dropping BsmtCond from df_test_noNull12
df_test_noNull12=df_test_noNull12.drop(['BsmtExposure'],axis=1)
df_test_noNull12.columns[df_test_noNull12.columns=='BsmtExposure']

# ----------------------------------------------------------------------
# BsmtFinType1 (categorical):Rating of basement finished area
df_test_noNull12['BsmtFinType1'].isnull().sum()
#42
# imputing values for BsmtExposure with None according to document
df_test_noNull12['BsmtFinType1']=df_test_noNull12['BsmtFinType1'].fillna('None')
df_test_noNull12['BsmtFinType1'].isnull().sum()
#0

# unique values
df_test_noNull12['BsmtFinType1'].unique()
#array(['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'None', 'LwQ']


# distribution within BsmtFinType1
test_distribution_column('BsmtFinType1',df_test_noNull12)

#Rec : Count = 155 and Percentage = 10.623714873200823
#ALQ : Count = 209 and Percentage = 14.324880054832077
#GLQ : Count = 431 and Percentage = 29.5407813570939
#Unf : Count = 421 and Percentage = 28.855380397532556
#BLQ : Count = 121 and Percentage = 8.293351610692255
#LwQ : Count = 80 and Percentage = 5.483207676490747
#None : Count = 42 and Percentage = 2.8786840301576424

# we see a pattern of this varaible variable with SalePrice: None<No<Mn<Av<Gd
BsmtFinType1_dict= {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
df_test_noNull12['BsmtFinType1_Ordinal']=df_test_noNull12.BsmtFinType1.map(BsmtFinType1_dict)
df_test_noNull12['BsmtFinType1'].unique()
df_test_noNull12['BsmtFinType1_Ordinal'].unique()
#array([6, 5, 1, 3, 4, 0, 2])

# dropping BsmtCond from df_test_noNull12
df_test_noNull12=df_test_noNull12.drop(['BsmtFinType1'],axis=1)
df_test_noNull12.columns[df_test_noNull12.columns=='BsmtFinType1']

# ----------------------------------------------------------------------
# BsmtFinType2 (categorical):Rating of basement finished area (if multiple types)
df_test_noNull12['BsmtFinType2'].isnull().sum()
#37
# imputing values for BsmtFinType2 with None according to document
df_test_noNull12['BsmtFinType2']=df_test_noNull12['BsmtFinType2'].fillna('None')
df_test_noNull12['BsmtFinType2'].isnull().sum()
#0

# unique values
df_test_noNull12['BsmtFinType2'].unique()
#array(['Unf', 'BLQ', 'None', 'ALQ', 'Rec', 'LwQ', 'GLQ']


# distribution within BsmtFinType1
test_distribution_column('BsmtFinType2',df_test_noNull12)

#LwQ : Count = 41 and Percentage = 2.8101439342015078
#Unf : Count = 1237 and Percentage = 84.78409869773817
#Rec : Count = 51 and Percentage = 3.495544893762851
#BLQ : Count = 35 and Percentage = 2.3989033584647017
#GLQ : Count = 20 and Percentage = 1.3708019191226868
#ALQ : Count = 33 and Percentage = 2.2618231665524333
#None : Count = 42 and Percentage = 2.8786840301576424

# we see a pattern of this varaible variable with SalePrice: None<No<Mn<Av<Gd
BsmtFinType2_dict= {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
df_test_noNull12['BsmtFinType2_Ordinal']=df_test_noNull12.BsmtFinType2.map(BsmtFinType2_dict)
df_test_noNull12['BsmtFinType2'].unique()
df_test_noNull12['BsmtFinType2_Ordinal'].unique()
#array([1, 4, 0, 5, 3, 2, 6])

# dropping BsmtCond from df_test_noNull12
df_test_noNull12=df_test_noNull12.drop(['BsmtFinType2'],axis=1)
df_test_noNull12.columns[df_test_noNull12.columns=='BsmtFinType2']


#-----------------------------------------------------------------------------
#BsmtFinSF1
df_test_noNull12['BsmtFinSF1'].isnull().sum()
#1

# imputing 0 there
df_test_noNull12['BsmtFinSF1']=df_test_noNull12['BsmtFinSF1'].fillna(0)

#-----------------------------------------------------------------------------
#BsmtFinSF2

df_test_noNull12['BsmtFinSF2'].isnull().sum()
#1

# imputing 0 there
df_test_noNull12['BsmtFinSF2']=df_test_noNull12['BsmtFinSF2'].fillna(0)


#-----------------------------------------------------------------------------
#BsmtUnfSF
df_test_noNull12['BsmtUnfSF'].isnull().sum()
#1

# imputing 0 there
df_test_noNull12['BsmtUnfSF']=df_test_noNull12['BsmtUnfSF'].fillna(0)


#-----------------------------------------------------------------------------
#TotalBsmtSF
df_test_noNull12['TotalBsmtSF'].isnull().sum()
#1

# imputing 0 there
df_test_noNull12['TotalBsmtSF']=df_test_noNull12['TotalBsmtSF'].fillna(0)

#-----------------------------------------------------------------------------
#BsmtFullBath
df_test_noNull12['BsmtFullBath'].isnull().sum()
#2

df_test_noNull12.dtypes['BsmtFullBath']

# imputing 0 there
df_test_noNull12['BsmtFullBath']=df_test_noNull12['BsmtFullBath'].fillna(0)

#-----------------------------------------------------------------------------
#BsmtHalfBath
df_test_noNull12['BsmtHalfBath'].isnull().sum()
#2

df_test_noNull12.dtypes['BsmtHalfBath']

# imputing 0 there
df_test_noNull12['BsmtHalfBath']=df_test_noNull12['BsmtHalfBath'].fillna(0)

# ----------------------------------------------------------------------------------------------------------------------------------------
# confirming if all the features related to basement doesnot have any missing value

base_cols=['BsmtQual_Ordinal','BsmtCond_Ordinal','BsmtExposure_Ordinal','BsmtFinType1_Ordinal',
           'BsmtFinSF1','BsmtFinType2_Ordinal','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']

for i in range(len(base_cols)):
    print('column = {c} and missing_value = {m}'.format(c=base_cols[i],m=df_test_noNull12[base_cols[i]].isnull().sum()))

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
df_test_noNull13=copy.deepcopy(df_test_noNull12)

df_test_noNull13['MasVnrType'].isnull().sum()
#16

# print all the 8 rows which has nan
MasVnrType_null_index=df_test_noNull13[df_test_noNull13['MasVnrType'].isnull()].index

df_test_noNull13[df_test_noNull13['MasVnrType'].isnull()][['MasVnrType','MasVnrArea']]
'''
     MasVnrType  MasVnrArea
231         NaN         NaN
246         NaN         NaN
422         NaN         NaN
532         NaN         NaN
544         NaN         NaN
581         NaN         NaN
851         NaN         NaN
865         NaN         NaN
880         NaN         NaN
889         NaN         NaN
908         NaN         NaN
1132        NaN         NaN
1150        NaN       198.0
1197        NaN         NaN
1226        NaN         NaN
1402        NaN         NaN
'''
# we see NaN in both the columns except at index = 1150. So lets predict MasVnrType at this index

# from EDA we conclude that we can replace MasVnrType by 2 - BrkFace at 1050 index

df_test_noNull13.loc[[1150],'MasVnrType']='BrkFace'

# So we can put None in MasVnrType and 0 in MasVnrArea
df_test_noNull13['MasVnrType']=df_test_noNull13['MasVnrType'].fillna('None')
df_test_noNull13['MasVnrType'].isnull().sum()
#0

# unique values
df_test_noNull13['MasVnrType'].unique()
#array(['BrkFace', 'None', 'Stone', 'BrkCmn']

# distribution within MasVnrType
test_distribution_column('MasVnrType',df_test_noNull13)

#None : Count = 893 and Percentage = 61.20630568882797
#BrkFace : Count = 435 and Percentage = 29.81494174091844
#Stone : Count = 121 and Percentage = 8.293351610692255
#BrkCmn : Count = 10 and Percentage = 0.6854009595613434

# we see a pattern of this varaible variable with SalePrice: None<BrkCmn<BrkFace<Stone
MasVnrType_dict= {'None':0,'BrkCmn':1,'BrkFace':2,'Stone':3}
df_test_noNull13['MasVnrType_Ordinal']=df_test_noNull13.MasVnrType.map(MasVnrType_dict)
df_test_noNull13['MasVnrType'].unique()
df_test_noNull13['MasVnrType_Ordinal'].unique()
#array([2, 0, 3, 1])

# dropping BsmtCond from df_test_noNull12
df_test_noNull13=df_test_noNull13.drop(['MasVnrType'],axis=1)
df_test_noNull13.columns[df_test_noNull13.columns=='MasVnrType']

#--------------------------------------------------------------------
#MasVnrArea (numeric): Masonry veneer area in square feet
df_test_noNull13['MasVnrArea'].isnull().sum()
#15

df_test_noNull13['MasVnrArea']=df_test_noNull13['MasVnrArea'].fillna(0)
df_test_noNull13['MasVnrArea'].isnull().sum()


# ----------------------------------------------------------------------------------------------------------------------------------------
#MSZoning (categorical): Identifies the general zoning classification of the sale
#MSZoning
df_test_noNull14=copy.deepcopy(df_test_noNull13)

df_test_noNull14['MSZoning'].isnull().sum()
#4

# unique values
df_test_noNull14['MSZoning'].unique()
#array(['RL', 'RM', 'C (all)', 'FV', 'RH']

# distribution within MSZoning
test_distribution_column('MSZoning',df_test_noNull14)

#RL : Count = 1151 and Percentage = 78.83561643835617
#RM : Count = 218 and Percentage = 14.931506849315069
#C (all) : Count = 10 and Percentage = 0.684931506849315
#FV : Count = 65 and Percentage = 4.4520547945205475
#RH : Count = 16 and Percentage = 1.095890410958904

# as there is None category in this variable, we dump these missing to other category

df_test_noNull14['MSZoning']=df_test_noNull14['MSZoning'].fillna('other')

# as all the elements of this features are not present in the train data, we decide to keep internal distribution of the feature 
# in mind and keep C (all) and RH in other

df_test_noNull14['MSZoning']=df_test_noNull14['MSZoning'].replace({'C (all)': 'other','RH':'other'})

#other : Count = 29 and Percentage = 1.9876627827278959
#RL : Count = 1114 and Percentage = 76.35366689513366
#RM : Count = 242 and Percentage = 16.58670322138451
#FV : Count = 74 and Percentage = 5.071967100753941

# converting into categorical variable
df_test_noNull14['MSZoning']=df_test_noNull14['MSZoning'].astype('category')


# ----------------------------------------------------------------------------------------------------------------------------------------
# Kitchen Variables - Kitchen quality and numer of Kitchens above grade
# KitchenQual (category) :Kitchen quality
df_test_noNull14['KitchenQual'].isnull().sum()
#1

df_test_noNull14[df_test_noNull14['KitchenQual'].isnull()][['KitchenQual','KitchenAbvGr']]

#   KitchenQual  KitchenAbvGr
#95         NaN             1

df_test_noNull14['KitchenAbvGr']=df_test_noNull14['KitchenAbvGr'].astype('category')
df_test_noNull14['KitchenAbvGr'].unique

# removing the nan index
corr_test_index=df_test_noNull14[df_test_noNull14['KitchenQual'].isnull()].index
corr_test=df_test_noNull14[df_test_noNull14['KitchenQual'].isnull()][['KitchenQual','KitchenAbvGr']]

corr_train=df_test_noNull14.loc[:,['KitchenQual','KitchenAbvGr']]
corr_train.drop(95,axis=0,inplace=True)

KitchenQual_dict= {'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}
corr_train['KitchenQual']=corr_train.KitchenQual.map(KitchenQual_dict)

test_distribution_column('KitchenAbvGr',corr_train)

#1 : Count = 1392 and Percentage = 95.47325102880659
#2 : Count = 64 and Percentage = 4.3895747599451305
#0 : Count = 2 and Percentage = 0.13717421124828533

corr_train['KitchenQual']=corr_train['KitchenQual'].astype('category')
corr_train['KitchenAbvGr']=corr_train['KitchenAbvGr'].astype('category')

# we find relation between 2 categorical variables using Spearman's Rank Correlation
import scipy
from scipy.stats import spearmanr
spearmanr_coefficient, p_value=spearmanr(corr_train['KitchenAbvGr'],corr_train['KitchenQual'])
spearmanr_coefficient
# -0.1756609459219278

# this shows very low correlation

# checking the distribution of column = KitchenQual when KitchenAbvGr = 1
test_distribution_column('KitchenQual',df_test_noNull14[df_test_noNull14['KitchenAbvGr']==1])
#TA : Count = 697 and Percentage = 50.03589375448672
#Gd : Count = 563 and Percentage = 40.41636755204594
#Ex : Count = 105 and Percentage = 7.5376884422110555
#Fa : Count = 27 and Percentage = 1.9382627422828427
#nan : Count = 0 and Percentage = 0.0

# hence distribution of TA is more so we impute index = 95 with TA
df_test_noNull14.loc[95,'KitchenQual']='TA'

df_test_noNull14['KitchenQual'].isnull().sum()
#0

# unique values
df_test_noNull14['KitchenQual'].unique()
# array(['Gd', 'TA', 'Ex', 'Fa']

# distribution within BsmtFinType1
test_distribution_column('KitchenQual',df_test_noNull14)

#TA : Count = 758 and Percentage = 51.953392734749826
#Gd : Count = 565 and Percentage = 38.7251542152159
#Ex : Count = 105 and Percentage = 7.196710075394106
#Fa : Count = 31 and Percentage = 2.1247429746401645

# we see a pattern of this varaible variable with SalePrice: 
KitchenQual_dict= {'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}
df_test_noNull14['KitchenQual_Ordinal']=df_test_noNull14.KitchenQual.map(KitchenQual_dict)
df_test_noNull14['KitchenQual'].unique()
df_test_noNull14['KitchenQual_Ordinal'].unique()
#array([3, 2, 4, 1])

# dropping BsmtCond from df_test_noNull14
df_test_noNull14=df_test_noNull14.drop(['KitchenQual'],axis=1)
df_test_noNull14.columns[df_test_noNull14.columns=='KitchenQual']


#-----------------------------------------------------------------------------
# KitchenAbvGr (numeric)
df_test_noNull14['KitchenAbvGr'].isnull().sum()
#0
# hence no operation is needed for this varaible

# ----------------------------------------------------------------------------------------------------------------------------------------
#Utilities (categorical): Type of utilities available
# Utilities

df_test_noNull14['Utilities'].isnull().sum()
#2

# unique values
df_test_noNull14['Utilities'].unique()

# distribution within Utilities
test_distribution_column('Utilities',df_test_noNull14)

#AllPub : Count = 1457 and Percentage = 99.86291980808772
#nan : Count = 0 and Percentage = 0.0

# we impute by AllPub as it has max frequency
df_test_noNull14['Utilities']=df_test_noNull14['Utilities'].fillna('AllPub')

# we see a pattern of this varaible variable with SalePrice: 
Utilities_dict= {'ELO':1,'NoSeWa':2,'NoSewr':3,'AllPub':4}
df_test_noNull14['Utilities_Ordinal']=df_test_noNull14.Utilities.map(Utilities_dict)
df_test_noNull14['Utilities'].unique()
df_test_noNull14['Utilities_Ordinal'].unique()
#array([4, 2])

# dropping BsmtCond from df_test_noNull14
df_test_noNull14=df_test_noNull14.drop(['Utilities'],axis=1)
df_test_noNull14.columns[df_test_noNull14.columns=='Utilities']


# ----------------------------------------------------------------------------------------------------------------------------------------
# Functional: Home functionality
# Functional
df_test_noNull15=copy.deepcopy(df_test_noNull14)

df_test_noNull15['Functional'].isnull().sum()
#2

# unique values
df_test_noNull15['Functional'].unique()
# array(['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev']

# distribution within Functional
test_distribution_column('Functional',df_test_noNull15)

#Typ : Count = 1357 and Percentage = 93.00891021247429
#Min2 : Count = 36 and Percentage = 2.4674434544208363
#Min1 : Count = 34 and Percentage = 2.3303632625085675
#Mod : Count = 20 and Percentage = 1.3708019191226868
#Maj1 : Count = 5 and Percentage = 0.3427004797806717
#Sev : Count = 1 and Percentage = 0.06854009595613433
#Maj2 : Count = 4 and Percentage = 0.27416038382453733
#nan : Count = 0 and Percentage = 0.0

# we impute by Typ as it has max frequency
df_test_noNull15['Functional']=df_test_noNull15['Functional'].fillna('Typ')

# we see a pattern of this varaible variable with SalePrice: 
Functional_dict= {'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7}
df_test_noNull15['Functional_Ordinal']=df_test_noNull15.Functional.map(Functional_dict)
df_test_noNull15['Functional'].unique()
df_test_noNull15['Functional_Ordinal'].unique()
#array([7, 6, 3, 5, 4, 2, 1])

# dropping Functional from df_test_noNull15
df_test_noNull15=df_test_noNull15.drop(['Functional'],axis=1)
df_test_noNull15.columns[df_test_noNull15.columns=='Functional']



# ----------------------------------------------------------------------------------------------------------------------------------------
#There are 4 exterior variables
# Exterior1st (categorical): Exterior covering on house

df_test_noNull16=copy.deepcopy(df_test_noNull15)

df_test_noNull16['Exterior1st'].isnull().sum()
#1

# unique values
df_test_noNull16['Exterior1st'].unique()
#array(['VinylSd', 'Wd Sdng', 'HdBoard', 'Plywood', 'MetalSd', 'CemntBd',
#       'WdShing', 'BrkFace', 'AsbShng', 'BrkComm', 'Stucco', 'AsphShn',
#       nan, 'CBlock']

# distribution within Functional
test_distribution_column('Exterior1st',df_test_noNull16)

'''
VinylSd : Count = 510 and Percentage = 34.95544893762851
Wd Sdng : Count = 205 and Percentage = 14.05071967100754
HdBoard : Count = 220 and Percentage = 15.078821110349555
Plywood : Count = 113 and Percentage = 7.74503084304318
MetalSd : Count = 230 and Percentage = 15.764222069910899
CemntBd : Count = 65 and Percentage = 4.455106237148732
WdShing : Count = 30 and Percentage = 2.0562028786840303
BrkFace : Count = 37 and Percentage = 2.5359835503769705
AsbShng : Count = 24 and Percentage = 1.6449623029472242
BrkComm : Count = 4 and Percentage = 0.27416038382453733
Stucco : Count = 18 and Percentage = 1.2337217272104182
AsphShn : Count = 1 and Percentage = 0.06854009595613433
nan : Count = 0 and Percentage = 0.0
CBlock : Count = 1 and Percentage = 0.06854009595613433
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

# we impute by Stone as it is present in train and not in test
df_test_noNull16['Exterior1st']=df_test_noNull16['Exterior1st'].fillna('Stone')


test_distribution_column('Exterior1st',df_test_noNull16[((df_test_noNull16['ExterQual']=='TA') & (df_test_noNull16['ExterCond']=='TA'))])
'''
VinylSd : Count = 510 and Percentage = 34.95544893762851
Wd Sdng : Count = 205 and Percentage = 14.05071967100754
HdBoard : Count = 220 and Percentage = 15.078821110349555
Plywood : Count = 113 and Percentage = 7.74503084304318
MetalSd : Count = 230 and Percentage = 15.764222069910899
CemntBd : Count = 65 and Percentage = 4.455106237148732
WdShing : Count = 30 and Percentage = 2.0562028786840303
BrkFace : Count = 37 and Percentage = 2.5359835503769705
AsbShng : Count = 24 and Percentage = 1.6449623029472242
BrkComm : Count = 4 and Percentage = 0.27416038382453733
Stucco : Count = 18 and Percentage = 1.2337217272104182
Other : Count = 2 and Percentage = 0.13708019191226867
Stone : Count = 1 and Percentage = 0.06854009595613433
'''

# we donot see all the categories in train so we decide put CBlock,ImStucc,AsphShn to other category as their contribution is least
df_test_noNull16['Exterior1st']=df_test_noNull16['Exterior1st'].replace({'CBlock':'Other','ImStucc':'Other','AsphShn':'Other'})

# create categorical since there is no considerable relation between them
df_test_noNull16['Exterior1st']=df_test_noNull16['Exterior1st'].astype('category')


# -----------------------------------------------------------------------------
# Exterior2nd (categorical): Exterior covering on house (if more than one material)
# Exterior2nd
df_test_noNull16['Exterior2nd'].isnull().sum()
#1

# unique values
df_test_noNull16['Exterior2nd'].unique()
#array(['VinylSd', 'Wd Sdng', 'HdBoard', 'Plywood', 'MetalSd', 'Brk Cmn',
#       'CmentBd', 'ImStucc', 'Wd Shng', 'AsbShng', 'Stucco', 'CBlock',
#       'BrkFace', 'AsphShn', nan, 'Stone']

# distribution within Functional
test_distribution_column('Exterior2nd',df_test_noNull16)

'''
VinylSd : Count = 510 and Percentage = 34.95544893762851
Wd Sdng : Count = 194 and Percentage = 13.296778615490062
HdBoard : Count = 199 and Percentage = 13.639479095270733
Plywood : Count = 128 and Percentage = 8.773132282385195
MetalSd : Count = 233 and Percentage = 15.969842357779301
Brk Cmn : Count = 15 and Percentage = 1.0281014393420151
CmentBd : Count = 66 and Percentage = 4.523646333104867
ImStucc : Count = 5 and Percentage = 0.3427004797806717
Wd Shng : Count = 43 and Percentage = 2.9472241261137766
AsbShng : Count = 18 and Percentage = 1.2337217272104182
Stucco : Count = 21 and Percentage = 1.4393420150788212
CBlock : Count = 2 and Percentage = 0.13708019191226867
BrkFace : Count = 22 and Percentage = 1.5078821110349554
AsphShn : Count = 1 and Percentage = 0.06854009595613433
nan : Count = 0 and Percentage = 0.0
Stone : Count = 1 and Percentage = 0.06854009595613433
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

df_test_noNull16[df_test_noNull16['Exterior2nd'].isnull()][['Exterior1st','Exterior2nd','ExterQual','ExterCond']]
'''
    Exterior1st Exterior2nd ExterQual ExterCond
691       Stone         NaN        TA        TA
'''
test_distribution_column('Exterior2nd',df_test_noNull16[((df_test_noNull16['ExterQual']=='TA') & (df_test_noNull16['ExterCond']=='TA'))])
'''
VinylSd : Count = 110 and Percentage = 14.666666666666666
Wd Sdng : Count = 150 and Percentage = 20.0
HdBoard : Count = 151 and Percentage = 20.133333333333333
Plywood : Count = 99 and Percentage = 13.2
Brk Cmn : Count = 13 and Percentage = 1.7333333333333334
MetalSd : Count = 132 and Percentage = 17.6
Wd Shng : Count = 29 and Percentage = 3.8666666666666667
Stucco : Count = 14 and Percentage = 1.8666666666666667
CBlock : Count = 2 and Percentage = 0.26666666666666666
CmentBd : Count = 22 and Percentage = 2.933333333333333
BrkFace : Count = 15 and Percentage = 2.0
AsbShng : Count = 10 and Percentage = 1.3333333333333333
ImStucc : Count = 1 and Percentage = 0.13333333333333333
nan : Count = 0 and Percentage = 0.0
Stone : Count = 1 and Percentage = 0.13333333333333333
'''
# we donot get anything considering as all the % are very close


test_distribution_column('Exterior2nd',(df_test_noNull16[df_test_noNull16['Exterior1st']=='Wd Sdng']))
'''
Wd Sdng : Count = 176 and Percentage = 85.85365853658537
Plywood : Count = 9 and Percentage = 4.390243902439025
Wd Shng : Count = 9 and Percentage = 4.390243902439025
Stucco : Count = 2 and Percentage = 0.975609756097561
HdBoard : Count = 5 and Percentage = 2.4390243902439024
MetalSd : Count = 3 and Percentage = 1.4634146341463414
VinylSd : Count = 1 and Percentage = 0.4878048780487805
'''
test_distribution_column('Exterior2nd',(df_test_noNull16[df_test_noNull16['Exterior1st']=='VinylSd']))
'''
VinylSd : Count = 503 and Percentage = 98.62745098039215
Wd Shng : Count = 4 and Percentage = 0.7843137254901961
MetalSd : Count = 2 and Percentage = 0.39215686274509803
Wd Sdng : Count = 1 and Percentage = 0.19607843137254902
'''
test_distribution_column('Exterior2nd',(df_test_noNull16[df_test_noNull16['Exterior1st']=='HdBoard']))
'''
HdBoard : Count = 190 and Percentage = 86.36363636363636
ImStucc : Count = 4 and Percentage = 1.8181818181818181
Wd Sdng : Count = 2 and Percentage = 0.9090909090909091
Plywood : Count = 18 and Percentage = 8.181818181818182
Wd Shng : Count = 4 and Percentage = 1.8181818181818181
BrkFace : Count = 1 and Percentage = 0.45454545454545453
Stucco : Count = 1 and Percentage = 0.45454545454545453
'''
# we see a pattern, whichever value we put in Exterior1st, Exterior2nd has max % of that. 
#Hecce we decide to fill miising index = 691 by Stone as it has Stone in Exterior1st

df_test_noNull16['Exterior2nd']=df_test_noNull16['Exterior2nd'].fillna('Stone')


# we donot see all the categories in train so we decide put CBlock,ImStucc,AsphShn to other category as their contribution is least
df_test_noNull16['Exterior2nd']=df_test_noNull16['Exterior2nd'].replace({'CBlock':'Other','ImStucc':'Other','AsphShn':'Other'})

# create categorical since there is no considerable relation between them
df_test_noNull16['Exterior2nd']=df_test_noNull16['Exterior2nd'].astype('category')

df_test_noNull16['Exterior2nd'].unique()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ExterQual(categorical): Evaluates the quality of the material on the exterior 
# ExterQual

df_test_noNull16['ExterQual'].isnull().sum()
#0

# unique values
df_test_noNull16['ExterQual'].unique()
# array(['Gd', 'TA', 'Ex', 'Fa']

# distribution within ExterQual
test_distribution_column('ExterQual',df_test_noNull16)

#Gd : Count = 488 and Percentage = 33.42465753424658
#TA : Count = 906 and Percentage = 62.054794520547944
#Ex : Count = 52 and Percentage = 3.5616438356164384
#Fa : Count = 14 and Percentage = 0.958904109589041

# we see a pattern of this varaible variable with SalePrice: 
ExterQual_dict= {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
df_test_noNull16['ExterQual_Ordinal']=df_test_noNull16.ExterQual.map(ExterQual_dict)
df_test_noNull16['ExterQual'].unique()
df_test_noNull16['ExterQual_Ordinal'].unique()
#array([4, 3, 5, 2])

# dropping Functional from df_test_noNull15
df_test_noNull16=df_test_noNull16.drop(['ExterQual'],axis=1)
df_test_noNull16.columns[df_test_noNull16.columns=='ExterQual']

# -----------------------------------------------------------------------------
# ExterCond (categorical): Evaluates the present condition of the material on the exterior
# ExterCond

df_test_noNull16['ExterCond'].isnull().sum()
#0

# unique values
df_test_noNull16['ExterCond'].unique()
# array(['TA', 'Gd', 'Fa', 'Po', 'Ex']

# distribution within ExterCond
test_distribution_column('ExterCond',df_test_noNull16)

#TA : Count = 1282 and Percentage = 87.8082191780822
#Gd : Count = 146 and Percentage = 10.0
#Fa : Count = 28 and Percentage = 1.917808219178082
#Po : Count = 1 and Percentage = 0.0684931506849315
#Ex : Count = 3 and Percentage = 0.2054794520547945

# we see a pattern of this varaible variable with SalePrice: 
ExterCond_dict= {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
df_test_noNull16['ExterCond_Ordinal']=df_test_noNull16.ExterCond.map(ExterCond_dict)
df_test_noNull16['ExterCond'].unique()
df_test_noNull16['ExterCond_Ordinal'].unique()
#array([4, 3, 5, 2])

# dropping Functional from df_test_noNull15
df_test_noNull16=df_test_noNull16.drop(['ExterCond'],axis=1)
df_test_noNull16.columns[df_test_noNull16.columns=='ExterCond']


# ------------------------------------------------------------------------------------------------------------------------------
# Electrical (categorical): Electrical system
# Electrical
df_test_noNull17=copy.deepcopy(df_test_noNull16)

df_test_noNull17['Electrical'].isnull().sum()
#0

# unique values
df_test_noNull17['Electrical'].unique()
# array(['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix', nan]

# distribution within ExterCond
test_distribution_column('Electrical',df_test_noNull17)

#SBrkr : Count = 1337 and Percentage = 91.6381082933516
#FuseA : Count = 94 and Percentage = 6.442769019876628
#FuseF : Count = 23 and Percentage = 1.5764222069910898
#FuseP : Count = 5 and Percentage = 0.3427004797806717

# we see a pattern of this varaible variable with SalePrice: 
Electrical_dict= {'Mix':1, 'FuseP':2, 'FuseF':3, 'FuseA':4, 'SBrkr':5}
df_test_noNull17['Electrical_Ordinal']=df_test_noNull17.Electrical.map(Electrical_dict)
df_test_noNull17['Electrical'].unique()
df_test_noNull17['Electrical_Ordinal'].unique()
#array([5, 3, 4, 2, 1])

# dropping Electrical from df_test_noNull15
df_test_noNull17=df_test_noNull17.drop(['Electrical'],axis=1)
df_test_noNull17.columns[df_test_noNull17.columns=='Electrical']

# --------------------------------------------------------------------------------------------------------------------------------------
# SaleType (categorical): Type of sale

df_test_noNull17['SaleType'].isnull().sum()
#1

# unique values
df_test_noNull17['SaleType'].unique()
# array(['WD', 'COD', 'New', 'ConLD', 'Oth', 'Con', 'ConLw', 'ConLI', 'CWD',nan]

# distribution within ExterCond
test_distribution_column('SaleType',df_test_noNull17)

'''
WD : Count = 1258 and Percentage = 86.223440712817
COD : Count = 44 and Percentage = 3.015764222069911
New : Count = 117 and Percentage = 8.019191226867717
ConLD : Count = 17 and Percentage = 1.1651816312542838
Oth : Count = 4 and Percentage = 0.27416038382453733
Con : Count = 3 and Percentage = 0.20562028786840303
ConLw : Count = 3 and Percentage = 0.20562028786840303
ConLI : Count = 4 and Percentage = 0.27416038382453733
CWD : Count = 8 and Percentage = 0.5483207676490747
nan : Count = 0 and Percentage = 0.0
'''
df_test_noNull17[df_test_noNull17['SaleType'].isnull()][['SaleType','SaleCondition']]

#     SaleType SaleCondition
#1029      NaN        Normal

test_distribution_column('SaleType',df_test_noNull17[df_test_noNull17['SaleCondition']=='Normal'])
'''
WD : Count = 1154 and Percentage = 95.84717607973423
COD : Count = 20 and Percentage = 1.6611295681063123
ConLD : Count = 15 and Percentage = 1.2458471760797343
ConLw : Count = 2 and Percentage = 0.16611295681063123
ConLI : Count = 1 and Percentage = 0.08305647840531562
Oth : Count = 1 and Percentage = 0.08305647840531562
Con : Count = 2 and Percentage = 0.16611295681063123
CWD : Count = 8 and Percentage = 0.6644518272425249
nan : Count = 0 and Percentage = 0.0
'''
# we see WD has max frequency when SaleCOndition = Normal, so we replace the missing value by WD

df_test_noNull17['SaleType']=df_test_noNull17['SaleType'].fillna('WD')


# from above distribution we see that all categories are not present - VWD, so we put VWD in other (Oth) 
df_test_noNull17['SaleType']=df_test_noNull17['SaleType'].replace({'VWD':'Oth'})

# create categorical since there is no considerable relation between them
df_test_noNull17['SaleType']=df_test_noNull17['SaleType'].astype('category')

df_test_noNull17['SaleType'].unique()

# ----------------------------------------------------------------------------
# SaleCondition (categorical): Condition of sale

df_test_noNull17['SaleCondition'].isnull().sum()
#0

# unique values
df_test_noNull17['SaleCondition'].unique()
# array(['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family']

# distribution within SaleCondition
test_distribution_column('SaleCondition',df_test_noNull17)
'''
Normal : Count = 1204 and Percentage = 82.52227553118574
Partial : Count = 120 and Percentage = 8.224811514736121
Abnorml : Count = 89 and Percentage = 6.100068540095956
Family : Count = 26 and Percentage = 1.7820424948594928
Alloca : Count = 12 and Percentage = 0.8224811514736121
AdjLand : Count = 8 and Percentage = 0.5483207676490747
'''
# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_test_noNull17['SaleCondition']=df_test_noNull17['SaleCondition'].astype('category')

# --------------------------------------------------------------------------------------------------------------------------------------
# Foundation (categorical): Type of foundation
df_test_noNull18=copy.deepcopy(df_test_noNull17)

df_test_noNull18['Foundation'].isnull().sum()
#0

# unique values
df_test_noNull18['Foundation'].unique()
# array(['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone']

# distribution within SaleCondition
test_distribution_column('Foundation',df_test_noNull18)

#CBlock : Count = 601 and Percentage = 41.19259766963674
#PConc : Count = 661 and Percentage = 45.3050034270048
#BrkTil : Count = 165 and Percentage = 11.309115832762165
#Stone : Count = 5 and Percentage = 0.3427004797806717
#Slab : Count = 25 and Percentage = 1.7135023989033584
#Wood : Count = 2 and Percentage = 0.13708019191226867

# here all the categories are present in the train data

# create catgorical since there is no considerable relation between them
df_test_noNull18['Foundation']=df_test_noNull18['Foundation'].astype('category')

# --------------------------------------------------------------------------------------------------------------------------------------
# Heating and Air condition
# Heating (categorical): Type of heating

df_test_noNull18['Heating'].isnull().sum()
#0

# unique values
df_test_noNull18['Heating'].unique()
# array(['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor']

# distribution within Heating
test_distribution_column('Heating',df_test_noNull18)

#GasA : Count = 1446 and Percentage = 99.10897875257025
#GasW : Count = 9 and Percentage = 0.6168608636052091
#Grav : Count = 2 and Percentage = 0.13708019191226867
#Wall : Count = 2 and Percentage = 0.13708019191226867

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_test_noNull18['Heating']=df_test_noNull18['Heating'].astype('category')

# ----------------------------------------------------------------------------
# HeatingQC (categorical): Heating quality and condition

df_test_noNull18['HeatingQC'].isnull().sum()
#0

# unique values
df_test_noNull18['HeatingQC'].unique()
# array(['Ex', 'Gd', 'TA', 'Fa', 'Po']

# distribution within Heating
test_distribution_column('HeatingQC',df_test_noNull18)

#TA : Count = 429 and Percentage = 29.40370116518163
#Gd : Count = 233 and Percentage = 15.969842357779301
#Ex : Count = 752 and Percentage = 51.54215215901302
#Fa : Count = 43 and Percentage = 2.9472241261137766
#Po : Count = 2 and Percentage = 0.13708019191226867

# here all the categories are present in the train data

# we see a pattern of this varaible variable with SalePrice: 
HeatingQC_dict= {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
df_test_noNull18['HeatingQC_Ordinal']=df_test_noNull18.HeatingQC.map(HeatingQC_dict)
df_test_noNull18['HeatingQC'].unique()
df_test_noNull18['HeatingQC_Ordinal'].unique()
#array([4, 3, 5, 2])

# dropping Functional from df_test_noNull15
df_test_noNull18=df_test_noNull18.drop(['HeatingQC'],axis=1)
df_test_noNull18.columns[df_test_noNull18.columns=='HeatingQC']

# ----------------------------------------------------------------------------
# CentralAir: Central air conditioning
df_test_noNull18['CentralAir'].isnull().sum()
#0

# unique values
df_test_noNull18['CentralAir'].unique()
# array(['Y', 'N']

# distribution within Heating
test_distribution_column('CentralAir',df_test_noNull18)

#Y : Count = 1358 and Percentage = 93.07745030843043
#N : Count = 101 and Percentage = 6.922549691569568

# here all the categories are present in the train data

df_test_noNull18['CentralAir']=df_test_noNull18['CentralAir'].replace({'N':0,'Y':1})


# --------------------------------------------------------------------------------------------------------------------------------------
# There are 2 variables that deal with the roof of houses.
# RoofStyle (categorical): Type of roof
df_test_noNull19=copy.deepcopy(df_test_noNull18)

df_test_noNull19['RoofStyle'].isnull().sum()
#0

# unique values
df_test_noNull19['RoofStyle'].unique()
# array(['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed']

# distribution within Heating
test_distribution_column('RoofStyle',df_test_noNull19)

#Gable : Count = 1169 and Percentage = 80.12337217272105
#Hip : Count = 265 and Percentage = 18.1631254283756
#Gambrel : Count = 11 and Percentage = 0.7539410555174777
#Flat : Count = 7 and Percentage = 0.47978067169294036
#Mansard : Count = 4 and Percentage = 0.27416038382453733
#Shed : Count = 3 and Percentage = 0.20562028786840303

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_test_noNull19['RoofStyle']=df_test_noNull19['RoofStyle'].astype('category')

# ----------------------------------------------------------------------------
# RoofMatl (categorical): Roof material
df_test_noNull19['RoofMatl'].isnull().sum()
#0

# unique values
df_test_noNull19['RoofMatl'].unique()
# array(['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv','Roll', 'ClyTile']

# distribution within Heating
test_distribution_column('RoofMatl',df_test_noNull19)

#CompShg : Count = 1442 and Percentage = 98.83481836874572
#Tar&Grv : Count = 12 and Percentage = 0.8224811514736121
#WdShake : Count = 4 and Percentage = 0.27416038382453733
#WdShngl : Count = 1 and Percentage = 0.06854009595613433

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_test_noNull19['RoofMatl']=df_test_noNull19['RoofMatl'].astype('category')


# --------------------------------------------------------------------------------------------------------------------------------------
# 2 variables that specify the flatness and slope of the propoerty.
# LandContour (categorical): Flatness of the property
df_test_noNull19['LandContour'].isnull().sum()
#0

# unique values
df_test_noNull19['LandContour'].unique()
# array(['Lvl', 'Bnk', 'Low', 'HLS']

# distribution within Heating
test_distribution_column('LandContour',df_test_noNull19)

#Lvl : Count = 1311 and Percentage = 89.85606579849212
#HLS : Count = 70 and Percentage = 4.797806716929403
#Bnk : Count = 54 and Percentage = 3.701165181631254
#Low : Count = 24 and Percentage = 1.6449623029472242

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_test_noNull19['LandContour']=df_test_noNull19['LandContour'].astype('category')

# ----------------------------------------------------------------------------
# LandSlope (categorical): Slope of property
df_test_noNull19['LandSlope'].isnull().sum()
#0

# unique values
df_test_noNull19['LandSlope'].unique()
# array(['Gtl', 'Mod', 'Sev']

# distribution within LandSlope
test_distribution_column('LandSlope',df_test_noNull19)

#Gtl : Count = 1396 and Percentage = 95.68197395476354
#Mod : Count = 60 and Percentage = 4.112405757368061
#Sev : Count = 3 and Percentage = 0.20562028786840303

# here all the categories are present in the train data

df_test_noNull19['LandSlope']=df_test_noNull19['LandSlope'].replace({'Sev':0,'Mod':1,'Gtl':2})


# --------------------------------------------------------------------------------------------------------------------------------------
#2 variables that specify the type and style of dwelling.
#BldgType: Type of dwelling
df_test_noNull20=copy.deepcopy(df_test_noNull19)
df_test_noNull20['BldgType'].isnull().sum()
#0

# unique values
df_test_noNull20['BldgType'].unique()
# array(['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs']

# distribution within BldgType
test_distribution_column('BldgType',df_test_noNull20)

#1Fam : Count = 1205 and Percentage = 82.59081562714188
#TwnhsE : Count = 113 and Percentage = 7.74503084304318
#Twnhs : Count = 53 and Percentage = 3.63262508567512
#Duplex : Count = 57 and Percentage = 3.906785469499657
#2fmCon : Count = 31 and Percentage = 2.1247429746401645

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_test_noNull20['BldgType']=df_test_noNull20['BldgType'].astype('category')

# ----------------------------------------------------------------------------
#HouseStyle: Style of dwelling
df_test_noNull20['HouseStyle'].isnull().sum()
#0

# unique values
df_test_noNull20['HouseStyle'].unique()
# array(['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf','2.5Fin']

# distribution within BldgType
test_distribution_column('HouseStyle',df_test_noNull20)

#1Story : Count = 745 and Percentage = 51.062371487320085
#2Story : Count = 427 and Percentage = 29.26662097326936
#SLvl : Count = 63 and Percentage = 4.318026045236463
#1.5Fin : Count = 160 and Percentage = 10.966415352981494
#SFoyer : Count = 46 and Percentage = 3.1528444139821796
#2.5Unf : Count = 13 and Percentage = 0.8910212474297464
#1.5Unf : Count = 5 and Percentage = 0.3427004797806717

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_test_noNull20['HouseStyle']=df_test_noNull20['HouseStyle'].astype('category')


# --------------------------------------------------------------------------------------------------------------------------------------
#3 variables that specify the physical location, and the proximity of ‘conditions’.
#Neighborhood (categorical): Physical locations within Ames city limits
df_test_noNull20['Neighborhood'].isnull().sum()
#0

# unique values
df_test_noNull20['Neighborhood'].unique()
# array(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
#       'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
#       'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
#       'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
#       'Blueste']

# distribution within BldgType
test_distribution_column('Neighborhood',df_test_noNull20)
'''
NAmes : Count = 218 and Percentage = 14.941740918437286
Gilbert : Count = 86 and Percentage = 5.894448252227553
StoneBr : Count = 26 and Percentage = 1.7820424948594928
BrDale : Count = 14 and Percentage = 0.9595613433858807
NPkVill : Count = 14 and Percentage = 0.9595613433858807
NridgHt : Count = 89 and Percentage = 6.100068540095956
Blmngtn : Count = 11 and Percentage = 0.7539410555174777
NoRidge : Count = 30 and Percentage = 2.0562028786840303
Somerst : Count = 96 and Percentage = 6.579849211788897
SawyerW : Count = 66 and Percentage = 4.523646333104867
Sawyer : Count = 77 and Percentage = 5.277587388622344
NWAmes : Count = 58 and Percentage = 3.9753255654557917
OldTown : Count = 126 and Percentage = 8.636052090472926
BrkSide : Count = 50 and Percentage = 3.427004797806717
ClearCr : Count = 16 and Percentage = 1.0966415352981493
SWISU : Count = 23 and Percentage = 1.5764222069910898
Edwards : Count = 94 and Percentage = 6.442769019876628
CollgCr : Count = 117 and Percentage = 8.019191226867717
Crawfor : Count = 52 and Percentage = 3.5640849897189857
Blueste : Count = 8 and Percentage = 0.5483207676490747
IDOTRR : Count = 56 and Percentage = 3.838245373543523
Mitchel : Count = 65 and Percentage = 4.455106237148732
Timber : Count = 34 and Percentage = 2.3303632625085675
MeadowV : Count = 20 and Percentage = 1.3708019191226868
Veenker : Count = 13 and Percentage = 0.8910212474297464
'''
# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_test_noNull20['Neighborhood']=df_test_noNull20['Neighborhood'].astype('category')

# ----------------------------------------------------------------------------
#Condition1 (categorical): Proximity to various conditions
df_test_noNull20['Condition1'].isnull().sum()
#0

# unique values
df_test_noNull20['Condition1'].unique()
# array(['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA','RRNe']

# distribution within BldgType
test_distribution_column('Condition1',df_test_noNull20)

#Feedr : Count = 83 and Percentage = 5.68882796435915
#Norm : Count = 1251 and Percentage = 85.74366004112406
#PosN : Count = 20 and Percentage = 1.3708019191226868
#RRNe : Count = 4 and Percentage = 0.27416038382453733
#Artery : Count = 44 and Percentage = 3.015764222069911
#RRNn : Count = 4 and Percentage = 0.27416038382453733
#PosA : Count = 12 and Percentage = 0.8224811514736121
#RRAn : Count = 24 and Percentage = 1.6449623029472242
#RRAe : Count = 17 and Percentage = 1.1651816312542838

# here all the categories are present in the train data

# create categorical since there is no considerable relation between them
df_test_noNull20['Condition1']=df_test_noNull20['Condition1'].astype('category')

# ----------------------------------------------------------------------------
#Condition2 (categorical): Proximity to various conditions (if more than one is present)
df_test_noNull20['Condition2'].isnull().sum()
#0

# unique values
df_test_noNull20['Condition2'].unique()
# array(['Norm', 'Artery', 'RRNn', 'Feedr', 'PosN', 'PosA', 'RRAn', 'RRAe']

# distribution within BldgType
test_distribution_column('Condition2',df_test_noNull20)

#Norm : Count = 1444 and Percentage = 98.97189856065799
#Feedr : Count = 7 and Percentage = 0.47978067169294036
#PosA : Count = 3 and Percentage = 0.20562028786840303
#PosN : Count = 2 and Percentage = 0.13708019191226867
#Artery : Count = 3 and Percentage = 0.20562028786840303

# according to train coding

df_test_noNull20['Condition2']=df_test_noNull20['Condition2'].replace({'PasA':'Other','RRAn':'Other','RRAe':'Other'})

# create categorical since there is no considerable relation between them
df_test_noNull20['Condition2']=df_test_noNull20['Condition2'].astype('category')


# --------------------------------------------------------------------------------------------------------------------------------------
# Pavement of streets
#Street: Type of road access to property
df_test_noNull20['Street'].isnull().sum()
#0

# unique values
df_test_noNull20['Street'].unique()
# array(['Pave', 'Grvl']

# distribution within BldgType
test_distribution_column('Street',df_test_noNull20)

#Pave : Count = 1453 and Percentage = 99.58875942426319
#Grvl : Count = 6 and Percentage = 0.41124057573680606

# here all the categories are present in the train data

df_test_noNull20['Street']=df_test_noNull20['Street'].replace({'Grvl':0,'Pave':1})

# ----------------------------------------------------------------------------
#PavedDrive (categorical): Paved driveway
df_test_noNull20['PavedDrive'].isnull().sum()
#0

# unique values
df_test_noNull20['PavedDrive'].unique()
# array(['Y', 'N', 'P']

# distribution within BldgType
test_distribution_column('PavedDrive',df_test_noNull20)

#Y : Count = 1301 and Percentage = 89.17066483893078
#N : Count = 126 and Percentage = 8.636052090472926
#P : Count = 32 and Percentage = 2.1932830705962987

# here all the categories are present in the train data

df_test_noNull20['PavedDrive']=df_test_noNull20['PavedDrive'].replace({'N':0,'P':1,'Y':2})


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

df_test_noNull21=copy.deepcopy(df_test_noNull20)

df_test_noNull21['MoSold'].isnull().sum()
#0

df_test_noNull21['MoSold']=df_test_noNull21['MoSold'].astype('category')

# distribution within BldgType
test_distribution_column('MoSold',df_test_noNull21)
'''
6 : Count = 250 and Percentage = 17.135023989033584
3 : Count = 126 and Percentage = 8.636052090472926
1 : Count = 64 and Percentage = 4.386566141192597
4 : Count = 138 and Percentage = 9.458533241946538
5 : Count = 190 and Percentage = 13.022618231665524
2 : Count = 81 and Percentage = 5.551747772446881
7 : Count = 212 and Percentage = 14.53050034270048
10 : Count = 84 and Percentage = 5.757368060315285
8 : Count = 111 and Percentage = 7.607950651130912
11 : Count = 63 and Percentage = 4.318026045236463
9 : Count = 95 and Percentage = 6.511309115832762
12 : Count = 45 and Percentage = 3.0843043180260454
'''
df_test_noNull21.shape


# --------------------------------------------------------------------------------------------------------------------------------------
#MSSubClass
df_test_noNull21['MSSubClass'].isnull().sum()
#0

df_test_noNull21['MSSubClass'].dtype
# int64

# converting to categorical variable
df_test_noNull21['MSSubClass']=df_test_noNull21['MSSubClass'].astype('category')

df_test_noNull21['MSSubClass'].dtype
#CategoricalDtype(categories=[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190], ordered=False)

# unique values
df_test_noNull21['MSSubClass'].unique()
# array([ 60,  20,  70,  50, 190,  45,  90, 120,  30,  85,  80, 160,  75, 180,  40])

# distribution within MSSubClass
test_distribution_column('MSSubClass',df_test_noNull21)
'''
20 : Count = 543 and Percentage = 37.21727210418094
60 : Count = 276 and Percentage = 18.917066483893077
120 : Count = 95 and Percentage = 6.511309115832762
160 : Count = 65 and Percentage = 4.455106237148732
80 : Count = 60 and Percentage = 4.112405757368061
30 : Count = 70 and Percentage = 4.797806716929403
50 : Count = 143 and Percentage = 9.801233721727211
90 : Count = 57 and Percentage = 3.906785469499657
85 : Count = 28 and Percentage = 1.9191226867717615
190 : Count = 31 and Percentage = 2.1247429746401645
45 : Count = 6 and Percentage = 0.41124057573680606
70 : Count = 68 and Percentage = 4.660726525017135
75 : Count = 7 and Percentage = 0.47978067169294036
180 : Count = 7 and Percentage = 0.47978067169294036
40 : Count = 2 and Percentage = 0.13708019191226867
150 : Count = 1 and Percentage = 0.06854009595613433
'''
# converting into categorical
df_test_noNull21['MSSubClass']=df_test_noNull21['MSSubClass'].astype('category')

# we donot see all the categories (150). So we introduce Other category which includes: 40, 180, 45,150
df_test_noNull21['MSSubClass']=df_test_noNull21['MSSubClass'].replace({40:'Other',180:'Other',45:'Other',150:'Other'})

# decoding the values for better understanding
df_test_noNull21['MSSubClass']=df_test_noNull21['MSSubClass'].replace({20:'1 story 1946+', 30:'1 story 1945-', 
                                                                            50:'1,5 story fin', 60:'2 story 1946+', 
                                                                           70:'2 story 1945-', 75:'2,5 story all ages', 80:'split/multi level', 
                                                                           85:'split foyer', 90:'duplex all style/age', 120:'1 story PUD 1946+', 
                                                                           150:'1,5 story PUD all', 160:'2 story PUD 1946+', 
                                                                           190:'2 family conversion'})

df_test_noNull21['MSSubClass'].unique()
'''
array(['2 story 1946+', '1 story 1946+', '2 story 1945-', '1,5 story fin',
       '2 family conversion', 'Other', 'duplex all style/age',
       '1 story PUD 1946+', '1 story 1945-', 'split foyer',
       'split/multi level', '2 story PUD 1946+', '2,5 story all ages']
'''
df_test_noNull21['MSSubClass']=df_test_noNull21['MSSubClass'].astype('category')

df_test_noNull21.isnull().sum().sum()
#0

#-------------------------------------------------------------------------------------------------------------------------------
# exporting df_test_noNull21 into csv
df_test_noNull21.to_csv(r'/Users/amanprasad/Documents/Kaggle/House Prices/df_test_noNull21_V2.csv', index = False)















