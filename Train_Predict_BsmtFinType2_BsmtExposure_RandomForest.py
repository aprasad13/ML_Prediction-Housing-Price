

df_train_noNull11=copy.deepcopy(df_train_noNull10)
BsmtExposure_null=copy.deepcopy(df_train_noNull11[df_train_noNull11['BsmtExposure'].isnull()==True][ll])
BsmtExposure_null[BsmtExposure_null['BsmtQual'].isnull()==False]


BsmtFinType2_null=copy.deepcopy(df_train_noNull11[df_train_noNull11['BsmtFinType2'].isnull()==True][ll])
BsmtFinType2_null[BsmtFinType2_null['BsmtQual'].isnull()==False]

# we see that BsmtExposure and BsmtFinType2 has 1 extra null value. We need to investigate those values
# we will predict this with random forest.

# prepare train test split
#BsmtExposure_null
BsmtExposure_null=copy.deepcopy(df_train_noNull11[df_train_noNull11['BsmtExposure'].isnull()==True][ll])
test=BsmtExposure_null[BsmtExposure_null['BsmtQual'].isnull()==False].index[1]
# 948

# from df_train_noNull11 extract basement columns and remove this element
df_base=copy.deepcopy(df_train_noNull11[ll])
df_base_testdata_BsmtExposure=copy.deepcopy(pd.DataFrame(df_base.loc[[test]]))
# removing BsmtExposure column
df_base_testdata_BsmtExposure=df_base_testdata_BsmtExposure.drop(['BsmtExposure'],axis=1)

df_base=copy.deepcopy(df_base.drop(test))

#BsmtFinType2
BsmtFinType2_null=copy.deepcopy(df_train_noNull11[df_train_noNull11['BsmtFinType2'].isnull()==True][ll])
test_BsmtFinType2=BsmtFinType2_null[BsmtFinType2_null['BsmtQual'].isnull()==False].index[0]
# 500
df_base_testdata_BsmtFinType2=copy.deepcopy(pd.DataFrame(df_base.loc[[test_BsmtFinType2]]))
# removing BsmtFinType2 column
df_base_testdata_BsmtFinType2=df_base_testdata_BsmtFinType2.drop(['BsmtFinType2'],axis=1)

df_base=copy.deepcopy(df_base.drop(test_BsmtFinType2))

# --------------------------------------------------------------------------------------------------------------------------------------
# preparation of train data - df_base

#BsmtQual
df_base['BsmtQual'].isnull().sum()
#37
# imputing values for GarageCond with None according to document
df_base['BsmtQual']=df_base['BsmtQual'].fillna('None')
df_base['BsmtQual'].isnull().sum()
#0

# unique values
df_base['BsmtQual'].unique()
#array(['Gd', 'TA', 'Ex', 'None', 'Fa']


# we see a pattern of this varaible variable with SalePrice: None<Po<Fa<TA<Gd<Ex
BsmtQual_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_base['BsmtQual_Ordinal']=df_base.BsmtQual.map(BsmtQual_dict)
df_base['BsmtQual'].unique()
df_base['BsmtQual_Ordinal'].unique()
#array([4, 3, 5, 0, 2])

# dropping GarageCond from df_base
df_base=df_base.drop(['BsmtQual'],axis=1)
df_base.columns[df_base.columns=='BsmtQual']

# ----------------------------------------------------------------------
# BsmtCond
df_base['BsmtCond'].isnull().sum()
#37
# imputing values for GarageCond with None according to document
df_base['BsmtCond']=df_base['BsmtCond'].fillna('None')
df_base['BsmtCond'].isnull().sum()
#0

# unique values
df_base['BsmtCond'].unique()
#array(['TA', 'Gd', 'None', 'Fa', 'Po']

# we see a pattern of this varaible variable with SalePrice: None<Po<Fa<TA<Gd<Ex
BsmtCond_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_base['BsmtCond_Ordinal']=df_base.BsmtCond.map(BsmtCond_dict)
df_base['BsmtCond'].unique()
df_base['BsmtCond_Ordinal'].unique()
#array([3, 4, 0, 2, 1])

# dropping BsmtCond from df_base
df_base=df_base.drop(['BsmtCond'],axis=1)
df_base.columns[df_base.columns=='BsmtCond']

# ----------------------------------------------------------------------
# BsmtExposure
df_base['BsmtExposure'].isnull().sum()
#37
# imputing values for BsmtExposure with None according to document
df_base['BsmtExposure']=df_base['BsmtExposure'].fillna('None')
df_base['BsmtExposure'].isnull().sum()
#0

# unique values
df_base['BsmtExposure'].unique()
#array(['No', 'Gd', 'Mn', 'Av', 'None']


# we see a pattern of this varaible variable with SalePrice: None<No<Mn<Av<Gd
BsmtExposure_dict= {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4}
df_base['BsmtExposure_Ordinal']=df_base.BsmtExposure.map(BsmtExposure_dict)
df_base['BsmtExposure'].unique()
df_base['BsmtExposure_Ordinal'].unique()
#array([1, 4, 2, 3, 0])

# dropping BsmtCond from df_base
df_base=df_base.drop(['BsmtExposure'],axis=1)
df_base.columns[df_base.columns=='BsmtExposure']

# ----------------------------------------------------------------------
# BsmtFinType1
df_base['BsmtFinType1'].isnull().sum()
#37

# imputing values for BsmtFinType1 with None according to document
df_base['BsmtFinType1']=df_base['BsmtFinType1'].fillna('None')
df_base['BsmtFinType1'].isnull().sum()
#0

df_base[df_base['BsmtFinType1'].isnull()]


# unique values
df_base['BsmtFinType1'].unique()
#array(['Unf', 'BLQ', None, 'ALQ', 'Rec', 'LwQ', 'GLQ']



BsmtFinType1_dict= {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
df_base['BsmtFinType1_Ordinal']=df_base.BsmtFinType1.map(BsmtFinType1_dict)
df_base['BsmtFinType1'].unique()
df_base['BsmtFinType1_Ordinal'].unique()
#array([1, 4, 0, 5, 3, 2, 6])

# dropping BsmtFinType1 from df_base
df_base=df_base.drop(['BsmtFinType1'],axis=1)
df_base.columns[df_base.columns=='BsmtFinType1']

# ----------------------------------------------------------------------
# BsmtFinType2
df_base['BsmtFinType2'].isnull().sum()
#37

# imputing values for BsmtFinType2 with None according to document
df_base['BsmtFinType2']=df_base['BsmtFinType2'].fillna('None')
df_base['BsmtFinType2'].isnull().sum()
#0

df_base[df_base['BsmtFinType2'].isnull()]


# unique values
df_base['BsmtFinType2'].unique()
#array(['Unf', 'BLQ', None, 'ALQ', 'Rec', 'LwQ', 'GLQ']

BsmtFinType2_dict= {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
df_base['BsmtFinType2_Ordinal']=df_base.BsmtFinType2.map(BsmtFinType2_dict)
df_base['BsmtFinType2'].unique()
df_base['BsmtFinType2_Ordinal'].unique()
#array([1, 4, 0, 5, 3, 2, 6])

# dropping BsmtFinType2 from df_base
df_base=df_base.drop(['BsmtFinType2'],axis=1)
df_base.columns[df_base.columns=='BsmtFinType2']

# ---------------------------------------------------------------------------------------------------------------------------------------
# test data preparation - df_base_testdata_BsmtExposure

df_base_testdata_BsmtExposure.columns
#(['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
#       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
#       'BsmtHalfBath']
# ----------------------------------------------------------------------
#BsmtQual

# unique values
df_base_testdata_BsmtExposure['BsmtQual'].unique()

# we see a pattern of this varaible variable with SalePrice: None<Po<Fa<TA<Gd<Ex
BsmtQual_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_base_testdata_BsmtExposure['BsmtQual_Ordinal']=df_base_testdata_BsmtExposure.BsmtQual.map(BsmtQual_dict)
df_base_testdata_BsmtExposure['BsmtQual'].unique()
df_base_testdata_BsmtExposure['BsmtQual_Ordinal'].unique()
#array([4, 3, 5, 0, 2])

# dropping GarageCond from df_base_testdata_BsmtExposure
df_base_testdata_BsmtExposure=df_base_testdata_BsmtExposure.drop(['BsmtQual'],axis=1)
df_base_testdata_BsmtExposure.columns[df_base_testdata_BsmtExposure.columns=='BsmtQual']

# ----------------------------------------------------------------------
# BsmtCond

# unique values
df_base_testdata_BsmtExposure['BsmtCond'].unique()
#array(['TA', 'Gd', 'None', 'Fa', 'Po']

# we see a pattern of this varaible variable with SalePrice: None<Po<Fa<TA<Gd<Ex
BsmtCond_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_base_testdata_BsmtExposure['BsmtCond_Ordinal']=df_base_testdata_BsmtExposure.BsmtCond.map(BsmtCond_dict)
df_base_testdata_BsmtExposure['BsmtCond'].unique()
df_base_testdata_BsmtExposure['BsmtCond_Ordinal'].unique()
#array([3, 4, 0, 2, 1])

# dropping BsmtCond from df_base_testdata_BsmtExposure
df_base_testdata_BsmtExposure=df_base_testdata_BsmtExposure.drop(['BsmtCond'],axis=1)
df_base_testdata_BsmtExposure.columns[df_base_testdata_BsmtExposure.columns=='BsmtCond']

'''
# ----------------------------------------------------------------------
# BsmtExposure

# unique values
df_base_testdata_BsmtExposure['BsmtExposure'].unique()
#array(['No', 'Gd', 'Mn', 'Av', 'None']


# we see a pattern of this varaible variable with SalePrice: None<No<Mn<Av<Gd
BsmtExposure_dict= {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4}
df_base_testdata_BsmtExposure['BsmtExposure_Ordinal']=df_base_testdata_BsmtExposure.BsmtExposure.map(BsmtExposure_dict)
df_base_testdata_BsmtExposure['BsmtExposure'].unique()
df_base_testdata_BsmtExposure['BsmtExposure_Ordinal'].unique()
#array([1, 4, 2, 3, 0])

# dropping BsmtCond from df_base_testdata_BsmtExposure
df_base_testdata_BsmtExposure=df_base_testdata_BsmtExposure.drop(['BsmtExposure'],axis=1)
df_base_testdata_BsmtExposure.columns[df_base_testdata_BsmtExposure.columns=='BsmtExposure']
'''
# ----------------------------------------------------------------------
# BsmtFinType1

# unique values
df_base_testdata_BsmtExposure['BsmtFinType1'].unique()
#array(['Unf', 'BLQ', None, 'ALQ', 'Rec', 'LwQ', 'GLQ']



BsmtFinType1_dict= {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
df_base_testdata_BsmtExposure['BsmtFinType1_Ordinal']=df_base_testdata_BsmtExposure.BsmtFinType1.map(BsmtFinType1_dict)
df_base_testdata_BsmtExposure['BsmtFinType1'].unique()
df_base_testdata_BsmtExposure['BsmtFinType1_Ordinal'].unique()
#array([1, 4, 0, 5, 3, 2, 6])

# dropping BsmtFinType1 from df_base_testdata_BsmtExposure
df_base_testdata_BsmtExposure=df_base_testdata_BsmtExposure.drop(['BsmtFinType1'],axis=1)
df_base_testdata_BsmtExposure.columns[df_base_testdata_BsmtExposure.columns=='BsmtFinType1']

# ----------------------------------------------------------------------
# BsmtFinType2

# unique values
df_base_testdata_BsmtExposure['BsmtFinType2'].unique()
#array(['Unf', 'BLQ', None, 'ALQ', 'Rec', 'LwQ', 'GLQ']

BsmtFinType2_dict= {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
df_base_testdata_BsmtExposure['BsmtFinType2_Ordinal']=df_base_testdata_BsmtExposure.BsmtFinType2.map(BsmtFinType2_dict)
df_base_testdata_BsmtExposure['BsmtFinType2'].unique()
df_base_testdata_BsmtExposure['BsmtFinType2_Ordinal'].unique()
#array([1, 4, 0, 5, 3, 2, 6])

# dropping BsmtFinType2 from df_base_testdata_BsmtExposure
df_base_testdata_BsmtExposure=df_base_testdata_BsmtExposure.drop(['BsmtFinType2'],axis=1)
df_base_testdata_BsmtExposure.columns[df_base_testdata_BsmtExposure.columns=='BsmtFinType2']


# ----------------------------------------------------------------------------------------------------------------------------------------
# test data preparation - df_base_testdata_BsmtFinType2

df_base_testdata_BsmtFinType2.columns
#(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
#       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
#       'BsmtHalfBath']
# ----------------------------------------------------------------------
#BsmtQual

# unique values
df_base_testdata_BsmtFinType2['BsmtQual'].unique()

# we see a pattern of this varaible variable with SalePrice: None<Po<Fa<TA<Gd<Ex
BsmtQual_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_base_testdata_BsmtFinType2['BsmtQual_Ordinal']=df_base_testdata_BsmtFinType2.BsmtQual.map(BsmtQual_dict)
df_base_testdata_BsmtFinType2['BsmtQual'].unique()
df_base_testdata_BsmtFinType2['BsmtQual_Ordinal'].unique()
#array([4, 3, 5, 0, 2])

# dropping GarageCond from df_base_testdata_BsmtFinType2
df_base_testdata_BsmtFinType2=df_base_testdata_BsmtFinType2.drop(['BsmtQual'],axis=1)
df_base_testdata_BsmtFinType2.columns[df_base_testdata_BsmtFinType2.columns=='BsmtQual']

# ----------------------------------------------------------------------
# BsmtCond

# unique values
df_base_testdata_BsmtFinType2['BsmtCond'].unique()
#array(['TA', 'Gd', 'None', 'Fa', 'Po']

# we see a pattern of this varaible variable with SalePrice: None<Po<Fa<TA<Gd<Ex
BsmtCond_dict= {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
df_base_testdata_BsmtFinType2['BsmtCond_Ordinal']=df_base_testdata_BsmtFinType2.BsmtCond.map(BsmtCond_dict)
df_base_testdata_BsmtFinType2['BsmtCond'].unique()
df_base_testdata_BsmtFinType2['BsmtCond_Ordinal'].unique()
#array([3, 4, 0, 2, 1])

# dropping BsmtCond from df_base_testdata_BsmtFinType2
df_base_testdata_BsmtFinType2=df_base_testdata_BsmtFinType2.drop(['BsmtCond'],axis=1)
df_base_testdata_BsmtFinType2.columns[df_base_testdata_BsmtFinType2.columns=='BsmtCond']


# ----------------------------------------------------------------------
# BsmtExposure

# unique values
df_base_testdata_BsmtFinType2['BsmtExposure'].unique()
#array(['No', 'Gd', 'Mn', 'Av', 'None']


# we see a pattern of this varaible variable with SalePrice: None<No<Mn<Av<Gd
BsmtExposure_dict= {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4}
df_base_testdata_BsmtFinType2['BsmtExposure_Ordinal']=df_base_testdata_BsmtFinType2.BsmtExposure.map(BsmtExposure_dict)
df_base_testdata_BsmtFinType2['BsmtExposure'].unique()
df_base_testdata_BsmtFinType2['BsmtExposure_Ordinal'].unique()
#array([1, 4, 2, 3, 0])

# dropping BsmtCond from df_base_testdata_BsmtFinType2
df_base_testdata_BsmtFinType2=df_base_testdata_BsmtFinType2.drop(['BsmtExposure'],axis=1)
df_base_testdata_BsmtFinType2.columns[df_base_testdata_BsmtFinType2.columns=='BsmtExposure']

# ----------------------------------------------------------------------
# BsmtFinType1

# unique values
df_base_testdata_BsmtFinType2['BsmtFinType1'].unique()
#array(['Unf', 'BLQ', None, 'ALQ', 'Rec', 'LwQ', 'GLQ']

BsmtFinType1_dict= {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
df_base_testdata_BsmtFinType2['BsmtFinType1_Ordinal']=df_base_testdata_BsmtFinType2.BsmtFinType1.map(BsmtFinType1_dict)
df_base_testdata_BsmtFinType2['BsmtFinType1'].unique()
df_base_testdata_BsmtFinType2['BsmtFinType1_Ordinal'].unique()
#array([1, 4, 0, 5, 3, 2, 6])

# dropping BsmtFinType1 from df_base_testdata_BsmtFinType2
df_base_testdata_BsmtFinType2=df_base_testdata_BsmtFinType2.drop(['BsmtFinType1'],axis=1)
df_base_testdata_BsmtFinType2.columns[df_base_testdata_BsmtFinType2.columns=='BsmtFinType1']
'''
# ----------------------------------------------------------------------
# BsmtFinType2

# unique values
df_base_testdata_BsmtFinType2['BsmtFinType2'].unique()
#array(['Unf', 'BLQ', None, 'ALQ', 'Rec', 'LwQ', 'GLQ']

BsmtFinType2_dict= {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
df_base_testdata_BsmtFinType2['BsmtFinType2_Ordinal']=df_base_testdata_BsmtFinType2.BsmtFinType2.map(BsmtFinType2_dict)
df_base_testdata_BsmtFinType2['BsmtFinType2'].unique()
df_base_testdata_BsmtFinType2['BsmtFinType2_Ordinal'].unique()
#array([1, 4, 0, 5, 3, 2, 6])

# dropping BsmtFinType2 from df_base_testdata_BsmtFinType2
df_base_testdata_BsmtFinType2=df_base_testdata_BsmtFinType2.drop(['BsmtFinType2'],axis=1)
df_base_testdata_BsmtFinType2.columns[df_base_testdata_BsmtFinType2.columns=='BsmtFinType2']
'''

# ----------------------------------------------------------------------------------------------------------------------------------------

for i in range(len(df_base.columns)):
    print('column = {c} and missing_value = {m}'.format(c=df_base.columns[i],m=df_base[df_base.columns[i]].isnull().sum()))

# no missing now

# print unique value
for i in range(len(df_base.columns)):
    print('column = {c} and unique_val = {u}'.format(c=df_base.columns[i],u=df_base[df_base.columns[i]].unique()))



#-----------------------------------------------------------------------------------------------------------------------------------------
# predict BsmtExposure_Ordinal
df_base2=copy.deepcopy(df_base)
df_base2.columns
y_train=df_base2['BsmtExposure_Ordinal']
X_train=df_base2.drop(['BsmtExposure_Ordinal'],axis=1)
y_train[1]
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
X_train['BsmtFinType2_Ordinal'].unique()
y_train.unique()
predictions = dtree.predict(df_base_testdata_BsmtExposure)
predictions
# array([1]) - no exposer

#--------------------------------------------------------------------
# predict BsmtFinType2_Ordinal
df_base2=copy.deepcopy(df_base)
df_base2.columns
y_train=df_base2['BsmtFinType2_Ordinal']
X_train=df_base2.drop(['BsmtFinType2_Ordinal'],axis=1)
y_train[1]
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
#X_train['BsmtFinType2_Ordinal'].unique()
y_train.unique()
predictions = dtree.predict(df_base_testdata_BsmtFinType2)
predictions
#array([5]) - ALQ : Average Living Quarters





















