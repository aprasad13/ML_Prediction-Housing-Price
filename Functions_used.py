# Function Used:

#-------------------------------------------------------------------------------------------------------------
# train function for distribution within a column
        
def train_distribution_column(column_name, dataFrame_name, target_var):
    G1_uniq=dataFrame_name[column_name].unique()
    len(dataFrame_name[column_name])
    # 1460
    sns.countplot(x=column_name,data=dataFrame_name)
    plt.show()
    sns.barplot(x=column_name,y=target_var,data=dataFrame_name)
    plt.show()

    for i in range(len(G1_uniq)):
        # print(i)
        print('{n} : Count = {c} and Percentage = {p}'.format(n=G1_uniq[i],c=dataFrame_name[dataFrame_name[column_name]==G1_uniq[i]].shape[0],
                 p=100*dataFrame_name[dataFrame_name[column_name]==G1_uniq[i]].shape[0]/len(dataFrame_name[column_name])))

    
    plt.figure(figsize=(10,5))
    chart1 = sns.boxplot(data=dataFrame_name,x=column_name,y=target_var)
    chart1.set_xticklabels(chart1.get_xticklabels(), rotation=45)
    plt.show()
    

#-------------------------------------------------------------------------------------------------------------
# test function for distribution within a column
       
def test_distribution_column(column_name, dataFrame_name):
    G1_uniq=dataFrame_name[column_name].unique()
    len(dataFrame_name[column_name])
    # 1460
    #sns.countplot(x=column_name,data=dataFrame_name)
    #plt.show()
    #sns.barplot(x=column_name,y=target_var,data=dataFrame_name)
    #plt.show()

    for i in range(len(G1_uniq)):
        # print(i)
        print('{n} : Count = {c} and Percentage = {p}'.format(n=G1_uniq[i],c=dataFrame_name[dataFrame_name[column_name]==G1_uniq[i]].shape[0],
                 p=100*dataFrame_name[dataFrame_name[column_name]==G1_uniq[i]].shape[0]/len(dataFrame_name[column_name])))

    
    #plt.figure(figsize=(10,5))
    #chart1 = sns.boxplot(data=dataFrame_name,x=column_name,y=target_var)
    #chart1.set_xticklabels(chart1.get_xticklabels(), rotation=45)
    #plt.show()
    

#-------------------------------------------------------------------------------------------------------------
# finding coorelation between numeric variables

def correlation_num_variable(dataFrame_name,n,target_var):
    num_col=dataFrame_name._get_numeric_data()
    cor_numVar=num_col.corr()
    cor_numVar_sorted=cor_numVar[target_var].sort_values(ascending=False)
    
    # considering variables which has correlation >n with SalePrice
    name=cor_numVar_sorted[cor_numVar_sorted>n].index
    final_cor_num=cor_numVar.loc[name,name]
    final_cor_num
    
    fig = plt.figure(figsize=(10,6), dpi=100)
    sns.heatmap(final_cor_num, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=3, linecolor='black')
    plt.title('Correlation Plot - Numerical Variables')
    plt.tight_layout


#-------------------------------------------------------------------------------------------------------------
# function to find ordinal and numeric variables
def separate_ordinal_numric_var(dataFrame_name,num_uniq_val):
    num_ordinal_col=dataFrame_name._get_numeric_data()
    num_ordinal_col_name=num_ordinal_col.columns
    
    num_col_name=[]
    ordinal_col_name=[]
    col_name=[]
    for i in range(len(num_ordinal_col_name)):
        if len(dataFrame_name[num_ordinal_col_name[i]].unique())>num_uniq_val:
            num_col_name.append(num_ordinal_col_name[i])
        else:
            ordinal_col_name.append(num_ordinal_col_name[i])
    col_name=[num_col_name,ordinal_col_name]
    return (col_name)



#-------------------------------------------------------------------------------------------------------------
# find missing values in columns

def missing_value(dataFrame):
    null_all=dataFrame.isnull().sum()
    
    null_all_desc=null_all[null_all>0].sort_values(ascending = False) 
    
    null_all_desc=pd.DataFrame(null_all_desc)
    null_all_desc = null_all_desc.reset_index()
    null_all_desc.columns = ['column', 'missing_count']
    return (null_all_desc)
    



#-------------------------------------------------------------------------------------------------------------
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




















