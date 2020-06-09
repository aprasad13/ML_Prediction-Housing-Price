# Prediction-Housing-Price
### Introduction

This project is aimed to develop a model that can efficiently predict the Sales Price of a house based on 79 explanatory variables describing (almost) every aspect of a residential home.

As this is a Regression problem. The project shows performance of several Machine Learning Regression Models such as:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Elastic Net
- Random Forest
- Gradient Boost

### Dataset Description

There are 2 separate files for training and testing the data. Training dataset contains 1460 records and 79 explanatory variables describing (almost) every aspect of residential homes and Sales Price as target. There were 36 numerical variables,  43 categorical variable (nominal and ordinal) and a numeric target variable (SalePrice). The same goes with testing dataset except it does not contain the target variable. This dataset contains information of the residential houses in Ames and Iowa.
There is also a file that contains the column description with each and every details. 

Click to access the below files: 
- [Data Description](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/data_description.txt)
- [House_price_train](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/House_price_train.csv)
- [House_price_test](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/House_price_test.csv)

### Data Visualization
I have tried to visualize all the important variables. This can help one to quickly get some insights about the data and can also help a person to decide a roadmap to proceed further. Click to access the [Visualization File](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/Train_House_Visualization_important_variables.ipynb)

### Splitting data into Training and Validation set
Stratified sampling has been used to split the data bewteen training and validation set. The most highly correlated categorical variable (OverallQual) with the target has been used as a basis for sampling.

### Data Preparation
All the columns have been segmented in some categories wherever it was possible.

*Outlier Treatment*
> Whenever there was an outlier in a variable, we tried to print all corresponding data in its related variables. Suppose variable X shows category1. then we check the distribution of column where we are checking outlier based on category1 in column X, we would check if the value that we are calling outlier is occuring most of the time or not. If yes then its not an outlier and if no then its an outlier and we need further investigation if we need to drop it or not. In this project there were outliers in column GrLivArea and the value was not occuring most of the time. So it was decided to drop them.

*Missing Value Treatment*

- Categorical Variable (Nominal and Ordinal)
> Wherever there was missing value, first it was checked if some other related variables/features have any value. If there are some other related variables, then the missing value was replaced with the help of other related variables using an algorithm like Random Forest Classifier. If there is no related variable, then Data Description was checked if this feature can contain None value. If yes, then missing data was replaced by None but if it cannot contain None then missing data was replaced with most appropriate value (Mode or any other suitable value this depends on the feature and the distribution within the varibale.

-	Numerical Variable
> There were very fewer missing columns which were numerical. Whenever there was any numerical column with missing data then it was checked if the other related variables have any value. If yes, then it was replaced with the help of related variables using an algorithm like Random Forest Regressor. If no, then it was replaced with most appropriate value (mean/median/mode, this depends on the feature and its distribution within the variable.

*Categorical Variable (Nominal and Ordinal) treatment* 
> After treating the missing data, Ordinal variables were encoded based on their levels into a suitable number and it was changed to Categorical data type. In case of Nominal, distribution of the categorical level within the variable was checked. It was checked if all the levels that can be possible based on Data Description are present. If all the levels are not present, then it was checked if the description contained ‘Other’ category. If yes, then levels with least distribution and levels which were not present were put to ‘Other’ category. If no, ‘Other’ category was created and then levels with least distribution and levels which were not present were put to ‘Other’ category.

Click on the below code file to access the Data Preparation Code for further details:

- [Training Splitted Data Preparation](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/Train_House_V2.py)
- [Validation Data Preparation](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/Validation_House.py)
- [Training All Data Preparation](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/Train_All_House.py)
- [Testing Data Preparation](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/Test_House_V1.py)


### Feature Engineering

- There were 4 bathroom variables. Individually, these variables were not very important. However, they were added up into one predictor named ‘Total number of Bathrooms’, this predictor became a strong one. The new feature had correlation of 0.631 with the target variable.

- Time Related Features

> There are 3 features related to year such as YearBuilt (Original construction date), YearRemodAdd (Remodel date ,same as construction date if no remodeling or additions) and YrSold (Year Sold (YYYY)). An age column (age = YrSold - YearRemodAdd) is created. YrSold is kept as it is, to capture the effect of the year when it is being sold. However, as parts of old constructions will always remain and only parts of the house might have been renovated, so, a column=Remodeled (value=Yes/No) is created. This should be seen as some sort of penalty parameter that indicates that if the Age is based on a remodeling date, it is probably worth less than houses that were built from scratch in that same year. A new column ‘IsNew’ is created which indicates ‘yes’ if the YearBuilt = YrSold and otherwise 'No'. After creating these 3 new columns/features, YearBuilt and YearRemodAdd were removed.

- Neighbourhood variable was binned. A bar chart of Neighbourhood with SalePrice wa plotted and it was noted that both the median and mean of Saleprice agree on 3 neighborhoods with substantially higher saleprices and 3 neighborhoods with low SalePrice. So, they were being put into high and low category respectively. Other neighborhoods were less clear, and Since they could have been ‘overbinned’, they were left as it is.

- As the total living space generally is very important when people buy houses, a predictor ‘TotalSqFeet’ was added that adds up the living space above (GrLivArea) and below ground (TotalBsmtSF). Also TotalBsmtSF was removed to avoid multicollinearity as GrLivArea had less correlation with target as compared to TotalBsmtSF.

- There were 5 variable related to Porch such as WoodDeckSF (Wood deck area in square feet), OpenPorchSF (Open porch area in square feet), EnclosedPorch (Enclosed porch area in square feet), 3SsnPorch (Three season porch area in square feet), ScreenPorch (Screen porch area in square feet). These all Porch variables were consolidated to one varaible called 'TotalPorchSF'.

### Preparing data for modeling

- Dropping Correlated Vraibles
> To avoid multicollinearity, two variables with high correlation were dropped. To find these correlated pairs, 
correlations matrix has been used. For instance: GarageCars and GarageArea had a correlation of 0.89. Of those two, the variable with the lowest correlation with SalePrice was dropped (which is GarageArea with a SalePrice correlation of 0.62. GarageCars has a SalePrice correlation of 0.64).

- Skewness 
> - Skewness is a measure of the symmetry in a distribution. A symmetrical dataset will have a skewness equal to 0. So, a normal distribution will have a skewness of 0. Skewness essentially measures the relative size of the two tails. As a rule of thumb, skewness should be between -1 and 1. In this range, data are considered fairly symmetrical
> - In order to fix the skewness, log was applied for all numeric predictors with an absolute skew greater than 0.8 (actually: log+1, to avoid division by zero issues).

- Data Normalization and Data Standardization
> - Data Normalization - Normalization refers to rescaling real valued numeric attributes into the range 0 and 1. It is useful to scale the input attributes for a model that relies on the magnitude of values, such as distance measures used in k-nearest neighbors and in the preparation of coefficients in regression
> - Data Standardization - Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance). It is useful to standardize attributes for a model that relies on the distribution of attributes such as Gaussian processes. 
> - It was decided to go with Standardization

- One hot encoding the categorical variables
> After this there are 21 categorical columns and in total of 74 columns including nominal. In total there were 225 independent variables.


Click on the following code file to see more details:
- [Training All & Splitted Feature Engineering](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/Train_House_AllAndSplitted_Feature_Engineering_V3.py)
- [Validation Feature Engineering](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/Validation_House_Feature_Engineering.py)
- [Testing Feature Engineering](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/Test_House_Feature_Engineering_V1.py)

Click on the following file generated after feature engineering to see more details:
- [Training Splitted Feature Engineering](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/df_train_Splitted_V1.csv)
- [Validation Feature Engineering](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/df_validation_FeatEngg_9.csv)
- [Training All Feature Engineering](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/df_train_ALL_FeatEngg_8.csv)
- [Testing Feature Engineering](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/df_test_FeatEngg_9_V2.csv)

### Model Performance

| Model               | Stochastic | PCA      | Training RMSE             | Validation RMSE       |
| -------------       | ---------- | ----     |    --------               | -------------         |
| Linear Regression   | No         | No       | 878549133.2543052         | 23736807855.154545    |
| Linear Regression   | Yes        | No       | NA                        | 0.903577214121022     |
| Linear Regression   | No         | Yes      | 0.11710506977706689       | 0.21866147149548182   |
| Linear Regression   | Yes        | Yes      | NA                        | 0.23171273738088263   |
| Ridge Regression    | No         | No       | 0.11382586579804059       | 0.15502651269939982   |
| Ridge Regression    | Yes        | No       | 0.2555794950868803        | 0.28638961850775196   |  
| Ridge Regression    | No         | Yes      | 0.11673464401746662       | 0.2159260272536825    |
| Ridge Regression    | Yes        | Yes      | 0.2989778654057608        | 0.3184118909709933    |
| Lasso Regression    | No         | No       | 0.11345006473735222       | 0.14954863164942334   |
| Lasso Regression    | Yes        | No       | 0.49130498580778337       | 0.8093628575728181    |
| Lasso Regression    | No         | Yes      | 0.11676191996399303       | 0.21583738578178066   |
| Lasso Regression    | Yes        | Yes      | 0.12214815903906791       | 0.22022969724341554   |
| Elastic Net         | No         | No       | 0.11347339693559097       | 0.14888627346166744   |
| Elastic Net         | No         | Yes      | 0.11675505949345945       | 0.21613504090408497   |
| Random Forest       | No         | No       | 0.12903180245135284       | 0.14272532051839343   |
| Random Forest       | No         | Yes      | 0.1418293940827534        | 0.1890364900292125    |
| Gradient Boost      | No         | No       | 0.11996227721676053       | 0.14045639753683603   |
| Gradient Boost      | No         | Yes      | 0.14648460664021323       | 0.19501973484749527   |
