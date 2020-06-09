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

There are 2 separate files for training and testing the data. Training dataset contains 1469 records and 79 explanatory variables describing (almost) every aspect of residential homes and Sales Price as target. The same goes with testing dataset except it does not contain the target variable. This dataset contains information of the residential houses in Ames and Iowa.
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
- Training All Data Preparation
- Testing Data Preparation


### Feature Engineering

- There were 4 bathroom variables. Individually, these variables were not very important. However, they were added up into one predictor named ‘Total number of Bathrooms’, this predictor became a strong one. The new feature had correlation of 0.631 with the target variable.

- Time Related Features

> There are 3 features related to year such as YearBuilt (Original construction date), YearRemodAdd (Remodel date ,same as construction date if no remodeling or additions) and YrSold (Year Sold (YYYY)). An age column (age = YrSold - YearRemodAdd) is created. YrSold is kept as it is, to capture the effect of the year when it is being sold. However, as parts of old constructions will always remain and only parts of the house might have been renovated, so, a column=Remodeled (value=Yes/No) is created. This should be seen as some sort of penalty parameter that indicates that if the Age is based on a remodeling date, it is probably worth less than houses that were built from scratch in that same year. A new column ‘IsNew’ is created which indicates ‘yes’ if the YearBuilt = YrSold and otherwise 'No'. After creating these 3 new columns/features, YearBuilt and YearRemodAdd were removed.

- Neighbourhood variable was binned. A bar chart of Neighbourhood with SalePrice wa plotted and it was noted that both the median and mean of Saleprice agree on 3 neighborhoods with substantially higher saleprices and 3 neighborhoods with low SalePrice. So, they were being put into high and low category respectively. Other neighborhoods were less clear, and Since they could have been ‘overbinned’, they were left as it is.

- As the total living space generally is very important when people buy houses, a predictor ‘TotalSqFeet’ was added that adds up the living space above (GrLivArea) and below ground (TotalBsmtSF). Also TotalBsmtSF was removed to avoid multicollinearity as GrLivArea had less correlation with target as compared to TotalBsmtSF.

- There were 5 variable related to Porch such as WoodDeckSF (Wood deck area in square feet), OpenPorchSF (Open porch area in square feet), EnclosedPorch (Enclosed porch area in square feet), 3SsnPorch (Three season porch area in square feet), ScreenPorch (Screen porch area in square feet). These all Porch variables were consolidated to one varaible called 'TotalPorchSF'.



