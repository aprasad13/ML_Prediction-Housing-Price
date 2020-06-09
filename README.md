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
I have tried to visualize all the important variables. This can help one to quickly get some insights about the data and can also help a person to decide a roadmap to proceed further. Click to access the [Visalization File](https://github.com/aprasad13/Prediction-Housing-Price/blob/master/Train_House_Visualization_important_variables.ipynb)

### Splitting data into Training and Validation set
Stratified sampling has been used to split the data bewteen training and validation set. The most highly correlated categorical variable (OverallQual) with the target has been used as a basis for sampling.

### Data Preparation
All the columns have been segmented in some categories wherever it was possible.

*Missing Value Treatment*

- Categorical Variable (Nominal and Ordinal)
> Wherever there was missing value, first it was checked if some other related variables/features have any value. If there are some other related variables, then the missing value was replaced with the help of other related variables using an algorithm like Random Forest Classifier. If there is no related variable, then Data Description was checked if this feature can contain None value. If yes, then missing data was replaced by None but if it cannot contain None then missing data was replaced with most appropriate value (Mode or any other suitable value this depends on the feature and the distribution within the varibale.

-	Numerical Variable
> There were very fewer missing columns which were numerical. Whenever there was any numerical column with missing data then it was checked if the other related variables have any value. If yes, then it was replaced with the help of related variables using an algorithm like Random Forest Regressor. If no, then it was replaced with most appropriate value (mean/median/mode, this depends on the feature and its distribution within the variable.

*Categorical Variable (Nominal and Ordinal) treatment* 
> After treating the missing data, Ordinal variables were encoded based on their levels into a suitable number and it was changed to Categorical data type. In case of Nominal, distribution of the categorical level within the variable was checked. It was checked if all the levels that can be possible based on Data Description are present. If all the levels are not present, then it was checked if the description contained ‘Other’ category. If yes, then levels with least distribution and levels which were not present were put to ‘Other’ category. If no, ‘Other’ category was created and then levels with least distribution and levels which were not present were put to ‘Other’ category.
