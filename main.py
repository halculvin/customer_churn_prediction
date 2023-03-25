# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:05:02 2023

@author: Harry McNinson

@Title: Predict the customer churn of a telecom company and find out the key drivers that lead to churn
(Customer churn, is the percentage of customers who stop doing business with an entity)
"""

# Import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, average_precision_score, f1_score, confusion_matrix, roc_auc_score, auc, accuracy_score, log_loss, roc_curve, precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Creating a function that does basic inspection of our dataset


def get_basic_stats(dfname):
    print("Shape of dataframe is " + str(dfname.shape), end="\n\n")
    print("First 5 rows of the dataframe", end="\n\n")
    print(dfname.head(), end="\n\n\n")
    print("Below are datatypes of columns in DF", end="\n\n")
    print(dfname.dtypes.sort_values(),  end="\n\n\n")
    print("Below are missing values in each column", end="\n\n")
    print(dfname.isna().sum().sort_values(),  end="\n\n\n")
    print("Below are the number of unique values taken by a column", end="\n\n")
    print(dfname.nunique().sort_values())


# A function to change categorical variables to binary
def cat_to_binary(df, varname, des):
    df[varname + '_num'] = df[varname].apply(lambda x: 1 if x == 'yes' else 0)
    print(des + " : Checking " + varname, end="\n\n")
    print(df.groupby([varname + '_num', varname]).size(), end="\n\n\n")
    return df


# A function to return feature importance
def get_FI(modelname, dfname):
    importance_list = pd.DataFrame(
        modelname.feature_importances_, columns=['importance'])
    varnames_list = pd.DataFrame(dfname.columns.tolist(), columns=['feature'])
    feature_importance = pd.concat([varnames_list, importance_list], axis=1)
    feature_importance = feature_importance.reindex(varnames_list.index)
    feature_importance = feature_importance.sort_values(
        by=['importance'], ascending=False)
    feature_importance['cum_importance'] = feature_importance['importance'].cumsum()
    return feature_importance


# read the required data files
trainer_df = pd.read_csv('data/telecom_train.csv')
tester_df = pd.read_csv('data/telecom_test.csv')

##### EDA Starts here ######
get_basic_stats(trainer_df)

get_basic_stats(tester_df)


# Create copies of both data frames
trainer = trainer_df.copy()
tester = tester_df.copy()

## Data Cleansing starts here#######################################################
# Remove unwanted columns
trainer = trainer.drop('Unnamed: 0', axis=1)
tester = tester.drop('Unnamed: 0', axis=1)

# Convert Categorical to numerical variables
convert_list = ['churn', 'international_plan', 'voice_mail_plan']

for varname in convert_list:
    trainer = cat_to_binary(trainer, varname, 'Train')
    tester = cat_to_binary(tester, varname, 'Test')


# Univariate analysis of non continuous variables

# Area Code
area_code = trainer["area_code"].value_counts()

plt.figure(figsize=(10, 5))
sns.barplot(x=area_code.index, y=area_code.values, alpha=0.8)
plt.title('Area Code Distribution')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Area Code', fontsize=12)
plt.show()


# State
state = trainer["state"].value_counts()

plt.figure(figsize=(20, 10))
sns.barplot(x=state.index, y=state.values, alpha=0.8)
plt.title('State Distribution')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('State', fontsize=12)
plt.show()


trainer.mean(numeric_only=True)

# Visualizing the churn variable
topie = trainer["churn"].value_counts(sort=True)

colorss = ["blue", "orange"]
plt.figure(figsize=(7, 7))
plt.pie(topie, labels=topie.index.values, explode=[
        0, 0.2], colors=colorss, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Percentage Churn in Training Data')
plt.show()


# Univariate Analysis - Continuous Variables

# Select continuous variables
continuous_vars = trainer.select_dtypes([np.number]).columns.tolist()
continuous_vars = [x for x in continuous_vars if '_num' not in x]

calls_vars = [x for x in continuous_vars if 'calls' in x]


trainer.boxplot(column=calls_vars, figsize=(20, 10))


df = pd.DataFrame(data=trainer, columns=calls_vars)
plt.figure(figsize=(20, 10))
sns.boxplot(x="variable", y="value", data=pd.melt(df))

plt.show()


# Univariate analysis of continuous variables
type_of_vars = ['intl', 'customer', 'minutes', 'calls', 'charge']
remaining_list = trainer.columns
for vartype in type_of_vars:
    temp_list = [x for x in remaining_list if vartype in x]
    remaining_list = list(set(remaining_list).difference(set(temp_list)))
    trainer.boxplot(column=temp_list, figsize=(20, 10))
    plt.title('Boxplot for ' + vartype + ' variables')
    plt.show()


# Bivariate analysis

# Drop the target variable
X = trainer.drop('churn_num', axis=1)

# Check correlations
all_corr = X.corr(numeric_only=True).unstack().reset_index()
corr_table = all_corr[all_corr['level_0'] > all_corr['level_1']]
corr_table.columns = ['var1', 'var2', 'corr_value']
corr_table['corr_abs'] = corr_table['corr_value'].abs()
corr_table = corr_table.sort_values(by=['corr_abs'], ascending=False)


# Using heatmap to visualize correlation
plt.figure(figsize=(12, 12))
sns.heatmap(X.corr(numeric_only=True))

# Using pairplot for bivariate analysis
sns.pairplot(X)


## -- Feature Creation -- ##

# Creating a charge per minute variable in both dataframes
# Intuitively, we expect customer with high value of this variable to have a higher churn rate
charge_vars = [x for x in trainer.columns if 'charge' in x]
minutes_vars = [x for x in trainer.columns if 'minutes' in x]

# Function to create charge per minute columns for both train and test data


def create_cpm(df):
    df['total_charges'] = 0
    df['total_minutes'] = 0
    for indexer in range(0, len(charge_vars)):
        df['total_charges'] += df[charge_vars[indexer]]
        df['total_minutes'] += df[minutes_vars[indexer]]
    df['charge_per_minute'] = np.where(
        df['total_minutes'] > 0, df['total_charges']/df['total_minutes'], 0)
    df.drop(['total_minutes', 'total_charges'], axis=1, inplace=True)
    print(df['charge_per_minute'].describe())
    return df


trainer = create_cpm(trainer)
tester = create_cpm(tester)

trainer.boxplot(column='charge_per_minute', figsize=(10, 10))


# Probability Distribution Function

# Plotting PDF of all variables based on Churn
def create_pdf(df, varname):
    plt.figure(figsize=(10, 5))
    plt.hist(list(df[df['churn_num'] == 0][varname]), bins=50,
             label='non-churned', density=True, color='g', alpha=0.8)
    plt.hist(list(df[df['churn_num'] == 1][varname]), bins=50,
             label='churned', density=True, color='r', alpha=0.8)
    plt.legend(loc='upper right')
    plt.xlabel(varname)
    plt.ylabel('Probability Distribution Function')
    plt.show()


for varname in trainer.columns:
    create_pdf(trainer, varname)


## Preparing Data for Modelling #######################################

# Drop highly correlated variables
drop_after_corr = ['total_day_charge', 'total_eve_charge',
                   'total_night_charge', 'total_intl_charge', 'voice_mail_plan_num']
trainer2 = trainer.drop(drop_after_corr, axis=1)
print(trainer2.shape)
tester2 = tester.drop(drop_after_corr, axis=1)


# Transform categorical columns using one hot encoding
cat_columns = ['state', 'area_code']
trainer2 = pd.concat([trainer2, pd.get_dummies(
    trainer2[cat_columns], drop_first=True)], axis=1)
print(trainer2.shape)
tester2 = pd.concat([tester2, pd.get_dummies(
    tester2[cat_columns], drop_first=True)], axis=1)


# Drop the categorical variables
trainer2 = trainer2.drop(convert_list + cat_columns, axis=1)
print(trainer2.shape)
tester2 = tester2.drop(convert_list + cat_columns, axis=1)

# Drop the target variable
X_train = trainer2.drop('churn_num', axis=1)
X_test = tester2.drop('churn_num', axis=1)

Y_train = trainer2['churn_num']
Y_test = tester2['churn_num']


## Modelling #################################################

# Logistic Regression with hyper parameter tuning
lr = LogisticRegression(random_state=42, solver='liblinear')
param_gridd = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 2, 3, 5]}
CV_lr = GridSearchCV(estimator=lr, param_grid=param_gridd,
                     cv=5)  # do this with 5 folds
CV_lr.fit(X_train, Y_train)
lr_best = CV_lr.best_estimator_
print(lr_best)

# Score test dataset
test_score_lr = lr_best.predict_proba(X_test)[:, 1]
pd.Series(test_score_lr).describe()

# Gradient Boosting with hyper-parameter tuning
gbr = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 500], 'max_features': [1.0], 'learning_rate': [0.01, 0.05, 0.1, 0.2]
}
CV_gbr = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5)
CV_gbr.fit(X_train, Y_train)
gbr_best = CV_gbr.best_estimator_
print(gbr_best)

test_score_gbm = gbr_best.predict_proba(X_test)[:, 1]
pd.Series(test_score_gbm).describe()

## Performance Comparison of Models #########################################


# Area Under ROC and PR curves for LR model
roc_auc = (roc_auc_score(Y_test, test_score_lr, average='macro'))
avg_pre = average_precision_score(Y_test, test_score_lr)
print("Area Under ROC and PR curves for LR model", end="\n\n")
print("roc auc score:", roc_auc)
print("avg precision score:", avg_pre)


# Area Under ROC and PR curves for GBM model
roc_auc_gbm = (roc_auc_score(Y_test, test_score_gbm, average='macro'))
avg_pre_gbm = average_precision_score(Y_test, test_score_gbm)
print("Area Under ROC and PR curves for GBM model", end="\n\n")
print("roc auc score:", roc_auc_gbm)
print("avg precision score:", avg_pre_gbm)

# Plot the ROC Curves
plt.figure(figsize=(10, 5))
fpr_gbm, tpr_gbm, _ = roc_curve(Y_test, test_score_gbm)
plt.plot(fpr_gbm, tpr_gbm, label='GBM')
fpr_lr, tpr_lr, _ = roc_curve(Y_test, test_score_lr)
plt.plot(fpr_lr, tpr_lr, label='LR')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

# PLot precision recall curve
plt.figure(figsize=(10, 5))
precision_gbm, recall_gbm, _ = precision_recall_curve(Y_test, test_score_gbm)
plt.plot(recall_gbm, precision_gbm, label='GBM')
precision_lr, recall_lr, _ = precision_recall_curve(Y_test, test_score_lr)
plt.plot(recall_lr, precision_lr, label='LR')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve')
plt.legend()


# Look at the confusion matrix for GB Model
cm = confusion_matrix(Y_test, (test_score_gbm >= 0.5))
ax = plt.subplot()
sns.heatmap(cm, annot=True,  ax=ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix for GB model')
ax.xaxis.set_ticklabels(['retained', 'churned'])
ax.yaxis.set_ticklabels(['retained', 'churned'])


# Look at the confusion matrix for LR Model
cm_lr = confusion_matrix(Y_test, (test_score_lr >= 0.5))
ax = plt.subplot()
sns.heatmap(cm_lr, annot=True,  ax=ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix for LR model')
ax.xaxis.set_ticklabels(['retained', 'churned'])
ax.yaxis.set_ticklabels(['retained', 'churned'])

# Calculate Accuracy Score
print(accuracy_score(Y_test, (test_score_lr >= 0.5), normalize=True))
print(accuracy_score(Y_test, (test_score_gbm >= 0.5), normalize=True))


# Get top 10% of customers who are likely to churn for followup
get_top10 = pd.concat(
    [pd.Series(test_score_gbm, name='model_score'), Y_test], axis=1)
get_top10 = get_top10.sort_values(by=['model_score'], ascending=False)
get_top10.head()
get_top10['rownum'] = np.arange(len(get_top10))
get_top10[get_top10['rownum'] <= Y_test.shape[0] /
          10]['churn_num'].value_counts()

# Call the get_FI function to return feature importance table
fi_importance_table = get_FI(gbr_best, X_train)


# Drop state and area code features from the data since they do not contribute much towards
# the predicting power of the model
state_vars = [x for x in X_train.columns if 'state' in x]
area_vars = [x for x in X_train.columns if 'area' in x]
rfe_vars = state_vars + area_vars
print(len(rfe_vars))
X_train_rfe = X_train.drop(rfe_vars, axis=1)
X_test_rfe = X_test.drop(rfe_vars, axis=1)
X_test_rfe.shape


# Gradient Boosting on reduced feature set
gbr = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 500], 'max_features': [1.0], 'learning_rate': [0.01, 0.05, 0.1, 0.2]
}
CV_gbr = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5)
CV_gbr.fit(X_train_rfe, Y_train)
gbr_best_rfe = CV_gbr.best_estimator_
print(gbr_best_rfe)

# Score the GB rfe model and compare test score with previous model
test_score_rfe = gbr_best_rfe.predict_proba(X_test_rfe)[:, 1]
print(pd.Series(test_score_rfe).describe())
roc_auc_gbm = (roc_auc_score(Y_test, test_score_rfe, average='macro'))
avg_pre_gbm = average_precision_score(Y_test, test_score_rfe)

print("Area Under ROC and PR curves for GB RFE model", end="\n\n")
print("roc auc score:", roc_auc_gbm)
print("avg precision score:", avg_pre_gbm)


## Take a look at the feature importance of the reduced GB model
FI = get_FI(gbr_best_rfe, X_train_rfe)
FI.index = np.arange(1, len(FI) + 1)
print(FI)

## Plot the key drivers of churn
vals = list(FI['importance'])
plt.figure(figsize=(10, 8))
plt.barh(FI['feature'], FI['importance'])
plt.title('Importance of different variables')
plt.gca().xaxis.grid(linestyle=':')


## Model Implementation
model_columns = list(X_train_rfe.columns)
pickle.dump(gbr_best_rfe, open('model/model.pkl', 'wb'))
pickle.dump(model_columns, open('model/model_columns.pkl', 'wb'))
