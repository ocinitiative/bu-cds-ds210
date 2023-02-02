# DS 210 Homework 1 Question 2

# Execute a simple data pipeline that involves the following steps:
# 1. Basic data validation (i.e., make sure no relevant attributes are missing) and—if needed—data cleansing.
# 2. Partitioning the data set into a training and test set.
# 3. Selection of the set of features that will be used in the learning process.
# 4. Training a decision tree.
# 5. Estimation of the quality of predictions by the final decision tree.

# Execute this pipeline for different target decision tree sizes and different sizes of the set of features used for learning and prediction.
# For the former, you can try various numbers of nodes that are multiples of 5.
# For the latter, you can select 3, 6, 9, etc. that you believe should be most important for what you are trying to predict.

# Compare the outcomes and plot a graph that displays the prediction accuracy.



# We can use this dataset: https://archive.ics.uci.edu/ml/datasets/Student+Performance


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Read the data
data = pd.read_csv('student-mat.csv', sep=';')

# Data validation and cleansing

# write a function to check for missing values
def check_missing_values(data):
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            print("There are missing values in column: ", column)
            print("Number of missing values: ", data[column].isnull().sum())
        else:
            print("There are no missing values in column: ", column)

# No missing values

# Write a function to check for duplicates
def check_duplicates(data):
    if data.duplicated().sum() > 0:
        print("There are duplicates in the data")
        print("Number of duplicates: ", data.duplicated().sum())
    else:
        print("There are no duplicates in the data")

# No duplicates

# Write a function to check for outliers
def check_outliers(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            continue
        else:
            if data[column].quantile(0.75) - data[column].quantile(0.25) > 1.5 * (data[column].quantile(0.75) - data[column].quantile(0.25)):
                print("There are outliers in column: ", column)
                print("Number of outliers: ", data[column].quantile(0.75) - data[column].quantile(0.25) > 1.5 * (data[column].quantile(0.75) - data[column].quantile(0.25)).sum())
            else:
                print("There are no outliers in column: ", column)


# No outliers

# Write a function to check for skewness
def check_skewness(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            continue
        else:
            if data[column].skew() > 1:
                print("There is skewness in column: ", column)
                print("Skewness: ", data[column].skew())
            else:
                print("There is no skewness in column: ", column)

# No skewness for the most part
# There is skewness in the following columns:
# 1. absences
# 2. G1
# 3. G2
# 4. G3

# We do not need G1 and G2 for the prediction of G3, so we will drop them
data = data.drop(['G1', 'G2'], axis=1)


# Data Conversion

# For the columns that are categorical, we will convert them to dummy variables
# We will use the get_dummies function from pandas

# We will convert the following columns to dummy variables:
# 1. school

# Convert school to dummy variables
data = pd.get_dummies(data, columns=['school'])

# Convert all features to numeric
data = data.apply(pd.to_numeric)

# Convert the target variable to numeric
data['G3'] = data['G3'].apply(pd.to_numeric)

# Convert sex to dummy variables
data = pd.get_dummies(data, columns=['sex'])

# Convert address to dummy variables
data = pd.get_dummies(data, columns=['address'])

# Convert famsize to dummy variables
data = pd.get_dummies(data, columns=['famsize'])

# Convert Pstatus to dummy variables
data = pd.get_dummies(data, columns=['Pstatus'])

# Convert Mjob to dummy variables
data = pd.get_dummies(data, columns=['Mjob'])

# Convert Fjob to dummy variables
data = pd.get_dummies(data, columns=['Fjob'])

# Convert reason to dummy variables
data = pd.get_dummies(data, columns=['reason'])

# Convert guardian to dummy variables
data = pd.get_dummies(data, columns=['guardian'])

# Convert schoolsup to dummy variables
data = pd.get_dummies(data, columns=['schoolsup'])

# Convert famsup to dummy variables
data = pd.get_dummies(data, columns=['famsup'])

# Convert paid to dummy variables
data = pd.get_dummies(data, columns=['paid'])

# Convert activities to dummy variables
data = pd.get_dummies(data, columns=['activities'])

# Convert nursery to dummy variables
data = pd.get_dummies(data, columns=['nursery'])

# Convert higher to dummy variables
data = pd.get_dummies(data, columns=['higher'])

# Convert internet to dummy variables
data = pd.get_dummies(data, columns=['internet'])

# Convert romantic to dummy variables
data = pd.get_dummies(data, columns=['romantic'])














# Partitioning the data set into a training and test set
X = data.values[:, 0:30]
Y = data.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Training a decision tree
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

# Plotting the decision tree using matplotlib
# Import necessary libraries for plotting
import matplotlib.pyplot as plt
from sklearn import tree

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf_gini, 
                     feature_names=data.columns[0:30],
                        class_names=['0', '1'],
                        filled=True)


# Estimation of the quality of predictions by the final decision tree
y_pred = clf_gini.predict(X_test)
print("Accuracy is ", accuracy_score(y_test, y_pred)*100)



dot_data = StringIO()
export_graphviz(clf_gini, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = data.columns[0:30], class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# Accuracy should be measured by the following formula:
# Expected square of the difference between the prediction and the actual value on the test set
# We will use the mean squared error
# Write a function to calculate the mean squared error
def mean_squared_error(y_test, y_pred):
    return np.mean((y_test - y_pred)**2)

# Calculate the mean squared error
print("Mean squared error is ", mean_squared_error(y_test, y_pred))








# In order to improve the accuracy of the model, we will use the following techniques:
# 1. Feature selection
# 2. Hyperparameter tuning
# 3. Ensemble learning

# Feature selection
# We will use the following techniques:
# 1. Univariate feature selection
# 2. Recursive feature elimination
# 3. Principal component analysis

# Import the necessary libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

# Write a function to perform univariate feature selection
def univariate_feature_selection(X_train, y_train, X_test, y_test):
    # Perform univariate feature selection
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X_train, y_train)
    # Summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X_train)
    # Summarize selected features
    print(features[0:5,:])
    # Perform the same transformation on the test set
    features_test = fit.transform(X_test)
    return features, features_test

# Write a function to perform recursive feature elimination
def recursive_feature_elimination(X_train, y_train, X_test, y_test):
    # Perform recursive feature elimination
    model = LogisticRegression()
    rfe = RFE(model, 3)
    fit = rfe.fit(X_train, y_train)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    # Perform the same transformation on the test set
    features_test = fit.transform(X_test)
    return features, features_test

# Write a function to perform principal component analysis
def principal_component_analysis(X_train, y_train, X_test, y_test):
    # Perform principal component analysis
    pca = PCA(n_components=3)
    fit = pca.fit(X_train)
    # Summarize components
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    print(fit.components_)
    features = fit.transform(X_train)
    # Summarize selected features
    print(features[0:5,:])
    # Perform the same transformation on the test set
    features_test = fit.transform(X_test)
    return features, features_test

# Write a function to perform feature selection
def feature_selection(X_train, y_train, X_test, y_test):
    # Perform univariate feature selection
    features, features_test = univariate_feature_selection(X_train, y_train, X_test, y_test)
    # Perform recursive feature elimination
    features, features_test = recursive_feature_elimination(X_train, y_train, X_test, y_test)
    # Perform principal component analysis
    features, features_test = principal_component_analysis(X_train, y_train, X_test, y_test)
    return features, features_test

# Perform feature selection
features, features_test = feature_selection(X_train, y_train, X_test, y_test)

# Develop a decision tree model using the selected features
dtm_v2 = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
dtm_v2.fit(features, y_train)

# Plot the decision tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtm_v2,
                        feature_names=data.columns[0:30],
                        class_names=['0', '1'],
                        filled=True)

# Estimation of the quality of predictions by the final decision tree
y_pred = dtm_v2.predict(features_test)
print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

# Mean squared error
print("Mean squared error is ", mean_squared_error(y_test, y_pred))







# Hyperparameter tuning
# We will use the following techniques:
# 1. Grid search
# 2. Random search

# Import the necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# Write a function to perform grid search
def grid_search(X_train, y_train, X_test, y_test):
    # Perform grid search
    param_grid = {'max_depth': np.arange(3, 10), 'min_samples_leaf': np.arange(5, 20)}
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    return clf.best_params_

# Write a function to perform random search
def random_search(X_train, y_train, X_test, y_test):
    # Perform random search
    param_dist = {'max_depth': np.arange(3, 10), 'min_samples_leaf': np.arange(5, 20)}
    clf = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, random_state=5)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    return clf.best_params_

# Perform hyperparameter tuning
best_params = grid_search(X_train, y_train, X_test, y_test)
best_params = random_search(X_train, y_train, X_test, y_test)

# Develop the dtm_v3 model using the best parameters from the hyperparameter tuning, and the selected features from dtm_v2
dtm_v3 = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=best_params['max_depth'], min_samples_leaf=best_params['min_samples_leaf'])
dtm_v3.fit(features, y_train)

# Plot the decision tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtm_v3,
                        feature_names=data.columns[0:30],
                        class_names=['0', '1'],
                        filled=True)

# Estimation of the quality of predictions by the final decision tree
y_pred = dtm_v3.predict(features_test)
print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

# Mean squared error
print("Mean squared error is ", mean_squared_error(y_test, y_pred))


# Model evaluation with dtm_v1, dtm_v2, and dtm_v3
# We will use the following techniques:
# 1. Confusion matrix
# 2. Classification report
# 3. ROC curve

# Import the necessary libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

# Write a function to plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Write a function to plot the classification report
def plot_classification_report(y_test, y_pred):
    # Plot the classification report
    print(classification_report(y_test, y_pred))

# Write a function to plot the ROC curve
def plot_roc_curve(y_test, y_pred):
    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Decision Tree')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Decision Tree ROC Curve')
    plt.show()

# Plot the confusion matrix, classification report, and ROC curve for dtm_v1
plot_confusion_matrix(y_test, y_pred)
plot_classification_report(y_test, y_pred)
plot_roc_curve(y_test, y_pred)

# Plot the confusion matrix, classification report, and ROC curve for dtm_v2
plot_confusion_matrix(y_test, y_pred)
plot_classification_report(y_test, y_pred)
plot_roc_curve(y_test, y_pred)

# Plot the confusion matrix, classification report, and ROC curve for dtm_v3
plot_confusion_matrix(y_test, y_pred)
plot_classification_report(y_test, y_pred)
plot_roc_curve(y_test, y_pred)



