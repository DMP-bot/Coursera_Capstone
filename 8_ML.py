"""
Created on Sat Dec 24 11:57:31 2022
"""

# %% Set up

### Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import os

# Current directory
os.chdir('C:/Users/user/Desktop/Coursera/10_CapstoneDS/Figures')

### Useful function
def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.savefig('ML.png') ## Notice all 4 graphs are the same. I will overwrite them, but does not cause any change
    plt.show() 
    
    
### Download and save data
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
data = pd.read_csv(URL1)
X = pd.read_csv(URL2)
data.to_csv('text1')
X.to_csv('text2')

# %% Task 1: Create a NumPy array from the column Class in data, by applying 
# the method to_numpy() then assign it to the variable Y,make sure the output 
# is a Pandas series (only one bracket df['name of column']).

Y = data['Class'].to_numpy()

# %% Task 2: Standardize the data in X then reassign it to the variable X using 
# the transform provided below.

transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)


# %% Task 3: Use the function train_test_split to split the data X and Y into 
# training and test data. Set the parameter test_size to 0.2 and random_state 
# to 2. The training data and test data should be assigned to the following labels.

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

Y_test.shape

# %% Task 4: Create a logistic regression object then create a GridSearchCV 
# object logreg_cv with cv = 10. Fit the object to find the best parameters 
# from the dictionary parameters

parameters ={'C':[0.01,0.1,1],# C: Regularization strength
             'penalty':['l2'], # l1 lasso l2 ridge
             'solver':['lbfgs']}
lr = LogisticRegression()
logreg_cv = GridSearchCV(lr, parameters, cv=10)
logreg_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_) # Optimal C=0.01
print("accuracy :",logreg_cv.best_score_) # 0.8196428571428571 accuracy in train data



# %% Task 5: Accuracy on test data and confusion matrix on test data

# Accuracy in test data
acc_logreg_test_data = logreg_cv.score(X_test, Y_test)
print("Accuracy on test data :", acc_logreg_test_data)

# Confusion matrix in test data
yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
#plt.savefig('ML_LogReg.png')
# Major problem is false positives

# %% Task 6: Create a support vector machine object then create a GridSearchCV 
# object svm_cv with cv - 10. Fit the object to find the best parameters from 
# the dictionary parameters
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(svm, parameters, cv=10)
svm_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)

# %% Task 7: Accuracy on test data and confusion matrix on test data

# Accuracy in test data
acc_svm_test_data = svm_cv.score(X_test, Y_test)
print("Accuracy on test data :", acc_svm_test_data)

# Confusion matrix in test data
yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

# %% Task 8: Create a decision tree classifier object then create a GridSearchCV 
# object tree_cv with cv = 10. Fit the object to find the best parameters from 
# the dictionary parameters

parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# %% Task 9: Accuracy on test data and confusion matrix on test data

# Accuracy in test data
acc_tree_test_data = tree_cv.score(X_test, Y_test)
print("Accuracy on test data :", acc_tree_test_data)

# Confusion matrix in test data
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

# %% Task 10: Create a k nearest neighbors object then create a GridSearchCV 
# object knn_cv with cv = 10. Fit the object to find the best parameters from 
# the dictionary parameters
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, parameters, cv=10)
knn_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)

# %% Task 11: Accuracy on test data and confusion matrix on test data
    
# Accuracy in test data
acc_knn_test_data = knn_cv.score(X_test, Y_test)
print("Accuracy on test data :", acc_knn_test_data)

# Confusion matrix in test data
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# %% Task 12: Compare

# Dictionary and df with test data accuracy
accuracy_test = {'Method': ['LogReg', 'SVM', 'Trees', 'KNN'],
     'Accuracy': [acc_logreg_test_data, acc_svm_test_data, acc_tree_test_data, acc_knn_test_data]}
df_accuracy_test  = pd.DataFrame(data=accuracy_test)
print(df_accuracy_test)

# Dictionary and df with train data accuracy
accuracy_train = {'Method': ['LogReg', 'SVM', 'Trees', 'KNN'],
     'Accuracy': [logreg_cv.best_score_, svm_cv.best_score_, tree_cv.best_score_, knn_cv.best_score_]}
df_accuracy_train  = pd.DataFrame(data=accuracy_train)
print(df_accuracy_train)

# Graph test data accuracy
sns.barplot(data=accuracy_test, x='Method', y='Accuracy',color='b')
plt.xlabel("Method",fontsize=10)
plt.xticks(rotation=45)
plt.ylabel("Accuracy (%)",fontsize=10)
plt.ylim(0.7,0.9)
plt.savefig('ML_TestAccuracy.png')

# Graph train data accuracy
sns.barplot(data=accuracy_train, x='Method', y='Accuracy',color='b')
plt.xlabel("Method",fontsize=10)
plt.xticks(rotation=45)
plt.ylabel("Accuracy (%)",fontsize=10)
plt.ylim(0.7,0.9)
plt.savefig('ML_TrainAccuracy.png')