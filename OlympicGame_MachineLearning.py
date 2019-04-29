
# coding: utf-8

# <h1 align="center"> 
# DATS 6501 —Capstone Project (Part III)
# </h1> 
# 
# <h1 align="center"> 
# Olympic Games Analysis — Machine Learning Continued
# </h1> 
# 
# <h4 align="center"> 
# Author: Xiaochi Ge ([gexiaochi@gwu.edu](mailto:gexiaochi@gwu.edu))
# </h4>

# ## 1. Import Packagea & Read Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


athletes = pd.read_csv('athlete_events.csv')
regions = pd.read_csv('noc_regions.csv')
olympic_data = pd.merge(athletes, regions, on='NOC', how='left')


olympic = olympic_data[['Sex','Age','Height','Weight','NOC','Sport','Medal']]
#olympic.head()


# ## 2. Fill-in Missing Data


olympic['Medal'].fillna(('Lose'), inplace=True)
olympic['Age'].fillna((olympic['Age'].median()), inplace=True)
olympic['Height'].fillna((olympic['Height'].median()), inplace=True)
olympic['Weight'].fillna((olympic['Weight'].median()), inplace=True)
olympic.head()


#If medals = Gold,Silver,and Bronze, change it to Win; Otherwise, Lose
def medal(olympic):
    if (olympic['Medal'] == 'Gold'):
        return 'Win'
    elif (olympic['Medal'] == 'Silver'):
        return 'Win'
    elif (olympic['Medal'] == 'Bronze'):
        return 'Win'
    else:
        return 'Lose'


olympic['Medal'] = olympic.apply(medal, axis=1)
olympic.head()


# ## 3. Mapping Construction


# construct medal mapping
# to convert class labels from strings to integers
#medal_mapping = {label: idx for idx, label in enumerate(np.unique(olympic['Medal']))}
#medal_mapping = {'Lose': -1,'Bronze': 0,'Silver': 1,'Gold':2}
medal_mapping = {'Lose': -1,'Win': 1}
medal_mapping

# construct NOC mapping
NOC_mapping = {label: idx for idx, label in enumerate(np.unique(olympic['NOC']))}
NOC_mapping

# construct sports mapping
sport_mapping = {label: idx for idx, label in enumerate(np.unique(olympic['Sport']))}
sport_mapping

olympic['Medal'] = olympic['Medal'].map(medal_mapping)
olympic['Sport'] = olympic['Sport'].map(sport_mapping)
olympic['NOC'] = olympic['NOC'].map(NOC_mapping)
olympic.head()


# ## 4. Female Athelets

#get female data
olympic_F = olympic[(olympic.Sex=='F')]
olympic_F.head(10)

olympic_F.info()


# ##  (a). Female Data Correlation

#draw a heatmap to check the correlation between each variable
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr = olympic_F.corr()
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


# ## （b). Train_Test Split

from sklearn.model_selection import train_test_split
Xf = olympic_F.iloc[:,1:6]
yf = olympic_F.iloc[:,6]
yf = yf.reshape(74522)
#Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.3, random_state=1, stratify=yf)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf,yf,test_size=0.33, random_state=42)


# ## （c). RandomForest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

scores = []

# Declare and train the model
clf = RandomForestClassifier(random_state = 0,n_estimators=25, n_jobs = 2)
clf.fit(Xf_train, yf_train)
yf_pred_RandomForestClassifier = clf.predict(Xf_test)
#Get Accuracy Score
score = accuracy_score(yf_pred_RandomForestClassifier,yf_test)
scores.append(score)

global importances
# Get the feature importances
importances = []
importances = clf.feature_importances_


# Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, Xf.columns)

# Sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# Make the bar Plot from f_importances 
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16,9), rot=45, fontsize=30)

# Show the plot
plt.tight_layout()
plt.show()

#Get cross validation score of random forest model
from sklearn.model_selection import cross_val_score
cv_scores = []

score_forest=cross_val_score(clf, Xf,yf, cv=10)
score_forest
print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (score_forest.mean(), score_forest.std() * 2))
cv_score = score_forest.mean()
cv_scores.append(cv_score)

from sklearn.metrics import classification_report
yf_predict = clf.predict(Xf_test)
print(classification_report(yf_test, yf_predict))


# ## (d). KNN

from sklearn.neighbors import KNeighborsClassifier
# Declare the model
clf = KNeighborsClassifier(n_neighbors=5)

# Train the model
clf.fit(Xf_train, yf_train)
yf_pred_KNeighborsClassifier = clf.predict(Xf_test)
#Get Accuracy Score
score = accuracy_score(yf_pred_KNeighborsClassifier,yf_test)
scores.append(score)

#Get cross validation score of K-Nearest Neighbors
score_knn=cross_val_score(clf, Xf,yf, cv=10)
print("K-Nearest Neighbors Acuracy: %0.2f (+/- %0.2f)" % (score_knn.mean(), score_knn.std() * 2))
cv_score = score_knn.mean()
cv_scores.append(cv_score)

from sklearn.metrics import classification_report
yf_predict = clf.predict(Xf_test)
print(classification_report(yf_test, yf_predict))


# ## （e). Perceptron

from sklearn.linear_model import Perceptron

# Declare the model
clf = Perceptron(n_iter=40, eta0=0.1, random_state=0)

# Train the model
clf.fit(Xf_train, yf_train)
yf_pred_Perceptron = clf.predict(Xf_test)
#Get Accuracy Score
score = accuracy_score(yf_pred_Perceptron,yf_test)
scores.append(score)

#Get cross validation score of Perceptron
score_Perceptron=cross_val_score(clf, Xf,yf, cv=10)
score_Perceptron
print("Perceptron Accuracy: %0.2f (+/- %0.2f)" % (score_Perceptron.mean(), score_Perceptron.std() * 2))
cv_score = score_Perceptron.mean()
cv_scores.append(cv_score)

from sklearn.metrics import classification_report
yf_predict = clf.predict(Xf_test)
print(classification_report(yf_test, yf_predict))


# ## (f). DecisionTree

from sklearn.tree import DecisionTreeClassifier

# Declare the model
clf = DecisionTreeClassifier()

#Training the Model
clf.fit(Xf_train,yf_train)
yf_pred_DecisionTree = clf.predict(Xf_test)

#Get Accuracy Score
score = accuracy_score(yf_pred_DecisionTree,yf_test)
scores.append(score)

#Get cross validation score of DecisionTree
score_DecisionTree=cross_val_score(clf, Xf,yf, cv=10)
print("DecisionTree Accuracy: %0.2f (+/- %0.2f)" % (score_DecisionTree.mean(), score_DecisionTree.std() * 2))
cv_score = score_DecisionTree.mean()
cv_scores.append(cv_score)

from sklearn.metrics import classification_report
yf_predict = clf.predict(Xf_test)
print(classification_report(yf_test, yf_predict))


# ## (g). Compare Algorithms
# ### Accuracy Score

from matplotlib.colors import ListedColormap

#Compare model among female data
#Convert the Accuracy Scores into one-dimensional 1darray with corresponding classifier names as axis labels

Acc_scores = pd.Series(scores, ['Random forest','KNeighborsClassifier','Perceptron','Decision tree'])

current_palette = sns.color_palette("muted", n_colors=4)
cmap = ListedColormap(sns.color_palette(current_palette).as_hex())

# Make the bar Plot from f_importances 
Acc_scores.plot(x='Classifiers', y='Accuracy scores',kind = 'bar',figsize=(16,9), rot=45, fontsize=30, colormap=cmap)

plt.xlabel('', fontsize=30)
plt.ylabel('Accuracy Score', fontsize=30)
plt.ylim([0.75,1])
# Show the plot
plt.tight_layout()
plt.show()


# ### Cross Validation Scores

# Convert the Cross Validation scores into one-dimensional 1darray with corresponding classifier names as axis labels

clf_scores = pd.Series(cv_scores, ['Random forest','KNeighborsClassifier','Perceptron','Decision tree'])

current_palette = sns.color_palette("muted", n_colors=4)
cmap = ListedColormap(sns.color_palette(current_palette).as_hex())

# Make the bar Plot from f_importances 
clf_scores.plot(x='Classifiers', y='Cross Validation scores',kind = 'bar',figsize=(16,9), 
                rot=45, fontsize=30, colormap=cmap)
#plt.bar(fscores,clfs)
plt.xlabel('', fontsize=30)
plt.ylabel('Cross Validation Score', fontsize=30)
plt.ylim([0.75,1])
# Show the plot
plt.tight_layout()
plt.show()                      


# # 5. Male Athelets

#get male data
olympic_M = olympic[(olympic.Sex=='M')]
olympic_M.head()

olympic_M.info()


# ## (a). Male Correlation

#draw a heatmap to check the correlation between each variable
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr_m = olympic_M.corr()
sns.heatmap(corr_m, annot=True,
            xticklabels=corr_m.columns.values,
            yticklabels=corr_m.columns.values)
plt.show()


# ## (b). Train_Test Split

from sklearn.model_selection import train_test_split
Xm = olympic_M.iloc[:,1:6]
ym = olympic_M.iloc[:,6]
ym = ym.reshape(196594)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm,ym,test_size=0.33, random_state=42)


# ## (c). RandomForest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

scores_m = []

# Declare and train the model
clf_m = RandomForestClassifier(random_state = 0,n_estimators=25, n_jobs = 2)
clf_m.fit(Xm_train, ym_train)
ym_pred_RandomForestClassifier = clf_m.predict(Xm_test)
#Get Accuracy Score
score_m = accuracy_score(ym_pred_RandomForestClassifier,ym_test)
scores_m.append(score_m)

global importances
# Get the feature importances
importances_m = []
importances_m = clf_m.feature_importances_


# Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances_m = pd.Series(importances_m, Xm.columns)

# Sort the array in descending order of the importances
f_importances_m.sort_values(ascending=False, inplace=True)

# Make the bar Plot from f_importances 
f_importances_m.plot(x='Features', y='Importance', kind='bar', figsize=(16,9), rot=45, fontsize=30)

# Show the plot
plt.tight_layout()
plt.show()

#Get cross validation score of random forest model
from sklearn.model_selection import cross_val_score
cv_scores_m = []

score_forest_m=cross_val_score(clf_m, Xm,ym, cv=10)
score_forest_m
print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (score_forest_m.mean(), score_forest_m.std() * 2))
cv_score_m = score_forest_m.mean()
cv_scores_m.append(cv_score_m)


from sklearn.metrics import classification_report
ym_predict = clf_m.predict(Xm_test)
print(classification_report(ym_test, ym_predict))


# ## (d). KNN

from sklearn.neighbors import KNeighborsClassifier
# Declare the model
clf_m = KNeighborsClassifier(n_neighbors=5)

# Train the model
clf_m.fit(Xm_train, ym_train)
ym_pred_KNeighborsClassifier = clf_m.predict(Xm_test)
#Get Accuracy Score
score_m = accuracy_score(ym_pred_KNeighborsClassifier,ym_test)
scores_m.append(score_m)


#Get cross validation score of K-Nearest Neighbors
score_knn_m=cross_val_score(clf_m, Xm,ym, cv=10)
print("K-Nearest Neighbors Acuracy: %0.2f (+/- %0.2f)" % (score_knn_m.mean(), score_knn_m.std() * 2))
cv_score_m = score_knn_m.mean()
cv_scores_m.append(cv_score_m)


from sklearn.metrics import classification_report
ym_predict = clf_m.predict(Xm_test)
print(classification_report(ym_test, ym_predict))


# ## (e). Perceptron

from sklearn.linear_model import Perceptron

# Declare the model
clf_m = Perceptron(n_iter=40, eta0=0.1, random_state=0)

# Train the model
clf_m.fit(Xm_train, ym_train)
ym_pred_Perceptron = clf_m.predict(Xm_test)
#Get Accuracy Score
score_m = accuracy_score(ym_pred_Perceptron,ym_test)
scores_m.append(score_m)

#Get cross validation score of Perceptron
score_perceptron_m=cross_val_score(clf_m, Xm,ym, cv=10)
score_perceptron_m
print("Perceptron Accuracy: %0.2f (+/- %0.2f)" % (score_perceptron_m.mean(), score_perceptron_m.std() * 2))
cv_score_m = score_perceptron_m.mean()
cv_scores_m.append(cv_score_m)


from sklearn.metrics import classification_report
ym_predict = clf_m.predict(Xm_test)
print(classification_report(ym_test, ym_predict))


# ## (f). DecisionTree

from sklearn.tree import DecisionTreeClassifier

# Declare the model
clf_m = DecisionTreeClassifier()

#Training the Model
clf_m.fit(Xm_train,ym_train)
ym_pred_DecisionTree = clf_m.predict(Xm_test)

#Get Accuracy Score
score_m = accuracy_score(ym_pred_DecisionTree,ym_test)
scores_m.append(score_m)

#Get cross validation score of DecisionTree
score_DecisionTree_m=cross_val_score(clf_m, Xm,ym, cv=10)
print("DecisionTree Accuracy: %0.2f (+/- %0.2f)" % (score_DecisionTree_m.mean(), score_DecisionTree_m.std() * 2))
cv_score_m = score_DecisionTree_m.mean()
cv_scores_m.append(cv_score_m)

from sklearn.metrics import classification_report
ym_predict = clf_m.predict(Xm_test)
print(classification_report(ym_test, ym_predict))


# ## (g). Compare Algorithms
# ### Accuracy Score

from matplotlib.colors import ListedColormap

#Compare model among female data
#Convert the Accuracy Scores into one-dimensional 1darray with corresponding classifier names as axis labels

Acc_scores_m = pd.Series(scores_m, ['Random forest','KNeighborsClassifier','Perceptron','Decision tree'])

current_palette_m = sns.color_palette("muted", n_colors=5)
cmap_m = ListedColormap(sns.color_palette(current_palette_m).as_hex())
#colors = np.random.randint(0,5,5)

# Make the bar Plot from f_importances 
Acc_scores_m.plot(x='Classifiers', y='Accuracy scores',kind = 'bar',figsize=(16,9), rot=45, fontsize=30, colormap=cmap)
#plt.bar(fscores,clfs)
plt.xlabel('', fontsize=30)
plt.ylabel('Accuracy Score', fontsize=30)
plt.ylim([0.75,1])
# Show the plot
plt.tight_layout()
plt.show()


# ### Cross Validation Scores

# Convert the Cross Validation scores into one-dimensional 1darray with corresponding classifier names as axis labels

clf_scores_m = pd.Series(cv_scores_m, ['Random forest','KNeighborsClassifier','Perceptron','Decision tree'])

current_palette_m = sns.color_palette("muted", n_colors=5)
cmap_m = ListedColormap(sns.color_palette(current_palette_m).as_hex())
#colors = np.random.randint(0,5,5)

# Make the bar Plot from f_importances 
clf_scores_m.plot(x='Classifiers', y='Cross Validation scores',kind = 'bar',figsize=(16,9), 
                rot=45, fontsize=30, colormap=cmap)
#plt.bar(fscores,clfs)
plt.xlabel('', fontsize=30)
plt.ylabel('Cross Validation Score', fontsize=30)
plt.ylim([0.75,1])
# Show the plot
plt.tight_layout()
plt.show()

