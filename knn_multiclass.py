
# coding: utf-8

# In[97]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[98]:


#lets read the training data
train_data = pd.read_csv('../input/training.csv')
print("The train data")
train_data.head()


# In[99]:


#loading the test data
test_data = pd.read_csv('../input/test (1).csv')
print("The test data")
test_data.head()


# In[100]:


#information on the train_features
train_data.info()


# In[101]:


#information on the test_features
test_data.info()


# In[102]:


#from the above we can see that train_Data has more columns
# check if the data has missing points with seaborn heatmap
import seaborn as sns
sns.heatmap(train_data.isnull(),yticklabels=False, cmap='viridis')


# In[103]:



#though unusual we can also look if the test data has some missing values
# check if the data has missing points with seaborn heatmap
import seaborn as sns
sns.heatmap(test_data.isnull(),yticklabels=False, cmap='viridis')


# In[104]:


#Checking for missing data
NAs = pd.concat([train_data.isnull().sum()], axis=1)
#keys=[‘train_data’]
NAs[NAs.sum(axis=1) > 0]


# In[105]:


# view the columns
# view the columns in train_data and test_data
train_data.columns,test_data.columns


# In[106]:


#lets check how thye submission csv is supposed to look like 
submission= pd.read_csv('../input/sample_submission (2).csv')
submission.head()


# In[107]:


#dropping the labels that you are supposed to predict and the excess from train_head
cols = ['ID','mobile_money', 'savings', 'borrowing','insurance']
train_data = train_data.drop(cols, axis=1)
x_test = test_data.drop(['ID'], axis=1)


# In[108]:


#lets first train the ANN using all the data without removing outliers
#using the all data
X = train_data.drop(['mobile_money_classification'], axis=1)
y = train_data['mobile_money_classification']


# knn has no attribute for feature importrance but there is work around that

# In[109]:


from sklearn.preprocessing import MinMaxScaler
names=X.columns
names1=x_test.columns
scaler = MinMaxScaler(feature_range=(0, 1))
X1 = scaler.fit_transform(X)
X2 = pd.DataFrame(X1, columns=names)
X_test=scaler.fit_transform(x_test)
X_test1 = pd.DataFrame(X_test, columns=names1)


# In[110]:


#looking at the structure of the two data frames
X2.columns,X_test1.columns


# In[111]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, random_state=42, stratify=y)

X_train.shape, y_train.shape, X_test.shape, y_test.shape,


# In[112]:


#training a base knn on the data
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)


# In[113]:


from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

test_pred = knn_classifier.predict_proba(X_test1)

test_pred = pd.DataFrame(knn_classifier.predict_proba(X_test1), columns=labels.classes_)
q = {'ID': test_data["ID"], 'no_financial_services': test_pred[0], 'other_only': test_pred[1],
    'mm_only': test_pred[2], 'mm_plus': test_pred[3]}
df_pred = pd.DataFrame(data=q)
df_pred = df_pred[['ID','no_financial_services', 'other_only', 'mm_only', 'mm_plus'  ]]


# In[114]:


df_pred.head()


# In[115]:


#df_pred.to_csv('pred_set.csv', index=False) #save to csv file#


# In[116]:


#now lets try tuning the knn parameters
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[117]:


# 10-fold (cv=10) cross-validation with K=5 (n_neighbors=5) for KNN (the n_neighbors parameter)

# instantiate model
knn = KNeighborsClassifier(n_neighbors=5)

# store scores in scores object
# scoring metric used here is 'accuracy' because it's a classification problem
# cross_val_score takes care of splitting X and y into the 10 folds that's why we pass X and y entirely instead of X_train and y_train
scores = cross_val_score(knn, X2, y, cv=10, scoring='accuracy')
print(scores)


# In[118]:


# use average accuracy as an estimate of out-of-sample accuracy

# scores is a numpy array so we can use the mean method
print(scores.mean())


# In[119]:


# search for an optimal value of K for KNN

# list of integers 1 to 30
# integers we want to try
k_range = range(1, 31)

# list of scores from k_range
k_scores = []

# 1. we will loop through reasonable values of k
for k in k_range:
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, X2, y, cv=10, scoring='accuracy')
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())
print(k_scores)


# In[120]:


# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# looking at the above depictio we can see that the k_neighbor is best at 20 and doesnt change the performance

# In[121]:


#using the best n_neighbors
#training a base knn on the data
from sklearn.neighbors import KNeighborsClassifier
knn_classifier1 = KNeighborsClassifier(n_neighbors=20)
knn_classifier1.fit(X_train,y_train)


# In[122]:


from sklearn.preprocessing import LabelEncoder

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

test_pred = knn_classifier1.predict_proba(X_test1)

test_pred = pd.DataFrame(knn_classifier1.predict_proba(X_test1), columns=labels.classes_)
q = {'ID': test_data["ID"], 'no_financial_services': test_pred[0], 'other_only': test_pred[1],
    'mm_only': test_pred[2], 'mm_plus': test_pred[3]}
df_pred1 = pd.DataFrame(data=q)
df_pred1 = df_pred1[['ID','no_financial_services', 'other_only', 'mm_only', 'mm_plus'  ]]


# In[123]:


df_pred1.head()


# In[ ]:


df_pred1.to_csv('pred_set.csv', index=False) #save to csv file#

