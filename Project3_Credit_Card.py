#!/usr/bin/env python
# coding: utf-8

# # Project 3: Credit Card Default Detection

# #### 1. Given the data below, please classify the two cluster data and find the boundary line to devide the two groups. Use visualization to show your answer. 

# In[1]:


## import the necessary packages
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

plt.style.use('ggplot')

X,Y = datasets.make_blobs(n_samples=200, n_features=8, centers=2, random_state=1234)
plt.scatter (X[:,0],X[:,1],c=Y)


# In[2]:


from sklearn import svm
svc = svm.SVC(kernel='linear', C=1.0).fit(X, Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(X, Y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=1.0).fit(X, Y)
lin_svc = svm.LinearSVC(C=1.0).fit(X, Y)


# In[3]:


h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# In[4]:


## concatenate numpy array:
## numpy.concatenate: https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
np.concatenate((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1)),axis=1)


# #### 2. Following you will need to solve a Credit Card Default Detection Case

# **Data Description:**
# 
# **id**: A unique Id field which represents a customer
# 
# **X1**: Credit line
# 
# **X2**: Gender (1 = male; 2 = female).
# 
# **X3**: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
# 
# **X4**: Marital status (1 = married; 2 = single; 3 = others).
# 
# **X5**: Age (year).
# 
# **X6 - X11**: History of past payment. 
# 
# **X6** = September, 2015;
# 
# **X7** = August, 2015;
# 
# **X11** =April, 2015. -1 = pay one month ahead; -2 = pay two month ahead; 0 = pay on time; Positive means the payment delayed months, 1 = delay 1 month, 2 = delay 2 months, etc.
# 
# **X12- X17**: Amount in bill statement.
# 
# **X12** = amount of bill statementSeptember, 2015
# 
# **X13** = amount of bill statementAugust, 2015
# 
# **X17** = amount of bill statementApril, 2015. 
# 
# **X18-X23**: Amount of previous payment
# 
# **X18** = amount paid in September, 2015; 
# 
# **X19** = amount paid in August, 2015; 
# 
# **X23** = amount paid in April, 2015.
# 
# **Y**: A binary response variable to indicate whether the customer is fraud (1) or not (0).
# 
# This is a real problem to classify multi-feature data into two groups.
# 
# 

# **1. Load the data**

# In[9]:


train_data=pd.read_csv('Project 3/dataset/train.csv')
test_data=pd.read_csv('Project 3/dataset/test.csv')
print(train_data.shape)
print(test_data.shape)


# In[10]:


train_data.columns


# In[11]:


train_data.iloc[0:10,6:12]


# In[12]:


train_data.dtypes


# In[13]:


train_data.isnull().sum()


# **2. Distinguish categorical and continuous variables**

# In[14]:


cat_v = []
con_v = []
for c in train_data.columns:
    if len(train_data[c].value_counts().index)<=15:
        cat_v.append(c)
    else:
        con_v.append(c)
cat_v.remove('Y')
target = ['Y']


# In[15]:


print("The continuous variables: ", con_v, "\n")
print("The categorical variables: ", cat_v)


# **3. Basic feature analysis**

# i. Check the pattern differences between the training data and testing data

# In[16]:


count=1
for i in range(len(cat_v)):
    fig = plt.figure(figsize=(30,80))
    plt.subplot(len(cat_v),2,count)
    plt.bar(train_data[cat_v[i]].value_counts().index, train_data[cat_v[i]].value_counts().values)
    plt.title("train "+cat_v[i])
    
    plt.subplot(len(cat_v),2,count+1)
    plt.bar(test_data[cat_v[i]].value_counts().index, test_data[cat_v[i]].value_counts().values)
    plt.title("test "+cat_v[i])
    count+=2


# In[17]:


count=1
for i in range(len(con_v)):
    fig = plt.figure(figsize=(20,100))
    plt.subplot(len(con_v),2,count)
    plt.violinplot(train_data[con_v[i]],showmeans=True)
    plt.title("train "+con_v[i])
    
    plt.subplot(len(con_v),2,count+1)
    plt.violinplot(test_data[con_v[i]],showmeans=True)
    plt.title("test "+con_v[i])
    count+=2


# ii. Check the if there are linear relationships between features

# In[18]:


def plot_corr(df,size=15):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr,cmap=plt.get_cmap('rainbow'))
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(cax)
plot_corr(train_data)


# iii. Check the pattern of the label

# In[19]:


fig = plt.figure(figsize=(20,10))
plt.bar(train_data['Y'].value_counts().index, train_data['Y'].value_counts().values)
plt.xticks(train_data['Y'].value_counts().index,fontsize=15)
plt.show()


# **4. Build a baseline model**

# In[20]:


from sklearn.model_selection import train_test_split
Y = train_data['Y']
X = train_data.drop(['Y', 'id'], axis= 1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)


# In[21]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
RF = RandomForestClassifier(class_weight = {0:1, 1:3})
RF = RF.fit(x_train, y_train)
y_pred = RF.predict(x_test)
print (metrics.classification_report(y_test, y_pred))


# **5. Basic parameter tuning: Grid Searching**

# In[22]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
def search_model(x_train, y_train, est, param_grid, n_jobs, cv):
    model = GridSearchCV(estimator  = est,
                                     param_grid = param_grid,
                                     scoring = 'f1_weighted',
                                     verbose = 10,
                                     n_jobs = n_jobs,
                                     iid = True,
                                     cv = cv)
    # Fit Grid Search Model
    model.fit(x_train, y_train)   
    return model


# In[23]:


param_grid = {'n_estimators':[100,300,500],
             'criterion':['gini', 'entropy'],
             'class_weight': [{0:1, 1:3}]}

RF = search_model(X.values
            , Y.values
            , RandomForestClassifier()
            , param_grid
            , -1
            , 5)


# In[27]:


print("Best score: %0.3f" % RF.best_score_)
print("Best parameters set:", RF.best_params_)


# **6. Model Ensemble**

# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

clf1 = RandomForestClassifier(n_estimators = 300, criterion = 'entropy',class_weight = {0:1, 1:3})
clf2 = KNeighborsClassifier(n_neighbors=3)
clf3 = LogisticRegression (class_weight = {0:1, 1:3})
# results from your gridsearch
eclf = VotingClassifier(estimators=[('Random_Forest',clf1), ('KNN', clf2),('Logistic', clf3)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, eclf], ['Random_Forest', 'KNN','Logistic', 'Ensemble']):
    scores = cross_val_score(clf, X, Y, cv=3, scoring='f1_weighted')
    print ("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# **7. Generate the final submission**

# In[26]:


eclf.fit(X, Y)
y = pd.DataFrame(eclf.predict(test_data.drop(['id'],axis=1)), columns=['y'])
predict_data = pd.concat([y, test_data['id']], axis =1)
predict_data.to_csv('Submmission.csv', index=False)


# In[ ]:





# In[ ]:




