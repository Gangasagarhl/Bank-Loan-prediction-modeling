#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data=pd.read_csv("C:/Users/RAGHAVENDRA/Downloads/Bank_Personal_Loan_Modelling.csv",index_col=0)


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.describe(include="all")


# In[8]:


data.cov()


# In[9]:


data.corr()


# In[10]:


sns.pairplot(data)


# In[11]:


sns.boxplot(x="Personal Loan",y="Income",data=data)


# In[12]:


sns.boxplot(x="Personal Loan",y="CCAvg",data=data)


# In[13]:


'''
From the correlation matrix,I got an Idea what to take as target and non-taret variable.
Correlation of non-target with target variables,has taught me  that,
I need to take the column correlation >0.13, eventhough correlation rate is not so high.

Preparing data set.

'''


# ### Preparing data set

# In[14]:


'''
Target:Personal Loan.

Non-Target:Income,CCAvg,CD Account,Mortgage,Education
'''


# In[15]:


x=data.loc[:,["Income","CCAvg","CD Account","Mortgage","Education","Family","Securities Account"]]
y=data["Personal Loan"]


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[18]:


Accuracy_score_final=0
model_name_final=""


# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# ### 1.Logistic Regression

# In[107]:


#Getting object of LogisticRegression
LogReg=LogisticRegression()

#Fitting Logistic Regression
LogReg.fit(x_train,y_train)

#Prediction for the split x_test data
y_pred=LogReg.predict(x_test)

print(confusion_matrix(y_test,y_pred))

acc=LogReg.score(x_test,y_test)
print("Accuracy Score thorugh Logistic Regression=",acc)

if acc>Accuracy_score_final:
    Accuracy_score_final=acc
    model_name_final="Logistic Regression"
    


# ### 2.Decision Tree Classifier

# In[109]:


from sklearn.tree import DecisionTreeClassifier

dectr=DecisionTreeClassifier()

dectr.fit(x_train,y_train)

y_pred=dectr.predict(x_test)


print(confusion_matrix(y_test,y_pred))
acc=dectr.score(x_test,y_test)
print("Accuarcy Score from Decison Tree classifier is:",acc)

if acc>Accuracy_score_final:
    Accuracy_score_final=acc
    model_name_final="Decision Tree Classifier"


# ### 3.RandomForestClassifer
# 

# In[110]:


from sklearn.ensemble import RandomForestClassifier

RFC=RandomForestClassifier()

RFC.fit(x_train,y_train)

y_predict=RFC.predict(x_test)
print(confusion_matrix(y_test,y_pred))
acc=RFC.score(x_test,y_test)
print("Accuarcy Score from Random Forest classifier is:",acc)

if acc>Accuracy_score_final:
    Accuracy_score_final=acc
    model_name_final="Random Forest Classifier"


# ### 4.SVM 
# 

# In[111]:


from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train,y_train)

y_predict=svc.predict(x_test)

print(confusion_matrix(y_test,y_pred))
acc=svc.score(x_test,y_test)
print("Accuarcy Score from SVC is:",acc)

if acc>Accuracy_score_final:
    Accuracy_score_final=acc
    model_name_final="Support Vector Machine Classifier"


# ### 5.Naive Bayes

# In[112]:


from sklearn.naive_bayes import GaussianNB

nbgnb= GaussianNB()

nbgnb.fit(x_train,y_train)

y_predict=nbgnb.predict(x_test)

print(confusion_matrix(y_test,y_pred))
acc=nbgnb.score(x_test,y_test)

print("Accuarcy Score from GaussianNB is:",acc)

if acc>Accuracy_score_final:
    Accuracy_score_final=acc
    model_name_final="Naive Bayes Classifier"


# ## Comparision Result

# In[113]:


print("Best Model among all is:' ",model_name_final," ' with its accuracy score: ' ", Accuracy_score_final," '")

