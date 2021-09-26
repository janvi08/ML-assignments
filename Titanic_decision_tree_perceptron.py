 
# ## Solve the following questions:
# 1. Based on logical reasoning, decide which variables are absolutely not required or relevant to the model. List them down and drop those columns. 
# 2. Plot a bar chart of the survived column. 
# 3. Plot a bar chart of the number of females who survived and not, and similarly males who survived and not. 
# 4. **BONUS(0.3%)** Plot a stacked bar chart of survived or not based on PClass(i.e how many survived and not for 1st, 2nd and 3rd class). The graph can be made by referring to https://matplotlib.org/examples/pylab_examples/bar_stacked.html
# 5. Check if there are missing values in any columns. Remove such rows in the dataset.  
# 6. Convert categorical values (if any) to numbers in the dataset.
# 7. Plot a heatmap of the correlation between all columns. There are columns are inter-related. Which are those colums? Can you drop either one of the inter-related column before proceeding ahead?
# 8. Now split your dataset into training & testing dataset with 80:20 ratio using train_test_split function. Use a random state which will give an approximately equal number of survived and non survived rows in the training test. Validate using a bar chart on the training dataset. 
# 
# ## Decision Tree Modelling
# #### Use a decision tree classifier with a minimum depth of 6, to train your model. Obtain the decision tree & confusion matrix for the predictions. Obtain the score of the model.
# 
# ## Perceptron Modelling
# #### Use a perceptron model with a max_iter value 400, to train your model. Obtain the confusion matrix for the predictions. Obtain the score of the model. 
#


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ### Based on logical reasoning, decide which variables are absolutely not required or relevant to the model. List them down and drop those columns.

# In[28]:


df = pd.read_csv("titanic-train.csv")
df.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
#df=df.drop(['Ticket', 'Pclass', 'Name', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked','PassengerId'], axis=1)
df=df.drop(['Name', 'Ticket','PassengerId','Cabin'], axis=1)
df = df.dropna()
mydict={'C':0, 'Q':1, 'S':2}
df["Embarked"] = df['Embarked'].apply(lambda x: mydict[x])
df.head()


# In[29]:


plt.figure(figsize=(20,10))
for i in range(0,8):
    plt.subplot(3,6,i+1)
    plt.xlabel(list(df.columns.values)[i])
    plt.ylabel('Survived')
    plt.scatter(df.iloc[:,i],df.iloc[:,0])


# ### Plot a bar chart of the survived column. 

# In[30]:


df['Survived'].value_counts().plot(kind="bar")
plt.xticks(rotation=0)


# ### Plot a bar chart of the number of females who survived and not, and similarly males who survived and not. 

# In[31]:


fem=df[df['Sex']=='female']
male=df[df['Sex']=='male']
fem['Survived'].value_counts().plot(kind="bar")
plt.title("Female");
plt.xticks(rotation=0)


# In[32]:


male['Survived'].value_counts().plot(kind="bar")
plt.title("Male");
plt.xticks(rotation=0)


# ### 4. **BONUS(0.3%)** Plot a stacked bar chart of survived or not based on PClass(i.e how many survived and not for 1st, 2nd and 3rd class). 

# In[33]:


import numpy as np
width=0.35
first=df[df['Pclass']==1]
f=first['Survived'].value_counts(sort=1)
sec=df[df['Pclass']==2]
s=sec['Survived'].value_counts(sort=1)
third=df[df['Pclass']==3]
t=third['Survived'].value_counts(sort=1)
x=(f[1], f[0])
y=(s[1], s[0])
z=(t[1], t[0])
bars =(x[0]+y[0], x[1]+y[1])

ind = np.arange(2)
p1 = plt.bar(ind, x, width, color='black', label='first class')
p2 = plt.bar(ind, y, width,
             bottom=x, color='navy', label='second class')
p2 = plt.bar(ind, z, width,
             bottom=bars, color='teal', label='Third class')
plt.ylabel('No. of people')
plt.xlabel('Classes')
plt.title('Survivors')
plt.xticks(ind, ('Survived', 'Died'))
plt.legend()


# ### 6. Convert categorical values (if any) to numbers in the dataset.

# In[34]:



my={'male':0, 'female':1}
df["Sex"] = df['Sex'].apply(lambda x: my[x])


# ### 7. Plot a heatmap of the correlation between all columns. There are columns are inter-related. Which are those colums? Can you drop either one of the inter-related column before proceeding ahead?

# In[35]:


import seaborn as sb
plt.figure(figsize=(12,10))
sb.heatmap(df.corr(),annot=True,fmt='.1f')


# In[36]:


dfa=df.drop(['SibSp'], axis=1)
dfa.head()


# ### Now split your dataset into training & testing dataset with 80:20 ratio using train_test_split function. Use a random state which will give an approximately equal number of survived and non survived rows in the training test. Validate using a bar chart on the training dataset.

# In[37]:


cols_to_consider = ['Pclass','Sex','Age','Parch','Fare','Embarked']
from sklearn.model_selection import train_test_split as tts 
x_train, x_test, y_train, y_test = tts(dfa[cols_to_consider],dfa['Survived'],test_size=0.2,random_state=5)
print(y_train.value_counts())


# In[38]:


y_train.value_counts().plot(kind="bar")
plt.xticks(rotation=0)


# In[39]:


dfa['Embarked'].value_counts()


# ### Decision Tree Modelling
# #### Use a decision tree classifier with a minimum depth of 6, to train your model. Obtain the decision tree & confusion matrix for the predictions. Obtain the score of the model.

# In[40]:


dfa.head()
survived = dfa[dfa['Survived']==1]
notsurvived = dfa[dfa['Survived']==0]


# In[41]:


dfa['BFare'] = np.where(dfa['Fare'] >263,True,False)
dfa['BAge'] = np.where(dfa['Age'] >75,True,False)
decisiontree = DecisionTreeClassifier(criterion='entropy',max_depth=6)
obtainedtree = decisiontree.fit(x_train,
                               y_train)
print("Extracted classes ",decisiontree.classes_)


# In[42]:


import seaborn
Predicted_Survived = obtainedtree.predict(x_test)
seaborn.heatmap(confusion_matrix(y_test,Predicted_Survived),
               annot=True,cmap="Pastel1",
               xticklabels = decisiontree.classes_,
               yticklabels = decisiontree.classes_)


# In[47]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error (y_true = y_test, y_pred = Predicted_Survived)
print(mse)


# In[25]:


import os
os.environ["PATH"] += os.pathsep + '/Users/JanviPhadtare/Desktop'
from io import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
DecisionTreeImg = StringIO()

export_graphviz(obtained_tree, out_file=DecisionTreeImg, filled = True, rounded = True, feature_names=["Age", 'Pclass', "Parch",'sval','eval'], special_characters=True)
graph = pydotplus.graph_from_dot_data(DecisionTreeImg.getvalue())
Image(graph.create_jpg())


# ### Use a perceptron model with a max_iter value 400, to train your model. Obtain the confusion matrix for the predictions. Obtain the score of the model.¶

# In[43]:


from sklearn.linear_model import Perceptron

Classifier=Perceptron(max_iter=400)
model=Classifier.fit(x_train,y_train)
predictions=model.predict(x_test)


# In[44]:


cfa= confusion_matrix(y_test, predictions)
sb.heatmap(cfa, annot=True, cmap='Pastel1', xticklabels=decisiontree.classes_, yticklabels=decisiontree.classes_)


# In[46]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error (y_true = y_test, y_pred = predictions)
print(mse)


# Which of the two models fairs better?
# The Decision tree model fairs better than the perceptron model.
