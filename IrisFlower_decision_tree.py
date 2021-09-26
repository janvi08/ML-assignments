
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'flower']
df = pd.read_csv('./data/iris.data.txt', header=None, names=cols)
df.head()


# In[3]:


mydict={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
df["bflower"] = df['flower'].apply(lambda x: mydict[x])
df.head()


# In[4]:


df['flower'].value_counts().plot(kind='bar')
plt.xticks(rotation=0)


# In[5]:


setosa = df[df['bflower']==0]
versicolor = df[df['bflower']==1]
virginica = df[df['bflower']==2]
plt.figure(figsize=(20,20))
for i in range(0,4):
    for j in range(0,4):
        plt.subplot(4,4,i*4+j+1)
        plt.scatter(setosa.iloc[:,i],setosa.iloc[:,j])
        plt.scatter(versicolor.iloc[:,i],versicolor.iloc[:,j])
        plt.scatter(virginica.iloc[:,i],virginica.iloc[:,j])
        plt.xlabel(list(df.columns.values)[i])
        plt.ylabel(list(df.columns.values)[j])


# In[40]:


mycols = ['petal_length','petal_width']
x = df[mycols]
y = df['bflower']


# In[42]:


from sklearn.model_selection import train_test_split as tts 
x_train, x_test, y_train, y_test = tts(x,y,test_size=0.2,random_state=9032)


# In[32]:


from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier(criterion='entropy', max_depth=2)
obtained_tree = decisiontree.fit(x_train, y_train)
print('Extracted Classes', decisiontree.classes_)


# In[37]:


import seaborn as sb


from sklearn.metrics import classification_report, confusion_matrix
Predicted_Species = obtained_tree.predict(x_test)
cf= confusion_matrix(y_test, Predicted_Species)
sb.heatmap(cf, annot=True, cmap='Pastel1', xticklabels=decisiontree.classes_, yticklabels=decisiontree.classes_)


# In[47]:


a=x_train[y_train==0]
b=x_train[y_train==1]
c=x_train[y_train==2]
x_min,x_max=0,8
y_min,y_max=0,3
xx,yy=np.meshgrid(np.arange(x_min,x_max, .02),np.arange(y_min,y_max, .02))

xx_lin=xx.ravel()
yy_lin=yy.ravel()
pred_input=np.c_[xx_lin,yy_lin]

dt_z=obtained_tree.predict(pred_input)
dt_z=dt_z.reshape(xx.shape)

# #contour plot requires 2D matrix
plt.contourf(xx,yy,dt_z,cmap=plt.cm.Pastel1)
plt.scatter(a['petal_length'], a['petal_width'], label="Setosa")
plt.scatter(b['petal_length'], b['petal_width'], label="Versicolor")
plt.scatter(c['petal_length'], c['petal_width'], label="Virginica",c='r')
plt.legend()
plt.grid()

