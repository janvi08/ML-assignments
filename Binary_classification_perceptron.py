
# Classify vehicle mileage as high or low based on two features, weight and horsepower using Perceptron model. High mileage is 1 and low mileage is 0.
# 
# ## Perform the following operations
# 1. Plot the scatter of weight vs horsepower. The vehicles with high mileage should be in blue color and low should be in red color. Give appropriate legends. 
# 2. Split the data set into training and testing data set. Use the train_test_split function to get a 80:20 split, using random_state 3. Plot a bar chart of number of 1's and 0's in the training dataset. 
# 3. Use the perceptron model to classify the data, and get predictions for test dataset.
# 4. Obtain the confusion matrix of the output.
# 5. Plot the separating hyperplane on the training dataset.
# 6. Consider a new test data. Weight values are 2, 3.3, 1.21, 5.32, 1.23, 4.8 and horsepower 20, 29, 13, 100, 40, 49 respectively, and mileage is 0, 0, 0, 1, 1, 1, 0. Obtain the confusion matrix for this test data. 
# 7. Split the original dataset with 80:20 ratio again, but with random_state 299. Get the output of steps 3, 4, 5, 6. What is the difference from the previous output?


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from io import StringIO
dataset =StringIO('''
weight,horsepower,mileage
1.4,43,1
1.5,8,0
2.71,44,1
3.55,23,0
2.5,15,0
2.9,19,0
3.85,112,1
3.96,87,1
4.62,108,1
4.7,21,0
2.67,34,1
2.5,20,0
1.932,12, 0
4.8,48,1
1.321,32,1
1.9013,52,1
4.13,19,0
2.8,29,0
''')
df = pd.read_csv(dataset)
df.head()


# ##### Plot the scatter of weight vs horsepower. The vehicles with high mileage should be in blue color and low should be in red color. Give appropriate legends.

# In[3]:


high_mileage=df[df['mileage']==1]
low_mileage=df[df['mileage']==0]
plt.scatter(high_mileage['weight' ],high_mileage['horsepower'],label='high mileage')
plt.scatter(low_mileage['weight' ],low_mileage['horsepower'],color="r",label='low_mileage')
plt.xlabel("Weight")
plt.ylabel("Horsepower")
plt.legend()
plt.axis([1,5,0,115])
plt.grid()


# ##### Split the data set into training and testing data set. Use the train_test_split function to get a 80:20 split, using random_state 3. Plot a bar chart of number of 1's and 0's in the training dataset. 

# In[4]:


from sklearn.model_selection import train_test_split as tts 
cols = ['weight','horsepower']
x_train, x_test, y_train, y_test = tts(df[cols],df['mileage'],test_size=0.2,random_state=3)


# In[5]:


y_train


# In[6]:


y_train.value_counts(sort=1).plot(kind='bar')
plt.xticks(rotation=0)


# In[7]:


df['mprediction']=df['mileage'].apply(lambda x:x==1)
from sklearn.linear_model import Perceptron
classifier=Perceptron(max_iter=1000)
model=classifier.fit(df[['weight','horsepower']],df['mprediction'])
df


# ##### Use the perceptron model to classify the data, and get predictions for test dataset.

# In[8]:


model.predict(x_test)


# ##### Obtain the confusion matrix of the output.

# In[9]:


from sklearn import metrics
metrics.confusion_matrix(y_test,model.predict(x_test))


# In[10]:


hm = x_train[y_train==1]
lm = x_train[y_train==0]


# In[11]:


import numpy as np
x_min, x_max=1,5
y_min, y_max=0,115

xx,yy=np.meshgrid(np.arange(x_min,x_max,.01),np.arange(y_min,y_max,.01))

print(xx);print(yy);
xx_lin=xx.ravel();print(xx_lin[:10]);
yy_lin=yy.ravel();print(yy_lin[:10]);
pred_input=np.c_[xx_lin,yy_lin];print(pred_input[:20])

z=model.predict(pred_input);print(z)
z=z.reshape(xx.shape);print(z)

#counter plot requires 2D matrix
plt.contourf(xx,yy,z,cmap=plt.cm.Pastel1)

plt.scatter(hm['weight'],hm['horsepower'],color='b',label="High Mileage")
plt.scatter(lm['weight'],lm['horsepower'],color='r',label="Low Mileage")
plt.axis([1,5,0,115])
#plt.legend()
plt.grid()


# In[12]:


collist = ['weight','horsepower']
x_testlist = pd.DataFrame([[2,20],[3.3,29],[1.21,13],[5.32,100],[1.23,40],[4.8,49]], columns = collist)
ytestlist = [0, 0, 0, 1, 1, 1]
from sklearn import metrics
metrics.confusion_matrix(ytestlist,model.predict(x_testlist))


# In[13]:


collist = ['weight','horsepower']
x_testlist = pd.DataFrame([[2,20],[3.3,29],[1.21,13],[5.32,100],[1.23,40],[4.8,49]], columns = collist)


# Split the original dataset with 80:20 ratio again, but with random_state 299. Get the output of steps 3, 4, 5, 6. What is the difference from the previous output?

# In[18]:


from sklearn.model_selection import train_test_split as tts 
cols = ['weight','horsepower']
x_train, x_test, y_train, y_test = tts(df[cols],df['mileage'],test_size=0.2,random_state=299)
df['mprediction']=df['mileage'].apply(lambda x:x==1)
from sklearn.linear_model import Perceptron
classifier=Perceptron(max_iter=1000)
model=classifier.fit(df[['weight','horsepower']],df['mprediction'])
from sklearn import metrics
metrics.confusion_matrix(y_test,model.predict(x_test))
hm = x_train[y_train==1]
lm = x_train[y_train==0]
import numpy as np
x_min, x_max=1,5
y_min, y_max=0,115

xx,yy=np.meshgrid(np.arange(x_min,x_max,.01),np.arange(y_min,y_max,.01))

print(xx);print(yy);
xx_lin=xx.ravel();print(xx_lin[:10]);
yy_lin=yy.ravel();print(yy_lin[:10]);
pred_input=np.c_[xx_lin,yy_lin];print(pred_input[:20])

z=model.predict(pred_input);print(z)
z=z.reshape(xx.shape);print(z)

#counter plot requires 2D matrix
plt.contourf(xx,yy,z,cmap=plt.cm.Pastel1)

plt.scatter(hm['weight'],hm['horsepower'],color='b',label="High Mileage")
plt.scatter(lm['weight'],lm['horsepower'],color='r',label="Low Mileage")
plt.axis([1,5,0,115])
#plt.legend()
plt.grid()
collist = ['weight','horsepower']
x_testlist = pd.DataFrame([[2,20],[3.3,29],[1.21,13],[5.32,100],[1.23,40],[4.8,49]], columns = collist)
ytestlist = [0, 0, 0, 1, 1, 1]
from sklearn import metrics
metrics.confusion_matrix(ytestlist,model.predict(x_testlist))

