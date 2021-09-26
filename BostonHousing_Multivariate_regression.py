
# # Exercise - Multivariate Regression with the Boston Housing Data set
# 
# The dataset contains prices of houses(MEDV) in Boston City(USA) based on various parameters. The dataset is available at https://www.kaggle.com/vikrishnan/boston-house-prices/data.
# We have to predict the house prices based on the variables provided in the dataset. 
# 
# ### The first step is analysis of the data. Plot & answer the following questions regarding the data.
# 
# 1. Determine if there any values missing in any rows/columns. Filter out such rows. 
# 2. Find out the correlation of this dataset using pandas. Plot a heatmap of this matrix. Which features have a higher correlation with MEDV? Are there any features which are correlated with other features? What are those? 
# 2. Plot different scatter plots of all feature variables with MEDV. Observe trends based on the plots. Which features are more likely to give a precise value for MEDV?
# 
# ### With the analysis done above of the dataset, remove the columns which are not likely to predict MEDV. Perform Linear regression on this new filtered dataset. 
# 1. Perform a 80:20 split with the train_test_split function, with random_state=0. Perform linear regression on the training dataset. Print the obtained co-efficients for every feature. Which features have more weightage? 
# 2. Plot a scatter of test prices vs obtained prices. Obtain the MSE, R^2 score of this model. 

import pandas as pd
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


cols = ['CRIM', "ZN", "INDUS", "CHAS", "NOX", "RM" , "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
df = pd.read_csv('housing.csv', names=cols, header=None)
df.head()


df = df.dropna()


plt.figure(figsize=(12,10))
sb.heatmap(df.corr(),annot=True,fmt='.1f')


dfc = df.corr()
dfca = dfc.loc[dfc['MEDV']<0]
dfc.iloc[:,-1]



plt.figure(figsize=(20,10))
for i in range(0,13):
    plt.subplot(3,6,i+1)
    plt.xlabel(list(df.columns.values)[i])
    plt.ylabel('MEDV')
    plt.scatter(df.iloc[:,i],df.iloc[:,-1])



cols_to_consider = ['CRIM','INDUS','NOX','RM','AGE','DIS','PTRATIO','LSTAT']
df.loc[:,cols_to_consider]



from sklearn.model_selection import train_test_split as tts 
x_train, x_test, y_train, y_test = tts(df[cols_to_consider],df['MEDV'],test_size=0.2,random_state=30)


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(x_train, y_train)
obtained_y_test = lm.predict(x_test)
print(lm.score(x_test,y_test))


from sklearn.metrics import mean_squared_error
mse = mean_squared_error (y_true = y_test, y_pred = obtained_y_test)
print(mse)


plt.scatter(y_test,obtained_y_test)
plt.axis([0,60,0,50])
plt.grid(True)

