
# coding: utf-8

# In[ ]:


import pandas as pd
air=pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")

#air.isnull().sum()
#print(air.count())
#air.dropna() #used to drop NaN values
#print(list(air['Operator'].value_counts().index)) #used to store index
#print(air['Operator'].value_counts().tolist()) #used to store value
air['Year'] =air['Date'].apply(lambda x: int(x.split("/")[-1]))
air['Month'] =air['Date'].apply(lambda x: int(x.split("/")[0]))
air.head()
newair = air.drop(['Flight #','cn/In'],axis=1)
newair.head()
mylist = air['Route'].value_counts().tolist()
less_risk_route = []
for i in range (0,len(mylist )):
   if(mylist[i] == min(mylist)):
     less_risk_route.append(air['Route'].value_counts().index[i])
print("Less risky routes are:",less_risk_route)


# # year and month

# In[21]:


air.head()


# In[22]:


newair.head()


# In[2]:


yearc=list(air['Year'].value_counts().index)
year_crashes=air['Year'].value_counts().tolist()
f=year_crashes[0]
print(yearc[year_crashes.index(f)],"year had the most airplane crashes")


# In[3]:


fatal_count = 0
for i in range(0,len(air)):
    if air.iloc[i][13] == 1972:
        fatal_count += air.iloc[i][10]
print("fatalities in the year 1972 is",fatal_count)


# In[4]:


s  = 0
air.dropna(subset=['Aboard','Fatalities'])
for i in range(0,len(air)):
    s += air.iloc[i][9]-air.iloc[i][10]


# In[5]:


aira = air.dropna(subset=['Aboard','Fatalities'])
mylist = list(aira["Aboard"]-aira["Fatalities"])
suma = 0 
for i in range(0,len(mylist)):
    suma += mylist[i]
#print(suma)
#print(sum(aira["Aboard"]))
per = suma/sum(aira["Aboard"]) * 100
print(" %.2f percentage people survived"%(per))


# In[6]:


c = 0
for i in range(0,len(air)):
    if pd.isnull(air["Registration"].iloc[i]):
        if((air["Aboard"].iloc[i]-air["Fatalities"].iloc[i])/(air["Aboard"].iloc[i])*100)>40.0:
            c +=1
print("no.of airplanes with no registration and more than 40% people survived is",c)


# In[7]:


print(air['Type'].value_counts().index[0],"is used by most operators") 


# In[8]:


amax = max(air['Fatalities'])
amin = min(air['Fatalities'])
airm = air.groupby(['Operator'], axis=0).mean()
airm['Fatalities']
for i in range(0,len(airm)):
    if airm['Fatalities'][i] == amax:
        print("Operators with max fatalities are",airm.iloc[i])
    elif airm['Fatalities'][i] == amin:
        print("Operators with min fatalities are",airm.iloc[i])
    


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908.csv')
data["month"]=data['Date'].apply(lambda x: (x[:2]))
data["year"]=data['Date'].apply(lambda x: (x[6:]))
newdata = data.drop(["Flight #","cn/In"],axis=1)


# In[15]:


year_sort=data.sort_values(by=['year'], axis=0)
year_count=list(year_sort['year'].value_counts().index.sort_values())
xcd=np.arange(len(year_count))

group_year = newdata.groupby(['year'])
#for name , group in group_year:
abc=[]
xyz=[]
for x in year_count:
    y = group_year.get_group(x)
    abc.append(y["Fatalities"].sum())
    xyz.append(y["Aboard"].sum())
plt.figure(figsize = (20,10))
plt.xlabel('Year')
plt.ylabel('Number')
plt.plot(abc,color='red',marker="*")
plt.plot(xyz,color='blue',marker="o")
plt.xticks(xcd,year_count,rotation=90)
plt.title('Number of Passengers and fatalities',fontsize=18)


# In[17]:


operator_list=data['Operator'].value_counts().tolist()
yaxis_operator=operator_list[:10]
operator_counta=list(data['Operator'].value_counts().index)
xaxis=operator_counta[:10]
col='rgbmkcy'
plt.figure(figsize=(20,10))
plt.bar(xaxis,yaxis_operator, width=0.8, color=col)
plt.xlabel("Operators")
plt.ylabel("No. of accidents")
plt.title("Top 10 Operators by accident")
plt.xticks(rotation=90)
plt.show()


# In[18]:


svm=data.sort_values(by=['month'], axis=0)
crashes_no=svm['year'].value_counts()
ac=list(svm['year'].value_counts().index.sort_values())
yc=[]
for everyval in ac:
    yc.append(crashes_no[everyval])

plt.figure(figsize=(20,15))
plt.bar(ac,yc,align="center", width=0.8, color=col)
plt.xlabel("Year")
plt.ylabel("Number of crashes")
plt.title("Number of Crashes per year")
plt.xticks(rotation=90)
plt.show()


# In[19]:


x_month=data.sort_values(by=['month'], axis=0)
month_count=x_month['month'].value_counts()
am=list(x_month['month'].value_counts().index.sort_values())
ym=[]
for everyval in am:
    ym.append(month_count[everyval])
plt.figure(figsize=(20,15))
plt.bar(am,ym,align="center", width=0.8, color=col)
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Number of crashes")
plt.show()


# In[20]:


operator_by = data.groupby(['Operator'])
aft = operator_by.get_group('Aeroflot')
aft["Fatalities"].sum()
aft1=aft.sort_values(by=['year'], axis=0)
aft2=list(aft1['year'].value_counts().index.sort_values())
aerof=[]
xl=np.arange((len(aft2)))
group_year = aft1.groupby(['year'])
aerof=[]
for f in aft2:
    y = group_year.get_group(f)
    aerof.append(y["Fatalities"].sum())
plt.figure(figsize=(20,10))
plt.bar(aft2,aerof,align="center", width=0.8, color=col)
plt.xticks(rotation=90)
plt.xlabel("Year")
plt.ylabel("No. of crashes")
plt.title("No. of Crashes per year by Aeroflot")
plt.show()

