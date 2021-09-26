
# ### From the DataFrame find the following information:
# 1. Find out all the common words across the five documents.
# 2. Find out all the uncommon words across the five documents.


f1=open("doc1.txt","r")
file_content = f1.read()
str1 = "".join(file_content.split("."))
str2 = "".join(str1.replace(",",""))
str3 = "".join(str2.split("'"))
str3 = "".join(str3.split('"'))
list_a = list(filter(lambda x:len(x)>4,str3.split()))
f2=open("doc2.txt","r")
file_content = f2.read()
str1 = "".join(file_content.split("."))
str2 = "".join(str1.replace(",",""))
str3 = "".join(str2.split("'"))
str3 = "".join(str3.split('"'))
list_b = list(filter(lambda x:len(x)>4,str3.split()))
f3=open("doc3.txt","r")
file_content = f3.read()
str1 = "".join(file_content.split("."))
str2 = "".join(str1.replace(",",""))
str3 = "".join(str2.split("'"))
str3 = "".join(str3.split('"'))
list_c = list(filter(lambda x:len(x)>4,str3.split()))
f4=open("doc4.txt","r")
file_content = f4.read()
str1 = "".join(file_content.split("."))
str2 = "".join(str1.replace(",",""))
str3 = "".join(str2.split("'"))
str3 = "".join(str3.split('"'))
list_d = list(filter(lambda x:len(x)>4,str3.split()))
f5=open("doc5.txt","r")
file_content = f5.read()
str1 = "".join(file_content.split("."))
str2 = "".join(str1.replace(",",""))
str3 = "".join(str2.split("("))
str3 = "".join(str3.split(')'))
list_e = list(filter(lambda x:len(x)>4,str3.split()))
list_x=list_a+list_b+list_c+list_d+list_e


# In[2]:


mylist=[]
for a in list_x:
    if a not in mylist:
        mylist.append(a)
mylist=sorted(mylist, key=str.lower)
mylist=mylist[1:]


# In[3]:


import pandas as pd
df=pd.DataFrame()
df['doc_id']=['doc_1', 'doc_2', 'doc_3','doc_4', 'doc_5']
for i in range(0,len(mylist)):
    df[mylist[i]]=[0,0,0,0,0]
    if mylist[i] in list_a:
        df.iloc[0, i+1]=1
    if mylist[i] in list_b:
        df.iloc[1, i+1]=1
    if mylist[i] in list_c:
        df.iloc[2, i+1]=1
    if mylist[i] in list_d:
        df.iloc[3, i+1]=1
    if mylist[i] in list_e:
        df.iloc[4, i+1]=1
df


# ### Find out all the common words across the five documents.

# In[6]:


c=0
for i in range(1, len(mylist)):
    if df[df.columns[i]].sum()==5 :
        print(df.columns[i])
        c+=1
if(c==0):
    print("No words are common across all the five documents")


# In[7]:


print("Words common in atleast 2 documents are:")
for i in range(1, len(mylist)):
    if df[df.columns[i]].sum()>1 :
        print(df.columns[i])


# ### Find out all the uncommon words across the five documents.

# In[5]:


print("Uncommon words are :")
for i in range(1, len(mylist)):
    if df[df.columns[i]].sum()==1 :
        print(df.columns[i])

