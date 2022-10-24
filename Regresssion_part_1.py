#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd 
import numpy as np 
np.random.seed(101)
y = np.random.randint(1 ,100 ,size = (5 ,5 ))
print(y)


# In[126]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
path = "C:\\important\\Hesham Assem\\dataset\\ML.csv"
my_data = pd.read_csv(path , header= None , names = 'population profit'.split()  )
my_data


# In[127]:


print (f'my_first_10_raws_in_my_data_is : \n    {my_data.head(11)}')
print("🎓" * 30)
print(f'my_describe_data_is : \n {my_data.describe()}')


# In[128]:


my_data.plot(kind = 'scatter' , x = 'population' , y = 'profit')


# In[129]:


my_data.insert(0 ,'Ones ', 1 )
print (f'my_first_10_raws_in_my_data_is : \n    {my_data.head(11)}')


# In[130]:


my_data


# In[131]:


w =pd.DataFrame(data = y )


# In[132]:


w.insert(1 ,'yy' ,10)


# In[133]:


w


# In[134]:


w.insert(1 ,'1' ,10)
print(w)


# In[135]:


my_data


# In[136]:


#my_data['OOnes'] = my_data['profit'] + my_data['population']


# In[137]:


my_data


# In[138]:


my_data


# In[145]:


cols = my_data.shape[1]


# In[146]:


cols


# In[150]:


X = my_data.iloc[ :  ,    0:cols-1]
y = my_data.iloc[  :  ,  cols-1 : cols  ]


# In[152]:


y 


# In[155]:


print(f'my_X_data_is : \n {X.head(10)}')
print("🎓" * 30)
print(f'my_y_data_is  : \n {y.head(10)}')


# In[199]:


X = np.matrix(X)
y = np.matrix(y)
theta = np.zeros((1 ,2))


# In[200]:


theta


# In[201]:


print(f'x_matrix_is : \n {X}')
print("🎓" * 30)
print(f'y_matrix_is : \n {y}')
print("🎓" * 30)
print(f'theta_in_matrix_is : \n {theta}')


# In[202]:


## Cost Function >>>>>>>>>>>>

def cost_function(X , y , theta):
    z = np.power(((X * theta.T) - y) , 2)
    return np.sum(z) / (2 * len(my_data))


# In[203]:


print(f'cost_function_is : \n {cost_function(X , y , theta)}')


# In[ ]:




