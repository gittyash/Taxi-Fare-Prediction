#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

df = pd.read_csv("taxi-fares.csv")
df.head()


# Rows and Colms in DataSet

# In[17]:


df.shape


# In[18]:


df.info()


# In[19]:


df.isnull().sum()


# In[20]:


sns.countplot(x=df['passenger_count'])


# Remove all rows with multiple passengers
# 
# Remove key column from dataset

# In[21]:


df = df[df['passenger_count'] == 1]
df = df.drop(['key', 'passenger_count'], axis=1)
df.head()


# In[22]:


df.shape


# In[23]:


#how much influence input variables such as latitude and longitude have on the values in the "fare_amount" column.

corr_matrix = df.corr()
corr_matrix['fare_amount'].sort_values(ascending=False)


# # Feature Engineering

# Add columns specifying the day of the week (0=Monday, 1=Sunday, and so on)
# 
# The hour of the day that the passenger was picked up (0-23)
# 
# Distance (through the air, not on the street) in miles that the ride covered
# 

# In[24]:


import datetime
from math import sqrt

for i, row in df.iterrows():
    dt = datetime.datetime.strptime(row['pickup_datetime'], '%Y-%m-%d %H:%M:%S UTC')
    df.at[i, 'day_of_week'] = dt.weekday()
    df.at[i, 'pickup_time'] = dt.hour
    x = (row['dropoff_longitude'] - row['pickup_longitude']) * 54.6 # 1 degree == 54.6 miles
    y = (row['dropoff_latitude'] - row['pickup_latitude']) * 69.0   # 1 degree == 69 miles
    distance = sqrt(x**2 + y**2)
    df.at[i, 'distance'] = distance
    
df.head()

Drop columns no longer required
# In[25]:


df.drop(columns=['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], inplace=True)
df.head()


# In[26]:


corr_matrix = df.corr()
corr_matrix["fare_amount"].sort_values(ascending=False)


# In[27]:


#still no strong correlation betwen distance travelled and fare amount


# In[28]:


df.describe()


# Removing outliers

# In[29]:


df = df[(df['distance'] > 1.0) & (df['distance'] < 10.0)]
df = df[(df['fare_amount'] > 0.0) & (df['fare_amount'] < 50.0)]
df.shape


# In[30]:


corr_matrix = df.corr()
corr_matrix["fare_amount"].sort_values(ascending=False)


# # Train a Regeression Model

# In[31]:


from sklearn.model_selection import train_test_split

x = df.drop(['fare_amount'], axis=1)
y = df['fare_amount']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[32]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)


# In[33]:


model.score(x_test, y_test)


# In[34]:


#Score the model again using 5-fold cross-validation.

from sklearn.model_selection import cross_val_score

cross_val_score(model, x, y, cv=5).mean()


# In[37]:


#Measure the model's mean absolute error (MAE).

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, model.predict(x_test))


# In[39]:


#RandomoForestRegressor

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=0)
model.fit(x_train, y_train)

cross_val_score(model, x, y, cv=5).mean()


# In[40]:


from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=0)
model.fit(x_train, y_train)

cross_val_score(model, x, y, cv=5).mean()


# In[41]:


#what it will cost to hire a taxi for a 2-mile trip at 5:00 p.m. on Friday afternoon.
model.predict([[4, 17, 2.0]])


# In[42]:


#predict the fare amount for a 2-mile trip taken at 5:00 p.m. one day later (on Saturday).
model.predict([[5, 17, 2.0]])


# Visualizing the fares v/s distance travelled
# 

# In[45]:


plt.figure(figsize=(10, 6))
plt.scatter(df['distance'], df['fare_amount'], alpha=0.3, color='blue')
plt.title('Fare Amount vs Travel Distance')
plt.xlabel('Distance (in miles)')
plt.ylabel('Fare Amount (in $)')
plt.grid(True)
plt.show()


# In[ ]:




