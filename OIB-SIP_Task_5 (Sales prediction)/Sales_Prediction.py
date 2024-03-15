# %% [markdown]
# # Import necessory libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# %% [markdown]
# # Loading the data

# %%
sales_data = pd.read_csv('Advertising.csv',index_col='Unnamed: 0')

# %%
sales_data.head()

# %% [markdown]
# # EDA

# %%
sales_data.shape

# %%
# check if any null value in data
sales_data.isnull().sum()

# %%
sales_data.describe()

# %%
# check for duplicate values in dataset
sales_data.duplicated().sum()

# %%
sales_data.info()

# %% [markdown]
# # Now visualize the Data

# %%
sns.pairplot(sales_data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.8)
plt.show()

# %% [markdown]
# # Check correlation between data

# %%
#Correlation matrix
corr_matrix = sales_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='CMRmap', fmt='.2f')
plt.show()

# %%
corr_matrix

# %% [markdown]
# # Now train the model

# %% [markdown]
# * split the dataset

# %%
X = sales_data[['TV', 'Radio', 'Newspaper']]
y = sales_data['Sales']

# %%
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# %% [markdown]
# * Choose and train the model

# %%
model = LinearRegression()

# %%
model.fit(X_train,y_train)

# %% [markdown]
# make prediction

# %%
# Make predictions on the training set
train_predictions = model.predict(X_train)

# %%
# Make predictions on the testing set
test_predictions = model.predict(X_test)

# %%
# Scatter plot for both training and test set predictions
plt.figure(figsize=(12, 6))
plt.scatter(y_train, train_predictions, c='blue', label='Training Set', marker='o')
plt.scatter(y_test, test_predictions, c='red', label='Test Set', marker='x')
plt.title('Scatter Plot: Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()

# %% [markdown]
# * Model evalution

# %%
print(f''' The train accuracy : {r2_score(y_train,train_predictions)} 
The test accuracy : {r2_score(y_test , test_predictions)}''')

# %% [markdown]
# Train model using random forest regression

# %%
rf = RandomForestRegressor()

# %%
rf.fit(X_train,y_train)

# %%
train_predict = rf.predict(X_train)
test_predict = rf.predict(X_test)

# %%
# Scatter plot for both training and test set predictions
plt.figure(figsize=(12, 6))
plt.scatter(y_train, train_predict, c='blue', label='Training Set', marker='o')
plt.scatter(y_test, test_predict, c='red', label='Test Set', marker='x')
plt.title('Scatter Plot: Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()

# %%
print(f''' The train accuracy : {r2_score(y_train,train_predict)} 
The test accuracy : {r2_score(y_test , test_predict)}''')

# %%



