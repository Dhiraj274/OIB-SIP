# %% [markdown]
# # Import necessory Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


# %% [markdown]
# # Load the Data

# %%
# load data into pandas DataFrame
car_data = pd.read_csv("car data.csv")

# %%
car_data.head()

# %% [markdown]
# # EDA

# %%
car_data.shape

# %%
car_data.isnull().sum()

# %%
car_data.describe()

# %%
car_data.duplicated().sum()

# %%
#lets remove duplicated data
car_data.drop_duplicates(inplace=True)
car_data.duplicated().sum()

# %%
car_data.info()

# %% [markdown]
# # Visualize Data

# %%
car_data.hist(bins = 100, figsize =(15,10))
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.scatter(car_data['Year'], car_data['Selling_Price'])
plt.title('Selling Price vs Year')
plt.xlabel('Year')
plt.ylabel('Selling Price')
plt.grid(True)
plt.show

# %%
plt.scatter(car_data['Owner'], car_data['Selling_Price'])
plt.title('Scatter Plot between Selling Price and Owner Type')
plt.xlabel('Owner Type')
plt.ylabel('Selling Price')
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.scatter(car_data['Driven_kms'], car_data['Selling_Price'], alpha=0.5)
plt.title('Scatter Plot: Selling Price vs Driven Kilometers')
plt.xlabel('Driven Kilometers')
plt.ylabel('Selling Price')
plt.grid(True)
plt.show()

# %% [markdown]
# # Data pre-processing

# %% [markdown]
# Label Encoding for Categorical Variables

# %%
# create a copy of Original data
data = car_data.copy()

# %%
le = LabelEncoder()
data["Fuel_Type"] = le.fit_transform(data["Fuel_Type"])
data["Transmission"] = le.fit_transform((data["Transmission"]))
data["Selling_type"] = le.fit_transform(data["Selling_type"])

# %%
data.head()

# %% [markdown]
# Feature Engineering

# %%
# Now Create new columns name as car_age
# It is very useful feature because as much as age of car increase then selling price is decreasing
current_year = 2024
data["car_age"] = current_year - data["Year"]

# %%
# Now drop year columns because "car_age" columns have similar data
new_data = data.drop('Year',axis=1)

# %%
new_data.head()

# %%
new_data = new_data.drop('Car_Name', axis=1)

# %% [markdown]
# Feature Scaling

# %%
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(new_data)

# %%
# Convert scaled data back to DataFrame (if needed)
data_scaled = pd.DataFrame(scaled_data, columns=new_data.columns)

# %%
data_scaled

# %% [markdown]
# Correlation between data

# %%
corr_matrix = data_scaled.corr()
corr_matrix

# %%
# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# # Train Machine Learning Model

# %% [markdown]
# Split the data

# %%
X = data_scaled.drop('Selling_Price', axis=1)
y = data_scaled['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)


# %% [markdown]
# Choose and train the model

# %%
model = LinearRegression()

# %%
model.fit(X_train,y_train)

# %% [markdown]
# Make prediction

# %%
# Make predictions on the training set
train_predictions = model.predict(X_train)

# %%
# Make predictions on the testing set
test_predictions = model.predict(X_test)

# %%
# Visualization of actual vs. predicted values with color differentiation
plt.figure(figsize=(10, 6))

# Scatter plot for actual values (in blue)
plt.scatter(y_test, y_test, color='blue', label='Actual Selling Price')

# Scatter plot for predicted values (in red)
plt.scatter(y_test, test_predictions, color='red', label='Predicted Selling Price')

plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs. Predicted Selling Price with Color Differentiation')
plt.legend()
plt.show()

# %% [markdown]
# # Evaluate model

# %%
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")



