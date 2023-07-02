import pandas as pd

# Import the dataset
df = pd.read_csv('Breach_level_index.csv')

# Remove irrelevant variables
df = df.drop(['Breach ID', 'Entity'], axis=1)

# Check for missing or erroneous data
print(df.isnull().sum())

# Impute missing data
df = df.fillna(df.mean())

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

import matplotlib.pyplot as plt

# Descriptive statistics
print(df.describe())

# Histogram of records lost
plt.hist(df['Records Lost'])
plt.xlabel('Records Lost')
plt.ylabel('Frequency')
plt.title('Distribution of Records Lost')
plt.show()

# Box plot of records lost by industry
plt.boxplot(df['Records Lost'], by=df['Industry'])
plt.xlabel('Industry')
plt.ylabel('Records Lost')
plt.title('Records Lost by Industry')
plt.xticks(rotation=90)
plt.show()

# Create new variable for breach size
df['Breach Size'] = df['Records Lost'] * df['Severity']

# Normalize the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_normalized[:,:-1], df_normalized[:,-1], test_size=0.3)

# Develop a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the performance of the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Forecast future trends using linear regression
from sklearn.linear_model import LinearRegression

X = df[['Year']]
y = df['Records Lost']
model = LinearRegression()
model.fit(X, y)
future_years = pd.DataFrame({'Year': [2022, 2023, 2024]})
future_predictions = model.predict(future_years)
print('Future Predictions:', future_predictions)

# Summarize the findings of the analysis
print('The average number of records lost in a data breach is', df['Records Lost'].mean())

# Visualize the trends and insights discovered
plt.scatter(df['Year'], df['Records Lost'])
plt.plot(future_years, future_predictions, color='red')
plt.xlabel('Year')
plt.ylabel('Records Lost')
plt.title('Trends in Data Breaches')
plt.show()

# Provide recommendations for improving cybersecurity measures
print('To improve cybersecurity measures, companies should consider implementing stronger password policies, increasing employee training on security best practices, and investing in advanced threat detection technologies.')
