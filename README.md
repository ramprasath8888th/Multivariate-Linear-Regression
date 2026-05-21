# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
Import the required libraries such as NumPy, Pandas, and the Linear Regression model from Scikit-learn.
### Step2
Load the dataset and separate it into feature variables (X) and target variable (y).
### Step3
Split the dataset into training and testing sets using the train_test_split() function.
### Step4

Create a Multivariate Linear Regression model and train it using the training data with the fit() method.
### Step5
Use the trained model to predict the output for the test data and evaluate the model performance by displaying coefficients, intercept, and accuracy score.

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Load diabetes dataset
diabetes = datasets.load_diabetes()

# Defining feature matrix (X) and response vector (y)
x = diabetes.data
y = diabetes.target

# Splitting X and y into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=1
)

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(x_train, y_train)

# Regression coefficients
print("Coefficients:")
print(reg.coef_)

# Variance score (R²)
print("Variance score: {:.2f}".format(reg.score(x_test, y_test)))

# Set plot style
plt.style.use("fivethirtyeight")

# Plot residual errors in training data
plt.scatter(
    reg.predict(x_train),
    reg.predict(x_train) - y_train,
    color='green',
    s=10,
    label='Training Data'
)

# Plot residual errors in test data
plt.scatter(
    reg.predict(x_test),
    reg.predict(x_test) - y_test,
    color='blue',
    s=10,
    label='Test Data'
)

# Plot line for zero residual error
plt.hlines(
    y=0,
    xmin=min(reg.predict(x_test)),
    xmax=max(reg.predict(x_test)),
    linewidth=2
)

# Plot legend and title
plt.legend(loc='upper right')
plt.title('Residual Errors')

# Axis labels
plt.xlabel('Predicted Values')
plt.ylabel('Residual Errors')

# Show plot
plt.show()





```
## Output:

<img width="948" height="722" alt="image" src="https://github.com/user-attachments/assets/59a6ceb5-dd5a-4456-b7cb-7a1fab45789b" />



## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
