# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Print the present data and placement data and salary data.
3. Using logistic regression find the predicted values of accuracy confusion matrices.
4. Display the results.

## Program and Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Shree Lekha.S
RegisterNumber: 212223110052
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Placement_Data.csv')
dataset
```

![image](https://github.com/user-attachments/assets/4794de60-4dd2-4d44-8952-0bda85310b21)


```
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')

dataset.dtypes

```

![image](https://github.com/user-attachments/assets/6497db18-f024-4f14-b769-79786f522830)


```
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

Y
```

![image](https://github.com/user-attachments/assets/349a6dd0-a37e-467c-8d88-d1554a25abf9)


```
theta = np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def gradient_descent(theta, X, y, alpha, num_iterations):
  m = len(y)
  for i in range(num_iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h - y) / m
    theta -= alpha * gradient
  return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

def predict(theta, X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5,1, 0)
  return y_pred

y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy: ", accuracy)

```

![image](https://github.com/user-attachments/assets/076bf539-0f79-4a14-8b61-f4304060ccc6)


```
print(y_pred)

```

![image](https://github.com/user-attachments/assets/ec38664a-ed4f-4e58-a9df-
ae069063de40)


```
print(Y)
```

![image](https://github.com/user-attachments/assets/b96bf4f5-fb8c-4652-9716-9240739238fd)


```
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
```

![image](https://github.com/user-attachments/assets/c93b7ec6-8cab-4985-9c3b-34be14997d0b)

```
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
```

![image](https://github.com/user-attachments/assets/2a881f31-f6db-4651-a078-078837ab34b3)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
