# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import Libraries and Load Data**
   - Use `pandas` to load and inspect the dataset.
   - Check for missing values and data types.

2. **Label Encoding**
   - Convert the categorical feature `Position` to numerical using `LabelEncoder`.

3. **Feature and Target Selection**
   - Input Features: `Position` and `Level`
   - Target Variable: `Salary`

4. **Train-Test Split**
   - Use `train_test_split()` to divide data into training and testing sets (60% test size in this case).

5. **Model Training**
   - Use `DecisionTreeRegressor` from scikit-learn and fit it with the training data.

6. **Prediction and Evaluation**
   - Predict salaries on test data.
   - Evaluate the model using `Mean Squared Error (MSE)` and `R² score`.

7. **New Prediction**
   - Predict salary for a new input `[Position=5, Level=6]`.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Abishek Priyan M
RegisterNumber:  212224240004
*/
```
```py
import pandas as pd

data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
X_test,X_train,Y_test,Y_train=train_test_split(x,y,test_size=0.4,random_state=4)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(X_train,Y_train)
y_pred=dt.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y_test,y_pred)
print("MSE =",mse)

r2=r2_score(Y_test,y_pred)
print("R2 = ",r2)
dt.predict([[5,6]])
```
## Output:
### Dataset Preview
![image](https://github.com/user-attachments/assets/82062341-f09c-49da-98eb-b93f03a40f56)

### df.info()
![image](https://github.com/user-attachments/assets/607f4ebb-de45-4e36-82d0-1b4788f1a401)

### Value of df.isnull().sum()
![image](https://github.com/user-attachments/assets/549d1ebd-8890-4bb1-90c5-5fe05a7f6e54)

### Data after encoding calculating Mean Squared Error
![image](https://github.com/user-attachments/assets/94eba274-6402-4c08-b5cb-667cdbd77268)

### MSE
![image](https://github.com/user-attachments/assets/4f6ad0af-2df0-4c2f-bd67-0305b91ee811)

### R2
![image](https://github.com/user-attachments/assets/124dc19e-e930-49d4-86ca-4b439e7995c1)

### Model prediction with [5,6] as input
![image](https://github.com/user-attachments/assets/f07d971b-fd5f-4529-b811-edf95470fb99)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
