# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values\
## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ISTIN B
RegisterNumber: 212223040068
*/

```
import pandas as pd
data=pd.read_csv("C:/Users/Aadhi/Documents/ML/Placement_Data ex 04.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") #Library for Large Linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![Screenshot 2024-03-12 092725](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/8c80c2f9-e157-4f1b-8384-d2e658ccb857)
![image](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/e86d4886-e799-4bf6-8308-164f87d7e7d0)
![image](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/c5196916-5dac-4bba-be95-68b9b32706d8)
![image](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/7b061b35-baf9-4e25-9432-ded1e2deeb00)
![image](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/209478f5-b241-434c-866b-35cc34dba07f)
![image](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/8a7ccb2d-a053-4f3b-9ec0-1281b5cd5c2f)
![image](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/c3ef3ee1-a773-494f-bc6c-50474d8f2182)
![image](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/36816028-3928-4c20-b0be-1701733160d6)
![image](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/689e295f-10e8-4672-a6f9-2d07782515c2)
![image](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/a6fbdf2b-9678-4285-b3df-a9eeacb95630)
![image](https://github.com/Dharma23012432/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152275002/0167b310-ffe2-47e5-b806-55f1e122f71b)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
