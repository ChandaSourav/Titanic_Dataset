import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import csv

train_data =pd.read_csv("train.csv")
test_data =pd.read_csv("test.csv")

y = train_data['Survived']

features=['Pclass', 'Sex','SibSp', 'Parch']

x_train=pd.get_dummies(train_data[features])

x_test=pd.get_dummies(test_data[features])

model = LogisticRegression(solver='liblinear', C=10.0, random_state=1)

model.fit(x_train, y)

prediction = model.predict(x_test)

print(prediction)
