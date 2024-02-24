from os import system, name
from concurrent.futures import ProcessPoolExecutor as PPE
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

dataset = pd.read_csv('../Perceptron/weight-height.csv')
c = 0

if __name__ == "__main__" :
    x = dataset[['Height', 'Weight','BMI']].values
    y = dataset['Fat or Not'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    PCT = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    PCT.fit(X_train, y_train)
    y_pred = PCT.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    with open('input_data.txt', 'r') as file:
      lines = file.readlines()

    for line in lines:
     c += 1
     Data = list(map(float, input("Enter height in cm and weight in kg separated by space: ").split()))
     #Data = list(map(float, line.strip().split()))
     BMI = float(round(Data[1]/((Data[0]/100)**2),1))
     user_input = np.array([[Data[0], Data[1], BMI]])
     user_scaled = scaler.transform(user_input)
     Predict = PCT.predict(user_scaled)
     print("Student : ", c , " Predicted :", Predict, " BMI : ", BMI) 