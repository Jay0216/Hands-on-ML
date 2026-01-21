print("This Model For Predic Heart Health Risk")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('F:/Machine Learning Hands On/Hands on ML/Second Day/Health_Data.csv')
print(data.head())


# first observation
# plt.scatter(data["age"], data["cholesterol"], c=data["risk"])
# plt.xlabel("Age")
# plt.ylabel("Cholesterol")
# plt.title("Heart Health Data")  

# second observation
#plt.scatter(data["bmi"], data["blood_pressure"], c=data["risk"])
#plt.xlabel("BMI")
# plt.ylabel("Blood_Pressure")
#plt.title("Heart Health Data")  

#plt.show()


# validate that observed patterns via numbers

def rule_1(row):
    return 1 if row["age"] > 50 and row["cholesterol"] > 240 else 0

def rule_2(row):
    return 1 if row["bmi"] > 30 and row["blood_pressure"] > 140 else 0

# most important rule and most important features for prediction
def rule_3(row):
    return 1 if row["cholesterol"] > 240 and row["blood_pressure"] > 140 else 0


data["rule1_pred"] = data.apply(rule_1, axis=1)
data["rule2_pred"] = data.apply(rule_2, axis=1)
data["rule3_pred"] = data.apply(rule_3, axis=1)

accuracy1 = (data["rule1_pred"] == data["risk"]).mean()
accuracy2 = (data["rule2_pred"] == data["risk"]).mean() 
accuracy3 = (data["rule3_pred"] == data["risk"]).mean()

actualmean = data[["age","bmi", "blood_pressure", "cholesterol"]].mean()

print("Actual Mean Values:")
print(actualmean)

actualstd = data[["age","bmi", "blood_pressure", "cholesterol"]].std()
print("Actual Standard Deviation Values:")
print(actualstd)

print("Rule 1 Accuracy:", accuracy1)
print("Rule 2 Accuracy:", accuracy2)
print("Rule 3 Accuracy:", accuracy3)



# now train logistic regression model to find trained weights for each feature

#X = data[["age","bmi","blood_pressure","cholesterol"]].values
#y = data["risk"].values


#X_mean = X.mean(axis=0)
#X_std = X.std(axis=0)

#X = (X - X_mean) / X_std

#def sigmoid(z):
    #return 1 / (1 + np.exp(-z))


#w = np.array([0.0, 0.0, 0.0, 0.0])   # age, bmi, bp, chol
#b = 0.0
#lr = 0.1


#for i in range(1000):
    #z = X @ w + b
    #y_pred = sigmoid(z)

    #error = y_pred - y

    #dw = (X.T @ error) / len(X)
    #db = error.mean()

    #w -= lr * dw
    #b -= lr * db

    # if i % 100 == 0:
        # loss = -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
        # print(i, "loss:", loss)

# print("Trained Weights (feature importance for heart health risk - age, bmi, blood_pressure, cholesterol):", w)
# print("Bias:", b)


# test the model with new data



# Example: your dataset stats (mean & std for normalization)
X_mean = np.array([40.16, 29.88, 140.72, 241.40])
X_std  = np.array([9.03, 4.19, 18.05, 36.07])

w = np.array([2.08352134, 2.26583022, 1.91822183, 1.97300018])
b = 1.1294684493360876

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_heart_risk(age, bmi, bp, chol):
    x = np.array([age, bmi, bp, chol])
    # normalize using dataset mean/std
    x_norm = (x - X_mean) / X_std
    z = x_norm @ w + b
    prob = sigmoid(z)
    risk = 1 if prob > 0.5 else 0
    return risk, prob


# Test different people
people = [
    (45, 28, 135, 235),  # older, medium risk
    (32, 22, 110, 180),  # young, healthy
    (55, 30, 150, 260),  # high risk
    (29, 20, 105, 170)   # very low risk
]

for age, bmi, bp, chol in people:
    risk, prob = predict_heart_risk(age, bmi, bp, chol)
    print(f"Age: {age}, BMI: {bmi}, BP: {bp}, Chol: {chol} -> Prediction: {'High Risk' if risk else 'Healthy'}, Probability: {prob:.2f}")

