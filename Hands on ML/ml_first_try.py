print("Hello, Machine Learning World!")

#import pandas as pd
#import matplotlib.pyplot as plt

# load the dataset and found any strong rules or patterns inside the data

#df = pd.read_csv("F:/Machine Learning Hands On/Hands on ML/student_details.csv")
#print(df.head())

#plt.scatter(df["hours_studied"], df["attendance"], c=df["pass"])
#plt.xlabel("Hours studied")
#plt.ylabel("Attendance")
#plt.title("Student Data")
#plt.show()

# validate my founded rule in scatter plot and its got 0.86 its a hudge sign my identified rule and pattern works.

# import pandas as pd

# df = pd.read_csv("F:/Machine Learning Hands On/Hands on ML/student_details.csv")

# def rule(row):
#     return 1 if row["hours_studied"] > 4 and row["attendance"] > 65 else 0

# df["rule_pred"] = df.apply(rule, axis=1)

#     accuracy = (df["rule_pred"] == df["pass"]).mean()
# print("Rule Accuracy:", accuracy)

# print(df[["hours_studied","attendance"]].mean())
#  print(df[["hours_studied","attendance"]].std())

# pick the weights (what feature how much matters to the prediction outcome to prove my actual hand written rule) and train it using logistic regreression formula.

import numpy as np
import pandas as pd

df = pd.read_csv("F:/Machine Learning Hands On/Hands on ML/student_details.csv")

#X = df[["hours_studied","attendance"]].values
#y = df["pass"].values

# normalize
#X = (X - X.mean(axis=0)) / X.std(axis=0)


#def sigmoid(z):
    #return 1 / (1 + np.exp(-z))


# w = np.array([0.0, 0.0])   # w1, w2
#b = 0.0
#lr = 0.1


# for i in range(1000):
    # z = X @ w + b
    # y_pred = sigmoid(z)

    #error = y_pred - y

    #dw = (X.T @ error) / len(X)
    #db = error.mean()

    #w -= lr * dw
    #b -= lr * db

    # if i % 100 == 0:
        # loss = -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
        # print(i, "loss:", loss)

#print("Weights (to predict pass or fail which feature matters more or less (hours, attendance)):", w)
#print("Bias:", b)



#import matplotlib.pyplot as plt

# plt.scatter(df["hours_studied"], df["attendance"], c=df["pass"], cmap='bwr', edgecolor='k')
# plt.xlabel("Hours studied")
# plt.ylabel("Attendance")
# plt.title("Student Data with Decision Boundary")

# plot boundary (rough approximation)
# z = w1*hours_norm + w2*attendance_norm + b = 0 -> convert to original scale
# hours_range = np.linspace(df["hours_studied"].min(), df["hours_studied"].max(), 100)
# attendance_boundary = -(w[0]*(hours_range - df[["hours_studied","attendance"]].mean()[0])/df[["hours_studied","attendance"]].std()[0] + b)/w[1]*df[["hours_studied","attendance"]].std()[1] + df[["hours_studied","attendance"]].mean()[1]

# plt.plot(hours_range, attendance_boundary, 'g--', label="Decision Boundary")
# plt.legend()
# plt.show()



# testing results


# raw features
X_raw = df[["hours_studied","attendance"]].values
y = df["pass"].values

X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)

# trained weights & bias from previous training
w = np.array([3.30638866, 3.1966561])
b = 1.2598256739265938

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_student(hours, attendance):
    x = np.array([hours, attendance])
    # normalize using mean/std from raw X
    x_norm = (x - X_mean) / X_std
    z = x_norm @ w + b
    prob = sigmoid(z)
    pred = 1 if prob > 0.5 else 0
    return pred, prob

# test
students = [(5, 70), (3, 60), (6, 50), (2, 30)]
for hours, attendance in students:
    pred, prob = predict_student(hours, attendance)
    print(f"Hours: {hours}, Attendance: {attendance} -> Prediction: {'Pass' if pred else 'Fail'}, Probability: {prob:.2f}")



