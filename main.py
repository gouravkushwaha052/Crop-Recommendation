import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Initialize Tkinter window
win = tk.Tk()

# Load and inspect dataset
df = pd.read_csv(r"D:\LNCT\hackathon\sistech\GUI\crop_recommendation.csv")

# Select features and target
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']  # Assuming 'label2' was a mistake

# Initialize accuracy and model lists
acc = []
model = []

# Splitting into train and test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

# Decision Tree Classifier
DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
DecisionTree.fit(Xtrain, Ytrain)
predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("Decision Tree's Accuracy is: ", x*100)
print(classification_report(Ytest, predicted_values))

# Save Decision Tree model
with open(r"D:\LNCT\hackathon\sistech\GUI\DecisionTree.pkl", 'wb') as DT_pkl:
    pickle.dump(DecisionTree, DT_pkl)

# Naive Bayes Classifier
NaiveBayes = GaussianNB()
NaiveBayes.fit(Xtrain, Ytrain)
predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x)
print(classification_report(Ytest, predicted_values))

# Save Naive Bayes model
with open(r"D:\LNCT\hackathon\sistech\GUI\NaiveBayes.pkl", 'wb') as NB_pkl:
    pickle.dump(NaiveBayes, NB_pkl)

# SVM Classifier
norm = MinMaxScaler().fit(Xtrain)
X_train_norm = norm.transform(Xtrain)
X_test_norm = norm.transform(Xtest)
SVM = SVC(C=1)
SVM.fit(X_train_norm, Ytrain)
predicted_values = SVM.predict(X_test_norm)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x)
print(classification_report(Ytest, predicted_values))

# Save SVM model
with open(r"D:\LNCT\hackathon\sistech\GUI\SVMClassifier.pkl", 'wb') as SVM_pkl:
    pickle.dump(SVM, SVM_pkl)

# Random Forest Classifier
RF = RandomForestClassifier(n_estimators=16, random_state=2)
RF.fit(Xtrain, Ytrain)
predicted_values = RF.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Random Forest')
print("Random Forest's Accuracy is: ", x)
print(classification_report(Ytest, predicted_values))

# Save Random Forest model
with open(r"D:\LNCT\hackathon\sistech\GUI\RandomForest.pkl", 'wb') as RF_pkl:
    pickle.dump(RF, RF_pkl)

# K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train_norm, Ytrain)
predicted_values = knn.predict(X_test_norm)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('KNN')
print("KNN's Accuracy is: ", x)
print(classification_report(Ytest, predicted_values))

# Save KNN model
with open(r"D:\LNCT\hackathon\sistech\GUI\KNN.pkl", 'wb') as KNN_pkl:
    pickle.dump(knn, KNN_pkl)

# Plotting accuracy comparison
plt.figure(figsize=[10, 5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=acc, y=model, palette='dark')
plt.show()

# Create GUI input fields and labels
lb1 = tk.Label(win, text="N Value").pack()
l1 = tk.Entry(win)
l1.pack()

lb2 = tk.Label(win, text="P Value").pack()
l2 = tk.Entry(win)
l2.pack()

lb3 = tk.Label(win, text="K Value").pack()
l3 = tk.Entry(win)
l3.pack()

lb4 = tk.Label(win, text="Temperature Value").pack()
l4 = tk.Entry(win)
l4.pack()

lb5 = tk.Label(win, text="Humidity Value").pack()
l5 = tk.Entry(win)
l5.pack()

lb6 = tk.Label(win, text="PH Value").pack()
l6 = tk.Entry(win)
l6.pack()

lb7 = tk.Label(win, text="Rainfall Value").pack()
l7 = tk.Entry(win)
l7.pack()

# Prediction function
def outs():
    list0 = []
    list1 = [0, 0, 0, 0, 0, 0, 0]
    list0 = [list1]
    list1[0] = float(l1.get())
    list1[1] = float(l2.get())
    list1[2] = float(l3.get())
    list1[3] = float(l4.get())
    list1[4] = float(l5.get())
    list1[5] = float(l6.get())
    list1[6] = float(l7.get())

    pre = RF.predict(list0)
    print("Prediction:", pre)
    tk.Label(win, text="The best crop is", height=2, width=20, font=("arial", 20, "italic")).pack()
    tk.Label(win, text=pre[0], height=2, width=30, font=("arial", 20, "bold")).pack()

# Create output button
b1 = tk.Button(win, text="Output", command=outs)
b1.pack()

# Run Tkinter loop
win.mainloop()
