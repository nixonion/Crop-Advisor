import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import classification_report
#import pickle'
#from sklearn.externals import joblib'


col_names = ['Crop', 'Temperature', 'Water', 'pH', 'Soil Type', 'Area', 'Yield', 'Growing Season', 'Label']
crop = pd.read_csv("Outcome.csv", header=None, names=col_names, skiprows=[0])
predict = pd.read_csv("predict.csv", header=None, names=col_names, skiprows=[0])

#print(predict)


feature_cols = ['Temperature', 'Water', 'pH', 'Area', 'Yield']
X = crop[feature_cols] 
y = crop.Label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
#print(classification_report(y_test,y_pred))

#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

####################### PREDICTION STARTS HERE#####################

X_predict = predict[feature_cols]

y_predict = clf.predict(X_predict)

print(y_predict)





