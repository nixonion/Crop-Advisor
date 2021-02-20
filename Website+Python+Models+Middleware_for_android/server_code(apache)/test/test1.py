print("nabeel")
import sys
sys.path.append('/home/ubuntu/.local/lib/python3.6/site-packages')
print(sys.path)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import classification_report
import pickle
from sklearn.externals import joblib


# clf = DecisionTreeClassifier()
# clf = clf.fit(X_train,y_train)

cropnames = ["Jowar", "Maize", "Bajra", "Sugarcane", "Wheat"]
col_names = ['Crop', 'Temperature', 'Water', 'pH', 'Soil Type', 'Area', 'Yield', 'Growing Season', 'Label']
feature_cols = ['Temperature', 'Water', 'pH']

predict = pd.read_csv("Input.csv", header=None, names=col_names, skiprows=[0])

for i in range(5):
	filename = 'finalized_model_'+cropnames[i]+'.sav'
	loaded_model = pickle.load(open(filename, 'rb')) 
	X_predict = predict[feature_cols]
	#print(X_predict)
	y_predict = loaded_model.predict(X_predict)
	if "1" in str(y_predict):
		print(cropnames[i])
print("done")
