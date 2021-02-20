import numpy as np 
from sklearn import datasets, linear_model, metrics 
import pandas as pd
import datetime
from sklearn.externals import joblib

col_names = ['Sl no.','District Name','Market Name', 'Commodity',	'Variety', 'Grade','Min Price (Rs/Quintal)','Max Price (Rs/Quintal)','modalprice','pricedate']
report = ['PriceJowar.csv','PriceMaize.csv','PriceBajra.csv','PriceWheat.csv']
cropnames = ["Jowar", "Maize", "Bajra", "Wheat"]

from datetime import datetime
cropnames = ["Jowar", "Maize", "Bajra", "Wheat"]

"""for i in range(4):
  cost = pd.read_csv("test.csv", header=None, names=col_names, skiprows=[0])
  cost['pricedate'] = cost['pricedate'].astype('str')
  joblib_model = joblib.load('regression_'+cropnames[i]+'.pkl')
  predict_date = datetime.strptime(cost.pricedate[0], "%Y-%m-%d")

  predict_date = predict_date.toordinal()
  predict_date = np.array(predict_date)
  predict_date = predict_date.reshape(-1,1)
#   print(predict_date)
  lol = joblib_model.predict(predict_date)
  print(lol)"""
cost = pd.read_csv("test.csv", header=None, names=col_names, skiprows=[0])
cost['pricedate'] = cost['pricedate'].astype('str')
cost['Commodity'] = cost['Commodity'].astype('str')
joblib_model = joblib.load('regression_'+cost.Commodity[0]+'.pkl')
predict_date = datetime.strptime(cost.pricedate[0], "%Y-%m-%d")
predict_date = predict_date.toordinal()
predict_date = np.array(predict_date)
predict_date = predict_date.reshape(-1,1)
#   print(predict_date)
lol = joblib_model.predict(predict_date)
lel=str(lol).replace('[','')
print(lel.replace(']',''))
