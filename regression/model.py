import numpy as np 
from sklearn import datasets, linear_model, metrics 
import pandas as pd
import datetime
from sklearn.externals import joblib

col_names = ['Sl no.','District Name','Market Name', 'Commodity',	'Variety', 'Grade','Min Price (Rs/Quintal)','Max Price (Rs/Quintal)','modalprice','pricedate']
report = ['PriceJowar.csv','PriceMaize.csv','PriceBajra.csv','PriceWheat.csv']
cropnames = ["Jowar", "Maize", "Bajra", "Wheat"]

for i in range(4):

  cost = pd.read_csv(report[i], header=None, names=col_names, skiprows=[0])
  feature_cols = ['modalprice']

  converted_date = []

  y = np.array(cost[feature_cols])
  for j in range(len(cost.pricedate)):
    
    new_date = datetime.datetime.strptime(cost.pricedate[j], "%d-%b-%y")
    converted_date.append(new_date.toordinal())

  X = np.array(converted_date)
  # splitting X and y into training and testing sets 
  from sklearn.model_selection import train_test_split 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                            random_state=1) 

  X_train= X_train.reshape(-1, 1)
  y_train= y_train.reshape(-1, 1)
  X_test= X_test.reshape(-1, 1)
  y_test= y_test.reshape(-1, 1)

  # create linear regression object 
  reg = linear_model.LinearRegression() 

  # train the model using the training sets 
  reg.fit(X_train, y_train) 

  # regression coefficients 
  print('Coefficients: \n', reg.coef_)

  # variance score: 1 means perfect prediction 
  print('Variance score: {}'.format(reg.score(X_test, y_test))) 
  joblib_file = 'regression_'+cropnames[i]+'.pkl'  
  joblib.dump(reg, joblib_file)