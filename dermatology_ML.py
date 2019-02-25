#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error


def calculate_vif_(X, thresh=5.0):
	''' Calculate tolerance / VIF and drop redundant features to avoid multicollinearity issues. '''
	variables = list(range(X.shape[1]))
	dropped = True
	while dropped:
	    dropped = False
	    vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
	           for ix in range(X.iloc[:, variables].shape[1])]

	    maxloc = vif.index(max(vif))
	    if max(vif) > thresh:
	        print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
	              '\' at index: ' + str(maxloc))
	        del variables[maxloc]
	        dropped = True

	print('Remaining variables:')
	print(X.columns[variables])
	return X.iloc[:, variables]


def assess_results(test_y, test_predictions):
	''' Calculate RMSE.  Assess percent correct.  Print some basic results. '''
	prediction_mse = mean_squared_error(test_y, test_predictions)
	prediction_rmse = int(round(np.sqrt(prediction_mse),2)*100)

	test_predict_df = pd.DataFrame(data=test_predictions[:,:], columns=label_columns)  # 1st row as the column names
	test_y = test_y.reset_index(drop=True)
	side_by_side = pd.concat([test_y.idxmax(axis=1), test_predict_df.idxmax(axis=1)], axis=1)
	side_by_side.columns=["Answer", "Prediction"]

	num_matching = 0
	total = 0

	for index, row in side_by_side.iterrows():
	    total += 1
	    if row['Answer'] == row['Prediction']:
	        num_matching += 1

	print(str(num_matching) + " / " + str(total) + " - " + str(round(num_matching/total,2)*100) + "%")
	print("RMSE: " + str(prediction_rmse) + "%")
	return



###################################################################
#########              Pre Processing !!!                 #########
###################################################################


data = pd.read_csv('dermatologywheaders.csv')
label_columns = ["psoriasis", "seboreic dermatitis", "lichen planus", "pityriasis rosea", "chronic dermatitis", "pityriasis rubra pilaris"]

# would be able to replace with median with imputer, except that doesn't use NaN with this dataset, uses "?". =/
data_without_unknown_ages = data[data.Age != "?"]
median_age = data_without_unknown_ages["Age"].median()
data["Age"]= data["Age"].str.replace("?", str(median_age))
data["Age"] = pd.to_numeric(data["Age"])


# make dummy values for family history
data["family history"] = data["family history"].astype(str)
data["family history"] = data["family history"].str.replace("0", "Neg Fam Hist")
data["family history"] = data["family history"].str.replace("1", "Pos Fam Hist")
famhist_dummies = pd.get_dummies(data["family history"])
data_famhist_encoded = pd.concat([data, famhist_dummies], axis=1)
data = data_famhist_encoded.drop('family history', axis=1).drop('Neg Fam Hist', axis=1)


# make dummy values for diagnosis.  
data["diagnosis"] = data["diagnosis"].astype(str)
data["diagnosis"] = data["diagnosis"].str.replace("1", label_columns[0])
data["diagnosis"] = data["diagnosis"].str.replace("2", label_columns[1])
data["diagnosis"] = data["diagnosis"].str.replace("3", label_columns[2])
data["diagnosis"] = data["diagnosis"].str.replace("4", label_columns[3])
data["diagnosis"] = data["diagnosis"].str.replace("5", label_columns[4])
data["diagnosis"] = data["diagnosis"].str.replace("6", label_columns[5])
diagnosis_dummies = pd.get_dummies(data["diagnosis"])
data_diagnoses_encoded = pd.concat([data, diagnosis_dummies], axis=1)
data = data_diagnoses_encoded.drop('diagnosis', axis=1)


# scale it.
scaler = preprocessing.MinMaxScaler((0,3))
data[["Age"]] = scaler.fit_transform(data[["Age"]])
data = data.round(3)


# make correlation array and write to csv
correlation_array = pd.DataFrame(data = np.corrcoef(data,rowvar=0)).round(2)
features_column = pd.DataFrame(data = {"Features":data.columns})  
correlations_with_titles = pd.concat([features_column, correlation_array], axis=1)  # add left headers to correlation array
correlations_with_titles.columns = ['Features'] + data.columns.tolist() # add top header row to correlation array
correlations_with_titles.to_csv('correlations2.csv', sep=',', index=False)


# Calculate tolerance / VIF and drop redundant features to avoid multicollinearity issues.
#data_x_trimmed = calculate_vif_(data.drop(label_columns, axis=1), thresh=5.0)  # don't want to trim labels
#data = pd.concat([data_x_trimmed, data.loc[:,label_columns]], axis=1)          # re-concatenate labels after redundant features dropped


# Test Train Split.  Usual stuff.
train_set, test_set = train_test_split(data, test_size=0.2)
train_x = train_set.drop(label_columns, axis=1)
train_y = train_set.loc[:,label_columns]
test_x = test_set.drop(label_columns, axis=1)
test_y = test_set.loc[:,label_columns]


# Write to a CSV file incase you want to inspect it.
train_x.to_csv('train_x.csv', sep=',', index=False)
test_x.to_csv('test_x.csv', sep=',', index=False)




###################################################################
#########              Run Algorithms !!!                 #########
###################################################################


print("\n\n*********** Linear Regression ***********")
lin_reg = LinearRegression()
lin_reg.fit(train_x, train_y)
test_predictions = lin_reg.predict(test_x)
assess_results(test_y, test_predictions)


print("\n\n*********** Logistic Regression -- One vs Rest ***********")
log_reg = OneVsRestClassifier(LogisticRegression())
log_reg.fit(train_x, train_y)
test_predictions = log_reg.predict(test_x)
assess_results(test_y, test_predictions)


print("\n\n*********** Random Forests ***********")
rand_forest = RandomForestClassifier(n_estimators=100)
rand_forest.fit(train_x, train_y)
test_predictions = rand_forest.predict(test_x)
assess_results(test_y, test_predictions)


print("\n\n*********** Artificial Neural Network \"Multi-Layer Perceptron Classifier\" ***********")
neural_net = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
neural_net.fit(train_x, train_y)
test_predictions = neural_net.predict(test_x)
assess_results(test_y, test_predictions)


