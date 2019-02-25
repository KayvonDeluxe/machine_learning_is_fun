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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


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
	''' Assess percent correct.  Print some basic results. '''
	test_predict_df = pd.DataFrame(data=test_predictions[:,:], columns=label_columns)  # 1st row as the column names
	test_y = test_y.reset_index(drop=True)

	test_y_argmax = test_y.idxmax(axis=1)
	test_predict_argmax = test_predict_df.idxmax(axis=1)
	side_by_side = pd.concat([test_y_argmax, test_predict_argmax], axis=1)
	side_by_side.columns=["Answer", "Prediction"]

	num_matching = 0
	total = 0
	for index, row in side_by_side.iterrows():
	    total += 1
	    if row['Answer'] == row['Prediction']:
	        num_matching += 1

	print("Accuracy: " + str(num_matching) + " / " + str(total) + " - " + str(round(num_matching/total,2)*100) + "%")
	print("\nConfusion Matrix:\n")
	print(confusion_matrix(test_y_argmax, test_predict_argmax))
	print("\n\nClassification Report:\n")
	print(classification_report(test_y_argmax, test_predict_argmax))
	return



###################################################################
#########              Pre Processing !!!                 #########
###################################################################


data = pd.read_csv('dermatologywheaders.csv')
label_columns = ["psoriasis", "seboreic dermatitis", "lichen planus", "pityriasis rosea", "chronic dermatitis", "pityriasis rubra pilaris"]


data.describe()

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


print("\n")
print("*****************************************")
print("*********** Linear Regression ***********")
print("*****************************************")
lin_reg = LinearRegression()
lin_reg.fit(train_x, train_y)
test_predictions = lin_reg.predict(test_x)
assess_results(test_y, test_predictions)



print("\n")
print("**********************************************************")
print("*********** Logistic Regression -- One vs Rest ***********")
print("**********************************************************")
log_reg = OneVsRestClassifier(LogisticRegression())
log_reg.fit(train_x, train_y)
test_predictions = log_reg.predict(test_x)
assess_results(test_y, test_predictions)



print("\n")
print("**************************************")
print("*********** Random Forests ***********")
print("**************************************")
rand_forest = RandomForestClassifier(n_estimators=100)
rand_forest.fit(train_x, train_y)
test_predictions = rand_forest.predict(test_x)
assess_results(test_y, test_predictions)



print("\n")
print("***************************************************************************************")
print("*********** Artificial Neural Network \"Multi-Layer Perceptron Classifier\" ***********")
print("***************************************************************************************")
neural_net = MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter=1500)
neural_net.fit(train_x, train_y)
test_predictions = neural_net.predict(test_x)
assess_results(test_y, test_predictions)



print("\n")
print("*******************************************")
print("*********** K Nearest Neighbors ***********")
print("*******************************************")
k_near_neighbor = KNeighborsClassifier(n_neighbors=3)
k_near_neighbor.fit(train_x, train_y)
test_predictions = k_near_neighbor.predict(test_x)
assess_results(test_y, test_predictions)



print("\n")
print("**********************************************")
print("*********** Support Vector Machine ***********")
print("**********************************************")
support_vect_mach = SVC(kernel='poly')
train_y_stacked = train_y.stack()
train_y_reverse_dummy = pd.Series(pd.Categorical(train_y_stacked[train_y_stacked!=0].index.get_level_values(1)))

test_y_stacked = test_y.stack()
test_y_reverse_dummy = pd.Series(pd.Categorical(test_y_stacked[test_y_stacked!=0].index.get_level_values(1)))

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']}
support_vect_mach = GridSearchCV(SVC(class_weight='balanced'), param_grid)
support_vect_mach = support_vect_mach.fit(train_x, train_y_reverse_dummy)
#print("Best estimator found by grid search:", support_vect_mach.best_estimator_)

support_vect_mach_best_params = SVC(**support_vect_mach.best_params_)
support_vect_mach_best_params = support_vect_mach_best_params.fit(train_x, train_y_reverse_dummy)
test_predictions = support_vect_mach_best_params.predict(test_x)
num_matching = 0
total = 0
for p, t in zip(test_predictions, test_y_reverse_dummy):
	total += 1
	if p == t:
		num_matching += 1

print("Accuracy: " + str(num_matching) + " / " + str(total) + " - " + str(round(num_matching/total,2)*100) + "%")
print("\nConfusion Matrix: ")
print(confusion_matrix(test_y_reverse_dummy, test_predictions))
print("\n\nClassification Report: ")
print(classification_report(test_y_reverse_dummy, test_predictions))

