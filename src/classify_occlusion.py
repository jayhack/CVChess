import os
import pickle
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

if __name__ == "__main__":

	data_dir = '../data/p1'

	#=====[ Step 1: change to correct directory	]=====
	os.chdir (data_dir)

	#=====[ Step 2: load in the data	]=====
	X, y = pickle.load (open('features.mat', 'r')), pickle.load (open('labels.mat', 'r'))

	#=====[ Step 3: split for train/test	]=====
	# X_train, X_test = X[:-64], X[-64:]
	# y_train, y_test = y[:-64], y[-64:]	
	X_train, X_test = X[:700], X[700:]
	y_train, y_test = y[:700], y[700:]	

	#=====[ Step 3: create/fit/score raw models	]=====
	lr = LogisticRegression ().fit (X_train, y_train)
	dt = DecisionTreeClassifier ().fit (X_train, y_train)
	rf = RandomForestClassifier ().fit (X_train, y_train)
	svm = SVC().fit (X_train, y_train)
	print "=====[ 	RAW SCORES ]====="
	print "LogisticRegression: ", lr.score (X_test, y_test)
	print "DecisionTree: ", dt.score (X_test, y_test)
	print "RandomForest: ", rf.score (X_test, y_test)		
	print "SVM: ", svm.score (X_test, y_test)			


	#=====[ Step 4: get only bottom rows	]=====
	X_br, y_br = [], []
	for i in range(14):
		start = i*64 + 56
		end = i*64 + 64
		X_br.append (X[start:end])
		y_br.append (y[start:end])
	X_br = np.concatenate (X_br, 0)
	y_br = np.concatenate (y_br)


	#=====[ Step 5: create/fit/score pca models	]=====
	# pca = PCA (n_components=10)
	# X = pca.fit_transform(X)
	# X_train, X_test = X[:700], X[700:]
	# y_train, y_test = y[:700], y[700:]
	# lr = LogisticRegression ().fit (X_train, y_train)
	# dt = DecisionTreeClassifier ().fit (X_train, y_train)
	# rf = RandomForestClassifier ().fit (X_train, y_train)
	# svm = SVC().fit (X_train, y_train)
	# print "=====[ 	WITH PCA TO 5 ]====="
	# print "LogisticRegression: ", lr.score (X_test, y_test)
	# print "DecisionTree: ", dt.score (X_test, y_test)
	# print "RandomForest: ", rf.score (X_test, y_test)		
	# print "SVM: ", svm.score (X_test, y_test)	
