import os
import pickle
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
# from CVAnalysis import GetCornerFeatures


def is_sift (f):
	return f[0] == 's'

def is_coords (f):
	return f[0] == 'i'

if __name__ == "__main__":

	data_dir = './corner_data'

	#=====[ Step 1: change to correct directory	]=====
	os.chdir (data_dir)

	#=====[ Step 2: load features	]=====
	features_pos = pickle.load (open('features.mat', 'r'))
	features_neg = pickle.load (open('features_neg.mat', 'r'))

	# fp_train, fp_test = features_pos[:300], features_pos[300:]
	# fn_train, fn_test = features_neg[:-300], features_neg[-300:]
	# yp_train, yp_test = np.ones ((fp_train.shape[0],)), np.ones ((fp_test.shape[0],))
	# yn_train, yn_test = np.zeros ((fn_train.shape[0],)), np.zeros ((fn_test.shape[0],))	
	# X_train = np.concatenate ([fp_train, fn_train], 0)
	# y_train = np.concatenate ([yp_train, yn_train])
	# X_test = np.concatenate ([fp_test, fn_test])
	# y_test = np.concatenate ([yp_test, yn_test])
	X_train = np.concatenate ([features_pos, features_neg])
	y_train = np.concatenate ([np.ones ((features_pos.shape[0],)), np.zeros((features_neg.shape[0],))])

	#=====[ Step 3: create/fit/score raw models	]=====
	lr = LogisticRegression ().fit (X_train, y_train)
	dt = DecisionTreeClassifier ().fit (X_train, y_train)
	rf = RandomForestClassifier ().fit (X_train, y_train)
	svm = SVC().fit (X_train, y_train)
	# print "=====[ 	RAW SCORES ]====="
	# print "LogisticRegression: ", lr.score (X_test, y_test)
	# print "DecisionTree: ", dt.score (X_test, y_test)
	# print "RandomForest: ", rf.score (X_test, y_test)		
	# print "SVM: ", svm.score (X_test, y_test)	


	