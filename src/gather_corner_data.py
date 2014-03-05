import os
import pickle
import numpy as np

def get_sift_name (i):
	return 'sift_desc_' + str(i) + '.obj'

def get_coords_name(i):
	return 'coords_desc_' + str(i) + '.obj'

if __name__ == '__main__':

	os.chdir ('./corner_data')

	sift_filenames = [	'sift_desc_1.obj', 
						'sift_desc_2.obj',
						'sift_desc_3.obj',
						'sift_desc_4.obj'
					]

	sift_features = []
	for f in sift_filenames:
		feat = pickle.load (open(f, 'r'))
		sift_features.append (feat)

	features = np.concatenate (sift_features, 0)


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