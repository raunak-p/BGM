import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neural_network import *


def eval(y_true, y_pred):
	return r2_score(y_true, y_pred)

def visualize(data, x, y):
	sns.stripplot(x=x, y=y, data=data)
	plt.show()

####################################################################################################
#										PREPARE DATA
####################################################################################################

sys.stdout.write("Reading data from csv file...")
sys.stdout.flush()
data = pd.read_csv('train.csv')
sys.stdout.write("DONE!\n")

sys.stdout.write("Preparing data...")
sys.stdout.flush() 

# Make categorical
object_labels = data.select_dtypes(include=[object]).columns
feats_catToNum = {}
feats_numToCat = {}

for object_label in object_labels:
	# Also make categories numerical
	all_categories = list(set(data[object_label]))
	feats_catToNum[object_label] = {all_categories[s]: s for s in range(len(all_categories))}
	feats_numToCat[object_label] = {s: all_categories[s] for s in range(len(all_categories))}

	for category in all_categories:
		data.loc[data[object_label] == category, object_label] = feats_catToNum[object_label][category]

	# Type-cast to categorical
	data[object_label] = data[object_label].astype('category', copy=False)

# Make bool
int_labels = data.select_dtypes(include=['int64']).columns
for int_label in int_labels:
	if int_label != 'ID':
		data[int_label] = data[int_label].astype('bool', copy=False)


####################################################################################################
#										FIT MODEL
####################################################################################################

# Split into X and y
X = data.select_dtypes(include=['category', 'bool'])
y = data.select_dtypes(include=['float64'])
sys.stdout.write("DONE!\n")

scores = []
kf = KFold(n_splits = 10, shuffle = False)
for train_index, test_index in kf.split(X):
	X_train, X_val = X.iloc[train_index], X.iloc[test_index]
	y_train, y_val = y.iloc[train_index], y.iloc[test_index]

	ols = ARDRegression()

	ols.fit(X_train, np.ravel(y_train))
	y_pred = ols.predict(X_val)

	scores += [eval(y_val, y_pred)]
	print "Score: ", eval(y_val, y_pred)

print np.mean(scores)


####################################################################################################
#										TEST PREDICTIONS
####################################################################################################


sys.stdout.write("Reading data from test csv file...")
sys.stdout.flush()
test_data = pd.read_csv('test.csv')
sys.stdout.write("DONE!\n")

sys.stdout.write("Preparing data...")
sys.stdout.flush() 

# Make categorical
object_labels = test_data.select_dtypes(include=[object]).columns

for object_label in object_labels:
	# Also make categories numerical
	all_categories = list(set(test_data[object_label]))
	for category in all_categories:
		if category not in feats_catToNum[object_label].keys():
			# handle unseen cats
			test_data.loc[test_data[object_label] == category, object_label] = len(all_categories)
		else:
			test_data.loc[test_data[object_label] == category, object_label] = feats_catToNum[object_label][category]

	# Type-cast to categorical
	test_data[object_label] = test_data[object_label].astype('category', copy=False)

# Make bool
int_labels = test_data.select_dtypes(include=['int64']).columns
for int_label in int_labels:
	if int_label != 'ID':
		test_data[int_label] = test_data[int_label].astype('bool', copy=False)
sys.stdout.write("DONE!\n")


X_test = test_data.select_dtypes(include=['category', 'bool'])
y_test = ols.predict(X_test)

output = pd.DataFrame(test_data['ID'].copy())
y_test = pd.Series(y_test)
output['y'] = y_test

output.to_csv('output.csv', index=False)
