# Load libraries
# Import libraries
import array
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Purpose of this lab is to predict what conditions could lead to forest fires
#Months -> Nov(1)-Oct(12) By seasons
#Days -> Sun(1)-Sat(7)
#Area -> No 0-0.01
#     -> Yes 0.1-1100

col_names = ['x', 'y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind','rain','area']

# load dataset
fires = pd.read_csv("forestfires_cleaned.csv")
fires.columns=col_names

#bin area
bins = [0,0.1,1100] #Yes/No bins
labels = [0,1]
fires.area = pd.cut(fires['area'], include_lowest=True, bins=bins, labels=labels)

# Normalizing data to 100
fires.FFMC = fires.FFMC/10
fires.DMC = fires.DMC/30
fires.DC = fires.DC/90
fires.temp = fires.temp/3
fires.RH = fires.RH/10


#split data into feature and target variables
feature_cols = ['x', 'y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind','rain']
X = fires[feature_cols] # Features
a = fires.area # Target variable

# Split dataset into training set and test set
X_train, X_test, a_train, a_test = train_test_split(X, a, test_size=0.2, random_state=1) # 80% training and 20% test

# Create Decision Tree classifer object & Train
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf = clf.fit(X_train,a_train)

#Predict the response for test dataset
a_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy (DT) :",metrics.accuracy_score(a_test, a_pred))
print(confusion_matrix(a_test, a_pred))

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('DT_YesNO_.2.png')
Image(graph.create_png())


### k-Nearest Neighbor

from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 7)

# Fit the classifier to the data
knn.fit(X_train,a_train)

#check accuracy of our model on the test data
score = knn.score(X_test, a_test)

knn_pred = knn.predict(X_test)

print("Accuracy (kNN) :",metrics.accuracy_score(a_test, knn_pred))

from sklearn.model_selection import cross_val_score

#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=7)

#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X_test, a_test, cv=5)

#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:',format(np.mean(cv_scores)))
