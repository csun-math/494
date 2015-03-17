import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import random

data = pd.read_csv('math-redacted.csv')

data['Graduated'].replace(to_replace=('N','Y'),value=(0,1),inplace=True)

X_train,X_test,y_train,y_test = train_test_split(data['Total_Cr'],data['Graduated'],test_size=0.2,random_state = int(random()*100))

print "\n\nUsing test_size=0.2 of total data"

X_train = X_train.reshape(len(X_train),1)
X_test = X_test.reshape(len(X_test),1)

classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)

print "\n\nLogistic regression coefficients: "
print classifier.coef_[0]

#graduated_proba = classifier.predict_proba(data[list(['Total_Cr'])])

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

print "\nConfusion Matrix for test data"
print confusion_matrix

plt.matshow(confusion_matrix)
plt.title("Test Data Set Confusion Matrix")
plt.colorbar()
plt.ylabel("true label")
plt.xlabel("predicted label")
plt.show()

print "\n\nClassification report"
print metrics.classification_report(y_test,y_pred)

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test,y_pred_proba[:,1])
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('True Positive Rate TP/(TP+FN) aka Recall')
plt.xlabel('False Positive Rate FP/(FP+TN) aka Fall-Out')
plt.show()

###############################################################

print "\n\n\n\n\n\n\nNow to try it on all math classes/columns..."
classData = data[data.columns[6:]]
X_train,X_test,y_train,y_test = train_test_split(classData,data['Graduated'],test_size=0.2,random_state = int(random()*100))
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)

print "\n\nLogistic regression coefficients: "
print classifier.coef_[0]

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

print "\nConfusion Matrix for test data"
print confusion_matrix

plt.matshow(confusion_matrix)
plt.title("Test Data Set Confusion Matrix")
plt.colorbar()
plt.ylabel("true label")
plt.xlabel("predicted label")
plt.show()

print "\n\nClassification report"
print metrics.classification_report(y_test,y_pred)

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test,y_pred_proba[:,1])
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('True Positive Rate TP/(TP+FN) aka Recall')
plt.xlabel('False Positive Rate FP/(FP+TN) aka Fall-Out')
plt.show()

print "\n\n\nHere are classes with coefficients > 0.5"


classesCoefs = zip(classifier.coef_[0],data.columns[6:].values)
classesCoefs.sort()

print "\n\n\nZipping up class names with their corresponding coefficients and then printing the first then and last then..."
print '\nFirst Ten'
print classesCoefs[:10]
print '\n\n'
print '\nLast Ten'
print classesCoefs[-10:]

print "\n\n\nHere's everything."
print classesCoefs


#####################################################

pca = PCA(n_components = 2)
reduced_X = pca.fit_transform(classData)
red_x,red_y = [],[]
blue_x,blue_y = [],[]
for i in range(len(reduced_X)):
	if data['Graduated'][i]==0:
		red_x.append(reduced_X[i][0])
		red_y.append(reduced_X[i][1])
	if data['Graduated'][i]==1:
		blue_x.append(reduced_X[i][0])
		blue_y.append(reduced_X[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b')
plt.show()