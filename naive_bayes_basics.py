# Can you predict if a candy is chocolate or not based on its other features?

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


candy_data = pd.read_csv(r'/home/ram/Downloads/kaggle/candy-data.csv')
print(candy_data.info())
print(candy_data.isnull().sum())

x=candy_data.drop(columns=["chocolate","competitorname"])
y=candy_data["chocolate"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
# gaussian NB: if features follow a normal distribution.
print("gaussian NB:---------------")
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pr=gnb.predict(x_test)
#accu=gnb.score(x_test,y_pr)
#print(accu)
print("Auuracy: gaussian NB: {}".format(accuracy_score(y_test, y_pr)))
print("confusion_matrix")
print(confusion_matrix(y_test,y_pr))
print("classification_report")
print(classification_report(y_test,y_pr))


# Multinomial NB: discrete counts in features of all attributes.
print("multinomial NB:---------------")
mnb=MultinomialNB()
mnb.fit(x_train,y_train)
y_pr=mnb.predict(x_test)
#accu=gnb.score(x_test,y_pr)
#print(accu)
print("Accuracy:Multinomial NB: {}".format(accuracy_score(y_test, y_pr)))
print("confusion_matrix")

print(confusion_matrix(y_test,y_pr))
print("classification_report")

print(classification_report(y_test,y_pr))

# Bernouli's NB: if feature vectors are binary.
print("Bernouli's NB:------------------------")
bnb=BernoulliNB()
bnb.fit(x_train,y_train)
y_pr=bnb.predict(x_test)
#accu=gnb.score(x_test,y_pr)
#print(accu)
print("Accuracy:Bernouli's NB: {}".format(accuracy_score(y_test, y_pr)))
print("confusion_matrix")

print(confusion_matrix(y_test,y_pr))
print("classification_report")

print(classification_report(y_test,y_pr))


