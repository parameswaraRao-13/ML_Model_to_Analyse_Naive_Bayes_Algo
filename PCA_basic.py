import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.decomposition import PCA
from scipy.stats import zscore

candy_data = pd.read_csv(r'/home/ram/Downloads/kaggle/candy-data.csv')
print(candy_data.info())
print(candy_data.isnull().sum())
candy_data=candy_data.drop(columns="competitorname")

x=candy_data.drop(columns=["chocolate"])
x=x.apply(zscore)
print(x.head())
y=candy_data["chocolate"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

# without PCA:
# gaussian NB: if features follow a normal distribution.
print("gaussian NB: without PCA---------------")
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pr=gnb.predict(x_test)
#accu=gnb.score(x_test,y_pr)
#print(accu)
print(accuracy_score(y_test, y_pr))

print(confusion_matrix(y_test,y_pr))
print(classification_report(y_test,y_pr))

# with PCA:
# gaussian NB: if features follow a normal distribution.
print("gaussian NB: with PCA all components---------------")

pca = PCA()
X_train = pca.fit_transform(x_train)
X_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_
print("explained_variance:")
print(explained_variance)
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pr=gnb.predict(X_test)
#accu=gnb.score(x_test,y_pr)
#print(accu)
print("Accuracy: {}".format(accuracy_score(y_test, y_pr)))
print("confusion_matrix")
print(confusion_matrix(y_test,y_pr))
print("classification_report")
print(classification_report(y_test,y_pr))

for i in range(len(explained_variance)):
    print("gaussian NB: with PCA {} components---------------".format(i+1))
    pca = PCA(i+1)
    X_train = pca.fit_transform(x_train)
    X_test = pca.transform(x_test)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pr = gnb.predict(X_test)
    # accu=gnb.score(x_test,y_pr)
    # print(accu)
    print("with {} component acc: {}".format(i+1,accuracy_score(y_test, y_pr)))
    print("confusion_matrix")

    print(confusion_matrix(y_test, y_pr))
    print("classification_report")
    print(classification_report(y_test, y_pr))


# conclusion 3 components are enough to get the same accuracy as with all componets.


