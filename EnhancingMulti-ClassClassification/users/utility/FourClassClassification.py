import os
from django.conf import settings
import pandas as pd
datapath = os.path.join(settings.MEDIA_ROOT,'winequality-white.csv')
data = pd.read_csv(datapath)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
class classification:
    def lr(self):
        from sklearn.model_selection import train_test_split
        from sklearn.multiclass import OneVsOneClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
        lr = OneVsOneClassifier(LogisticRegression(random_state=0), n_jobs=2)
        lr.fit(X_train, y_train)
        ypred_lr = lr.predict(X_test)
        lr_score = lr.score(X_test, y_test)
        cm = confusion_matrix(y_test,ypred_lr)
        
        #Binary based LR
        bin_lr = LogisticRegression(solver='lbfgs')
        bin_lr.fit(X_train, y_train)

        # Test the classifier on the test data
        bin_lr_y_pred = bin_lr.predict(X_test)
        bin_lr_cm = confusion_matrix(y_test,bin_lr_y_pred)

        # Evaluate the classifier's performance
        bin_lr_score = bin_lr.score(X_test, y_test)
        
        
        return cm,lr_score,bin_lr_cm,bin_lr_score


    def svm(self):
        from sklearn.model_selection import train_test_split
        from sklearn.multiclass import OneVsOneClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import confusion_matrix
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=0)
        svm = OneVsOneClassifier(SVC(random_state=0), n_jobs=2)
        svm.fit(X_train, y_train)
        ypred_svm = svm.predict(X_test)
        cm = confusion_matrix(y_test,ypred_svm)
        svm_score = svm.score(X_test, y_test)
        
        #Binary based SVM
        bin_svm = SVC(random_state=0,C=1,kernel='linear')
        bin_svm.fit(X_train, y_train)
        ypred_bin_svm = bin_svm.predict(X_test)
        bin_svm_cm = confusion_matrix(y_test,ypred_bin_svm)
        bin_svm_score = bin_svm.score(X_test, y_test)
        
        return cm,svm_score,bin_svm_cm,bin_svm_score


