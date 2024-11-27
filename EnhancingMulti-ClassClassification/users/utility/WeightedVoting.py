class classification:
    def Voting(self):
        # import necessary libraries
        import os
        from django.conf import settings
        import pandas as pd        
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        

        datapath = os.path.join(settings.MEDIA_ROOT,'winequality-red.csv')
        data = pd.read_csv(datapath)
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
        # create the models
        lr = LogisticRegression()
        svm = SVC(C=1,probability=True)

        # create the ensemble model
        wv_svm = VotingClassifier(estimators=[('svm', svm)], voting='hard',weights=[4])
        wv_lr = VotingClassifier(estimators=[('lr', lr)], voting='hard',weights=[4])

        # fit the ensemble model on the training data
        wv_svm.fit(X_train, y_train)
        wv_lr.fit(X_train, y_train)

        # make predictions on the test data
        ypred_svm = wv_svm.predict(X_test)
        ypred_lr = wv_lr.predict(X_test)

        # evaluate the ensemble model
        acc_svm = accuracy_score(y_test, ypred_svm)
        acc_lr = accuracy_score(y_test, ypred_lr)
        print('*'*100)
        return acc_svm,acc_lr