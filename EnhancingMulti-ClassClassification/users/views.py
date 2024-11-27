from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.core.files.storage import FileSystemStorage
from matplotlib import pyplot as plt


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})
def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def FourClass(request):
    from .utility.FourClassClassification import classification
    algo = classification()
    lr_cm,lr_score,bin_lr_cm,bin_lr_score = algo.lr()
    svm_cm,svm_score,bin_svm_cm,bin_svm_score = algo.svm()
    print('-'*100)
    print('lr_score:',lr_score)
    print('svm_score:',svm_score)
    print('bin_lr_score:',bin_lr_score)
    print('bin_svm_score:',bin_svm_score)
    print(lr_cm)
    print('-'*100)
    print(svm_cm)
    scores = [bin_svm_score,svm_score,bin_lr_score,lr_score]

    # Create a list of the algorithms
    algorithms = ['BinaryBased SVM','3-ClassBased SVM','BinaryBased LR','3-ClassBased LR', ]

    # Create a bar plot of the scores
    plt.bar(algorithms, scores)
    plt.xlabel('Time with different Strategies')
    plt.ylabel('Time')
    # plt.title('Accuracy Comparison of LR and SVM')
    plt.show()
    return render(request,'users/FourClass.html',
                  {'lr_score':lr_score,
                    'svm_score':svm_score,
                    'bin_lr_score':bin_lr_score,
                    'bin_svm_score':bin_svm_score,
                    }
                  )

    
def FiveClass(request):
    from .utility.FiveClassClassification import classification
    algo = classification()
    lr_cm,lr_score,bin_lr_cm,bin_lr_score = algo.lr()
    svm_cm,svm_score,bin_svm_cm,bin_svm_score = algo.svm()
    print('-'*100)
    print('lr_score:',lr_score)
    print('svm_score:',svm_score)
    print('bin_lr_score:',bin_lr_score)
    print('bin_svm_score:',bin_svm_score)
    
    print(lr_cm)
    print('-'*100)
    print(svm_cm)
    scores = [bin_svm_score,svm_score,bin_lr_score,lr_score]

    # Create a list of the algorithms
    algorithms = ['BinaryBased SVM','3-ClassBased SVM','BinaryBased LR','3-ClassBased LR', ]

    # Create a bar plot of the scores
    plt.bar(algorithms, scores)
    plt.xlabel('Efficiency with different Strategies')
    plt.ylabel('Accuracy/Time')
    # plt.title('Accuracy Comparison of LR and SVM')
    plt.show()
    return render(request,'users/FiveClass.html',
                  {'lr_score':lr_score,
                    'svm_score':svm_score,
                    'bin_lr_score':bin_lr_score,
                    'bin_svm_score':bin_svm_score,
                    }
                  )
    
    
def WeightedVoting(request):
    from .utility.WeightedVoting import classification
    algo = classification()
    acc_svm,acc_lr = algo.Voting()
    return render(request,'users/WeightedVoting.html',{'acc_svm':acc_svm,'acc_lr':acc_lr})

def PredictQuality(request):
    if request.method == 'POST':
        import os
        from django.conf import settings
        import pandas as pd        
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        FixedAcidity = request.POST.get('FixedAcidity')
        VolatileAcidity = request.POST.get('VolatileAcidity')
        CitricAcid = request.POST.get('CitricAcid')
        ResidualSugar = request.POST.get('ResidualSugar')
        chlorides = request.POST.get('chlorides')
        FreeSulfurDioxide = request.POST.get('FreeSulfurDioxide')
        TotalSulfurDioxide = request.POST.get('TotalSulfurDioxide')
        density = request.POST.get('density')
        pH = request.POST.get('pH')
        sulphates = request.POST.get('sulphates')
        alcohol = request.POST.get('alcohol')
        print(type(FixedAcidity))
        test_set = [
            FixedAcidity,
            VolatileAcidity,
            CitricAcid,
            ResidualSugar,
            chlorides,
            FreeSulfurDioxide,
            TotalSulfurDioxide,
            density,
            pH,
            sulphates,
            alcohol,
        ]
        
        test_set = list(map(float, test_set))
        

        datapath = os.path.join(settings.MEDIA_ROOT,'winequality-red.csv')
        data = pd.read_csv(datapath)
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
        # create the models
        lr = LogisticRegression()

        # create the ensemble model
        wv_lr = VotingClassifier(estimators=[('lr', lr)], voting='hard',weights=[4])
        wv_lr.fit(X_train, y_train)

        # make predictions on the test data
        res = wv_lr.predict([test_set])
        return render(request,'users/predict.html',{'res':res})
    else:
        return render(request,'users/predict.html')