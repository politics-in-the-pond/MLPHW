from pandas.core import frame
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler	
import numpy as np
import pandas as pd

#load data
PATH = "./breast-cancer-wisconsin.data"
dftmp = pd.read_csv(PATH)

datalist = dftmp.values.tolist()
new_data = pd.DataFrame(datalist, columns=['ID','Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 
'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])

#delete '?' data
is_not_NaN = new_data['Bare Nuclei'] != '?'
df = new_data[is_not_NaN]

#setting dataset and label
X = np.array(pd.DataFrame(df, columns=['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 
'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']))
Y = np.array(pd.DataFrame(df, columns=['Class']))

trynum = 100
knum = [2, 4, 8, 15]
#parameters of Decision Tree
depths = [1, 6, 16, 30]
#parameters of SVM
params = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [1, 3, 5, 10]
}
#parameters of Logistic Regression
params2 = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'C': [1, 3, 5, 10]
}

#testing scalers
for scalenum in range(0, 4) :
    if(scalenum == 0):
        #Scaler 1
        scaler = StandardScaler()
        print("using Standard Scaler.")

    if(scalenum == 1):
        #Scaler 2
        scaler = MinMaxScaler()
        print("using MinMax Scaler.")

    if(scalenum == 2):
        #Scaler 3
        scaler = MaxAbsScaler()
        print("using MaxAbs Scaler.")

    if(scalenum == 3):
        #Scaler 4
        scaler = RobustScaler()
        print("using Robust Scaler.") 

    #testing KFold
    for kidx ,k in enumerate(knum) :
        kfold = KFold(n_splits=k)
        print("   k in kfold: " + str(k))

        X_scale = scaler.fit_transform(X)
        sum1 = np.zeros(len(depths))
        sum2 = np.zeros(len(depths))
        
        for train_idx, test_idx in kfold.split(X_scale):
            X_train, Y_train = X_scale[train_idx], Y[train_idx]
            X_test, Y_test = X_scale[test_idx], Y[test_idx]

            for i in range(0, trynum) :

                for didx, d in enumerate(depths):
                    dt_clfg = DecisionTreeClassifier(criterion='gini', max_depth=d)
                    dt_clfg = dt_clfg.fit(X_train, np.ravel(Y_train, order = "c"))
                    dt_clfg_score = dt_clfg.score(X_test, Y_test)
                    sum1[didx] += dt_clfg_score

                    dt_clfe = DecisionTreeClassifier(criterion='entropy', max_depth=d)
                    dt_clfe = dt_clfe.fit(X_train, np.ravel(Y_train, order = "c"))
                    dt_clfe_score = dt_clfe.score(X_test, Y_test)
                    sum2[didx] += dt_clfe_score

            lr_t = LogisticRegression()
            lr = GridSearchCV(lr_t, param_grid=params2, cv=3, refit=False)
            lr = lr.fit(X_train, np.ravel(Y_train, order = "c"))

            svm_t = SVC()
            svm = GridSearchCV(svm_t, param_grid=params, cv=3, refit=False)
            svm.fit(X_train, np.ravel(Y_train, order = "c"))

        print("      best avg score of gini Decision Tree: " + str(max(sum1) / (trynum*k)) + ", depth = " + str(depths[np.argmax(sum1)]))
        print("      best avg score of entropy Decision Tree: " + str(max(sum2) / (trynum*k)) + ", depth = " + str(depths[np.argmax(sum2)]))
        print('      best score of Logistic Regression: ' + str(lr.best_score_) + ", parameters:" , lr.best_params_)
        print('      best score of SVM: ' + str(svm.best_score_) + ", parameters:" , svm.best_params_)
