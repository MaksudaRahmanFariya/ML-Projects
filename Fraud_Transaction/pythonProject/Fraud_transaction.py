import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import optuna
import xgboost as xgb
import lightgbm as lgbm
import catboost as catb
from sklearn.metrics import accuracy_score, confusion_matrix
transction_data= pd.read_csv("C:\Users\User\Downloads\PS_20174392719_1491204439457_log.csv")
transction_data.isnull().any()
transction_data.head()
transction_data.rename(columns={'newbalanceOrig':'newbalanceOrg'},inplace=True)
transction_data.drop(labels=['nameOrig','nameDest'],axis=1,inplace=True)
print('Minimum value of Amount, Old/New Balance of Origin/Destination:')
transction_data[['amount','oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest']].min()
transction_data.loc[transction_data.isFraud == 1].type.unique()
sns.heatmap(transction_data.corr(),cmap='coolwarm');
fraud = transction_data.loc[transction_data.isFraud == 1]
non_fraud = transction_data.loc[transction_data.isFraud == 0]
fraudcount = fraud.isFraud.count()
nonfraudcount = non_fraud.isFraud.count()
data_fraud = pd.read_csv('C:\Users\User\Downloads\PS_20174392719_1491204439457_log.csv')
data_fraud = data_fraud.replace(to_replace={'PAYMENT':1,'TRANSFER':2,'CASH_OUT':3,
                                            'CASH_IN':4,'DEBIT':5,'No':0,'Yes':1})
data_fraud.head()
X = data_fraud.drop(['isFraud'],axis=1)
y = data_fraud[['isFraud']]
def my_print(s):
    a = 4
    for i in s:
        a+=1
    return print('-' * a + '\n' + '| ' + s + ' |' + '\n' + '-' * a)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=122)
N_SPLITS = 2  # previous:300, increasing N_SPLITS to remove error due to randomness

lgbm_preds = []
xgb_preds = []
catb_preds = []

prob = []

folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)

for fold, (train_id, test_id) in enumerate(tqdm(folds.split(X, y), total=N_SPLITS)):
    my_print(f'fold {fold + 1}')

    X_train, y_train = X.iloc[train_id], y.iloc[train_id]
    X_valid, y_valid = X.iloc[test_id], y.iloc[test_id]

    lgbm_model = lgbm.LGBMClassifier()
    xgb_model = xgb.XGBClassifier()
    catb_model = catb.CatBoostClassifier(verbose=0)

    lgbm_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    catb_model.fit(X_train, y_train)

    my_print(
        f'Training Accuracy   :- {(lgbm_model.score(X_train, y_train) * 100).round(2)}% | {(xgb_model.score(X_train, y_train) * 100).round(2)}% | {(catb_model.score(X_train, y_train) * 100).round(2)}%')
    my_print(
        f'Validation Accuracy :- {(lgbm_model.score(X_valid, y_valid) * 100).round(2)}% | {(xgb_model.score(X_valid, y_valid) * 100).round(2)}% | {(catb_model.score(X_valid, y_valid) * 100).round(2)}%')

    prob1, prob2, prob3 = lgbm_model.predict_proba(X_test), xgb_model.predict_proba(X_test), catb_model.predict_proba(
        X_test)
    prob.append((prob1 + prob2 + prob3) / 3)
my_print('Model Trained !!!')
final = [[0, 0]]
for i in range(N_SPLITS):
    final = final + prob[i]

final = final / N_SPLITS
y_pred = pd.Series([np.argmax([i]) for i in final])
my_print(f'Test Accuracy:- {accuracy_score(y_test, y_pred)*100}%')
sns.heatmap(confusion_matrix(y_test, y_pred), cmap='cool_r', linewidths=0.5, annot=True);

