#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from pystacknet.pystacknet import StackNetClassifier
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.metrics import auc

import os 
os.chdir('c:/kaggle/Santander')

train_df = pd.read_csv('train.csv')
target = train_df['target']

train_df = train_df.set_index('ID_code')
label_train = train_df.target
train_df = train_df.drop(columns = ['target'])

train_df_q = StandardScaler().fit_transform(train_df.values)

train_df = pd.DataFrame(train_df_q
                        , columns = train_df.columns.values
                        , index = train_df.index.values)

train_df = pd.concat([label_train, train_df], axis=1)
train_df = train_df.reset_index()
train_df = train_df.rename(columns = {'index':'ID_code'}) 
features = [c for c in train_df.columns if c not in ['ID_code', 'target']] #basic features

test_df  = pd.read_csv('test.csv')
test_df = test_df.set_index('ID_code')
test_df_q = StandardScaler().fit_transform(test_df.values)
test_df = pd.DataFrame(test_df_q
                        , columns = test_df.columns.values
                        , index = test_df.index.values)

test_df = test_df.reset_index()
test_df = test_df.rename(columns = {'index':'ID_code'})    
#%%

models=[ 
            
            [LogisticRegression(C=1,  random_state=1),
             LogisticRegression(C=3,  random_state=1),
             Ridge(alpha=0.1, random_state=1),
             LogisticRegression(penalty="l1", C=1, random_state=1),
             ExtraTreesClassifier(max_depth = 3
                                    , n_jobs = 1
                                    , criterion='entropy'
                                    , class_weight='balanced_subsample'
                                    , random_state=0
                                    , n_estimators=300),

             ExtraTreesClassifier(max_depth = 7
                                    , n_jobs = 1
                                    , criterion='entropy'
                                    , class_weight='balanced_subsample'
                                    , random_state=0
                                    , n_estimators=300),
            
             LGBMClassifier(boosting_type='gbdt', num_leaves=13, max_depth=-1
                            , learning_rate=0.008, n_estimators=100, subsample_for_bin=1000
                            , objective="binary", min_child_samples=80, subsample=0.35, subsample_freq=5
                            , colsample_bytree=0.05, reg_alpha=0.1, reg_lambda=0.35, random_state=1, n_jobs=-1),

             LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=-1
                            , learning_rate=0.008, n_estimators=100, subsample_for_bin=1000
                            , objective="binary", min_child_samples=80, subsample=0.35, subsample_freq=5
                            , colsample_bytree=0.05, reg_alpha=0.1, reg_lambda=0.35, random_state=1, n_jobs=-1)       
             ],
            
            [RandomForestClassifier (n_estimators=300, criterion="entropy", max_depth=6, max_features=0.5, random_state=1)]
            
            ]
    
model=StackNetClassifier(models, metric="auc", folds=5,restacking=False,use_retraining=True, use_proba=True, random_state=0,n_jobs=8, verbose=1)

model.fit(train_df.iloc[:,2:].values, train_df.iloc[:,1].values)
preds=model.predict_proba(test_df.iloc[:,1:].values)
sub = test_df.iloc[:,:2].drop(columns = ['var_0'])
sub['target'] = preds[:,1]
sub.to_csv('submission.csv', index = False)

