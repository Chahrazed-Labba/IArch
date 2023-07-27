from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap
import numpy as np
import tensorflow as tf 
from sklearn.cluster import KMeans 
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import silhouette_score

import configparser

def randomForest(): 
    # Create an instance of ConfigParser
    config = configparser.ConfigParser()
    config.read('config.ini')
    clf=RandomForestClassifier(n_estimators= int(config.get('rf_config', 'n_estimators')),
                               max_depth=int(config.get('rf_config', 'max_depth')),
                               min_samples_split=int(config.get('rf_config', 'min_samples_split')),
                               min_samples_leaf=int(config.get('rf_config', 'min_samples_leaf')),
                               random_state=int(config.get('rf_config', 'random_state')))
    
    
    return clf


def svm():
    config = configparser.ConfigParser()
    config.read('config.ini')
    svm = SVC(C=int(config.get('svm_config', 'C')), kernel=str(config.get('svm_config', 'kernel')),
              gamma=float(config.get('svm_config', 'C')),
              probability=bool(config.get('svm_config', 'probability')))
    return svm


def ann(train_X):
    classifier= Sequential([ Dense(units=10,
     input_shape=(train_X.shape[1],),
      activation='relu'),
      Dense(units=10, activation='relu'),
      Dense(units=1, activation='sigmoid')])
    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
     loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier

def xgboost():
    config = configparser.ConfigParser()
    config.read('config.ini')
    xgb_cl = XGBClassifier(n_estimators=int(config.get('xgboost_config', 'n_estimators')),
                           max_depth=int(config.get('xgboost_config', 'max_depth')),
                           colsample_bytree=int(config.get('xgboost_config', 'colsample_bytree')),
                           learning_rate=float(config.get('xgboost_config', 'learning_rate')))
    return xgb_cl
#Tree based models 
def compute_score_model(model,train_X,train_y):
    config = configparser.ConfigParser()
    config.read('config.ini')
    train_y=train_y.astype('int')
    sk_folds = StratifiedKFold(n_splits = int(config.get('CrossValidation', 'n_splits')), 
                               shuffle=bool(config.get('CrossValidation', 'shuffle')),
                               random_state=int(config.get('CrossValidation', 'random_state'))) 
    
    scores = cross_val_score(model, train_X, train_y, cv = sk_folds)
    return scores

#ANN models 
def compute_score_model_ANN(classifier,train_X,train_y):
    config = configparser.ConfigParser()
    config.read('config.ini')
    train_y=train_y.astype('int')
    sk_folds = StratifiedKFold(n_splits = int(config.get('CrossValidation', 'n_splits')), 
                               shuffle=bool(config.get('CrossValidation', 'shuffle')),
                               random_state=int(config.get('CrossValidation', 'random_state'))) 
    
    list_scores=[]
    for train , test in sk_folds.split(train_X, train_y):
        classifier.fit(x= train_X.iloc[train], y= train_y.iloc[train], batch_size=2, epochs=100, verbose=0)
        scores = classifier.evaluate(train_X.iloc[test], train_y.iloc[test], verbose=0)
        list_scores.append(scores[1]*100)
    
    return list_scores



def fit_test_model_(model, train_X,train_y,test_X, test_y):
    train_y=train_y.astype('int')
    test_y=test_y.astype('int')
    model.fit(train_X, train_y)
    y_pred_model=model.predict(test_X)
    y_pred_train_model=model.predict(train_X)
    train_accuracy=metrics.accuracy_score(train_y, y_pred_train_model)
    # compute the accuracy
    accuracy = accuracy_score(test_y, y_pred_model)
    # compute the precision
    precision = precision_score(test_y, y_pred_model,average='macro')
    # compute the recall
    recall = recall_score(test_y, y_pred_model, average='macro')
    # compute the F1-score
    f1 = f1_score(test_y, y_pred_model,average='macro')
    return train_accuracy, accuracy, precision, recall, f1, model

def fit_test_model_NN(model, train_X,train_y,test_X, test_y):
    train_y=train_y.astype('int')
    test_y=test_y.astype('int')
    model.fit(x= train_X, y= train_y, batch_size=2, epochs=100, verbose=0)
    y_pred_ann=model.predict(test_X)
    y_pred_ann_classes = np.argmax(y_pred_ann, axis=1)
    # compute the accuracy
    accuracy = accuracy_score(test_y, y_pred_ann_classes)
    # compute the precision
    precision = precision_score(test_y, y_pred_ann_classes,average='macro')
    # compute the recall
    recall = recall_score(test_y, y_pred_ann_classes,average='macro')
    # compute the F1-score
    f1 = f1_score(test_y, y_pred_ann_classes,average='macro')
    return  accuracy, precision, recall, f1, model


def explain_results_Tree(model,data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data) 
    
    return shap_values


def explain_results_Kernal_SVM(model,data):
    explainer = shap.KernelExplainer(model.predict_proba, data)
    shap_values = explainer.shap_values(data) 
    
    return shap_values

def explain_results_Kernal_ANN(model,data):
    explainer = shap.KernelExplainer(model, data)
    shap_values = explainer.shap_values(data) 
    print(explainer.expected_value)
    
    return shap_values


def best_clustering(data, id, max_clusters = 10):
    n_clusters_list=[]
    silhouette_list=[]
    cols=[col for col in data.columns if col != id]
    data_std = data[cols]
        
    for n_c in range(2,max_clusters+1): 
        kmeans_model = KMeans(n_clusters=n_c, random_state=42).fit(data_std) 
        labels = kmeans_model.labels_
        n_clusters_list.append(n_c)
        silhouette_list.append(silhouette_score(data_std, labels, metric='euclidean'))
    
    # Best Parameters
    param1 = n_clusters_list[np.argmax(silhouette_list)]
    param2 = max(silhouette_list)
    best_params = param1,param2
    
    # Data labeling with the best model
    kmeans_best = KMeans(n_clusters= param1 , random_state=42).fit(data_std) 
    labels_best = kmeans_best.labels_
    labeled_data = np.concatenate((data,labels_best.reshape(-1,1)),axis=1)
        
    
    return best_params, labeled_data, n_clusters_list,silhouette_list


def generate_cluster_df(data, params):
    listid=[]
    listcluster=[]
    for j in range(params[0]): 
        for i in range(len(data)):
            size=len(data[0])-1
            if(data[i][size]==j):
                listid.append(data[i][0])
                listcluster.append(j)
    # initialize data of lists.
    data_ = {'id_sujet':listid ,
            'cluster':listcluster}
    # Create DataFrame
    df = pd.DataFrame(data_)
    
    return df

#add clusters to the orginal data 
def addcluster(data_f, ids):
    list_cluster_ordred=[]
    for i in range(len(ids)):
        df=data_f.loc[data_f['id_sujet']==ids[i]]
        for index,row in df.iterrows():
            list_cluster_ordred.append(row['cluster'])
            
    return list_cluster_ordred 

def train_classifier(model, train_X, train_y):
    train_y=train_y.astype('int')
    model.fit(train_X, train_y)

    return model