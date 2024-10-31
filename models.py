import pandas as pd
import shap
import numpy as np
import configparser

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score,recall_score, log_loss, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import tensorflow as tf 
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import LeaveOneOut




############## Création des algorithmes ########################
# Création de l'algorithme pour le random forest
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


# Création de l'algorithme pour le support vector machine
def svm():
    config = configparser.ConfigParser()
    config.read('config.ini')
    svm = SVC(C=int(config.get('svm_config', 'C')), kernel=str(config.get('svm_config', 'kernel')),
              gamma=float(config.get('svm_config', 'C')),
              probability=bool(config.get('svm_config', 'probability')))
    return svm

# Création de l'algorithme pour le XGBoost
def xgboost():
    config = configparser.ConfigParser()
    config.read('config.ini')
    xgb = XGBClassifier(n_estimators=int(config.get('xgboost_config', 'n_estimators')),
                           max_depth=int(config.get('xgboost_config', 'max_depth')),
                           colsample_bytree=int(config.get('xgboost_config', 'colsample_bytree')),
                           learning_rate=float(config.get('xgboost_config', 'learning_rate')))
    return xgb

# Création de l'algorithme avec du leave-one-out

##################################################################

############## Cross Validation ##################################
# pour les algorithmes basés sur des arbres
def compute_score_model(model,train_X,train_y):
    config = configparser.ConfigParser()
    config.read('config.ini')
    train_y=train_y.astype('int')
    sk_folds = StratifiedKFold(n_splits = int(config.get('CrossValidation', 'n_splits')), 
                               shuffle=bool(config.get('CrossValidation', 'shuffle')),
                               random_state=int(config.get('CrossValidation', 'random_state'))) 
    
    scores = cross_val_score(model, train_X, train_y, cv = sk_folds)
    return scores

##################################################################
############## Calcul des accuracy ###############################

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


##################################################################
############## Production des graphiques SHAP ####################
def explain_results_Tree(model,data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data) 
    
    return shap_values

def explain_results_Kernal_SVM(model,data):
    explainer = shap.KernelExplainer(model.predict_proba, data)
    shap_values = explainer.shap_values(data) 
    
    return shap_values


##################################################################
############## Présentation résultats K-means ####################

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

# ajout des cluster à la suite des données
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



