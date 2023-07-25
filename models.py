from sklearn.metrics import accuracy_score
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


def ann():
    return 

def xgboost():
    config = configparser.ConfigParser()
    config.read('config.ini')
    xgb_cl = XGBClassifier(n_estimators=int(config.get('xgboost_config', 'n_estimators')),
                           max_depth=int(config.get('xgboost_config', 'max_depth')),
                           colsample_bytree=int(config.get('xgboost_config', 'colsample_bytree')),
                           learning_rate=float(config.get('xgboost_config', 'learning_rate')))
    return xgb_cl

def compute_score_model(model,train_X,train_y):
    config = configparser.ConfigParser()
    config.read('config.ini')
    train_y=train_y.astype('int')
    sk_folds = StratifiedKFold(n_splits = int(config.get('CrossValidation', 'n_splits')), 
                               shuffle=bool(config.get('CrossValidation', 'shuffle')),
                               random_state=int(config.get('CrossValidation', 'random_state'))) 
    
    scores = cross_val_score(model, train_X, train_y, cv = sk_folds)
    return scores

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
    return train_accuracy, accuracy, precision, recall, f1


def explain_results_Tree(model,data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data) 
    
    return shap_values
    


    
    
