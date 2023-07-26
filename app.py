import streamlit as st
import Data_profile as dp
import Data_prepare as data_p
import pandas as pd
import models as model
import numpy as np
import shap
from statistics import mean
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


# Define functions for different tabs
def display_tab1():
    st.title("Data Profiling")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        st.text("Dataset uploaded successfully")
        # return the data as a dataframe
        df=dp.load_data(file)
        #Do the profiling of the data
        st.write('<span style="color: red;"> Please be patient: The data is currently being characterized, this may take some time depending on the size of your data...</span>', unsafe_allow_html=True)
        profile_html=dp.profiling(df)
        st.write('<span style="color: green;"> Profiling report is ready now ! </span>', unsafe_allow_html=True)
        # Display the profile using st.components.v1.html
        components.html(profile_html, width=800, height=600, scrolling=True)
        return df

def display_tab2(df):
    # Define the container dimensions
    container_width = 500
    container_height = 300
    st.title("Classification")
    st.write("Start Classification ...")
    #List of the models 
    options=["SVM","RF","ANN","XGboost"]
    options_metrics=["Accuracy"]
    #The target variable
    selected_options_target = st.selectbox('Select The Target Variable', df.columns)
    only_feature_= [option for option in df.columns if option != selected_options_target]
    #The list of features
    selected_options_features=st.multiselect('Select The Data Features', only_feature_)
    print(selected_options_features,selected_options_target)
    #The list of models 
    selected_options_models = st.multiselect('Select  The ML Models', options,default=options)
    #the list of metrics
    selected_options_metrics = st.multiselect('Select  The Metrics', options_metrics,default=options_metrics)
    button_clicked = st.button('Start Classification!')
    if button_clicked:
        train_X,test_X, train_y, test_y=data_p.pipeline_data_prepare(df,selected_options_features,selected_options_target)
        #Results of training RF
        container = st.container()
        with container:
            tabs = st.tabs(["RF","SVM","ANN","XGboost"])
            tab_rf = tabs[0]
            tab_svm=tabs[1]
            tab_ann=tabs[2]
            tab_xgboost=tabs[3]
            with tab_rf:
                if "RF" in selected_options_models:
                    st.header('**Random Forest Results**')
                    st.subheader('Cross Validation Training Phase')
                    clf=model.randomForest()
                    plot_cross_val_results(clf,train_X,train_y)
                    st.subheader('Train-Test (80-20) Results')
                    clf=train_test_results(clf,train_X,train_y,test_X,test_y,model_type=None)
                    container1 = st.container()
                    with container1:
                        shap_values=model.explain_results_Tree(clf, test_X)
                        plot_shap(shap_values,test_X,test_y)
                    
                else :
                     st.write('<span style="color: green;">Random Forest was not Selected!</span>', unsafe_allow_html=True)
            with tab_svm:
                if "SVM" in selected_options_models:
                    st.header('**Support Vector Machine Results**')
                    st.subheader('Cross Validation Training Phase')
                    svm=model.svm()
                    plot_cross_val_results(svm,train_X,train_y)
                    st.subheader('Train-Test (80-20) Results')
                    svm=train_test_results(svm,train_X,train_y,test_X,test_y,model_type=None)
                    container1 = st.container()
                    with container1:
                        shap_values=model.explain_results_Kernal_SVM(svm, test_X)
                        plot_shap(shap_values,test_X,test_y)
                else : 
                     st.write('<span style="color: green;">SVM was not Selected!</span>', unsafe_allow_html=True)
            with tab_ann:
                if "ANN" in selected_options_models:
                    st.write('<span style="color: green;">ANN  Results : Training Phase !</span>', unsafe_allow_html=True)
                    st.header('**Artificial Neural Network Results**')
                    st.subheader('Cross Validation Training Phase')
                    ann=model.ann(train_X)
                    plot_cross_val_results_ANN(ann,train_X, train_y)
                    st.subheader('Train-Test (80-20) Results')
                    ann=train_test_results(ann,train_X,train_y,test_X,test_y,model_type='ann')
                    


                else : 
                     st.write('<span style="color: green;">ANN was not Selected!</span>', unsafe_allow_html=True)
            with tab_xgboost:
                if "XGboost" in selected_options_models:
                    st.header('**eXtreme Gradient Boosting Results**')
                    st.subheader('Cross Validation Training Phase')
                    xgb=model.xgboost()
                    plot_cross_val_results(xgb,train_X,train_y)
                    st.subheader('Train-Test (80-20) Results')
                    xgb=train_test_results(xgb,train_X,train_y,test_X,test_y,model_type=None)
                    container1 = st.container()
                    with container1:
                        shap_values=model.explain_results_Tree(xgb, test_X)
                        plot_shap(shap_values,test_X,test_y)
                else : 
                     st.write('<span style="color: green;">XGboost was not Selected!</span>', unsafe_allow_html=True)
                     
    
           
def plot_cross_val_results(alg,train_X,train_y):
    scores= model.compute_score_model(alg, train_X, train_y)
    x_values = np.arange(1, len(scores) + 1)
    # Calculate statistics
    max_score = np.max(scores)
    min_score = np.min(scores)
    average_score=scores.mean()
    std_score = np.std(scores)
            
    # Create columns layout
    col1, col2 = st.columns(2)

    # Display the values in the first column
    with col1:
        st.markdown("**:blue[Statistics About the Cross Validation Training]**")
        st.write("Max Score : ", max_score)
        st.write("Min Score : ", min_score)
        st.write("Average score : ", average_score)
        st.write("Standard Deviation : ", std_score)
        # Plot the scores
    with col2:
        st.markdown("**:blue[Cross-Validation Scores]**")
        fig1, ax1 = plt.subplots()
        ax1.plot(x_values,scores)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('')
        st.pyplot(fig1)  

def plot_cross_val_results_ANN(alg,train_X,train_y):
    scores=model.compute_score_model_ANN(alg,train_X, train_y)
    x_values = np.arange(1, len(scores) + 1)
    max_score = max(scores)
    min_score = min(scores)
    average_score=mean(scores)
    std_score = np.std(scores, ddof=1)

    # Create columns layout
    col1, col2 = st.columns(2)

    # Display the values in the first column
    with col1:
        st.markdown("**:blue[Statistics About the Cross Validation Training]**")
        st.write("Max Score : ", max_score)
        st.write("Min Score : ", min_score)
        st.write("Average score : ", average_score)
        st.write("Standard Deviation : ", std_score)
        # Plot the scores
    with col2:
        st.markdown("**:blue[Cross-Validation Scores]**")
        fig1, ax1 = plt.subplots()
        ax1.plot(x_values,scores)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('')
        st.pyplot(fig1)  



def train_test_results(alg,train_X,train_y,test_X,test_y, model_type=None): 
    if model_type==None:
        train_accuracy,accuracy,precision, recall, f1, fit_model=model.fit_test_model_(alg, train_X,train_y,test_X,test_y)
        #st.markdown("**:blue[Statistics About the Train - Test phase]**")
        st.write("Training Accuracy : ", train_accuracy)
        st.write("Test Accuracy : ", accuracy)
        st.write("Precision : ", precision)
        st.write("Recall : ", recall)
        st.write("F1 : ", f1)
    else: 
       accuracy,precision, recall, f1, fit_model=model.fit_test_model_NN(alg, train_X,train_y,test_X,test_y) 
       st.write("Test Accuracy : ", accuracy)
       st.write("Precision : ", precision)
       st.write("Recall : ", recall)
       st.write("F1 : ", f1)
    
    return fit_model
    
    
def plot_shap(shap_values, data, test_y):
    st.subheader("Explainability with SHAP")
    # Generate the summary plot
    for label in np.unique(test_y):
        st.markdown(f'**:blue[Waterfall plot for the Label: {label}]**')
        fig1, ax1 = plt.subplots()
        shap.summary_plot(shap_values[label], data)
        st.pyplot(fig1)
       

def display_tab3(df):
    st.title("Clustering and Classification to generate new hypothesis")
    st.write("Start ...")
    #List of the models 
    options_classification=["SVM","RF","ANN","XGboost"]
    options_clustering=["K-means"]
    
    #The list of features
    selected_options_features=st.multiselect('Select The Data Features', df.columns)
    #The list of models 
    selected_options_models_classif = st.multiselect('Select  The ML Models for classification', options_classification,default=options_classification)
    #the list of metrics
    selected_options_models_clustering= st.multiselect('Select The ML Models for clustering ', options_clustering,default=options_clustering)
    button_clicked = st.button('Start !')       

# Define function for the welcome page
def display_welcome_page():
    st.title("IArch : Dig Deeper into your Data")
    st.write("IArch is a tool that allows archaeologists to do Explainable Artificial Intelligence (XAI) data analyses without having specific programming expertise. The platform covers the complete ML workflow, from data processing and feature selection to applying the ML models and explaining the predictions using the SHapley Additive exPlanations (SHAP)")
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/dig1.png", width=300)
    with col2:
        st.image("images/dig1.png", width=300)


# Render the application
df=pd.DataFrame()
def main():
    st.set_page_config(page_title="IArch Tool", page_icon="🚀")
    st.sidebar.title("Welcome!")
    page = st.sidebar.radio("Go to", ("About IArch", "Services"))

    if page == "About IArch":
        display_welcome_page()
    elif page == "Services":
        tabs=st.sidebar.radio("Select a tab", ["Profiling","Classification", "Clustering"], key="tabs", index=0, on_change=None)
        if tabs == "Profiling":
            df=display_tab1()
            st.session_state.df=df
            
        elif tabs == "Classification":
            df=st.session_state.df
            display_tab2(df)
        
        elif tabs == "Clustering":
            df=st.session_state.df
            display_tab3(df)
            
   
if __name__ == "__main__":
    main()
