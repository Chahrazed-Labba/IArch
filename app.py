import configparser
import time
import streamlit as st
import pandas as pd
import csv
import tempfile
import os
import streamlit.components.v1 as components
import numpy as np
import shap
from statistics import mean
import matplotlib.pyplot as plt
import data_upload
import data_prepare
import models as model
import io 
from fpdf import FPDF

################## Write the home page ###############################
title_format = f'<p style="text-align: center; font-family: ' \
               f'Arial; color: #808080; font-size: 40px; ' \
               f'font-weight: bold;">Welcome!</p>'

subtitle_format = f'<p style="text-align: center; font-family: ' \
               f'Arial; color: #808080; font-size: 20px; ' \
               f'font-weight: bold;">You are using IArch app, an App for digging deeper into your data!</p>'

explanable_texte_format = f'<p style="text-align: center; font-family: ' \
               f'Arial; color: #black; font-size: 15px; ' \
               f'font-weight: normal;">IArch is a tool that allows archaeologists to do Explainable Artificial Intelligence (XAI) data analyses without having specific programming expertise. The platform covers the complete ML workflow, from data processing and feature selection to applying the ML models and explaining the predictions using the SHapley Additive exPlanations (SHAP).</p>'

citation_usage_format = f'<p style="text-align: left; font-family: ' \
                        f'Arial; color: #black; font-size: 20px; ' \
                        f'font-weight: bold;">To cite IArch in publication use:</p>'

url_format = f'<p style="text-align: center; font-family'\
             f'Arial; color: #blue; font-size: 10px;'\
             f'font-weight: normal;"https://cagt.cnrs.fr/alcouffe-ameline/</p>'

def display_welcome_page():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #F8D78C; /* Green background */
            border: none; /* Remove borders */
            color: black; /* White text */
            padding: 8px 15px; /* Some padding */
            text-align: center; /* Center the text */
            text-decoration: none; /* Remove underline */
            display: inline-block; /* Make the buttons appear inline */
            font-size: 14px; /* Increase font size */
            margin: 20px 1px; /* Add some margin */
            cursor: pointer; /* Add a pointer cursor on hover */
            border-radius: 8px; /* Rounded corners */
        }
        .stButton>button:hover {
            background-color: #262730; /* Darker orange on hover */
        }
        </style>
    """, unsafe_allow_html=True)

    # Add title and subtitle
    st.image("images/logo.png", width=750)
    st.markdown(title_format, unsafe_allow_html=True)
    st.markdown(subtitle_format, unsafe_allow_html=True)
    st.markdown(explanable_texte_format, unsafe_allow_html=True)
    
    # Add illustrations
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/dig2.JPG", width=230, caption='Yakutia')
    with col2:
        st.image("images/dig4.jpg", width=230, caption='Predynastic Egypt')
    with col3:
        st.image("images/dig1.png", width=230, caption='Bronze Age Mongolia')
    
    # Add button for more informations
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])
    with col3:
        button0 = st.button("More")
    with col4:
        button1 = st.button("Team")
    with col5:
        button2 = st.button("Contact")

    if button0:
        st.write("Archaeologists and anthropologists often face a considerable amount of data, making comprehensive and global analysis challenging. Even for modest-sized datasets, this task can be time-consuming. The IArch © application aims to address this challenge by simplifying post-excavation data analysis. It provides an overview using machine learning algorithms, thereby revealing underlying structures often unnoticed by traditional analysis methods. Another major obstacle is the frequent presence of missing data in archaeology. Although the application does not explicitly address this issue at present, it could predict some missing data with expert assistance, thereby improving the reliability and completeness of the analysis.\n\nThree main features have been developed to meet the specific needs of archaeologists: (1) A detailed data description, including information such as correlations, number of individuals and variables, and characteristics of each variable. (2) Hypothesis testing to classify individuals according to user-defined categories and compare predicted data with actual data. (3) Hypothesis generation using the K-means algorithm, which creates clusters without initial biases, allowing for the discovery of new perspectives without prior prejudice.\n\nIArch © promises to clarify results, highlight potential discoveries hidden within a constant flow of data, expedite certain analyses, and even integrate disciplines previously unfamiliar to archaeology. It provides a user-friendly interface, enabling archaeologists to perform explainable artificial intelligence (XAI) data analysis without requiring in-depth programming knowledge. The platform supports the entire AI process, from data preprocessing and feature selection to model application and prediction explanation using the SHapley Additive exPlanations (SHAP) library.\n\n The development team is continually striving to enhance the application's features to meet users' evolving needs.")
    if button1:
        st.write("This innovative application was jointly developed by the Bird team at the Lorraine University (LORIA), specializing in artificial intelligence, and researchers from the Toulouse University (CAGT), experts in biological anthropology and archaeology. It is the result of a close collaboration aimed at bridging the gap between AI expertise and the needs of archaeologists. Designed within the framework of a biological anthropology thesis scheduled for completion by the end of 2025, this application aims to make learning algorithms accessible to archaeologists who are not proficient in computer language.")
        col1, col2 = st.columns(2)
        with col1:
            col1.subheader("Ameline Alcouffe")
            st.image("images/ameline.png", width=250)
            st.markdown('https://cagt.cnrs.fr/alcouffe-ameline/')
        with col2:
            col2.subheader("Chahrazed Labba")
            st.image("images/chahrazed.png", width=250)
            st.markdown('https://members.loria.fr/CLabba/accueil/')
    if button2:
        st.write("If you have any concerns or issues about the App, please contact us at ameline.alcouffe@univ-tlse3.fr or chahrazed.labba@loria.fr.")
    # Add other informations
    st.markdown("<span style='font-size:20px'><b>Application development:</b></span> Chahrazed Labba, AI Researcher", unsafe_allow_html=True)
    st.markdown("<span style='font-size:20px'><b>Anthropologist Expert:</b></span> Ameline Alcouffe, PhD student", unsafe_allow_html=True)
    st.write(citation_usage_format, unsafe_allow_html=True)
    st.markdown("<span style='color:#F63366'> C. Labba, A. Alcouffe, E. Crubézy and A. Boyer, IArch: An AI Tool for Digging Deeper into Archaeological Data, 2023 IEEE 35th International Conference on Tools with Artificial Intelligence (ICTAI), Atlanta, GA, USA, 2023, pp. 22-29, doi: 10.1109/ICTAI59109.2023.00012.</span>", unsafe_allow_html=True)    
    
    # Add a foot page
    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image("images/lorraine.png", width=115)
    with col2:
        st.image("images/loria.png", width=75)
    with col3:
        st.image("images/CNRS.png", width=75)
    with col4:
        st.image("images/CAGT.jpg", width=80)
    with col5:
        st.image("images/Logo_UT3.jpg", width=175)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='text-align: center'>Developp with Streamlit 1.31.1</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<p style='text-align: center'>Edited with Visual Studio Code</p>", unsafe_allow_html=True)

################### Write the Tutorial page ###############################
def display_tutorial_page():
    st.image("images/logo.png", width=750)
    st.title("Welcome to the Tutorial")
    st.subheader("IArch - July 2024 version")
    st.write("<b>Reminder</b>: IArch is a tool that allows archaeologists to do Explainable Artificial Intelligence (XAI) data analyses without having specific programming expertise.", unsafe_allow_html=True)
    st.write("IArch is completely <b><u>free</u></b>, so please cite it into your scientific productions. Thanks!", unsafe_allow_html=True)
    st.markdown("<span style='color:#f69a14'> C. Labba, A. Alcouffe, E. Crubézy and A. Boyer, IArch: An AI Tool for Digging Deeper into Archaeological Data, 2023 IEEE 35th International Conference on Tools with Artificial Intelligence (ICTAI), Atlanta, GA, USA, 2023, pp. 22-29, doi: 10.1109/ICTAI59109.2023.00012.</span>", unsafe_allow_html=True)    
    st.write(":warning::warning: <b>If this is your first time with IArch, please download the tutorial.</b> :warning::warning:", unsafe_allow_html=True)

    # création / importation du fichier pdf pour téléchargement en pdf
    with open("D:\IArch-Visual studio - streamlit/IArch_tutorial_final_V1.pdf", "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    
    st.download_button(
        label="Télécharger le tutoriel en PDF",
        data=pdf_bytes,
        file_name='IArch_Tutorial.pdf',
        mime='application/pdf')

    
################### Write the Uploading page ###############################
def display_uploading_profiling_page():
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #F8D78C; /* Green background */
        border: ; /* Remove borders */
        color: black; /* White text */
        padding: 8px 10px; /* Some padding */
        text-align: center; /* Center the text */
        text-decoration: none; /* Remove underline */
        display: inline-block; /* Make the buttons appear inline */
        font-size: 14px; /* Increase font size */
        margin: 20px 1px; /* Add some margin */
        cursor: pointer; /* Add a pointer cursor on hover */
        border-radius: 8px; /* Rounded corners */
    }
    .stButton>button:hover {
        background-color: #262730; /* Darker orange on hover */
    }
    </style>
    """, unsafe_allow_html=True)
        
    # Uploading part
    st.title("Uploading your data")
    file = st.file_uploader("Upload a CSV file.", type=["csv"])
    if file is not None:
        st.markdown("<span style='color:#35DD51'> Dataset uploaded successfully.</span>", unsafe_allow_html=True)
        df = data_upload.upload_data(file)
        st.write("Data Overview")
        st.write(df)
    else:
        st.markdown("<span style='color:#FF0000'> No file uploaded yet</span>", unsafe_allow_html=True)

    
    # Profiling part
    st.title("Profiling")
    st.subheader("A dataset Profiling Report provides essential insights on:")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown("""
            <div style='background-color: #AA8F66; padding: 12px; border-radius: 10px'><p style='text-align: center'>Data overview</p></div>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 13px;'>Summarizes the total number of rows (observations) and columns (variables), along with data types (numeric, text, date, etc.).</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style='background-color: #ED9B40; padding: 12px; border-radius: 10px'><p style='text-align: center'>Descriptive statistics</p>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 13px;'>Includes summary statistics such as mean, median, standard deviation, min/max values for numeric columns. For text columns, it may show average string length and unique values.</p>", unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style='background-color: #FFEEDB; padding: 12px; border-radius: 10px'><p style='text-align: center'>Data distribution</p>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 13px;'>Uses visualizations like histograms and box plots to illustrate the distribution of values in numeric columns, highlighting outliers, trends, and distribution properties.</p>", unsafe_allow_html=True)
    with col5:
        st.markdown("""
            <div style='background-color: #BA3B46; padding: 12px;border-radius: 10px'><p style='text-align: center'>Correlation matrix</p>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 13px;'>Provides a correlation matrix for numeric variables, revealing relationships and potential dependencies between data points.</p>", unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div style='background-color: #61C9A8; padding: 12px;border-radius: 10px'><p style='text-align: center'>Missing values</p>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 13px;'>Analyzes columns with missing data, indicating the percentage of missing values. This helps decide whether to impute missing data or exclude columns from analysis.</p>", unsafe_allow_html=True)
    with col6:
        st.markdown("""
            <div style='background-color: #A3C3D9; padding: 12px;border-radius: 10px'><p style='text-align: center'>Data quality</p>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 13px;'>Evaluates data quality by checking for duplicates, outliers, inconsistencies, or invalid values in columns, ensuring the reliability of data for analysis.</p>", unsafe_allow_html=True)

    if file is None:
        st.markdown("<span style='color:#FF0000'> Please upload your file first!</span>", unsafe_allow_html=True)
    
    # Add button for profiling
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])
    with col4:
        button3 = st.button("Start Profiling")
    
    if button3:
        with st.spinner('Please be patient: The data is currently being characterized, this may take some time depending on the size of your data...'):
            profile_html = data_upload.profiling(df)        
        
        st.success('Profiling report is ready now!')
        components.html(profile_html, width=800, height=600, scrolling=True)

        # Save the profile_html content to a file
        with open("data_profile.html", "w", encoding="utf-8") as file:
            file.write(profile_html)
        
        # Read the content of the saved file for downloading
        with open("data_profile.html", "r", encoding="utf-8") as file:
            download_data = file.read()
        
        st.markdown("<span style='color:#FF0000'> You can download the profile report directly as an HTML file, but please note that the profile report has already been created in the app folder!</span>", unsafe_allow_html=True)
        st.download_button(
            label="Download profiling report as HTML",
            data=download_data,
            file_name="data_profile.html",
            mime="text/html"
        )  

        return df

#################### Write the Classification page ###############################
def display_classification_page(df):
    st.title('Classification')
    container_width = 500
    container_height = 300
    options = ["RFC", "SVM", "XGBoost"]
    options_metrics = ["Accuracy"]
    
    selected_option_target = st.selectbox('Select the target...', df.columns)
    st.caption(":red[Warning]: the target can only be a <b>categorical variable</b> with at least two modalities (for example: 'A' or 'B'). If you try to define the target as a quantitative or binary variable (for example: '1' or '2'), you will encounter the following error: <i> :blue['ValueError: list.remove(x): x not in list']</i>", unsafe_allow_html=True)
    
    only_feature_ = [option for option in df.columns if option != selected_option_target]
    selected_option_features = st.multiselect('Select the features...', only_feature_)
    features_selected = bool(selected_option_features)
    if not features_selected:
        st.error("Please select at least one feature")
    print(selected_option_features, selected_option_target)

    selected_options_models = st.multiselect('Select the models you want...', options, default=options)
    models_selected = bool(selected_options_models)
    if not models_selected:
        st.error("Please select at least one model.")
    
    selected_options_metrics = st.multiselect('Select the metrics you want...', options_metrics, default=options_metrics)
    metrics_selected = bool(selected_options_metrics)
    if not metrics_selected:
        st.error("Please select at least one metric.")
    

    # Création d'une section pour pouvoir choisir les paramètres des tests
    def load_config():
        config = configparser.ConfigParser()
        config.read('config.ini')
        return config
    
    def save_config(config):
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
    
    config = load_config()

    with st.expander('Advanced settings'):
        param_choice = st.selectbox('', ['Select the option...','Cross Validation Parameters', 'Proportion of the Train Set', 'Random Forest Parameters', 'Support Vector Machine Parameters', 'XGBoost Parameters'])
    
        if param_choice == 'Cross Validation Parameters':
            st. caption('<b> In machine learning, cross-validation is an essential technique for evaluating model performance and robustly selecting hyperparameters. Cross-validation is used to estimate the performance of a model on unseen data by simulating multiple different splits of the training and validation data. This allows assessing the model‘s ability to generalize to new data.', unsafe_allow_html = True)
            st.subheader('Choose the Cross Validation parameters')

            config['CrossValidation']['n_splits'] = st.text_input('n_splits', config['CrossValidation']['n_splits'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='n_splits_details', label_visibility="visible"):
                st.caption('n_splits controls how many times the data is split and used to train and validate the model, thereby influencing the robustness of the model performance estimation. It is typically chosen based on the dataset size and desired accuracy of the performance estimate. Common values include 5, 10, or sometimes more, depending on the dataset size. A <u> higher n_splits</u> can provide a more reliable estimate of performance as it uses more data for training and validation. However, this also increases computational cost since the model needs to be trained and evaluated K times.Must be at least 2. <b> Default = 5',unsafe_allow_html=True)
            
            config['CrossValidation']['shuffle'] = st.text_input('Shuffle', config['CrossValidation']['shuffle'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='shuffle_details'):
                st.caption('True: The data is randomly shuffled before being divided into training and test sets. Attention, if you change the parameters to "False", you risk a data distribution bias. <b> Default = True',unsafe_allow_html=True)
            
            config['CrossValidation']['random_state'] = st.text_input('Cross Validation random state', config['CrossValidation']['random_state'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='random_state_details'):
                st.caption('Random state affects the ordering of the indices, which controls the randomness of each fold. Setting random state to a specific value (such as 0 in this example) ensures the reproducibility of results across executions, as the same random sequence will be used each time. It is <b> <u> not </b> </u> recommended to change this value. <b> Default = 0',unsafe_allow_html=True)
    
        elif param_choice == 'Proportion of the Train Set':
            st.subheader('Choose the proportion of the Train Set')
            config['Split_Train_Test']['test_size'] = st.text_input('Test size', config['Split_Train_Test']['test_size'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='test_size_details'):
                st.caption('Test size represents the proportion of your dataset dedicated to test phase. The testing phase then concerns 80 percent of your sample. If you choose a testing proportion greater than 0.3 (equivalent to 70 percent of your dataset in the training phase), you risk poor performance on the training data and insufficient generalization to the test data. Conversely, if you choose a testing proportion less than 0.2 (equivalent to more than 80 percent of your dataset in the training phase), you risk overfitting and a model that does not generalize well. <b> Default = 0.2   Recommended value = between 0.2 and 0.3', unsafe_allow_html=True)
            config['Split_Train_Test']['random_state'] = st.text_input('Train set Random state', config['Split_Train_Test']['random_state'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='random_state_rf_details'):
                st.caption('Setting the "random state" to 0 ensures that the data split will be reproducible every time the code is executed. It is not recommended to change this value. <b> Default = 0', unsafe_allow_html=True)

        elif param_choice == 'Random Forest Parameters':
            st.subheader('Choose the Random Forest parameters')
            config['rf_config']['n_estimators'] = st.text_input('Number of estimators', config['rf_config']['n_estimators'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='n_estimator_details'):
                st.caption('The number of trees in the forest. This parameter specifies how many decision trees the algorithm should create and aggregate to form the final model. For <u>smaller datasets</u>, fewer trees (e.g., 50-200) might be sufficient to achieve good performance. For <u>larger datasets</u>, it is common to use more trees (e.g., 200-500). In cases of <u>very large datasets</u> or when very high accuracy is needed, even more trees (e.g., 500-1000 or more) might be used. <b> Default = 400', unsafe_allow_html=True)
            config['rf_config']['max_depth'] = st.text_input('Random Forest Maximum depth', config['rf_config']['max_depth'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='max_depth_details'):
                st.caption('The maximum depth of the tree. It specifies the maximum number of levels that any tree in the forest can grow. A tree grows by splitting nodes based on features of the data, and the maximum depth limits how deep the tree can go. Setting a smaller maximumn depth (3 to 10) results in shallow trees, where each tree makes simpler decisions based on fewer features. Shallow trees tend to be less complex and less prone to overfitting but may underfit the data if set too low. A larger maximum depth (10 to 20 and more) allows the trees to capture more complex relationships in the data, potentially leading to better performance on training data. However, deeper trees can also overfit if the model learns noise in the training data. <b> For Text recognition = 5-15 ; For Image recognition = 20 or more ; Default = 10', unsafe_allow_html=True)
            config['rf_config']['min_samples_split'] = st.text_input('Minimum sample split', config['rf_config']['min_samples_split'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='min_sample_split_details'):
                st.caption('It determines the smallest number of samples that a node must have before it can be split into child nodes. If a node has fewer samples than the minimum samples split, it will not be split, and it will become a leaf node.  A <u>high value</u> of min_samples_split (20 to 30) implies that each node in the tree requires a substantial number of samples to be eligible for further splitting. High values of min_samples_split can lead to simpler trees with fewer nodes and less detailed decision boundaries.  A <u>low value</u> of min_samples_split (2 to 5) allows nodes with fewer samples to be split further, potentially leading to more complex and detailed decision boundaries. <b> For small to medium Dataset = 2 to 20 ; For Large Datasets = 20 to 100 ; Default = 4', unsafe_allow_html=True)
            config['rf_config']['min_samples_leaf'] = st.text_input('Minimum sample leaf', config['rf_config']['min_samples_leaf'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='min_sample_leaf_details'):
                st.caption('It specifies the minimum number of samples required to be at a leaf node i.e the endpoints where no further splitting occurs. Each leaf node represents a final decision or prediction. A <u>high value</u> of min_samples_leaf (10 to 20 or more) leads to trees with fewer leaf nodes, potentially simplifying the model by reducing overfitting and making it less sensitive to noise in the training data. A <u>low value</u> of min_samples_leaf (1 to 5) leads to more complex trees with more leaf nodes. For a <u>complex dataset</u>, you might start with a higher min_samples_leaf value (e.g., 10) to initially simplify the model and prevent overfitting. For <u>simpler data or when precision is critical</u>, lower min_samples_leaf values (e.g., 1 or 2) might be explored to allow for more detailed models. <b> Default = 1', unsafe_allow_html=True)
            config['rf_config']['random_state'] = st.text_input('Random Forest Random State', config['rf_config']['random_state'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='random_state_rfc_details'):
                st.caption('Random state affects the ordering of the indices, which controls the randomness of each fold. Setting random state to a specific value (such as 42 here) ensures the reproducibility of results across executions, as the same random sequence will be used each time. The number 42 itself is arbitrary and has no special mathematical significance in this context. It is not recommended to change this value. <b> Default = 42', unsafe_allow_html=True)
        
        elif param_choice == 'Support Vector Machine Parameters':
            st.subheader('Choose the Support Vector Machine parameters')
            config['svm_config']['C_'] = st.text_input('C', config['svm_config']['C'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='c_details'):
                st.caption('The parameter C controls the penalty for classification errors in SVM models. It plays a crucial role in determining the margin and robustness of the SVM model. <u>A higher value</u> of implies a higher penalty for classification errors on the training data, which can lead to a narrower margin but better performance on the training data. <u>A lower value </u>of allows for a wider margin, as the model is less sensitive to individual training errors, which can result in better generalization but potentially lower performance on the training data. <b> Default = 1000.', unsafe_allow_html = True)
            config['svm_config']['kernel'] = st.text_input('Kernel', config['svm_config']['kernel'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='kernel_details'):
                st.caption('The radial basis function (RBF) kernel measures similarity between two data points in infinite dimensions and then approaches classification by majority vote. It is very flexible and capable of handling complex separations. It is not recommended to modify this parameter. <b> Default = ‘rbf’.', unsafe_allow_html = True)
            config['svm_config']['gamma'] = st.text_input('Gamma', config['svm_config']['gamma'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='gamma_details'):
                st.caption('Gamma is a hyperparameter that defines how much influence a single training example has. The higher the value of gamma, the closer other examples must be to be affected. <u>Low gamma value:</u> This means that the model will have a more generalized decision boundary. Each data point has a large influence, leading to smoother and more generalized decision boundaries. <u>High gamma value:</u> This means that the model will have a more complex decision boundary. Each data point has a small influence, allowing the model to capture more complex patterns but also increasing the risk of overfitting. <b> Default = 0.01', unsafe_allow_html = True)
            config['svm_config']['probability'] = st.text_input('Probability', config['svm_config']['probability'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='probability_details'):
                st.caption('The probability option allows for calculating the classification probabilities for each class. This can be useful in cases where you not only need the class prediction but also an estimate of the confidence in that prediction. <b>Default = True.', unsafe_allow_html = True)

        elif param_choice == 'XGBoost Parameters':
            st.subheader('Choose the XGBoost parameters')
            config['xgboost_config']['n_estimators'] = st.text_input('Number of estimators', config['xgboost_config']['n_estimators'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='n_estimators_gbc_details'):
                st.caption('This is the number of trees to be created in the XGBoost algorithm. Increasing n_estimators increases the number of trees in the model, which can potentially improve model performance in terms of accuracy. However, too many trees can lead to overfitting and increase computation time. <b> Recommended values = 100 - 1000; Default = 200.', unsafe_allow_html = True)
            config['xgboost_config']['max_depth'] = st.text_input('XGBoost Maximum depth', config['xgboost_config']['max_depth'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='max_depth_gbc_details'):
                st.caption('This is the maximum depth of each decision tree in the algorithm. Deeper trees can capture more complex relationships in the training data, but they can also lead to overfitting if the value is too high. <b> Recommended values = 3 – 10; Default = 10.', unsafe_allow_html = True)
            config['xgboost_config']['colsample_bytree'] = st.text_input('Number of columns by tree', config['xgboost_config']['colsample_bytree'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='colsample_details'):
                st.caption('This is the fraction of columns to use for constructing each tree. It controls feature subsampling during tree construction. A value less than 1.0 reduces variance and overfitting by introducing more diversity among the trees. <b>Recommended values = 0.3 – 0.8 </b>; If you want to consider all columns/features, choose 1. <b>Default = 1.', unsafe_allow_html = True)
            config['xgboost_config']['learning_rate'] = st.text_input('Learning rate', config['xgboost_config']['learning_rate'])
            if st.checkbox(':orange[Click here for more details]:grey_question:',key='learning_rate_details'):
                st.caption('This is the learning rate that controls the contribution of each tree to correcting the model‘s prediction errors. It regulates how much each tree adjusts the predictions of the existing model. A <u>lower learning_rate</u> requires more trees to achieve the same performance but can improve model generalization by reducing the risk of overfitting. <b> Recommended values = 0.01 – 0.3; Default = 0.1.', unsafe_allow_html = True)

        if st.button('Save modifications'):
            save_config(config)
            st.success('Configurations successfully saveds')
    
    button_start_classification = st.button("Start Classification", disabled=not (models_selected and metrics_selected and features_selected))
    if button_start_classification:
        train_X, test_X, train_y, test_y = data_prepare.pipeline_data_prepare(df, selected_option_features, selected_option_target)
        container = st.container()
        with container:
            tabs = st.tabs(["RFC","SVM","XGBoost"])
            tab_rf = tabs[0]
            tab_svm = tabs[1]
            tab_xgb = tabs[2]

            with tab_rf:
                if "RFC" in selected_options_models:
                    st.header('**Random Forest Results**')
                    st.subheader('Cross Validation Training Phase')
                    #Entrainement du modele
                    clf=model.randomForest()
                    plot_cross_val_results(clf,train_X,train_y)
                    st.caption(":orange[To download the plot, click on the ⤡ icon, and save the plot with right click.]", unsafe_allow_html=True)
                    st.subheader('Train-Test (80-20) Results')
                    clf=train_test_results(clf,train_X,train_y,test_X,test_y,model_type=None)
                    container1 = st.container()
                    with container1:
                        shap_values = model.explain_results_Tree(clf, test_X)
                        plot_shap(shap_values,test_X,test_y)
                        st.caption(":orange[To download the plot, click on the ⤡ icon, and save the plot with right click.]", unsafe_allow_html=True)
                else :
                     st.write('<span style="color: red;">Random Forest was not Selected!</span>', unsafe_allow_html=True)
            with tab_svm:
                if "SVM" in selected_options_models:
                    st.header('**Support Vector Machine Results**')
                    st.subheader('Cross Validation Training Phase')
                    svm=model.svm()
                    plot_cross_val_results(svm,train_X,train_y)
                    st.caption(":orange[To download the plot, click on the ⤡ icon, and save the plot with right click.]", unsafe_allow_html=True)
                    st.subheader('Train-Test (80-20) Results')
                    svm=train_test_results(svm,train_X,train_y,test_X,test_y,model_type=None)
                    container1 = st.container()
                    with container1:
                        shap_values = model.explain_results_Kernal_SVM(svm, test_X)
                        plot_shap(shap_values,test_X,test_y)
                        st.caption(":orange[To download the plot, click on the ⤡ icon, and save the plot with right click.]", unsafe_allow_html=True)
                else : 
                     st.write('<span style="color: red;">SVM was not Selected!</span>', unsafe_allow_html=True)
            with tab_xgb:
                if "XGBoost" in selected_options_models:
                    st.header('**eXtreme Gradient Boosting Results**')
                    st.subheader('Cross Validation Training Phase')
                    xgb=model.xgboost()
                    plot_cross_val_results(xgb,train_X,train_y)
                    st.caption(":orange[To download the plot, click on the ⤡ icon, and save the plot with right click.]", unsafe_allow_html=True)
                    st.subheader('Train-Test (80-20) Results')
                    xgb=train_test_results(xgb,train_X,train_y,test_X,test_y,model_type=None)
                    container1 = st.container()
                    with container1:
                        shap_values=model.explain_results_Tree(xgb, test_X)
                        plot_shap(shap_values,test_X,test_y)

                        st.caption(":orange[To download the plot, click on the ⤡ icon, and save the plot with right click.]", unsafe_allow_html=True)
                else : 
                     st.write('<span style="color: red;">XGboost was not Selected!</span>', unsafe_allow_html=True)

def plot_cross_val_results(alg,train_X,train_y):
    scores= model.compute_score_model(alg, train_X, train_y)
    x_values = np.arange(1, len(scores) + 1)
    # Calculate statistics
    max_score = np.max(scores)
    min_score = np.min(scores)
    average_score = scores.mean()
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

def plot_clustering_results(n_clusters_list,silhouette_list):
    st.markdown("**:blue[Clustering Results using K-means]**")
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.plot(n_clusters_list,silhouette_list, linewidth=3,label = "Silhouette Score Against # of Clusters")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")
    ax.set_title('Silhouette score according to number of clusters')
    ax.grid(True)
    param1 = n_clusters_list[np.argmax(silhouette_list)]
    param2 = max(silhouette_list)
    plt.plot(param1,param2, "tomato", marker="*",markersize=20, label = 'Best Silhouette Score')
    plt.legend(loc="best",fontsize = 'large')
    st.pyplot(plt)

#################### Write the Clustering page ###############################
def display_clustering_page(df):
    st.title('Clustering')
    container_width = 500
    container_height = 300
    options_classification=["RF"]
    options_clustering=["K-means"]

    selected_options_id = st.selectbox('Select The Identifier to keep track of your data subjects', df.columns)
    only_feature_= [option for option in df.columns if option != selected_options_id]
    #The list of features
    selected_options_features = st.multiselect('Select The Data Features', only_feature_)
    features_selected = bool(selected_options_features)
    if not features_selected:
        st.error("Please select at least one feature")
    
    #The list of models 
    selected_options_models_classif = st.multiselect('Select  The ML Models for classification', options_classification,default=options_classification)
    models_selected = bool(selected_options_models_classif)
    if not models_selected:
        st.error("Please select at least one classification model.")
    
    #the list of metrics
    selected_options_models_clustering= st.multiselect('Select The ML Models for clustering ', options_clustering,default=options_clustering)
    cluster_selected = bool(selected_options_models_clustering)
    if not cluster_selected:
        st.error("Please select at least one clustering model.")

    button_start_clustering = st.button("Start Clustering", disabled=not (models_selected and cluster_selected and features_selected))
    if button_start_clustering:
        #data to be clustered
        data_encoded=data_prepare.pipeline_data_prepare_clustering(df,selected_options_features,selected_options_id)
        # Cluster the data using K-means 
        best_params, labeled_data, n_clusters_list,silhouette_list=model.best_clustering(data_encoded,selected_options_id)
        plot_clustering_results(n_clusters_list,silhouette_list)
        # Save the clusterd data with full information into a CSV file 
        df_cluster=model.generate_cluster_df(labeled_data,best_params)
        ids=data_encoded[selected_options_id].unique()
        list_cluster=model.addcluster(df_cluster,ids)
        data_encoded['cluster']=list_cluster
        csv = data_encoded.to_csv(index=False)
        # Apply RF algorithm 
        st.header('**Random Forest Results**')
        clf=model.randomForest()
        train_X, train_y=data_prepare.trainig_data_clustering(data_encoded,selected_options_id)
        clf=model.train_classifier(clf,train_X,train_y)
        # Apply Shap Value 
        container1 = st.container()
        with container1:
            shap_values=model.explain_results_Tree(clf, train_X)
            plot_shap(shap_values,train_X,train_y)

        st.download_button(label="Download results as CSV", data=csv,file_name="clustering.csv")


##################### render the App ###############################
df = pd.DataFrame()
def main():
    st.set_page_config(page_title="IArch", page_icon="☠️")
    st.sidebar.title("Welcome in IArch")
    page = st.sidebar.radio("Go to", ("Home Page", "Tutorial", "Services"))

    if page == "Home Page":
        display_welcome_page()
    
    elif page == "Tutorial":
        display_tutorial_page()

    elif page == "Services":
        tabs = ["Uploading and Profiling"]
        if 'df' in st.session_state:
            tabs.append("Classification")
            tabs.append("Clustering")
        
        selected_tabs=st.sidebar.radio("Select a tab", tabs, key="tabs", index=0, on_change=None)

        if selected_tabs == "Uploading and Profiling":
            df=display_uploading_profiling_page()
            st.session_state.df=df
        elif selected_tabs == "Classification":
            df=st.session_state.df
            display_classification_page(df)
        elif selected_tabs == "Clustering":
            df=st.session_state.df
            display_clustering_page(df)




if __name__ == "__main__":
    main()