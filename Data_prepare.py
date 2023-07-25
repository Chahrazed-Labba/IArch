import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import configparser

#return the dataframe to be used in the classification or clustering 
def build_data(dataframe,listfeatures_, target_):  
    listfeatures_.append(target_)
    df_=dataframe[listfeatures_]
    return df_

#Return the column type object, float, ...   
def determine_column_type(dataframe, column_name):
    column_type = dataframe.dtypes[column_name]
    return column_type

#Return the list of catgorical Data 
def categorical_column(dataframe,target_):
    list_column_categorical=[]
    for column_name in dataframe.columns:
        type_col=determine_column_type(dataframe,column_name)
        if (type_col=='object' or type_col=='bool' or type_col=='category'):
            list_column_categorical.append(column_name)
    list_column_categorical.remove(target_)
    return list_column_categorical

#Encode the cetagorical data 
def encoding_categorical_features(dataframe, cols):
    #creating instance of one-hot-encoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    #perform one-hot encoding on 'the following' columns 
    encoder_df = pd.DataFrame(encoder.fit_transform(dataframe[cols]).toarray())
    feature_names = encoder.get_feature_names_out(cols)
    encoder_df.columns = feature_names
    #merge one-hot encoded columns back with original DataFrame
    data_encoded = dataframe.join(encoder_df)
    #Drop the original categorical columns
    data_encoded.drop(cols, axis=1, inplace=True)
    return data_encoded

#Encode the target variable 
def encode_target(dataframe, target_):
    #get the unisue value in the column 
    unique_values = dataframe[target_].unique()
    target_dict = {index: value for index, value in enumerate(unique_values)}
    # Loop through the DataFrame
    for index, row in dataframe.iterrows():
        # Get the current value in the column
        current_value = row[target_]
        # Check if the current value needs to be replaced
        if current_value in target_dict.values():
            found_keys = [key for key, value in target_dict.items() if value == current_value]
            # Replace the value
            dataframe.loc[index, target_] = found_keys[0]

#Return the encoded data 
def pipeline_data_prepare(dataframe,listfeatures_, target_):
    #Build the data to be used in training and testing
    data=build_data(dataframe,listfeatures_, target_)
    #encode the target 
    encode_target(data,target_)
    #return categorical columns in the data
    list_categoric=categorical_column(data,target_)
    #encoding the categorical data 
    data_encoded=encoding_categorical_features(data,list_categoric)
    train_X,test_X, train_y, test_y=cross_Validation_splits(data_encoded,target_)
    
    return train_X,test_X, train_y, test_y

# Normalize the Data 

def normalizeData(dataframe): 
    # define min max scaler
    scaler = MinMaxScaler()
    # transform data
    scaled_data = scaler.fit_transform(dataframe)
    return scaled_data


#Split the data into train and Test  
#The test size is specified in the config file (as well as other parameters)
#Define cross validation splitter 
def cross_Validation_splits(dataframe, target_):
    # Create an instance of ConfigParser
    config = configparser.ConfigParser()
    #Read the config file
    config.read('config.ini')
    # Separate the target variable and the features
    X = dataframe.drop(target_, axis=1)
    y = dataframe[target_]
    X_scaled=normalizeData(X)
    X_scaled_df=pd.DataFrame(X_scaled, columns = X.columns)
    # Split the data into training and test sets
    train_X, test_X, train_y, test_y = train_test_split(X_scaled_df, y,
                                                        test_size=float(config.get('Split_Train_Test', 'test_size')),
                                                        random_state=int(config.get('Split_Train_Test', 'random_state')))

    
    return train_X,test_X, train_y, test_y


    