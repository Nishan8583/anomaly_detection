#%% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MaxAbsScaler

column_names=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urgent","hot","num_failed_logins","logged_in",
"num_compromised","root_shell","su_attempted","num_root","num_file_creations",
"num_shells","num_access_files","num_outbound_cmds","is_host_login",
"is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",
"rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
"dst_host_diff_srv_rate","dst_host_same_src_port_rate",
"dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
"dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]

#%% Function declaration
def load_data(train_filename,test_filename):

    # loading the csv file, im giving column names
    data = pd.read_csv(train_filename,names=column_names)
    print(data.shape[1])
    #data2 = pd.read_csv(test_filename,names=column_names)
    # the second column is protocol type in string, need to convert it to number
    # one hot convert converts to values from -1 to 0
    '''
    encoder = OneHotEncoder(sparse_output=False)
    column = data.iloc[:,1] # get the 1 column
    column = column.values.reshape(-1,1) # change it to 2D array, where each [[value1],[value2]]
    print(column)
    encoded_column = encoder.fit_transform(column) # returns 2d array where each element is array of 3 elements [0,0,1]
    print(encoded_column)
    s = np.unique(encoded_column)
    data[["proto_0","proto_1","proto_2"]]=encoded_column
    data.drop(data.columns[1], axis=1, inplace=True)

    fig, ax = plt.subplots()
    data.plot(legend=True, ax=ax)
    plt.show()

    fig2,ax2=plt.subplots()
    data2.plot(legend=True,ax=ax2)
    plt.show()
    '''
    # we just want a numerical categorization for the texts
    # I know labelencoder is supposed to be used for output labels
    # but one hotencoder outputs an array, wen can use labelencoder to get 
    # numbers easily compared to oneHotencoder
    encoder = LabelEncoder()
    proto_col = encoder.fit_transform(data["protocol_type"])
    service_col = encoder.fit_transform(data["service"])
    flag_col = encoder.fit_transform(data["flag"])
    attack_col = encoder.fit_transform(data["attack"])
    print(data["attack"].unique())
    data["protocol_type"] = proto_col
    data["service"] = service_col
    data["flag"] = flag_col
    data["attack"] = attack_col

    # get the labels
    targets = data["attack"].to_numpy().reshape(-1,1)
    data.drop("attack",axis=1)
    
    # feature array
    narray = data.to_numpy()

    # Feature Scaling: Scales the data to a common range, 
    # usually between 0 and 1, using the minimum and maximum values of each feature.
    scaler = MaxAbsScaler()

    targets = scaler.fit_transform(targets)
    narray = scaler.fit_transform(narray)
    #print(narray)
    return narray,targets

#%% train
def train(x,y):
    model = Sequential()

    # create layers
    # use 10 neurons, number of features is 43
    model.add(Dense(43,input_dim=43,activation='softmax'))
    # input_dim is 10, since previous layer had 10 neurons
    model.add(Dense(10,input_dim=10,activation='softmax'))
    # since out
    model.add(Dense(1,activation='softmax'))

    #optimizer=Adam(lr=0.001)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.fit(x,y,epochs=100,verbose=2)
    
#%% Main declaration
def main():
    x,y = load_data("./dataset/Train.txt","./dataset/Test.txt")
    train(x,y)

#%% Run main
main()
# %%
