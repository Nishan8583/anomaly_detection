#%% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
import keras
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
    #attack_col = encoder.fit_transform(data["attack"])
    print(data["attack"].unique())
    data["protocol_type"] = proto_col
    data["service"] = service_col
    data["flag"] = flag_col
    #data["attack"] = attack_col


    # filter by data that is not an attack, which will be our normal traffic
    normal_traffic = data[data["attack"]=="normal"]
    anamolous_traffic= data[data["attack"] != "normal"]
    normal_traffic = normal_traffic.drop("attack",axis=1)
    anamolous_traffic = anamolous_traffic.drop("attack",axis=1)


    normal_numpy= normal_traffic.to_numpy()
    anamolous_numpy = anamolous_traffic.to_numpy()

    scaler = MaxAbsScaler()
    normal_numpy = scaler.fit_transform(normal_numpy)
    anamolous_numpy = scaler.fit_transform(anamolous_numpy)

    return normal_numpy,anamolous_numpy


#%% train
def train(x):
    model =Sequential(
    [
        layers.Input(shape=(x.shape[0], x.shape[1])),
        layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.summary()
    history = model.fit(
    x,
    x,
    epochs=50,
    batch_size=4,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)

    
#%% Main declaration
def main():
    normal,anamolous=load_data("./dataset/Train.txt","./dataset/Test.txt")
    print("loaded data")
    print(normal.shape)
    print(normal,anamolous)
    train(normal)

#%% Run main
main()
# %%
