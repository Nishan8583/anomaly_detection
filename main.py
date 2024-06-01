#%% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
import keras
from keras.optimizers import Adam
import ipaddress,csv
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MaxAbsScaler, MinMaxScaler
from keras.layers import LSTM,Dense

TIME_STEPS=100
def create_sequences(values,time_steps=TIME_STEPS):
    output=[]
    for i in range(len(values)-time_steps+1):
        #print("--SEQUENCE--")
        print(values[i:(i+time_steps)].shape)
        #print("--SEQUENCE--")
        output.append(values[i:(i+time_steps)])
    return np.stack(output)

def convert_ip(ip):
    return int(ipaddress.IPv4Address(ip))

#%% Function declaration
def load_data(train_filename):

    # loading the csv file, im giving column names
    data = pd.read_csv(train_filename)
    
    print(data.shape)  # gives rows x columns
 
    encoder = LabelEncoder()

    
    l4_proto_col = encoder.fit_transform(data["l4_proto"])
    l7_proto_col = encoder.fit_transform(data["l7_proto"])
    print(l7_proto_col)
    data["l4_proto"] = l4_proto_col
    data["l7_proto"] = l7_proto_col
    data["source_ip"] = data["source_ip"].apply(convert_ip)
    data["destination_ip"] = data["destination_ip"].apply(convert_ip)
    #data["attack"] = attack_col
    data["source_port"] = data["source_port"].fillna(0)
    data["destination_port"] = data["destination_port"].fillna(0)
    print(data.isnull().sum())

    scaler = MinMaxScaler()
    np_data = data.to_numpy()
    np_data = scaler.fit_transform(np_data)
    print(np_data)
    return np_data
    #return normal_numpy,anamolous_numpy


#%% Train
def train(x):
    model =Sequential(
    [
        layers.Input(shape=(x.shape[1], x.shape[2])),
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
    epochs=1,
    batch_size=1,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)
    return model

   
    
#%% Main declaration
def main():
    
    normal=load_data("./normal.csv")
    normal= create_sequences(normal)
    print("Normal shape",normal.shape)
    anamolous = load_data("anamoly.csv")
    print(anamolous)
    anamolous = create_sequences(anamolous)
    print("anamolous data shape",anamolous.shape)

    input_dim = normal.shape[1]
    encoding_dim = 7  # Dimension of the encoded representation
 
    model = train(normal)
    
    
    reconstructed = model.predict(anamolous)
    mse = np.mean(np.power(anamolous - reconstructed, 2), axis=1)

    # Set a threshold for anomaly detection
    threshold = np.percentile(mse, 95)  # Example: using the 95th percentile

    # Detect anomalies
    anomalies = mse > threshold
    print(anomalies)
    model.save("model_lstm.keras")
    # Print the corresponding lines from the anomaly.csv file
    anomaly_indices = np.where(anomalies)[0]  # Get the indices of anomalies
    with open("anamoly.csv", "r") as file:
        reader = csv.reader(file)
        anomaly_lines = [row for i, row in enumerate(reader) if i in anomaly_indices]

    print("Anomaly lines:")
    for line in anomaly_lines:
        print(line)
    return
    # Train autoencoder
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(normal.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(normal, epochs=1, batch_size=1, verbose=2)   
    
    
    #input_layer = layers.Input(shape=(input_dim,))
    #encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    #decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    #autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
    #autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    #autoencoder.summary()
    #normal_seq= create_sequences(normal)
    #print("loaded data")
    #print(normal.shape,normal_seq.shape)
    #print(normal,anamolous,normal_seq)
    #train(normal_seq)
    #history = autoencoder.fit(normal, normal,
    #                      epochs=100,
    #                      batch_size=256,
    #                      shuffle=True,
    #                      validation_split=0.2,
    #                      verbose=1)
    #anamolous = load_data("anamoly.csv")
    

#%% Run main
main()

#%%
data = load_data("normal.csv")
#print(data=[0])
# %%
 