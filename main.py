#%% Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
import keras
from keras.optimizers import Adam
import ipaddress
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MaxAbsScaler

TIME_STEPS=1
def create_sequences(values,time_steps=TIME_STEPS):
    output=[]
    for i in range(len(values)-time_steps+1):
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
    data["l4_proto"] = l4_proto_col
    data["l7_proto"] = l7_proto_col
    data["source_ip"] = data["source_ip"].apply(convert_ip)
    data["destination_ip"] = data["destination_ip"].apply(convert_ip)
    #data["attack"] = attack_col


    scaler = MaxAbsScaler()
    np_data = data.to_numpy()
    np_data = scaler.fit_transform(np_data)
    print(np_data)
    return np_data
    #return normal_numpy,anamolous_numpy


#%% Train
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
    normal=load_data("./normal.csv")
    input_dim = normal.shape[1]
    encoding_dim = 7  # Dimension of the encoded representation
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.summary()
    #normal_seq= create_sequences(normal)
    #print("loaded data")
    #print(normal.shape,normal_seq.shape)
    #print(normal,anamolous,normal_seq)
    #train(normal_seq)
    history = autoencoder.fit(normal, normal,
                          epochs=100,
                          batch_size=256,
                          shuffle=True,
                          validation_split=0.2,
                          verbose=1)
    anamolous = load_data("anamoly.csv")
    reconstructed = autoencoder.predict(anamolous)
    mse = np.mean(np.power(anamolous - reconstructed, 2), axis=1)

    # Set a threshold for anomaly detection
    threshold = np.percentile(mse, 95)  # Example: using the 95th percentile

    # Detect anomalies
    anomalies = mse > threshold
    print(anomalies)
    autoencoder.save("model.keras")

#%% Run main
main()
#load_data("output.csv")
# %%
