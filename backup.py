# %% Imports
# from matplotlib import pyplot as plt
# from keras.models import Sequential
# from keras import layers
# import keras
# from keras.optimizers import Adam
import ipaddress

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, OneHotEncoder

TIME_STEPS = 1


def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


def convert_ip(ip):
    return int(ipaddress.IPv4Address(ip))


# %% Function declaration
def load_data(train_filename):

    # loading the csv file, im giving column names
    data = pd.read_csv(train_filename)
    print(data.shape[1])
    # data2 = pd.read_csv(test_filename,names=column_names)
    # the second column is protocol type in string, need to convert it to number
    # one hot convert converts to values from -1 to 0
    """
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
    """
    # we just want a numerical categorization for the texts
    # I know labelencoder is supposed to be used for output labels
    # but one hotencoder outputs an array, wen can use labelencoder to get
    # numbers easily compared to oneHotencoder
    encoder = LabelEncoder()

    proto_col = encoder.fit_transform(data["l4_proto"])
    data["l4_proto"] = proto_col
    data["source_ip"] = data["source_ip"].apply(convert_ip)
    data["destination_ip"] = data["destination_ip"].apply(convert_ip)
    # data["attack"] = attack_col

    print(data)
    # scaler = MaxAbsScaler()
    # return normal_numpy,anamolous_numpy


# %% train
def train(x):
    model = Sequential(
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


# %% Main declaration
def main():
    normal, anamolous = load_data("./dataset/Train.txt", "./dataset/Test.txt")
    input_dim = normal.shape[1]
    encoding_dim = 7  # Dimension of the encoded representation
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)
    decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)
    autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.summary()
    # normal_seq= create_sequences(normal)
    # print("loaded data")
    # print(normal.shape,normal_seq.shape)
    # print(normal,anamolous,normal_seq)
    # train(normal_seq)
    history = autoencoder.fit(
        normal,
        normal,
        epochs=1000,
        batch_size=256,
        shuffle=True,
        validation_split=0.2,
        verbose=1,
    )

    reconstructed = autoencoder.predict(normal)
    mse = np.mean(np.power(normal - reconstructed, 2), axis=1)

    # Set a threshold for anomaly detection
    threshold = np.percentile(mse, 95)  # Example: using the 95th percentile

    # Detect anomalies
    anomalies = mse > threshold
    print(anomalies)
    autoencoder.save("model.h5")


# %% Run main
# main()
load_data("output.csv")
# %%
