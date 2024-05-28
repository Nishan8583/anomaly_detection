# Network Anamoly Detection

- Dataset pulld from https://www.kaggle.com/datasets/anushonkar/network-anamoly-detection?resource=download
- Plan is to use keras and create a neural network to detect anamoly in the dataset.\
- Reference: https://keras.io/examples/timeseries/timeseries_classification_transformer/

- I need to change the data, remove all the column values with attack values set to normal

# TODO
 - [X] Generate a pcap file with network data, the original network data I used does not appear to be a time series data.
 - [X] Read the PCAP file and generate training data.
 - [X] Then proceed with ML autoencoder stuffs.
 - [X] Create version_1.py, that does basic anomaly detection without timeseries stuff.
 - [ ] Create a version which uses time series as well.