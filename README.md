# Network Anamoly Detection

- For timeseries anamoly used this doc as reference https://keras.io/examples/timeseries/timeseries_anomaly_detection/
- For dataset, generate network data from my Computer, saved the PCAP, and the used prepare_data to generate csv, and worked with that.
- I have removed dataset, because the PCAP might contain sensitive information like cookies, so if u want to use this script, please generate a PCAP, pass that as input to prepare_data and the script will generate a csv file for you, which can then be passed to main.py.
- Plan is to use keras and create a neural network to detect anamoly in the dataset.\
- Reference: https://keras.io/examples/timeseries/timeseries_classification_transformer/

# TODO
 - [X] Generate a pcap file with network data, the original network data I used does not appear to be a time series data.
 - [X] Read the PCAP file and generate training data.
 - [X] Then proceed with ML autoencoder stuffs.
 - [X] Create version_1.py, that does basic anomaly detection without timeseries stuff.
 - [X] Create a version which uses time series as well.

# End Result:
 - version_1.py was able to detect anamolies, like IP address that we did not communicate with.
 - main.py which is time series anamoly generated a lot of false positive, I believe  this is dues to few training dataset.