#%% Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#%% Function declaration
def load_data(filename: str):

    # loading the csv file, im giving column names
    data = pd.read_csv(filename,names=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urgent","hot","num_failed_logins","logged_in",
"num_compromised","root_shell","su_attempted","num_root","num_file_creations",
"num_shells","num_access_files","num_outbound_cmds","is_host_login",
"is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",
"rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
"dst_host_diff_srv_rate","dst_host_same_src_port_rate",
"dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
"dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]
)

    # the second column is protocol type in string, need to convert it to number
    # one hot convert converts to values from -1 to 0
    encoder = OneHotEncoder(sparse_output=False)
    column = data.iloc[:,1] # get the 1 column
    column = column.values.reshape(-1,1) # change it to 2D array, where each [[value1],[value2]]
    print(column)
    encoded_column = encoder.fit_transform(column) # returns 2d array where each element is array of 3 elements
    print(encoded_column)
    s = np.unique(encoded_column)
    data[["proto_0","proto_1","proto_2"]]=encoded_column
    data.drop(data.columns[1], axis=1, inplace=True)
    return data

#%% Main declaration
def main():
    print(load_data("./dataset/Train.txt"))

#%% Run main
main()
# %%
