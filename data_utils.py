import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def data_transform1(data, n_his=<YOUR_HIS_STEP>, n_pred=<YOUR_PRED_STEP>, device):
    n_slot = <YOUR_TOTAL_SLOT> - n_his - n_pred + 1
    x = np.zeros([n_slot, 1, n_his, <YOUR_NODE_NUM>])
    y = np.zeros([n_slot, 1, n_pred, <YOUR_NODE_NUM>])
    for j in range(n_slot):
        t = j
        s = j
        e = s + n_his
        x[t, :, :, :] = data[:,s:e,:].reshape(1, n_his, <YOUR_NODE_NUM>)
        y[t, :, :, :] = data[:,e:e+n_pred,:].reshape(1, n_pred, <YOUR_NODE_NUM>)
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def get_adjacency_matrix(distance_df_filename, num_of_vertices=<YOUR_NODE_NUM>, type_='connectivity', id_filename=None):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        df = pd.read_csv(distance_df_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            A[id_dict[i], id_dict[j]] = 1
            A[id_dict[j], id_dict[i]] = 1
        return A

    df = pd.read_csv(distance_df_filename)
    for row in df.values:
        if len(row) != 3:
            continue
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        if type_ == 'connectivity':
            A[i, j] = 1
            A[j, i] = 1
        elif type_ == 'distance':
            A[i, j] = 1 / distance
            A[j, i] = 1 / distance
        else:
            raise ValueError("type_ error, must be connectivity or distance!")
    return A

def load_data():
    data = np.load("<YOUR_DATA_PATH>")
    traffic_data = data["data"]
    feature_0_data = traffic_data[:, :, 0]
    f = torch.from_numpy(feature_0_data).float()
    flow = torch.reshape(f, ((1, <YOUR_TOTAL_SLOT>, <YOUR_NODE_NUM>)))
    return flow