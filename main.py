import torch
import numpy as np
import random
from torch.utils import data
from sklearn.model_selection import train_test_split

from models import STGAT
from data_utils import load_data, data_transform1, get_adjacency_matrix
from trainer import train_model



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed()

    total_flow = load_data()

    x_total, y_total = data_transform1(total_flow, device=device)
    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.4)
    x_train = x_train.permute(0, 2, 3, 1)
    x_test = x_test.permute(0, 2, 3, 1)
    y_train = y_train.permute(0, 2, 3, 1)
    y_test = y_test.permute(0, 2, 3, 1)

    adj = get_adjacency_matrix("<YOUR_ADJ_CSV_PATH>")
    adj = torch.from_numpy(adj).float().to(device)

    model = STGAT(
        input_dim=1,
        hidden_dim= < YOUR_HIDDEN_DIM >,
    output_dim = 1,
    num_nodes = < YOUR_NODE_NUM >
    ).to(device)

    train_data = data.TensorDataset(x_train, y_train)
    train_iter = data.DataLoader(train_data, batch_size= < YOUR_BATCH_SIZE >, shuffle = True)
    test_data = data.TensorDataset(x_test, y_test)
    test_iter = data.DataLoader(test_data, batch_size= < YOUR_BATCH_SIZE >)

    loss_list, val_list = train_model(
        model=model,
        train_iter=train_iter,
        test_iter=test_iter,
        adj=adj,
        device=device
    )