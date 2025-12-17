import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils import data


def evaluate_model(model, data_iter, adj, device):
    model.eval()
    l_sum1, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            y_pred = model(x, adj)
            y_pred1 = y_pred[:, :, 0]
            y_real1 = y[:, 0, :, 0]
            l1 = torch.sqrt(torch.mean((y_pred1 - y_real1) ** 2) / torch.mean((y_real1) ** 2))

            n += y.shape[0]
            l_sum1 += l1.item() * y.shape[0]
        return l_sum1 / n


def train_model(model, train_iter, test_iter, adj, device, epochs=< YOUR_EPOCHS >, lr=< YOUR_LR >,
                save_path="<YOUR_SAVE_PATH>"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    min_val_loss = np.inf
    loss_list, val_list = [], []

    for epoch in range(1, epochs + 1):
        l_sum1, n = 0.0, 0
        model.train()

        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            y_pred = model(x, adj)

            y_pred1 = y_pred[:, :, 0]
            y_real1 = y[:, 0, :, 0]
            l1 = torch.sqrt(torch.mean((y_pred1 - y_real1) ** 2) / torch.mean((y_real1) ** 2))

            optimizer.zero_grad()
            l1.backward()
            optimizer.step()

            l_sum1 += l1.item() * y.shape[0]
            n += y.shape[0]

        val_loss = evaluate_model(model, test_iter, adj, device)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Train Loss: {l_sum1 / n:.4f}, Valid Loss: {val_loss:.4f}")

        loss_list.append(l_sum1 / n)
        val_list.append(val_loss)

    return loss_list, val_list