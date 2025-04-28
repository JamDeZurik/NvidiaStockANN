import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# NVIDIA Stock Data
# https://www.kaggle.com/datasets/meharshanali/nvidia-stocks-data-2025
df = pd.read_csv("NVDA.csv")

class NvidiaData(Dataset):
    def __init__(self):
        self.X = torch.tensor(df.iloc[:, [1,2,4,5,6]].values, dtype=torch.float32) # features
        self.y = torch.tensor(df.iloc[:, 3].values, dtype=torch.float32)  # high prices
        self.len = len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len


class NvidiaFit(nn.Module):
    def __init__(self):
        super(NvidiaFit, self).__init__()
        self.norm = nn.BatchNorm1d(5)
        self.in_to_h1 = nn.Linear(5, 4)
        self.h1_to_h2 = nn.Linear(4, 8)
        self.h2_to_out = nn.Linear(8, 1)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.in_to_h1(x))
        x = F.relu(self.h1_to_h2(x))
        return self.h2_to_out(x)

# trains the neural network (ANN) provided otherwise saves a new one
def trainNN(epochs=200, batch_size=16, lr=0.001, epoch_display=25, trained_network=None, save_file="nvidaNN.pt"):
    nd = NvidiaData()
    loader = DataLoader(nd, batch_size=batch_size, drop_last=False, shuffle=True)

    nvidia_nn = NvidiaFit()
    if trained_network is not None:
        nvidia_nn.load_state_dict(trained_network)
        nvidia_nn.train()

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(nvidia_nn.parameters(), lr=lr)

    # train each epoch; go in batches
    running_loss = 0.0
    for epoch in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            output = nvidia_nn(x).view(-1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % epoch_display == epoch_display - 1:
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {running_loss / (len(loader) * epoch_display):.6f} (${np.sqrt(running_loss / (len(loader) * epoch_display)):.2f})")
            running_loss = 0.0
    torch.save(nvidia_nn.state_dict(), save_file)

# uses given trained model to predict the high price
def predict_today_high(trained_model_path="nvidaNN.pt"):
    model = NvidiaFit()
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    print("Enter yesterday's stock data:")
    open_price = float(input("Open: "))
    low = float(input("Low: "))
    close = float(input("Close: "))
    volume = float(input("Volume (ex: 178902400): ") or 0)
    input_tensor = torch.tensor([[close, close, low, open_price, volume]], dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    print(f"Predicted High for Today: ${prediction:.2f}")

#trainNN(epochs=200) #use to save a new model or bolster an existing one
predict_today_high()