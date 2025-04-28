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
        # open, low, high, close, volume
        raw_X = df.iloc[:, [4, 3, 2, 1, 6]].values
        self.mean = raw_X.mean(axis=0)
        self.std = raw_X.std(axis=0)
        self.X = torch.tensor((raw_X - self.mean) / self.std, dtype=torch.float32)
        # label is open low high close for tomorrow (shift over 1)
        self.y = torch.tensor(df.iloc[:, [4, 3, 2, 1]].shift(-1).iloc[:-1].values, dtype=torch.float32)  # [Next Open, Next Low, Next High, Next Close]
        self.len = len(self.X) - 1 #exclude the last row (could possibly include copy of previous last row)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

class NvidiaFit(nn.Module):
    def __init__(self):
        super(NvidiaFit, self).__init__()
        self.norm = nn.BatchNorm1d(5)
        self.in_to_h1 = nn.Linear(5, 8)
        self.h1_to_h2 = nn.Linear(8, 16)
        self.h2_to_out = nn.Linear(16, 4)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.in_to_h1(x))
        x = F.relu(self.h1_to_h2(x))
        return self.h2_to_out(x)

# trains the neural network (ANN) provided otherwise saves a new one
def trainNN(epochs=200, batch_size=16, lr=0.001, epoch_display=25, trained_network=None, save_file="nvidiaNN.pt"):
    nd = NvidiaData()
    loader = DataLoader(nd, batch_size=batch_size, drop_last=False, shuffle=False) #don't shuffle to preserve order

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
            output = nvidia_nn(x).view(-1, 4)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % epoch_display == epoch_display - 1:
            print(
                f"Epoch {epoch + 1}/{epochs} - Avg Loss: {running_loss / (len(loader) * epoch_display):.6f} (${np.sqrt(running_loss / (len(loader) * epoch_display)):.2f})")
            running_loss = 0.0
    torch.save(nvidia_nn.state_dict(), save_file)

# uses given trained model to predict the next day's Open, Low, High, and Close prices
def predict_tomorrow_prices(trained_model_path="nvidiaNN.pt"):
    model = NvidiaFit()
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    raw_X = df.iloc[:, [4, 3, 2, 1, 6]].values
    mean = raw_X.mean(axis=0)
    std = raw_X.std(axis=0)

    print("Enter stock data for Today:")
    open_price = float(input("Open: "))
    low = float(input("Low: "))
    high = float(input("High: "))
    close = float(input("Close: "))
    volume = float(input("Volume (ex: 192837465): ") or 0)

    # input tensor for prediction
    input_arr = np.array([[open_price, low, high, close, volume]])
    input_norm = (input_arr - mean) / std
    input_tensor = torch.tensor(input_norm, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).squeeze().tolist()

    # display predicted values
    predicted_open, predicted_low, predicted_high, predicted_close = prediction
    print(f"\nPredicted values for Tomorrow:")
    print(f"Open: ${predicted_open:.2f}")
    print(f"Low: ${predicted_low:.2f}")
    print(f"High: ${predicted_high:.2f}")
    print(f"Close: ${predicted_close:.2f}")

#trainNN(epochs=200) # use to save new model or bolster existing one
predict_tomorrow_prices()
