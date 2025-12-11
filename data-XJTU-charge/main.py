import os
import torch
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# batch 1, charge: 2C CCCV to 4.2V, discharge: 1C to 2.5V
matFile = 'data-XJTU-charge/RawData/batch-1.mat'
battery = sio.loadmat(matFile, squeeze_me=True, struct_as_record=False)
hdfeatureFile = 'data-XJTU-charge/RawData/handcraft_features/batch-1_features.xlsx'
savePath = 'data-XJTU-charge/Results/'
# battery 1
battery1 = battery['battery'][0]

# Prepare data as a single numpy array for efficiency
data_np = np.array([
    np.stack([
        cycle.relative_time_min,
        cycle.current_A,
        cycle.voltage_V,
        cycle.temperature_C
    ])
    for cycle in battery1.cycles
], dtype=np.float32)
data_mean = data_np.mean(axis=(0, 2), keepdims=True)
data_std = data_np.std(axis=(0, 2), keepdims=True)
data_np = (data_np - data_mean) / data_std
capacity = torch.tensor([cycle.capacity for cycle in battery1.cycles], dtype=torch.float32)
data = torch.from_numpy(data_np)
rSOH = capacity / 2.0  # normal capacity is 2.0 Ah
# print(data_np.shape)

# Define Dataset class
class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# network architecture
class CNN_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.fc(x)

# Prepare dataset and dataloader
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data = data[:train_size]
train_rSOH = rSOH[:train_size]
test_data = data[train_size:]
test_rSOH = rSOH[train_size:]
train_dataset = BatteryDataset(train_data, train_rSOH)
test_dataset = BatteryDataset(test_data, test_rSOH)
full_dataset = BatteryDataset(data, rSOH)
batch_size = 8
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
full_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN_small().to(device)

# Loss and optimizer
critierion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 200
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in tqdm(train_loader):
        if len(X_batch) != batch_size:
            continue
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch).squeeze(-1)
        loss = critierion(pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")



model.eval()
pred_plt = []
for dataset_ in full_dataset:
    with torch.no_grad():
        sample_X, sample_y = dataset_
        sample_X = sample_X.unsqueeze(0).to(device)
        pred = model(sample_X).squeeze(-1)
        pred_plt.append(pred.item())
    print(f"Real SOH: {sample_y.item():.4f}, Predicted SOH: {pred.item():.4f}")


plt.plot(range(len(rSOH)), rSOH.numpy(), label='Real SOH')
plt.plot(range(len(pred_plt)), pred_plt, label='Predicted SOH (Full)')
plt.xlabel('Cycle')
plt.ylabel('SOH')
plt.legend()
plt.savefig(os.path.join(savePath, 'SOH_prediction_CNN_full.png'))
plt.show()