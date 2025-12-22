import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as sio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

mat = sio.loadmat('data-XJTU-charge/RawData/batch-1.mat', squeeze_me=True, struct_as_record=False)
data, SOH = [], []
for battery in mat['battery']:
    data.append(np.array([
        np.stack([
            cycle.relative_time_min,
            cycle.current_A,
            cycle.voltage_V,
            cycle.temperature_C
        ], axis=1) for cycle in battery.cycles
    ], dtype=np.float32))
    SOH.append(np.array([cycle.capacity for cycle in battery.cycles], dtype=np.float32) / 2.0)

'''
数据结构：
一个batch中一共有8个battery
每个battery有若干个cycle，每个cycle是一个充电过程
每个cycle包含固定128个时间步，每个时间步有4个特征：相对时间（分钟）、电流（安培）、电压（伏特）、温度（摄氏度）
'''

train_data_raw = np.concatenate(data[:6], axis=0)
val_data_raw = np.concatenate(data[6:7], axis=0)
test_data_raw = np.concatenate(data[7:], axis=0)
train_flat = train_data_raw.reshape(-1, 4)
min = train_flat.min(axis=0)
max = train_flat.max(axis=0)
train_data = (train_data_raw - min) / (max - min)
val_data = (val_data_raw - min) / (max - min)
test_data = (test_data_raw - min) / (max - min)
train_SOH = np.concatenate(SOH[:6], axis=0)
val_SOH = np.concatenate(SOH[6:7], axis=0)
test_SOH = np.concatenate(SOH[7:], axis=0)


class BatteryDataset(Dataset):
    def __init__(self, data, capacity):
        super().__init__()
        self.X = torch.tensor(data, dtype=torch.float32)
        self.Y = torch.tensor(capacity, dtype=torch.float32)
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
train_dataset = BatteryDataset(train_data, train_SOH)
val_dataset = BatteryDataset(val_data, val_SOH)
test_dataset = BatteryDataset(test_data, test_SOH)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class myCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1), # lenth=128
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # lenth=64
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # lenth=64
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # lenth=32
        )
        self.fc = nn.Linear(32 * 32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, 128, 4)
        x = x.transpose(1, 2)  # (batch_size, 4, 128) - Conv1d需要 (batch, channels, length)
        x = self.layer1(x)  # (batch_size, 16, 64)
        x = self.layer2(x)  # (batch_size, 32, 32)
        x = x.view(x.size(0), -1)  # 展平: (batch_size, 32*32=1024)
        x = self.fc(x)  # (batch_size, 1)
        return x



if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = myCNN().to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        total_loss = 0
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        return total_loss / len(loader.dataset)

    def evaluate(model, loader, device):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, Y in loader:
                X = X.to(device)
                pred = model(X).squeeze().cpu().numpy()
                preds.append(pred)
                targets.append(Y.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        return mae, rmse, preds, targets

    num_epochs = 100
    patience = 10
    best_rmse = float('inf')
    patience_counter = 0
    train_losses = []
    val_rmses = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model=model, loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
        train_losses.append(train_loss)
        val_mae, val_rmse, _, _ = evaluate(model=model, loader=val_loader, device=device)
        val_rmses.append(val_rmse)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}')
        scheduler.step(val_rmse)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), 'Results/best_model.pth')
            patience_counter = 0
            print('Model saved. RMSE:', best_rmse)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered.')
            break

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_rmses, label='Validation RMSE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('Results/training_validation_curve.png')
    plt.show()
    plt.close()

    np.savez('Results/scaler.npz', min=min, max=max)
    print('Scaler saved.')