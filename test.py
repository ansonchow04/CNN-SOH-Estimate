import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io as sio
from train import BatteryDataset, myCNN

mat = sio.loadmat('data-XJTU-charge/RawData/batch-1.mat', 
                  squeeze_me=True, struct_as_record=False)
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
    SOH.append(np.array([cycle.capacity for cycle in battery.cycles], 
               dtype=np.float32) / 2.0)

test_data_raw = np.concatenate(data[7:], axis=0)  # 第8颗电池
test_SOH = np.concatenate(SOH[7:], axis=0)
scaler = np.load('Results/scaler.npz')
min_vals = scaler['min']
max_vals = scaler['max']
test_data = (test_data_raw - min_vals) / (max_vals - min_vals)
    
test_dataset = BatteryDataset(test_data, test_SOH)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = myCNN().to(device)
model.load_state_dict(torch.load('Results/best_model.pth'))
model.eval()

pred_list = []
target_list = []
with torch.no_grad():
    for X, Y in test_loader:
        X = X.to(device)
        pred = model(X).squeeze()
        pred_list.append(pred.cpu().numpy())
        target_list.append(Y.numpy())
test_preds = np.concatenate(pred_list)
test_targets = np.concatenate(target_list)
mae = np.mean(np.abs(test_preds - test_targets))
rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
print(f'Test MAE: {mae:.4f}, RMSE: {rmse:.4f}')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(test_targets, label='True SOH', marker='o', markersize=3)
plt.plot(test_preds, label='Predicted SOH', marker='x', markersize=3)
plt.xlabel("sample index")
plt.ylabel("SOH")
plt.title("SOH Prediction on Test Set (RMSE: {:.4f})".format(rmse))
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.scatter(test_targets, test_preds, alpha=0.6, label='Predicted')
plt.plot([test_targets.min(), test_targets.max()],
         [test_targets.min(), test_targets.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.title('Scatter Plot of True vs Predicted SOH')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('Results/test_results.png', dpi=300)
plt.show()
print('测试结果图已保存')