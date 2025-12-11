import os
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

# batch 1, charge: 2C CCCV to 4.2V, discharge: 1C to 2.5V
matFile = 'data-XJTU-charge/RawData/batch-1.mat'
battery = sio.loadmat(matFile, squeeze_me=True, struct_as_record=False)
battery1 = battery['battery'][0]

data = []
for cycle in battery1.cycles:
    data.append(np.stack([
        cycle.relative_time_min,
        cycle.current_A,
        cycle.voltage_V,
        cycle.temperature_C
    ]))


