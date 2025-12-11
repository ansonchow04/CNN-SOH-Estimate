import os
import matplotlib.pyplot as plt
import scipy.io as sio

dataFile = 'data-XJTU-charge/RawData/batch-'
saveFile = 'data-XJTU-charge/draw/batch-'

battery = sio.loadmat(dataFile, squeeze_me=True, struct_as_record=False)


for i in range(1, 6+1):
    print('drawing batch ' + str(i))
    dataFile_ = dataFile + str(i) + '.mat'
    saveFile_ = saveFile + str(i) + '/'
    batteries = sio.loadmat(dataFile_, squeeze_me=True, struct_as_record=False)
    j = 1
    for battery in batteries['battery']:
        print('\tdrawing battery ' + str(j))
        saveFile__ = saveFile_ + 'battery' + str(j) + '/'
        k = 1
        for cycle in battery.cycles:
            print('\t\tdrawing cycle ' + str(k))
            saveFile___ = saveFile__ + 'current_A' + '/' + 'batch' + str(i) + 'battery' + str(j) + 'cycle' + str(k) + 'current' + '.png'
            print('\t\t\t' + saveFile___)
            os.makedirs(os.path.dirname(saveFile___), exist_ok=True)
            os.path.join(saveFile___)
            plt.plot(cycle.current_A)
            plt.savefig(saveFile___)
            plt.close()
            saveFile___ = saveFile__ + 'voltage_V' + '/' + 'batch' + str(i) + 'battery' + str(j) + 'cycle' + str(k) + 'voltage' + '.png'
            print('\t\t\t' + saveFile___)
            os.makedirs(os.path.dirname(saveFile___), exist_ok=True)
            os.path.join(saveFile___)
            plt.plot(cycle.voltage_V)
            plt.savefig(saveFile___)
            plt.close()
            k += 1
        j += 1
