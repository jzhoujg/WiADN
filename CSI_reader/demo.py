import matplotlib.pyplot as plt
from wifilib import *

path = r"./user6-1-1-1-3-r5.dat"
bf = read_bf_file(path)
csi_list = list(map(get_scale_csi,bf))
csi_np = (np.array(csi_list))
csi_amp = np.abs(csi_np)
print("csi shape: ",csi_np.shape)

data = process_csidata(path)
fig = plt.figure()
plt.plot(data[::1,0,0,3]) # N_t*N_r*N_s*N
plt.show()