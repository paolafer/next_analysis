import os
import sys
import random
import tables as tb
import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


file_name = sys.argv[1]
evt_number = int(sys.argv[2])

hits = pd.read_hdf(file_name, 'MC/hits')

evt_hits = hits[hits.event_id == evt_number]

x = evt_hits.x
y = evt_hits.y
z = evt_hits.z
e = evt_hits.energy


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')

max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()/2.0
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
p = ax.scatter(x,y,z,cmap='coolwarm',c=e)
cb = fig.colorbar(p, ax=ax)
cb.set_label('Energy (MeV)')

plt.show()

