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

"""
This script plots the true MC hits of a nexus event.
"""

file_name = sys.argv[1]
evt_number = int(sys.argv[2])

hits = pd.read_hdf(file_name, 'MC/hits')

evt_hits = hits[hits.event_id == evt_number]
evt_hits = evt_hits[evt_hits.label == 'ACTIVE']

x = evt_hits.x
y = evt_hits.y
z = evt_hits.z
e = evt_hits.energy*1000


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')

x_range = (x.max()-x.min()) * 0.5
y_range = (y.max()-y.min()) * 0.5
z_range = (z.max()-z.min()) * 0.5
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - x_range, mid_x + x_range)
ax.set_ylim(mid_y - y_range, mid_y + y_range)
ax.set_zlim(mid_z - z_range, mid_z + z_range)
p = ax.scatter(x,y,z,cmap='viridis',c=e) #coolwarm
cb = fig.colorbar(p, ax=ax)
cb.set_label('Energy (keV)')

plt.show()

