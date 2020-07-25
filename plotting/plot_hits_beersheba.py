import os
import sys
import numpy  as np
import tables as tb
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
This scripts plots the hits reconstructed with Beersheba.
"""

cmap = plt.cm.get_cmap('viridis')
#plt.rcParams.update({'font.size': 14})

the_file = sys.argv[1]
evt_number = int(sys.argv[2])

with tb.open_file(the_file) as h5in:
    table = getattr(getattr(h5in.root, 'DECO'), 'Events').read()
    hits_df = pd.DataFrame.from_records(table)
    this_evt_df = hits_df[hits_df.event == evt_number]
    the_hits = []

    x = this_evt_df.X
    y = this_evt_df.Y
    z = this_evt_df.Z
    e = this_evt_df.E*1000

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')

x_range = (x.max()-x.min())/2.
y_range = (y.max()-y.min())/2.
z_range = (z.max()-z.min())/2.
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
