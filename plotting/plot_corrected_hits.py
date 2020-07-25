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
This script plots reconstructed hits, already corrected for
geometry and lifetime.
"""

the_file = sys.argv[1]
evt_number = int(sys.argv[2])

with tb.open_file(the_file) as h5in:
    table = getattr(getattr(h5in.root, 'CHITS'), 'highTh').read()
    hits_df = pd.DataFrame.from_records(table)
    this_evt_df = hits_df[hits_df.event == evt_number]

    x = this_evt_df.X
    y = this_evt_df.Y
    z = this_evt_df.Z
    e = this_evt_df.Ec*1000

    xa = np.array(x)
    ya = np.array(y)
    za = np.array(z)
    ea = np.array(e)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    #max_range = np.array([xa.max()-xa.min(), ya.max()-ya.min(), za.max()-za.min()]).max()/2.0

    x_range = (x.max()-x.min()) * 0.5
    y_range = (y.max()-y.min()) * 0.5
    z_range = (z.max()-z.min()) * 0.5
    mid_x = (xa.max()+xa.min()) * 0.5
    mid_y = (ya.max()+ya.min()) * 0.5
    mid_z = (za.max()+za.min()) * 0.5
    ax.set_xlim(mid_x - x_range, mid_x + x_range)
    ax.set_ylim(mid_y - y_range, mid_y + y_range)
    ax.set_zlim(mid_z - z_range, mid_z + z_range)
    p = ax.scatter(xa,ya,za,cmap='viridis',c=ea)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label('Energy (keV)')

    plt.show()
