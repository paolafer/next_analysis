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

from invisible_cities.io.mcinfo_io import load_mchits
from invisible_cities.io.mcinfo_io import read_mcinfo

the_file = sys.argv[1]
evt_number = int(sys.argv[2])

with tb.open_file(the_file, mode='r') as h5in:

    h5extents = h5in.root.MC.extents
    events_in_file = len(h5extents)

    for i in range(events_in_file):
        if h5extents[i]['evt_number'] == evt_number:
            evt_line = i
            break

    hits_dict = load_mchits(the_file, (evt_line, evt_line+1))

    x = [h.X for h in list(hits_dict.values())[0]]
    y = [h.Y for h in list(hits_dict.values())[0]]
    z = [h.Z for h in list(hits_dict.values())[0]]
    e = [h.E for h in list(hits_dict.values())[0]]

    xa = np.array(x)
    ya = np.array(y)
    za = np.array(z)
    ea = np.array(e)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    max_range = np.array([xa.max()-xa.min(), ya.max()-ya.min(), za.max()-za.min()]).max()/2.0
    mid_x = (xa.max()+xa.min()) * 0.5
    mid_y = (ya.max()+ya.min()) * 0.5
    mid_z = (za.max()+za.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    p = ax.scatter(xa,ya,za,cmap='coolwarm',c=ea)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label('Energy (MeV)')

    plt.show()

