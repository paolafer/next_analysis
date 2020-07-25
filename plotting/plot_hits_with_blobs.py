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
This script plots reconstructed hits and blobs.
The black crosses are the blob centres.
Only 1-track events can be displayed.
"""

the_file = sys.argv[1]
evt_number = int(sys.argv[2])
radius = int(sys.argv[3])

with tb.open_file(the_file) as h5in:
    table = getattr(getattr(h5in.root, 'CHITS'), 'highTh').read()
    hits_df = pd.DataFrame.from_records(table)
    this_evt_df = hits_df[hits_df.event == evt_number]

    x = this_evt_df.X
    y = this_evt_df.Y
    z = this_evt_df.Z
    e = this_evt_df.Ec*1000

    tracks_df    = pd.read_hdf(the_file, 'Tracking/Tracks')
    this_evt_trk = tracks_df[tracks_df.event == evt_number]
    if len(this_evt_trk) > 1:
        print(f'Event {evt_number} cannot be plotted, since it has more than one track')
        exit()

    x_b1 = this_evt_trk.extreme1_x.values
    y_b1 = this_evt_trk.extreme1_y.values
    z_b1 = this_evt_trk.extreme1_z.values
    x_b2 = this_evt_trk.extreme2_x.values
    y_b2 = this_evt_trk.extreme2_y.values
    z_b2 = this_evt_trk.extreme2_z.values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()*0.5
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    p = ax.scatter(x, y, z, cmap='viridis', c=e)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label('Energy (keV)')

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
    x_1 = x_b1+np.cos(u)*np.sin(v)*radius
    y_1 = y_b1+np.sin(u)*np.sin(v)*radius
    z_1 = z_b1+np.cos(v)*radius
    ax.plot_wireframe(x_1, y_1, z_1, color="red", alpha=0.1)

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
    x_2 = x_b2+np.cos(u)*np.sin(v)*radius
    y_2 = y_b2+np.sin(u)*np.sin(v)*radius
    z_2 = z_b2+np.cos(v)*radius
    ax.plot_wireframe(x_2, y_2, z_2, color="red", alpha=0.1)


    ### Extremes ###
    bx = np.array([x_b1, x_b2])
    by = np.array([y_b1, y_b2])
    bz = np.array([z_b1, z_b2])
    ax.scatter(bx, by, bz, marker='x', s=100, linewidth=5, color='black')

    plt.show()
