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


hit_file = sys.argv[1]
blob_file = sys.argv[2]
evt_number = int(sys.argv[3])
mc = int(sys.argv[4])
radius = int(sys.argv[5])

drift_velocity = 0.920869862957205
if mc:
    drift_velocity = 1

with tb.open_file(hit_file) as h5in:
    table = getattr(getattr(h5in.root, 'RECO'), 'Events').read()
    hits_df = pd.DataFrame.from_records(table)
    #hits_df = h5in.root.RECO.Event
    this_evt_df = hits_df[hits_df.event == evt_number]
## exclude NN hits from plot
    x = this_evt_df[this_evt_df.Q >= 0].X
    y = this_evt_df[this_evt_df.Q >= 0].Y
    t = this_evt_df[this_evt_df.Q >= 0].Z
    e = this_evt_df[this_evt_df.Q >= 0].E

    z = [time*drift_velocity for time in t]

    blobs_df = pd.read_hdf(blob_file, 'tracks')
    blob_onet = blobs_df[blobs_df.numb_of_tracks == 1]
    this_evt_blobs = blob_onet[blob_onet.event == evt_number]

    x_b1 = this_evt_blobs.extreme1_x.values
    y_b1 = this_evt_blobs.extreme1_y.values
    z_b1 = this_evt_blobs.extreme1_z.values
    x_b2 = this_evt_blobs.extreme2_x.values
    y_b2 = this_evt_blobs.extreme2_y.values
    z_b2 = this_evt_blobs.extreme2_z.values

    xa = np.array(x)
    ya = np.array(y)
    za = np.array(z)
    ea = np.array(e)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    #ax.set_xlim([60, 150])
    #ax.set_ylim([-50, 130])
    #ax.set_zlim([140, 220])
    max_range = np.array([xa.max()-xa.min(), ya.max()-ya.min(), za.max()-za.min()]).max()/2.0
    mid_x = (xa.max()+xa.min()) * 0.5
    mid_y = (ya.max()+ya.min()) * 0.5
    mid_z = (za.max()+za.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    p = ax.scatter(xa,ya,za,cmap='coolwarm',c=ea)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label('Energy (pes)')

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
    x_1 = x_b1+np.cos(u)*np.sin(v)*radius
    y_1 = y_b1+np.sin(u)*np.sin(v)*radius
    z_1 = z_b1+np.cos(v)*radius
    ax.plot_wireframe(x_1, y_1, z_1, color="green", alpha=0.1)

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
    x_2 = x_b2+np.cos(u)*np.sin(v)*radius
    y_2 = y_b2+np.sin(u)*np.sin(v)*radius
    z_2 = z_b2+np.cos(v)*radius
    ax.plot_wireframe(x_2, y_2, z_2, color="green", alpha=0.1)

    bx = np.array([x_b1, x_b2])
    by = np.array([y_b1, y_b2])
    bz = np.array([z_b1, z_b2])
    ax.scatter(bx, by, bz, marker='x',s=100,color='black')

   # ax.set_xlabel('x (mm)')
   # ax.set_ylabel('y (mm)')
   # ax.set_zlabel('z (mm)')
    plt.show()


