import os
import sys

import numpy  as np
import tables as tb
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as clrs

"""
This scripts plots the hits reconstructed with Beersheba in a nice way,
hiding less energetic hit. To be used for marketing.
"""

#plt.rcParams.update({'font.size': 14})
scale_norm = 1.5

the_file = sys.argv[1]
evt_number = int(sys.argv[2])

with tb.open_file(the_file) as h5in:
    table = getattr(getattr(h5in.root, 'DECO'), 'Events').read()
    hits_df = pd.DataFrame.from_records(table)
    hits_deco = hits_df[hits_df.event == evt_number]
    
    x = hits_deco.X
    y = hits_deco.Y
    z = hits_deco.Z
    e = hits_deco.E*1000

    cl = np.where(e>e.mean()/3, e/e.max(), 0)
    
    colors       = plt.cm.viridis(cl)
    colors[:,-1] = np.where(e>e.mean()/3, (e / e.max())**scale_norm, 0)

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d')

    p = ax.scatter3D(hits_deco.X, hits_deco.Y, hits_deco.Z, s=(50*(e)/e.max()),
                     c=colors, marker='s', depthshade=False)
    ax.set_xlabel('X (mm)', linespacing=0, labelpad=0)
    ax.set_ylabel('Y (mm)', linespacing=0, labelpad=0)
    ax.set_zlabel('Z (mm)', linespacing=0, labelpad=0)

    x_range = (x.max()-x.min())/2.
    y_range = (y.max()-y.min())/2.
    z_range = (z.max()-z.min())/2.
    mid_x   = (x.max()+x.min())/2.
    mid_y   = (y.max()+y.min())/2.
    mid_z   = (z.max()+z.min())/2.

    lims = []
    lims.append((mid_x - x_range, mid_x + x_range))
    lims.append((mid_y - y_range, mid_y + y_range))
    lims.append((mid_z - z_range, mid_z + z_range))


    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    ax.tick_params(pad=2)

    cb = fig.colorbar(p, ax=ax)
    cb.set_label('Energy (keV)')

    plt.show()
