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

import matplotlib.cm as cm
from matplotlib.colors import Normalize

from invisible_cities.evm.event_model        import Cluster, Hit
from invisible_cities.types.ic_types         import xy
from invisible_cities.reco.paolina_functions import voxelize_hits

"""
This script plots 2D projections of Beersheba-reconstructed voxels
"""

cmap = plt.cm.get_cmap('viridis')
#plt.rcParams.update({'font.size': 14})

the_file   = sys.argv[1]
evt_number = int(sys.argv[2])
base_vsize = float(sys.argv[3])
proj       = str(sys.argv[4]) # xy, zx or zy

with tb.open_file(the_file) as h5in:
    table = getattr(getattr(h5in.root, 'DECO'), 'Events').read()
    hits_df = pd.DataFrame.from_records(table)
    this_evt_df = hits_df[hits_df.event == evt_number]
    the_hits = []

    xs = this_evt_df.X
    ys = this_evt_df.Y
    zs = this_evt_df.Z
    es = this_evt_df.E

    for x, y, z, e in zip(xs, ys, zs, es):
        h = Hit(0, Cluster(0, xy(x,y), xy(0,0), 0), z, e*1000, xy(0,0))
        the_hits.append(h)

voxels = voxelize_hits(the_hits,
                       np.array([base_vsize, base_vsize, base_vsize]), False)

vsizex = voxels[0].size[0]
vsizey = voxels[0].size[1]
vsizez = voxels[0].size[2]

x = np.array([v.X for v in voxels])
y = np.array([v.Y for v in voxels])
z = np.array([v.Z for v in voxels])
e = np.array([v.E for v in voxels])


x_range = (x.max() + vsizex/2 - (x.min() - vsizex/2))/2.
y_range = (y.max() + vsizey/2 - (y.min() - vsizey/2))/2.
z_range = (z.max() + vsizez/2 - (z.min() - vsizez/2))/2.
mid_x   = (x.max() + vsizex/2 + (x.min() - vsizex/2))/2.
mid_y   = (y.max() + vsizey/2 + (y.min() - vsizey/2))/2.
mid_z   = (z.max() + vsizez/2 + (z.min() - vsizez/2))/2.

n_x = int(x_range * 2 / vsizex)
n_y = int(y_range * 2 / vsizey)
n_z = int(z_range * 2 / vsizez)

fig = plt.figure()

if proj == 'xy':
    plt.hist2d(x, y, weights=e, bins=(n_x, n_y),
               range=((mid_x-x_range, mid_x+x_range),(mid_y-y_range, mid_y+y_range)),
               cmin=0.0001)
   # plt.scatter(xt, yt, cmap='viridis', c=et)
   # plt.scatter(bx, by, marker='x', s=200, linewidth=5, color='red')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

elif proj == 'zx':
    plt.hist2d(z, x, weights=e, bins=(n_z, n_x),
               range=((mid_z-z_range, mid_z+z_range),(mid_x-x_range, mid_x+x_range)),
               cmin=0.0001)
    #plt.scatter(zt, xt, cmap='viridis', c=et)
    #plt.scatter(bz, bx, marker='x', s=200, linewidth=5, color='red')
    plt.xlabel('z (mm)')
    plt.ylabel('x (mm)')

elif proj == 'zy':
    plt.hist2d(z, y, weights=e, bins=(n_z, n_y),
               range=((mid_z-z_range, mid_z+z_range),(mid_y-y_range, mid_y+y_range)),
               cmin=0.0001)
   # plt.scatter(zt, yt, cmap='viridis', c=et)
   # plt.scatter(bz, by, marker='x', s=60, linewidth=1, color='red')
    plt.xlabel('z (mm)')
    plt.ylabel('y (mm)')

else:
    print(f'Wrong string for projection: try xy, zx or zy')
    exit()

cb = plt.colorbar(label='Energy (keV)')

plt.show()
