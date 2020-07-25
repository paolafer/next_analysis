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

"""
This script plots 2D projections of Beersheba-reconstructed hits
+ the true MC hits
The true end-point of the primary electron (for single e-) or of
the two primary electrons (bb0nu) are also shown in red.
The positions of the extremes reconstructed with skeleton are shown
in black.
"""

cmap = plt.cm.get_cmap('viridis')
plt.rcParams.update({'font.size': 14})

the_file   = sys.argv[1]
skel_file  = sys.argv[2]
evt_number = int(sys.argv[3])
proj       = str(sys.argv[4])
kind       = str(sys.argv[5])

z_bin = 2. #mm

with tb.open_file(the_file) as h5in:
    table = getattr(getattr(h5in.root, 'DECO'), 'Events').read()
    hits_df = pd.DataFrame.from_records(table)
    this_evt_df = hits_df[hits_df.event == evt_number]
    the_hits = []

    x = this_evt_df.X
    y = this_evt_df.Y
    z = this_evt_df.Z
    e = this_evt_df.E*1000

x_range = (x.max()-x.min())/2.
y_range = (y.max()-y.min())/2.
z_range = (z.max()-z.min())/2.
mid_x   = (x.max()+x.min())/2.
mid_y   = (y.max()+y.min())/2.
mid_z   = (z.max()+z.min())/2.

particles = pd.read_hdf(the_file, 'MC/particles')
evt_part = particles[particles.event_id == evt_number]

skel = pd.read_hdf(skel_file, 'Tracks')
skel = skel[skel.event == evt_number]
x_skel1 = skel.blob1_x
y_skel1 = skel.blob1_y
z_skel1 = skel.blob1_z
x_skel2 = skel.blob2_x
y_skel2 = skel.blob2_y
z_skel2 = skel.blob2_z

if kind == 'bb0nu':
    primaries = evt_part[evt_part.primary == True]
    if len(primaries) != 2:
        exit()

    part1 = primaries[primaries.particle_id == 1]
    part2 = primaries[primaries.particle_id == 2]
    x_e1 = part1.final_x.values[0]
    y_e1 = part1.final_y.values[0]
    z_e1 = part1.final_z.values[0]
    x_e2 = part2.final_x.values[0]
    y_e2 = part2.final_y.values[0]
    z_e2 = part2.final_z.values[0]

else:
    the_part = evt_part[evt_part.primary == True]
    if len(the_part) != 1:
        exit()
    x_e1 = the_part.initial_x.values[0]
    y_e1 = the_part.initial_y.values[0]
    z_e1 = the_part.initial_z.values[0]
    x_e2 = the_part.final_x.values[0]
    y_e2 = the_part.final_y.values[0]
    z_e2 = the_part.final_z.values[0]

bx = np.array([x_e1, x_e2])
by = np.array([y_e1, y_e2])
bz = np.array([z_e1, z_e2])

sx = np.array([x_skel1, x_skel2])
sy = np.array([y_skel1, y_skel2])
sz = np.array([z_skel1, z_skel2])


hits = pd.read_hdf(the_file, 'MC/hits')

evt_hits = hits[hits.event_id == evt_number]
evt_hits = evt_hits[evt_hits.label == 'ACTIVE']

xt = evt_hits.x
yt = evt_hits.y
zt = evt_hits.z
et = evt_hits.energy*1000

fig = plt.figure()

xbins = int(x.max()-x.min())
ybins = int(y.max()-y.min())
zbins = int((z.max()-z.min())/z_bin)

if proj == 'xy':
    plt.hist2d(x, y, weights=e, bins=(xbins, ybins),
               range=((mid_x - x_range, mid_x + x_range),
                      (mid_y - y_range, mid_y + y_range)), cmin=0.0001)
    plt.scatter(xt, yt, cmap='viridis', c=et)
    plt.scatter(bx, by, marker='x', s=200, linewidth=3, color='red')
    plt.scatter(sx, sy, marker='x', s=200, linewidth=3, color='black')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

if proj == 'zx':
    plt.hist2d(z, x, weights=e, bins=(zbins, xbins),
               range=((mid_z - z_range, mid_z + z_range),
                      (mid_x - x_range, mid_x + x_range)),
               cmin=0.0001)
    plt.scatter(zt, xt, cmap='viridis', c=et)
    plt.scatter(bz, bx, marker='x', s=200, linewidth=3, color='red')
    plt.scatter(sz, sx, marker='x', s=200, linewidth=3, color='black')
    plt.xlabel('z (mm)')
    plt.ylabel('x (mm)')

if proj == 'zy':
    plt.hist2d(z, y, weights=e, bins=(zbins, ybins),
               range=((mid_z - z_range, mid_z + z_range),
                      (mid_y - y_range, mid_y + y_range)),
               cmin=0.0001)
    plt.scatter(zt, yt, cmap='viridis', c=et)
    plt.scatter(bz, by, marker='x', s=200, linewidth=3, color='red')
    plt.scatter(sz, sy, marker='x', s=200, linewidth=3, color='black')
    plt.xlabel('z (mm)')
    plt.ylabel('y (mm)')

cb = plt.colorbar(label='Energy (keV)')

plt.show()

#    x_range = (x.max()-x.min())/2.
#    y_range = (y.max()-y.min())/2.
#    z_range = (z.max()-z.min())/2.
#    mid_x = (x.max()+x.min()) * 0.5
#    mid_y = (y.max()+y.min()) * 0.5
#    mid_z = (z.max()+z.min()) * 0.5
#    ax.set_xlim(mid_x - x_range, mid_x + x_range)
#    ax.set_ylim(mid_y - y_range, mid_y + y_range)
#    ax.set_zlim(mid_z - z_range, mid_z + z_range)
