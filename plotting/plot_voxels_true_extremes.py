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
This script plots voxels starting from reconstructed Esmeralda hits.
The true end-point of the primary electron (for single e-) or of
the two primary electrons (bb0nu) are also shown in black.
Only these two kinds of events can be displayed.
Only MC events can be displayed.
"""

cmap = plt.cm.get_cmap('viridis')
#plt.rcParams.update({'font.size': 14})

the_file = sys.argv[1]
evt_number = int(sys.argv[2])
base_vsize = float(sys.argv[3])
kind       = str(sys.argv[4])

with tb.open_file(the_file) as h5in:
    table = getattr(getattr(h5in.root, 'CHITS'), 'highTh').read()
    hits_df = pd.DataFrame.from_records(table)
    this_evt_df = hits_df[hits_df.event == evt_number]
    the_hits = []

    xs = this_evt_df.X
    ys = this_evt_df.Y
    zs = this_evt_df.Z
    es = this_evt_df.Ec

    for x, y, z, e in zip(xs, ys, zs, es):
        h = Hit(0, Cluster(0, xy(x,y), xy(0,0), 0), z, e*1000, xy(0,0))
        the_hits.append(h)

voxels = voxelize_hits(the_hits,
                       np.array([base_vsize, base_vsize, base_vsize]), False)

vsizex = voxels[0].size[0]
vsizey = voxels[0].size[1]
vsizez = voxels[0].size[2]

min_corner_x = min(v.X for v in voxels) - vsizex/2.
min_corner_y = min(v.Y for v in voxels) - vsizey/2.
min_corner_z = min(v.Z for v in voxels) - vsizez/2.

x = [np.round(v.X/vsizex) for v in voxels]
y = [np.round(v.Y/vsizey) for v in voxels]
z = [np.round(v.Z/vsizez) for v in voxels]
e = [v.E for v in voxels]

x_min = int(min(x))
y_min = int(min(y))
z_min = int(min(z))

x_max = int(max(x))
y_max = int(max(y))
z_max = int(max(z))

#print(f'X min = {x_min}, X max = {x_max}')
#print(f'Y min = {y_min}, Y max = {y_max}')
#print(f'Z min = {z_min}, Z max = {z_max}')

VOXELS = np.zeros((x_max-x_min+1, y_max-y_min+1, z_max-z_min+1))

# sort through the event set the "turn on" the hit voxels
cmap = cm.viridis
norm = Normalize(vmin=0, vmax=max(e))

colors = np.empty(VOXELS.shape, dtype=object)
for q in range(0,len(z)):
    VOXELS[int(x[q])-x_min][int(y[q])-y_min][int(z[q])-z_min] = 1
    colors[int(x[q])-x_min][int(y[q])-y_min][int(z[q])-z_min] = cmap(norm(e[q]))

# and plot everything
fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection='3d')
#a,b,c is spacing in mm needs an extra dim
a,b,c = np.indices((x_max-x_min+2, y_max-y_min+2, z_max-z_min+2))
a = a*vsizex + min_corner_x
b = b*vsizey + min_corner_y
c = c*vsizez + min_corner_z

ax.voxels(a,b,c, VOXELS, facecolors=colors , edgecolor='k',alpha=0.6)

ax.set_xlabel('x (mm)',fontsize=16)
ax.set_ylabel('y (mm)',fontsize=16)
ax.set_zlabel('z (mm)',fontsize=16)

#ax.set_xlim([x_min*vsizex-1.*vsizex, x_max*vsizex+1.*vsizex])
#ax.set_ylim([y_min*vsizey-1.*vsizey, y_max*vsizey+1.*vsizey])
#ax.set_zlim([z_min*vsizez-1.*vsizez, z_max*vsizez+1.*vsizez])
#ax.view_init(30, 210)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm)
cb.set_label('Energy (keV)')

particles = pd.read_hdf(the_file, 'MC/particles')
evt_part = particles[particles.event_id == evt_number]

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

print('*** True end-points of main track ***')
print(f'{x_e1}, {y_e1}, {z_e1}')
print(f'{x_e2}, {y_e2}, {z_e2}')

bx = np.array([x_e1, x_e2])
by = np.array([y_e1, y_e2])
bz = np.array([z_e1, z_e2])
ax.scatter(bx, by, bz, marker='x', s=200, linewidth=5, color='red')

## to check that voxels are drawn correctly, draw the center of one of them
#bx = np.array([169.8091666666642])
#by = np.array([-320.4477777777771])
#bz = np.array([22.469493489580834])
#ax.scatter(bx, by, bz, marker='x', s=200, linewidth=5, color='red')

plt.show()
