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

#from itertools import product, combinations

from invisible_cities.io.mcinfo_io import load_mchits
from invisible_cities.io.mcinfo_io import read_mcinfo

from invisible_cities.evm.event_model        import Cluster, Hit
from invisible_cities.types.ic_types         import xy
from invisible_cities.reco.paolina_functions import voxelize_hits

cmap = plt.cm.get_cmap('jet')

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

    xs = [h.X for h in list(hits_dict.values())[0]]
    ys = [h.Y for h in list(hits_dict.values())[0]]
    zs = [h.Z for h in list(hits_dict.values())[0]]
    es = [h.E for h in list(hits_dict.values())[0]]

    the_hits = []
    for x, y, z, e in zip(xs, ys, zs, es):
        h = Hit(0, Cluster(0, xy(x,y), xy(0,0), 0), z, e, xy(0,0))
        the_hits.append(h)

voxels = voxelize_hits(the_hits, np.array([10., 10., 10]), False)

fig = plt.figure() #figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
 #ax = fig.gca(projection='3d')
 #ax.set_aspect("equal")

energies = [v.E for v in voxels]
energies = np.array(energies)
min_energy = energies.min()
max_energy = energies.max()
print('Minimum energy = {}, maximum energy = {}'.format(min_energy, max_energy))

max_x = max_y = max_z = -1.e6
min_x = min_y = min_z = 1.e6

for v in voxels:

    x = [v.X - v.size[0] / 2., v.X + v.size[0] / 2. ]
    y = [v.Y - v.size[1] / 2., v.Y + v.size[1] / 2. ]
    z = [v.Z - v.size[2] / 2., v.Z + v.size[2] / 2. ]

    if x[1] > max_x: max_x = x[1]
    if x[0] < min_x: min_x = x[0]
    if y[1] > max_y: max_y = y[1]
    if y[0] < min_y: min_y = y[0]
    if z[1] > max_z: max_z = z[1]
    if z[0] < min_z: min_z = z[0]

    energy = v.E
    fraction = (energy - min_energy) /(max_energy - min_energy)
    rgba = cmap(fraction)

    #for s, e in combinations(np.array(list(product(x, y, z))), 2):
   #     if np.isclose(np.linalg.norm(np.abs(s-e)), v.size[0]) or np.isclose(np.linalg.norm(np.abs(s-e)), v.size[1]) or np.isclose(np.linalg.norm(np.abs(s-e)), v.size[2]):
    #        ax.plot3D(*zip(s, e), color="b")

    xx, yy = np.meshgrid(x, y)
    normal = np.array([0, 0, 1])
    d = -z[0]
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    ax.plot_surface(xx, yy, zz, alpha=.9, color=rgba)
    d = -z[1]
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    ax.plot_surface(xx, yy, zz, alpha=.9, color=rgba)

    xx, zz = np.meshgrid(x, z)
    normal = np.array([0, 1, 0])
    d = -y[0]
    yy = (-normal[0] * xx - normal[2] * zz - d) * 1. /normal[1]
    ax.plot_surface(xx, yy, zz, alpha=.9, color=rgba)
    d = -y[1]
    yy = (-normal[0] * xx - normal[2] * zz - d) * 1. /normal[1]
    ax.plot_surface(xx, yy, zz, alpha=.9, color=rgba)

    yy, zz = np.meshgrid(y, z)
    normal = np.array([1, 0, 0])
    d = -x[0]
    xx = (-normal[1] * yy - normal[2] * zz - d) * 1. /normal[0]
    ax.plot_surface(xx, yy, zz, alpha=.9, color=rgba)
    d = -x[1]
    xx = (-normal[1] * yy - normal[2] * zz - d) * 1. /normal[0]
    ax.plot_surface(xx, yy, zz, alpha=.9, color=rgba)

norm = clrs.Normalize(vmin=min_energy, vmax=max_energy)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm)
cb.set_label('Energy (MeV)')

max_range = np.array([max_x-min_x, max_y-min_y, max_z-min_z]).max()/2.0
mid_x = (min_x+max_x) * 0.5
mid_y = (min_y+max_y) * 0.5
mid_z = (min_z+max_z) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')
plt.show()
