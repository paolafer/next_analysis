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

from invisible_cities.evm.event_model        import Cluster, Hit
from invisible_cities.types.ic_types     import NN
from invisible_cities.types.ic_types         import xy
import invisible_cities.reco.dst_functions     as dstf
from   invisible_cities.io.dst_io              import load_dst
from invisible_cities.evm.event_model     import HitCollection
from invisible_cities.reco.paolina_functions import voxelize_hits


def load_hits(DST_file_name, evt_number):
    """Return the Hits as PD DataFrames."""

    dst = load_dst(DST_file_name,'RECO','Events')
    dst_size = len(dst)
    all_events = {}

    event = dst.event.values
    time  = dst.time .values
    npeak = dst.npeak.values
    X     = dst.X    .values
    Y     = dst.Y    .values
    Z     = dst.Z    .values
    Q     = dst.Q    .values
    E     = dst.E    .values

    for i in range(dst_size):
        if event[i] == evt_number:
            current_event = all_events.setdefault(event[i],
                                                  HitCollection(event[i], time[i] * 1e-3))
            hit = Hit(npeak[i],
                      Cluster(Q[i], xy(X[i], Y[i]), xy(0, 0),
                              0, Z[i], E[i]), Z[i], E[i], xy(0, 0))
            current_event.hits.append(hit)
    return all_events


def load_hits_skipping_NN(DST_file_name, evt_number):
    """Return the Hits as PD DataFrames."""

    dst = load_dst(DST_file_name,'RECO','Events')
    dst_size = len(dst)
    all_events = {}

    event = dst.event.values
    time  = dst.time .values
    npeak = dst.npeak.values
    X     = dst.X    .values
    Y     = dst.Y    .values
    Z     = dst.Z    .values
    Q     = dst.Q    .values
    E     = dst.E    .values

    for i in range(dst_size):
        if event[i] == evt_number:
            current_event = all_events.setdefault(event[i],
                                                  HitCollection(event[i], time[i] * 1e-3))
            hit = Hit(npeak[i],
                      Cluster(Q[i], xy(X[i], Y[i]), xy(0, 0),
                              0, Z[i], E[i]), Z[i], E[i], xy(0, 0))
            if(hit.Q != NN):
                current_event.hits.append(hit)

    good_events = {}
    for event, hitc in all_events.items():
        if len(hitc.hits) > 0:
            good_events[event] = hitc

    return good_events


def merge_NN_hits(hits_all, hits_nonNN):
    for evt, hc in hits_nonNN.items():
        hc_all = hits_all[evt]
        # Add energy from all NN hits to hits in closest slice.
        for h1 in hc_all.hits:
            if h1.Q == NN:
                # Find the hits to which the energy will be added.
                zdist_min = -1
                h_add = []
                for h2 in hc.hits:
                    if h1.npeak == h2.npeak:
                        zdist = np.abs(h1.Z - h2.Z)
                        if (zdist_min < 0) or (zdist < zdist_min):
                            zdist_min = zdist
                            h_add = []
                            h_add.append(h2)
                        elif zdist == zdist_min:
                            h_add.append(h2)
                # Add the energy.
                if zdist_min > 0:
                    hadd_etot = sum([ha.E for ha in h_add])
                    for ha in h_add:
                        ha.energy += h1.E*(ha.E/hadd_etot)


cmap = plt.cm.get_cmap('jet')

the_file = sys.argv[1]
evt_number = int(sys.argv[2])
mc = int(sys.argv[3])
vxl_size = float(sys.argv[4])

corrections    = "/Users/paola/Software/ic_data/corrections/corrections_run6352.h5"
time_evolution = "/Users/paola/Software/ic_data/corrections/Time_evolution_6352.h5"
LTcorrection = dstf.load_lifetime_xy_corrections(corrections,
                                                 group="XYcorrections",
                                                 node="Elifetime")
XYcorrection  = dstf.load_xy_corrections(corrections,
                                    group = "XYcorrections",
                                    node = f"Egeometry",
                                    norm_strategy = "index",
                                    norm_opts = {"index": (40,40)})

dv_dst  = load_dst(time_evolution, group="parameters", node="test")
dvs = dv_dst.dv.values
drift_velocity = dvs.mean()

if mc:
    correctionsLT   = "/Users/paola/Software/ic_data/corrections/corrections_run6198.h5"
    correctionsXY = "/Users/paola/Software/ic_data/corrections/corrections_MC_4734.h5"
    LTcorrection = dstf.load_lifetime_xy_corrections(correctionsLT,
                                                 group="XYcorrections",
                                                 node="Elifetime")

    XYcorrection  = dstf.load_xy_corrections(correctionsXY,
                                         group="XYcorrections",
                                         node="GeometryE_6.7mm",
                                         norm_strategy = "index",
                                         norm_opts = {"index": (40, 40)})
    drift_velocity = 1


rhits = load_hits(the_file, evt_number)
good_hits = load_hits_skipping_NN(the_file, evt_number)
merge_NN_hits(rhits, good_hits)

bad_evt = False
corr_hits = []

for hh in good_hits[evt_number].hits:
    if XYcorrection(hh.X, hh.Y).value == 0:
        bad_evt = True
        break
    e_corr = hh.E * LTcorrection(hh.Z, hh.X, hh.Y).value * XYcorrection(hh.X, hh.Y).value
    z_corr = hh.Z * drift_velocity
    hcorr = Hit(0,Cluster(0, xy(hh.X,hh.Y), xy(0,0), 0), z_corr, e_corr, xy(0,0))
    corr_hits.append(hcorr)

voxels = voxelize_hits(corr_hits, np.array([vxl_size, vxl_size, vxl_size]), False)

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
cb.set_label('Energy (pes)')

max_range = np.array([max_x-min_x, max_y-min_y, max_z-min_z]).max()/2.0
#print(x[0], x[1], y[0], y[1], z[0], z[1])
#print(np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]]).max())
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
