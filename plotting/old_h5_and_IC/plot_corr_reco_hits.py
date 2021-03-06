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

from invisible_cities.types.ic_types     import NN
from invisible_cities.types.ic_types     import xy
import invisible_cities.reco.dst_functions     as dstf
from   invisible_cities.io.dst_io              import load_dst
from invisible_cities.evm.event_model     import HitCollection
from invisible_cities.evm.event_model     import Hit, Cluster

#from   invisible_cities.io.hits_io             import load_hits
#from   invisible_cities.io.hits_io             import load_hits_skipping_NN

def load_hits(DST_file_name, evt_number):
    """Return the Hits as PD DataFrames."""

    dst = load_dst(DST_file_name,'RECO','Events')
    dst_size = len(dst)
    all_events = {}

    event = dst.event.values
    time  = dst.time .values
    npeak = dst.npeak.values
#    nsipm = dst.nsipm.values
    X     = dst.X    .values
    Y     = dst.Y    .values
#    Xrms  = dst.Xrms .values
#    Yrms  = dst.Yrms .values
    Z     = dst.Z    .values
    Q     = dst.Q    .values
    E     = dst.E    .values

#    Xpeak = getattr(dst, 'Xpeak', [-1000] * dst_size)
#    Ypeak = getattr(dst, 'Ypeak', [-1000] * dst_size)

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
#    nsipm = dst.nsipm.values
    X     = dst.X    .values
    Y     = dst.Y    .values
#    Xrms  = dst.Xrms .values
#    Yrms  = dst.Yrms .values
    Z     = dst.Z    .values
    Q     = dst.Q    .values
    E     = dst.E    .values

#    Xpeak = getattr(dst, 'Xpeak', [-1000] * dst_size)
#    Ypeak = getattr(dst, 'Ypeak', [-1000] * dst_size)

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


the_file = sys.argv[1]
evt_number = int(sys.argv[2])
mc = int(sys.argv[3])

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
hc_corr = []

x, y, z, e = [], [], [], []
for hh in good_hits[evt_number].hits:
    if XYcorrection(hh.X, hh.Y).value == 0:
        bad_evt = True
        break
    e_corr = hh.E * LTcorrection(hh.Z, hh.X, hh.Y).value * XYcorrection(hh.X, hh.Y).value
    z_corr = hh.Z * drift_velocity
    x.append(hh.X)
    y.append(hh.Y)
    z.append(z_corr)
    e.append(e_corr)

if bad_evt:
    print('No reconstructed hits in this event.')
    exit()

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
cb.set_label('Energy (pes)')

plt.show()

