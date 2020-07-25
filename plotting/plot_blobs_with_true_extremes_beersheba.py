import os
import sys
import tables as tb
import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
This scripts plots the hits reconstructed with Beersheba and the blobs
centred in the reconstructed end-points of the track (in green).
The true end-point of the primary electron (for single e-) or of
the two primary electrons (bb0nu) are also shown in black.
Only these two kinds of events can be displayed.
Only MC events can be displayed.
"""


hit_file   = sys.argv[1]
track_file  = sys.argv[2]
evt_number = int(sys.argv[3])
radius     = int(sys.argv[4])
kind       = str(sys.argv[5])# bb0nu for signal, whatever else for single e-

with tb.open_file(hit_file) as h5in:
    table = getattr(getattr(h5in.root, 'DECO'), 'Events').read()
    hits_df = pd.DataFrame.from_records(table)
    this_evt_df = hits_df[hits_df.event == evt_number]

    x = this_evt_df.X
    y = this_evt_df.Y
    z = this_evt_df.Z
    e = this_evt_df.E*1000

    tracks_df    = pd.read_hdf(track_file, 'Tracks')
    this_evt_trk = tracks_df[tracks_df.event == evt_number]
    main_trk = this_evt_trk[this_evt_trk.energy == this_evt_trk.energy.max()]
    if len(main_trk) > 1:
        print(f'Event {evt_number} cannot be plotted, since it has more tracks withe same largest energy')
        exit()

    x_b1 = main_trk.extreme1_x.values
    y_b1 = main_trk.extreme1_y.values
    z_b1 = main_trk.extreme1_z.values
    x_b2 = main_trk.extreme2_x.values
    y_b2 = main_trk.extreme2_y.values
    z_b2 = main_trk.extreme2_z.values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()/2.0
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
    ax.scatter(bx, by, bz, marker='x', s=100, linewidth=5, color='green')


    #### True end-points ###
    particles = pd.read_hdf(hit_file, 'MC/particles')
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
    ax.scatter(bx, by, bz, marker='x', s=200, linewidth=5, color='black')

    plt.show()
