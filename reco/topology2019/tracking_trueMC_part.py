
import os
import sys
import random
import tables as tb
import numpy  as np
import pandas  as pd

import invisible_cities.reco.paolina_functions as plf
import invisible_cities.reco.dst_functions     as dstf

from   invisible_cities.io.mcinfo_io           import load_mchits
from   invisible_cities.io.mcinfo_io           import load_mcparticles


start = int(sys.argv[1])
numb = int(sys.argv[2])
size = float(sys.argv[3])
blob_radius = float(sys.argv[4])

vox_size = np.array([size,size,size],dtype=np.float16)    # voxel size
pe2keV = 1.

loop_events = []
event, track_ID = [], []
maxR, minX, maxX, minY, maxY, minZ, maxZ = [], [], [], [], [], [], []
evt_energy, energy = [], []
length, numb_of_hits, numb_of_voxels, numb_of_tracks = [], [], [], []
v_size_x, v_size_y, v_size_z = [], [], []
extreme1_x, extreme1_y, extreme1_z = [], [], []
extreme2_x, extreme2_y, extreme2_z = [], [], []
eblob1, eblob2 = [], []
eblob1_bary, eblob2_bary = [], []
blob1_bary_x, blob1_bary_y, blob1_bary_z = [], [], []
blob2_bary_x, blob2_bary_y, blob2_bary_z = [], [], []
event_vxls, track_ID_vxls = [], []
voxel_x, voxel_y, voxel_z = [], [], []
voxel_e = []
signal = []

hits_file = ''
events_in = 0
for n in range(start,start+numb):

    for part in range(10):
        hits_file = '/home/paolafer/data/MC/Tl_upper_port/hits/Tl208_NEW_v1_03_01_nexus_v5_03_04_UPPER_PORT_10.2bar_run4_1hit_perSiPM_hits.{0}_{1}.h5'.format(n, part)

        if not os.path.isfile(hits_file):
            print('{0} not existing'.format(hits_file))
            continue

        print('Analyzing {0}'.format(hits_file))

        hits_dict = load_mchits(hits_file)
        p_dict = load_mcparticles(hits_file)

        events_in += len(hits_dict)

        for nevt, hitc in hits_dict.items():

            tot_e = sum([hh.E for hh in hitc])

            ### smear hit energy to create 1% FWHM resolution at 1592 keV
            sigma_e = 0.01/2.35 * np.sqrt(1.592/tot_e) ### remember, this is relative!
            smeared_tot_e = tot_e + tot_e*np.random.normal(0., 1.) * sigma_e
            sm_factor = smeared_tot_e / tot_e
            #print(tot_e, smeared_tot_e)
            for h in hitc:
                h.energy = h.energy * sm_factor

            voxels = plf.voxelize_hits(hitc, vox_size)
            trks = plf.make_track_graphs(voxels)

            ### Is it a e+e- events?
            positron = False
            for _, particle in p_dict[nevt].items():
                if (particle.name == 'e+') & (len(particle.hits) > 0):
                    positron = True

            for c, t in enumerate(trks, 0):

                etrk = sum([vox.E for vox in t.nodes()])
                extr1, extr2 = plf.find_extrema(t)

                ## first way to calculate blobs: using hits within a sphere from the extremes 
                e_blob1 = e_blob2 = 0.
                for h in hitc:
                    dist1 = np.linalg.norm(h.pos - extr1.pos)
                    dist2 = np.linalg.norm(h.pos - extr2.pos)
                    if dist1 < blob_radius:
                        e_blob1 += h.E
                    if dist2 < blob_radius:
                        e_blob2 += h.E

                if (e_blob2 > e_blob1):
                    e_blob1, e_blob2 = e_blob2, e_blob1

                ## second way to calculate blob (a la Michel)
                positions1 = [h.pos for h in extr1.hits]
                qs1 = [h.E for h in extr1.hits]
                if sum(qs1):
                    bary_pos1 = np.average(positions1, weights=qs1, axis=0)
                else:
                    bary_pos1 = extr1.pos

                positions2 = [h.pos for h in extr2.hits]
                qs2 = [h.E for h in extr2.hits]
                if sum(qs2):
                    bary_pos2 = np.average(positions2, weights=qs2, axis=0)
                else:
                    bary_pos2 = extr2.pos

                e_blob1_bary = e_blob2_bary = 0.
                for h in hitc:
                    dist1 = np.linalg.norm(h.pos - bary_pos1)
                    dist2 = np.linalg.norm(h.pos - bary_pos2)
                    if dist1 < blob_radius:
                        e_blob1_bary += h.E
                    if dist2 < blob_radius:
                        e_blob2_bary += h.E

                if (e_blob2_bary > e_blob1_bary):
                    e_blob1_bary, e_blob2_bary = e_blob2_bary, e_blob1_bary


                ## event-related
                event += [nevt]
                signal += [positron]
                evt_energy += [tot_e/pe2keV]
                numb_of_hits += [len(hitc)]
                v_size_x += [voxels[0].size[0]]
                v_size_y += [voxels[0].size[1]]
                v_size_z += [voxels[0].size[2]]

                ## track-related
                track_ID += [c]
                length += [plf.length(t)]
                energy += [etrk/pe2keV]
                numb_of_voxels += [len(t.nodes())]
                numb_of_tracks += [len(trks)]
                extreme1_x += [extr1.X]
                extreme1_y += [extr1.Y]
                extreme1_z += [extr1.Z]
                extreme2_x += [extr2.X]
                extreme2_y += [extr2.Y]
                extreme2_z += [extr2.Z]
                eblob1 += [e_blob1/pe2keV]
                eblob2 += [e_blob2/pe2keV]
                eblob1_bary += [e_blob1_bary/pe2keV]
                eblob2_bary += [e_blob2_bary/pe2keV]
                blob1_bary_x += [bary_pos1[0]]
                blob1_bary_y += [bary_pos1[1]]
                blob1_bary_z += [bary_pos1[2]]
                blob2_bary_x += [bary_pos2[0]]
                blob2_bary_y += [bary_pos2[1]]
                blob2_bary_z += [bary_pos2[2]]

                min_x = 1e+06
                max_x = -1e+06
                min_y = 1e+06
                max_y = -1e+06
                min_z = 1e+06
                max_z = 0.
                max_r = 0

                for v in t.nodes():
                    ## voxel-related
                    event_vxls = event_vxls + [nevt]
                    track_ID_vxls = track_ID_vxls + [c]
                    voxel_x = voxel_x + [v.X]
                    voxel_y = voxel_y + [v.Y]
                    voxel_z = voxel_z + [v.Z]
                    voxel_e = voxel_e + [v.E]

                    for h in v.hits:
                        if h.X < min_x:
                            min_x = h.X
                        if h.X > max_x:
                            max_x = h.X
                        if h.Y < min_y:
                            min_y = h.Y
                        if h.Y > max_y:
                            max_y = h.Y
                        if h.Z < min_z:
                            min_z = h.Z
                        if h.Z > max_z:
                            max_z = h.Z
                        if np.sqrt(h.X*h.X + h.Y*h.Y) > max_r:
                            max_r = np.sqrt(h.X*h.X + h.Y*h.Y)

                minX += [min_x]
                maxX += [max_x]
                minY += [min_y]
                maxY += [max_y]
                minZ += [min_z]
                maxZ += [max_z]
                maxR += [max_r]

loop_events = [events_in]
blob_radius = [blob_radius]

df = pd.DataFrame({ 'event': event, 'evt_energy': evt_energy, 'signal': signal,
                    'minX': minX, 'maxX': maxX, 'minY': minY, 'maxY': maxY, 'minZ': minZ, 'maxZ': maxZ,
                    'maxR': maxR,
                    'numb_of_hits': numb_of_hits, 'energy': energy,
                    'numb_of_tracks': numb_of_tracks, 'length': length, 'track_ID': track_ID,
                    'numb_of_voxels': numb_of_voxels,
                    'voxel_size_x': v_size_x, 'voxel_size_y': v_size_y,
                    'voxel_size_z': v_size_z, 'eblob1': eblob1, 'eblob2': eblob2,
                    'extreme1_x': extreme1_x, 'extreme1_y': extreme1_y, 'extreme1_z': extreme1_z,
                    'extreme2_x': extreme2_x, 'extreme2_y': extreme2_y, 'extreme2_z': extreme2_z,
                    'eblob1_bary': eblob1_bary, 'eblob2_bary': eblob2_bary,
                    'blob1_bary_x': blob1_bary_x, 'blob1_bary_y': blob1_bary_y, 'blob1_bary_z': blob1_bary_z,
                    'blob2_bary_x': blob2_bary_x, 'blob2_bary_y': blob2_bary_y, 'blob2_bary_z': blob2_bary_z,
                     })
df_vxls = pd.DataFrame({'event': event_vxls, 'track_ID': track_ID_vxls,
                        'voxel_x': voxel_x, 'voxel_y': voxel_y, 'voxel_z': voxel_z, 'voxel_e': voxel_e
                        })

df_run_info = pd.DataFrame({'events_in': loop_events, 'blob_radius': blob_radius
                           })

out_name = '/home/paolafer/analysis/tracking_trueinfo_TlMC_run4_vxl{0}mm_R{1}mm_{2}_{3}.hdf5'.format(int(size), int(blob_radius[0]), start, numb)

store = pd.HDFStore(out_name, "w", complib=str("zlib"), complevel=4)
store.put('tracks', df, format='table', data_columns=True)
store.put('voxels', df_vxls, format='table', data_columns=True)
store.put('run_info', df_run_info, format='table', data_columns=True)
store.close()
