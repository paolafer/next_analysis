
import os
import sys
import random
import tables as tb
import numpy  as np
import pandas  as pd
from pytest import approx

from   invisible_cities.database               import load_db

import invisible_cities.reco.paolina_functions as plf
from   invisible_cities.reco.dst_functions     import load_xy_corrections
import invisible_cities.reco.dst_functions     as dstf

from   invisible_cities.io.dst_io              import load_dst
from   invisible_cities.io.hits_io             import load_hits
from   invisible_cities.io.hits_io             import load_hits_skipping_NN
from   invisible_cities.io.mcinfo_io           import load_mcparticles

from   invisible_cities.types.ic_types         import xy
from   invisible_cities.types.ic_types         import NN

from   invisible_cities.evm.event_model        import Cluster, Hit


def merge_NN_hits(hits_all, hits_nonNN, lost_energy):

    # Iterate through the nonNN dictionary and update the energies including the NN hits from the "all" dictionary.
    for evt, hc in hits_nonNN.items():

        # Get the corresponding collection of all hits.
        hc_all = hits_all[evt]

        not_assigned_energy = 0.
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
                else:
                    not_assigned_energy += h1.E
        lost_energy[evt] = not_assigned_energy



correctionsLT   = "/home/paolafer/corrections/corrections_run6198.h5"
correctionsXY = "/home/paolafer/corrections/corrections_MC_4734.h5"

LTcorrection = dstf.load_lifetime_xy_corrections(correctionsLT,
                                                 group="XYcorrections",
                                                 node="Elifetime")

XYcorrection  = dstf.load_xy_corrections(correctionsXY,
                                         group="XYcorrections",
                                         node="GeometryE_6.7mm",
                                         norm_strategy = "index",
                                         norm_opts = {"index": (40, 40)})


drift_velocity = 1.


start = int(sys.argv[1])
numb = int(sys.argv[2])
size = float(sys.argv[3])
blob_radius = float(sys.argv[4])
threshold = int(sys.argv[5])

vox_size = np.array([size,size,size],dtype=np.float16)    # voxel size
pe2keV = 1.
voxel_cut = 6071.84 # corresponds to 20 keV
#voxel_cut = 3035.42 # corresponds to 10 keV

loop_events, not_fid_events = [], []
event, track_ID = [], []
raw_evt_energy, lost_raw_evt_energy = [], []
maxR, minX, maxX, minY, maxY, minZ, maxZ = [], [], [], [], [], [], []
evt_energy, energy = [], []
length, numb_of_hits, numb_of_voxels, numb_of_tracks = [], [], [], []
v_size_x, v_size_y, v_size_z = [], [], []
extreme1_x, extreme1_y, extreme1_z = [], [], []
extreme2_x, extreme2_y, extreme2_z = [], [], []
eblob1, eblob2 = [], []
blob1_x, blob1_y, blob1_z = [], [], []
blob2_x, blob2_y, blob2_z = [], [], []
event_vxls, track_ID_vxls = [], []
voxel_x, voxel_y, voxel_z = [], [], []
voxel_e = []
signal, blob_overlap = [], []


hits_file = ''
not_fid = 0
events_in = 0
for n in range(start,start+numb):

    for part in range(10):

        hits_file = '/data_extra/paolafer/SimMC/Tl208/hits/Tl208_NEW_v1_03_01_nexus_v5_03_04_UPPER_PORT_10.2bar_run4_1hit_perSiPM_{0}pes_hits.{1}_{2}.h5'.format(threshold, n, part)

        if not os.path.isfile(hits_file):
            print('{0} not existing'.format(hits_file))
            continue

        print('Analyzing {0}'.format(hits_file))

        lost_energy = {}

        rhits = load_hits(hits_file)
        good_hits = load_hits_skipping_NN(hits_file)
        merge_NN_hits(rhits, good_hits, lost_energy)

        hits_evt = {}
        raw_energy = {}
        bad_event = {}

        for ee, hc in good_hits.items():
            bad_evt = False
            hc_corr = []

            hX = [hh.X for hh in hc.hits]
            hY = [hh.Y for hh in hc.hits]
            hZ = [hh.Z for hh in hc.hits]
            hE = [hh.E for hh in hc.hits]

            hEcorr = hE * LTcorrection(hZ, hX, hY).value * XYcorrection(hX, hY).value

            for h, ec in zip(hc.hits, hEcorr):
                if ec == 0:
                    bad_evt = True
                    break
                hcorr = Hit(0, Cluster(0, xy(h.X,h.Y), xy(0,0), 0), h.Z*drift_velocity, ec, xy(h.Xpeak,h.Ypeak))
                hc_corr.append(hcorr)

            hits_evt[ee] = hc_corr
            bad_event[ee] = bad_evt
            raw_energy[ee] = sum(hE)

        p_dict = load_mcparticles(hits_file)

        events_in += len(hits_evt)
        for nevt, hitc in hits_evt.items():

            if bad_event[nevt]:
                not_fid += 1
                continue

            tot_e = sum(h.E for h in hitc)

            voxels     = plf.voxelize_hits(hitc, vox_size)
            mod_voxels = plf.drop_end_point_voxels(voxels, voxel_cut, min_vxls=3)

            ### Make tracks with the new voxels
            trks = plf.make_track_graphs(mod_voxels)

            ### Is it a e+e- events?
            positron = False
            for _, particle in p_dict[nevt].items():
                if (particle.name == 'e+') & (len(particle.hits) > 0):
                    positron = True

            for c, t in enumerate(trks, 0):
                etrk   = sum(vox.E for vox in t.nodes())
                n_hits = sum(len(v.hits) for v in t.nodes())

                extr1, extr2 = plf.find_extrema(t)
                e_blob1, e_blob2, hits_blob1, hits_blob2, pos1, pos2 = plf.blob_energies_hits_and_centres(t, blob_radius)

                overlap = False
                if len(set(hits_blob1).intersection(hits_blob2)) > 0:
                    overlap = True


                ## event-related
                event += [nevt]
                signal += [positron]
                raw_evt_energy += [raw_energy[nevt]]
                lost_raw_evt_energy += [lost_energy[nevt]]
                evt_energy += [tot_e/pe2keV]
                v_size_x += [voxels[0].size[0]]
                v_size_y += [voxels[0].size[1]]
                v_size_z += [voxels[0].size[2]]

                ## track-related
                track_ID += [c]
                length += [plf.length(t)]
                energy += [etrk/pe2keV]
                numb_of_voxels += [len(t.nodes())]
                numb_of_tracks += [len(trks)]
                numb_of_hits += [n_hits]
                extreme1_x += [extr1.X]
                extreme1_y += [extr1.Y]
                extreme1_z += [extr1.Z]
                extreme2_x += [extr2.X]
                extreme2_y += [extr2.Y]
                extreme2_z += [extr2.Z]
                eblob1 += [e_blob1/pe2keV]
                eblob2 += [e_blob2/pe2keV]
                blob_overlap += [overlap]

                blob1_x += [pos1[0]]
                blob1_y += [pos1[1]]
                blob1_z += [pos1[2]]
                blob2_x += [pos2[0]]
                blob2_y += [pos2[1]]
                blob2_z += [pos2[2]]

                min_x = min([h.X for v in t.nodes() for h in v.hits])
                max_x = max([h.X for v in t.nodes() for h in v.hits])
                min_y = min([h.Y for v in t.nodes() for h in v.hits])
                max_y = max([h.Y for v in t.nodes() for h in v.hits])
                min_z = min([h.Z for v in t.nodes() for h in v.hits])
                max_z = max([h.Z for v in t.nodes() for h in v.hits])
                max_r = max([np.sqrt(h.X*h.X + h.Y*h.Y) for v in t.nodes() for h in v.hits])

                minX += [min_x]
                maxX += [max_x]
                minY += [min_y]
                maxY += [max_y]
                minZ += [min_z]
                maxZ += [max_z]
                maxR += [max_r]

                for v in t.nodes():
                    ## voxel-related
                    event_vxls += [nevt]
                    track_ID_vxls += [c]
                    voxel_x += [v.X]
                    voxel_y += [v.Y]
                    voxel_z += [v.Z]
                    voxel_e += [v.E]


loop_events = [events_in]
not_fid_events = [not_fid]
blob_radius = [blob_radius]
voxel_cut = [voxel_cut]
threshold_SiPMs = [threshold]

df = pd.DataFrame({'event': event, 'signal': signal,
                   'raw_evt_energy': raw_evt_energy, 'lost_raw_evt_energy': lost_raw_evt_energy,
                   'evt_energy': evt_energy,  'energy': energy,
                   'minX': minX, 'maxX': maxX, 'minY': minY, 'maxY': maxY, 'minZ': minZ, 'maxZ': maxZ,
                   'maxR': maxR,
                   'numb_of_hits': numb_of_hits, 'length': length, 'track_ID': track_ID,
                   'numb_of_tracks': numb_of_tracks,
                   'numb_of_voxels': numb_of_voxels,
                   'voxel_size_x': v_size_x, 'voxel_size_y': v_size_y,
                   'voxel_size_z': v_size_z,
                   'extreme1_x': extreme1_x, 'extreme1_y': extreme1_y, 'extreme1_z': extreme1_z,
                   'extreme2_x': extreme2_x, 'extreme2_y': extreme2_y, 'extreme2_z': extreme2_z,
                   'eblob1': eblob1, 'eblob2': eblob2, 'blob_overlap': blob_overlap,
                   'blob1_x': blob1_x, 'blob1_y': blob1_y, 'blob1_z': blob1_z,
                   'blob2_x': blob2_x, 'blob2_y': blob2_y, 'blob2_z': blob2_z,
                   })
df_vxls = pd.DataFrame({'event': event_vxls, 'track_ID': track_ID_vxls,
                        'voxel_x': voxel_x, 'voxel_y': voxel_y, 'voxel_z': voxel_z, 'voxel_e': voxel_e
                        })

df_run_info = pd.DataFrame({'events_in': loop_events, 'not_fid': not_fid_events,
                            'blob_radius': blob_radius, 'voxel_cut': voxel_cut, 'threshold_SiPMs': threshold_SiPMs
                           })
out_name = '/home/paolafer/analysis/10bar/tracking/tracking_drop_voxels_1hit_perSiPM_{0}pes_TlMC_run4_check_vxl{1}mm_R{2}mm_{3}_{4}.hdf5'.format(threshold, int(size), int(blob_radius[0]), start, numb)

store = pd.HDFStore(out_name, "w", complib=str("zlib"), complevel=4)
store.put('tracks', df, format='table', data_columns=True)
store.put('voxels', df_vxls, format='table', data_columns=True)
store.put('run_info', df_run_info, format='table', data_columns=True)
store.close()
