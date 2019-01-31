
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

def neighbours(va, vb):
    return np.linalg.norm((va.pos - vb.pos) / va.size) < 1.8


corrections    = "/home/paolafer/corrections/corrections_run6352.h5"
time_evolution = "/home/paolafer/corrections/Time_evolution_6352.h5"
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


run_number = int(sys.argv[1])
start = int(sys.argv[2])
numb = int(sys.argv[3])
size = float(sys.argv[4])
blob_radius = float(sys.argv[5])

vox_size = np.array([size,size,size],dtype=np.float16)    # voxel size
pe2keV = 1.
voxel = 6259.85 # corresponds to 20 keV

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
eblob1_bary, eblob2_bary = [], []
blob1_bary_x, blob1_bary_y, blob1_bary_z = [], [], []
blob2_bary_x, blob2_bary_y, blob2_bary_z = [], [], []
event_vxls, track_ID_vxls = [], []
voxel_x, voxel_y, voxel_z = [], [], []
voxel_e = []
energy_corr, eblob1_corr_bary, eblob2_corr_bary = [], [], []
time = []


hits_file = ''
events_in = 0
not_fid = 0
for n in range(start,start+numb):

    hits_file = '/data_extra/paolafer/data/r{0}/hits/hits_{0}_trigger2_v0.9.9_20180921_krth1300.{1:04d}.h5'.format(run_number, n)

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
        times[ee] = hc.time
        raw_energy[ee] = sum(hE)

    events_in += len(hits_evt)
    for nevt, hitc in hits_evt.items():

        if bad_event[nevt]:
            not_fid += 1
            continue

        tot_e = sum([hh.E for hh in hitc])
        voxels = plf.voxelize_hits(hitc, vox_size)
        trks = plf.make_track_graphs(voxels)

        while True:
            modified_voxels = 0

            trks = plf.make_track_graphs(voxels)
            vxl_size = voxels[0].size

            for t in trks:
                if len(t.nodes()) < 3:
                    continue

                extr1, extr2 = plf.find_extrema(t)

                if extr1.E < voxel_cut:
                    modified_voxels += 1
                    ### remove voxel from list of voxels
                    voxels.remove(extr1)
                    ### remove hits from the list of hits
                    pos1 = [h.pos for h in extr1.hits]
                    qs1 = [h.E for h in extr1.hits]
                    bary_extr1_pos = np.average(pos1, weights=qs1, axis=0)
                    for h in extr1.hits:
                        hitc.remove(h)

                    ### find hit with minimum distance, only among neighbours
                    min_dist = 1e+06
                    min_hit = hitc[0]
                    min_v = voxels[0]
                    for v in voxels:
                        if neighbours(extr1, v):
                            for hh in v.hits:
                                dist = np.linalg.norm(bary_extr1_pos - hh.pos)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_hit = hh
                                    min_v = v
                    ### add voxel energy to hit and voxel, separately
                    min_hit.energy += extr1.E
                    min_v.energy   += extr1.E

                if extr2.E < voxel_cut:
                    modified_voxels += 1
                    ### remove voxel from list of voxels
                    voxels.remove(extr2)
                    ### remove hits from the list of hits
                    pos2 = [h.pos for h in extr2.hits]
                    qs2 = [h.E for h in extr2.hits]
                    bary_extr2_pos = np.average(pos2, weights=qs2, axis=0)
                    for h in extr2.hits:
                        hitc.remove(h)

                    ### find hit with minimum distance, only among neighbours
                    min_dist = 1e+06
                    min_hit = hitc[0]
                    min_v = voxels[0]
                    for v in voxels:
                        if neighbours(extr2, v):
                            for hh in v.hits:
                                dist = np.linalg.norm(bary_extr2_pos - hh.pos)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_hit = hh
                                    min_v = v
                    ### add voxel energy to hit and voxel, separately
                    min_hit.energy += extr2.E
                    min_v.energy   += extr2.E

            if modified_voxels == 0: break

        ### Make voxels with the new list of hits
        trks = plf.make_track_graphs(voxels)
        tot_e = sum([hh.E for hh in hitc])

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

            ## second way to calculate blob (à la Michel)
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


            ### Save quantities before E correction
            raw_evt_energy += [raw_energy[nevt]]
            lost_raw_evt_energy += [lost_energy[nevt]]
            evt_energy += [tot_e/pe2keV]
            energy += [etrk/pe2keV]

            eblob1 += [e_blob1/pe2keV]
            eblob2 += [e_blob2/pe2keV]
            eblob1_bary += [e_blob1_bary/pe2keV]
            eblob2_bary += [e_blob2_bary/pe2keV]

            min_x = min([h.X for v in t.nodes() for h in v.hits])
            max_x = max([h.X for v in t.nodes() for h in v.hits])
            min_y = min([h.Y for v in t.nodes() for h in v.hits])
            max_y = max([h.Y for v in t.nodes() for h in v.hits])
            min_z = min([h.Z for v in t.nodes() for h in v.hits])
            max_z = max([h.Z for v in t.nodes() for h in v.hits])
            max_r = max([np.sqrt(h.X*h.X + h.Y*h.Y) for v in t.nodes() for h in v.hits])

            z_factor = 1 - 3.4e-04 * (max_z - min_z)

            for v in t.nodes():
                for h in v.hits:
                    h.energy = h.energy/z_factor
                v.energy = v.energy/z_factor

        ### we need to create tracks again, because of networkx algorithms
        trks = plf.make_track_graphs(voxels)

        for c, t in enumerate(trks, 0):
            etrk_corr = sum([vox.E for vox in t.nodes()])
            extr1, extr2 = plf.find_extrema(t)

            ### method 1
            e_blob1_corr = e_blob2_corr = 0.
            for v in t.nodes():
                for h in v.hits:
                    dist1 = np.linalg.norm(h.pos - extr1.pos)
                    dist2 = np.linalg.norm(h.pos - extr2.pos)
                    if dist1 < blob_radius:
                        e_blob1_corr += h.E
                    if dist2 < blob_radius:
                        e_blob2_corr += h.E

            if (e_blob2_corr > e_blob1_corr):
                e_blob1_corr, e_blob2_corr = e_blob2_corr, e_blob1_corr

            ### method 2
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

            e_blob1_corr_bary = e_blob2_corr_bary = 0.
            for v in t.nodes():
                for h in v.hits:
                    dist1 = np.linalg.norm(h.pos - bary_pos1)
                    dist2 = np.linalg.norm(h.pos - bary_pos2)
                    if dist1 < blob_radius:
                        e_blob1_corr_bary += h.E
                    if dist2 < blob_radius:
                        e_blob2_corr_bary += h.E

            if (e_blob2_corr_bary > e_blob1_corr_bary):
                e_blob1_corr_bary, e_blob2_corr_bary = e_blob2_corr_bary, e_blob1_corr_bary  


            ## event-related
            event += [nevt]
            numb_of_hits += [len(hitc)]
            v_size_x += [voxels[0].size[0]]
            v_size_y += [voxels[0].size[1]]
            v_size_z += [voxels[0].size[2]]

            ## track-related
            track_ID += [c]
            length += [plf.length(t)]
            numb_of_voxels += [len(t.nodes())]
            numb_of_tracks += [len(trks)]
            extreme1_x += [extr1.X]
            extreme1_y += [extr1.Y]
            extreme1_z += [extr1.Z]
            extreme2_x += [extr2.X]
            extreme2_y += [extr2.Y]
            extreme2_z += [extr2.Z]

            blob1_bary_x += [bary_pos1[0]]
            blob1_bary_y += [bary_pos1[1]]
            blob1_bary_z += [bary_pos1[2]]
            blob2_bary_x += [bary_pos2[0]]
            blob2_bary_y += [bary_pos2[1]]
            blob2_bary_z += [bary_pos2[2]]

            energy_corr += [etrk_corr/pe2keV]
            eblob1_corr += [e_blob1_corr/pe2keV]
            eblob2_corr += [e_blob2_corr/pe2keV]
            eblob1_corr_bary += [e_blob1_corr_bary/pe2keV]
            eblob2_corr_bary += [e_blob2_corr_bary/pe2keV]

            minX += [min_x]
            maxX += [max_x]
            minY += [min_y]
            maxY += [max_y]
            minZ += [min_z]
            maxZ += [max_z]
            maxR += [max_r]

            time += [times[nevt]]

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

#print(len(numb_of_tracks_before), len(numb_of_tracks))
df = pd.DataFrame({'event': event,
                   'raw_evt_energy': raw_evt_energy, 'lost_raw_evt_energy': lost_raw_evt_energy,
                   'evt_energy': evt_energy,  'energy': energy, 'time': time,
                   'minX': minX, 'maxX': maxX, 'minY': minY, 'maxY': maxY, 'minZ': minZ, 'maxZ': maxZ, 'maxR': maxR,
                   'numb_of_hits': numb_of_hits, 'length': length, 'track_ID': track_ID,
                   'numb_of_tracks': numb_of_tracks,g
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
df_run_info = pd.DataFrame({'events_in': loop_events, 'not_fid': not_fid_events,
                            'blob_radius': blob_radius
                           })

out_name = '/home/paolafer/analysis/10bar/tracking/qthr3_qlm35_rebin2_nsipm6/tracking_drop_voxels_mod_corona_r{0}_vxl{1}mm_R{2}mm_{3}_{4}.hdf5'.format(run_number, int(size), int(blob_radius[0]), start, numb)
store = pd.HDFStore(out_name, "w", complib=str("zlib"), complevel=4)
store.put('tracks', df, format='table', data_columns=True)
store.put('voxels', df_vxls, format='table', data_columns=True)
store.put('run_info', df_run_info, format='table', data_columns=True)
store.close()

