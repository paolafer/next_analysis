import pandas as pd
import numpy as np

import os, sys

import tables as tb

import invisible_cities.io    .dst_io          as dstio
import invisible_cities.io    .mcinfo_io       as mcio

from invisible_cities.evm                       import event_model as evm
from invisible_cities.evm  .event_model    import Cluster, Hit
from invisible_cities.types.ic_types       import xy

from functools   import partial
from typing      import Tuple
from typing      import Callable
from typing      import Optional

from invisible_cities.reco                import paolina_functions    as plf

"""
This script takes true MC hits as an input and voxelizes them and create tracks
using paolina algorithms.
"""


def convert_df_to_hits(df):
    return [Hit(0, Cluster(0, xy(h.x,h.y), xy(0,0), 0), h.z, h.energy, xy(0, 0))
            for h in df.itertuples(index=False)]


def track_blob_info_creator_extractor(vox_size         : [float, float, float],
                                      energy_type      : evm.HitEnergy        ,
                                      strict_vox_size  : bool                 ,
                                      blob_radius      : float
                                     ) -> Callable:
    """
    For a given paolina parameters returns a function that extract tracks / blob information from a HitCollection.

    Parameters
    ----------
    vox_size         : [float, float, float]
        (maximum) size of voxels for track reconstruction
    energy_type      : HitEnergy
        class HitEnergy(AutoNameEnumBase):
            E        = auto()
            Ec       = auto()
        energy attribute to use for voxelization/ tracks

    strict_vox_size  : bool
        if False allows per event adaptive voxel size,
        smaller of equal thatn vox_size.
    blob_radius      : float
        radius of blob

    Returns
    ----------
    A function that from a given HitCollection returns a pandas DataFrame with per track information.
    """
    def create_extract_track_blob_info(hitc):
        voxels     = plf.voxelize_hits(hitc.hits, vox_size, strict_vox_size, energy_type)
        tracks = plf.make_track_graphs(voxels)

        df = pd.DataFrame(columns=['event', 'trackID', 'energy', 'length', 'numb_of_voxels',
                                   'numb_of_hits', 'numb_of_tracks', 'x_min', 'y_min', 'z_min',
                                   'x_max', 'y_max', 'z_max', 'r_max', 'x_ave', 'y_ave', 'z_ave',
                                   'extreme1_x', 'extreme1_y', 'extreme1_z',
                                   'extreme2_x', 'extreme2_y', 'extreme2_z',
                                   'blob1_x', 'blob1_y', 'blob1_z',
                                   'blob2_x', 'blob2_y', 'blob2_z',
                                   'eblob1', 'eblob2', 'ovlp_blob_energy',
                                   'vox_size_x', 'vox_size_y', 'vox_size_z'])

        if (len(voxels) == 0):
            return df, None

        vox_size_x = voxels[0].size[0]
        vox_size_y = voxels[0].size[1]
        vox_size_z = voxels[0].size[2]

        def get_track_energy(track):
            return sum([vox.Ehits for vox in track.nodes()])
        #sort tracks in energy
        tracks     = sorted(tracks, key = get_track_energy, reverse = True)

        track_hits = []

        for c, t in enumerate(tracks, 0):
            tID = c
            energy = get_track_energy(t)
            length = plf.length(t)
            numb_of_hits = len([h for vox in t.nodes() for h in vox.hits])
            numb_of_voxels = len(t.nodes())
            numb_of_tracks = len(tracks   )

            min_x = min([h.X for v in t.nodes() for h in v.hits])
            max_x = max([h.X for v in t.nodes() for h in v.hits])
            min_y = min([h.Y for v in t.nodes() for h in v.hits])
            max_y = max([h.Y for v in t.nodes() for h in v.hits])
            min_z = min([h.Z for v in t.nodes() for h in v.hits])
            max_z = max([h.Z for v in t.nodes() for h in v.hits])
            max_r = max([np.sqrt(h.X*h.X + h.Y*h.Y) for v in t.nodes() for h in v.hits])

            pos = [h.pos for v in t.nodes() for h in v.hits]
            e   = [getattr(h, energy_type.value) for v in t.nodes() for h in v.hits]
            ave_pos = np.average(pos, weights=e, axis=0)

            extr1, extr2 = plf.find_extrema(t)
            extr1_pos = extr1.XYZ
            extr2_pos = extr2.XYZ

            blob_pos1, blob_pos2 = plf.blob_centres(t, blob_radius)

            e_blob1, e_blob2, hits_blob1, hits_blob2 = plf.blob_energies_and_hits(t, blob_radius)
            overlap = False
            overlap = sum([h.E for h in set(hits_blob1).intersection(hits_blob2)])
            list_of_vars = [hitc.event, tID, energy, length, numb_of_voxels,
                            numb_of_hits, numb_of_tracks,
                            min_x, min_y, min_z, max_x, max_y, max_z, max_r,
                            ave_pos[0], ave_pos[1], ave_pos[2],
                            extr1_pos[0], extr1_pos[1], extr1_pos[2],
                            extr2_pos[0], extr2_pos[1], extr2_pos[2],
                            blob_pos1[0], blob_pos1[1], blob_pos1[2],
                            blob_pos2[0], blob_pos2[1], blob_pos2[2],
                            e_blob1, e_blob2, overlap,
                            vox_size_x, vox_size_y, vox_size_z]

            df.loc[c] = list_of_vars
            try:
                types_dict
            except NameError:
                types_dict = dict(zip(df.columns, [type(x) for x in list_of_vars]))

            for vox in t.nodes():
                for hit in vox.hits:
                    hit.track_id = tID
                    track_hits.append(hit)


        track_hitc = evm.HitCollection(hitc.event, hitc.time)
        track_hitc.hits = track_hits
        #change dtype of columns to match type of variables
        df = df.apply(lambda x : x.astype(types_dict[x.name]))

        return df, track_hitc

    return create_extract_track_blob_info


voxel   = 2
radius  = 21

vox_size = [voxel, voxel, voxel]

#### PAOLINA SETUP
track_creator = track_blob_info_creator_extractor(vox_size=vox_size, energy_type = evm.HitEnergy.E,
                                                  strict_vox_size=False, blob_radius=radius)


start = int(sys.argv[1])
numb  = int(sys.argv[2])
prod  = str(sys.argv[3])

folder = '/scratch/paola/SimMC/beershe/'

base = 'next100.{}'.format(prod)

for i in range(start, start+numb):
    ifile = folder + 'hits/'   + '{0}.{1}.deconv.h5'.format(base, i)
    ofile = '/scratch/paola/SimMC/true/' + '{0}.{1}.true_tracks.h5'.format(base, i)

    try:
        df = pd.read_hdf(ifile, 'MC/hits')
    except:
        print(f'File {ifile} not good.')
        continue

    print(f'Analyzing file {ifile}')

    evts   = df.event_id.unique()
    tracks = []

    for i, evt in enumerate(evts):
        df_evt = df[df.event_id == evt]
        hitcol      = evm.HitCollection(evt, 0)
        hitcol.hits = convert_df_to_hits(df_evt)
        if len(hitcol.hits) == 0:
            print(f'Event {evt} has no reconstructed hits.')
            continue
        trk_df, _   = track_creator(hitcol)

        if len(trk_df) == 0: continue

        tracks.append(trk_df)


    df_all = pd.concat(tracks)
    df_all.to_hdf(ofile, key='Tracks', mode='w')
