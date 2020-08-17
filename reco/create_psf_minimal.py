import os, sys
import glob
import numpy  as np
import pandas as pd
import tables as tb

from invisible_cities.reco.psf_functions    import create_psf
from invisible_cities.reco.psf_functions    import hdst_psf_processing
from invisible_cities.reco.psf_functions    import add_empty_sensors_and_normalize_q
from invisible_cities.reco.psf_functions    import add_variable_weighted_mean

import invisible_cities.core.core_functions as     coref
import invisible_cities.io  .dst_io         as     dstio

from invisible_cities.database              import load_db
from invisible_cities.io      .kdst_io      import psf_writer

"""
This script takes the penthesilea output of Kr as an input
and returns files to be combined to generate the full point spread functions.
"""

# Database
run = -1
the_db = load_db.DataSiPM('next100', run)

z_max  = 1210
z_step = 20
zrange = []
zrange.extend(np.arange(0, z_max, z_step))

### PSF binning and range
bin_size   = 1
Xrange     = [ -100, 100]
Yrange     = [ -100, 100]
ranges     = [Xrange, Yrange]
nbinX      = int(np.diff(Xrange)/bin_size)
nbinY      = int(np.diff(Yrange)/bin_size)

xstep    = [-100, 100]
ystep    = [-100, 100]

# Input and output path
psf_path   = '/scratch/paola/SimMC/psf/'
hits_path  = '/scratch/paola/SimMC/psf/kr_penthesilea/'
filename   = hits_path + 'next100.kr83m.{}.psf_hits.h5'


def compute_psf(df, fnumber):
    out_psf = psf_path + 'psf_{}.h5'.format(fnumber)

    with tb.open_file(out_psf, 'w') as outfile:
        # Declare the PSF writer
        writer = psf_writer(outfile)

        for z in zrange:
            z_sel = coref.in_range(df.Z, z, z+z_step)
            # Preprocess the data before extracting the PSF
            hdst = hdst_psf_processing(df[z_sel], [Xrange, Yrange], the_db)
            # Safety check (single sipm events not considered to be physical)
            hdst = hdst[hdst.nsipm > 1]

            # Loop to extract the PSF in different regions.
            bin_edges = [np.linspace(*rr, [nbinX, nbinY][i]+1) for i, rr in enumerate([Xrange, Yrange])]
            psf_new, entries_new, bins_new = create_psf((hdst.RelX.values, hdst.RelY.values), hdst.NormQ,
                                                         bin_edges)

            writer(bins_new[0], bins_new[1], [0], 0., 0., z + z_step/2,
                   np.asarray([psf_new]).transpose((1, 2, 0)),
                   np.asarray([entries_new]).transpose((1, 2, 0)))


start = int(sys.argv[1])
numb  = int(sys.argv[2])
for i in range(start, start+numb):
    thefile = filename.format(i)
    try:
        df = dstio.load_dsts([thefile], 'RECO', 'Events').drop(['Xrms', 'Yrms', 'Qc', 'Ec', 'track_id'], axis='columns').reset_index(drop=True)
    except:
        print(f'File {thefile} not found or corrupted.')
        continue
    compute_psf(df, i)
