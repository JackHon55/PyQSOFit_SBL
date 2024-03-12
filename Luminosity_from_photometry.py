# I need the EW and Raw intensity
# EW is flux / intensity, both can be relative
# I need the relative intensity of the continuum
# EW is the ratio that converts relative flux into absolute flux by
# scaling relative intensity to photometric predicted intensity
# Input here should provide: The needed spec_fluxes, the needed phot_fluxes, id, z, embv
# 1. Read spec for ctm
# 2. Calculate EW
# 3. Convert to estimated flux
# 4. Extinction correct the flux
# 5. Calculate DL
# 6. Calculate Lum

# this file only works on my computer
# To run in outside requires major directory modifications

import extinction
import numpy as np
import matplotlib.pyplot as plt
from rebin_spec import rebin_spec
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.stats import median_abs_deviation as mad
from tqdm import tqdm
from Spectra_handling.AllSpectrum import *
from Spectra_handling.Spectrum_utls import *
from extinction import remove as rm, fitzpatrick99 as fp
from astropy.cosmology import FlatLambdaCDM

HB = 4861.333
OR = 5006.843
MG = 2799.117


def linear_interp(y1, y2, x1, x2, xline):
    m = (y1 - y2) / (x1 - x2)
    c = y1 - m * x1
    return m * xline + c


def flux2L(flux, xz):
    """Transfer flux to luminoity assuming a flat Universe"""
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DL = cosmo.luminosity_distance(xz).value * 10 ** 6 * 3.08 * 10 ** 18  # unit cm
    L = flux * 4. * np.pi * DL ** 2  # erg/s/A
    return L


def sixdfgs_reader(xlocs, xfilelocs, z_arr, xres_path, path_write=1):
    c1s, c2s = [], []
    s1s, s2s = [], []
    loc = 'D:/ALL_6DFGS_Q3-4/'
    loc2 = 'D:/ALL_6DFGS_Q3-4_60-120/'
    loc3 = 'D:/ALL_6DFGS_Q3-4_120-360/'
    with open(xres_path, 'a') as wr_data:
        if path_write == 1:
            wr_data.write('\n')
        for xloc, xfile, xz in tqdm(zip(xlocs, xfilelocs, z_arr)):
            print(xfile)
            if xloc == 1:
                a = fits.open(loc + ('0000000' + str(int(xfile)))[-7:] + '_1d.fits')
            elif xloc == 2:
                a = fits.open(loc2 + ('0000000' + str(int(xfile)))[-7:] + '_1d.fits')
            else:
                a = fits.open(loc3 + ('0000000' + str(int(xfile)))[-7:] + '_1d.fits')
            rawflux = np.asarray([a[1].data[i][1] for i in range(len(a[1].data))])
            rawwave = np.asarray([a[1].data[i][0] for i in range(len(a[1].data))])

            rawwave = rawwave[rawflux > 1e-5]
            rawflux = rawflux[rawflux > 1e-5]

            datamean = np.mean(rawflux)
            dataflux = g_filt(rawflux, 0.5) / datamean
            spec_w, spec_f = blueshifting([rawwave, dataflux], xz)

            a1, b1 = 4700, 5100
            # a1, b1 = 2500, 3000
            if spec_w[-1] > a1:
                cont_1 = np.median(spec_f[id_finder(spec_w, a1)-10:id_finder(spec_w, a1)+10])
                snr_1 = cont_1 / 1.4826 / mad(spec_f[id_finder(spec_w, a1)-10:id_finder(spec_w, a1)+10])
            else:
                cont_1 = 0
                snr_1 = 0

            if spec_w[0] < b1:
                cont_2 = np.median(spec_f[id_finder(spec_w, b1)-10:id_finder(spec_w, b1)+10])
                snr_2 = cont_2 / 1.4826 / mad(spec_f[id_finder(spec_w, b1) - 10:id_finder(spec_w, b1) + 10])
            else:
                cont_2 = 0
                snr_2 = 0

            c1s.append(cont_1)
            c2s.append(cont_2)
            s1s.append(snr_1)
            s2s.append(snr_2)
            # print(str(int(xfile)) + '\t' + str(cont_1) + '\t' + str(cont_2)
            #      + '\t' + str(snr_1) + '\t' + str(snr_2))

        return np.asarray(c1s), np.asarray(c2s), np.asarray(s1s), np.asarray(s2s)


path = 'D:/6dfgs_classfication/8-10/take2/'
xdata = np.genfromtxt(path + 'xx.csv', delimiter=',')[1:].T
spec_id, z = xdata[0], xdata[1]
flx1, flx2 = xdata[2], xdata[3]
err1, err2 = xdata[4], xdata[5]
mg, mr, eg, er = xdata[6], xdata[7], xdata[8], xdata[9]
embv = xdata[10]

L1, L2 = HB, OR
# L1, L2 = MG, MG

# Converting magnitude into flux in frequency
fvg = np.asarray([10 ** ((i-8.9)/-2.5) if i != 0 else 0 for i in mg])
fvr = np.asarray([10 ** ((i-8.9)/-2.5) if i != 0 else 0 for i in mr])
eg = 2.303 * fvg * eg
er = 2.303 * fvr * er

# Converting flux in freqeuncy into lambda
flg = fvg/33400/5100/5100
flr = fvr/33400/6170/6170
eg = eg * flg/fvg
er = er * flr/fvr
eg = np.where(~np.isnan(eg), eg, 0)
er = np.where(~np.isnan(er), er, 0)

# Estimating continuum flux based on photometric flux in lambda
flhb = []
flo3 = []
for f1, f2, xz in zip(flg, flr, z):
    if f1 != 0 and f2 != 0:
        flhb.append(linear_interp(f1, f2, 5100, 6170, L1*(1+xz)))
        flo3.append(linear_interp(f1, f2, 5100, 6170, L2*(1+xz)))
    elif f1 == 0:
        flhb.append(f2)
        flo3.append(f2)
    elif f2 == 0:
        flhb.append(f1)
        flo3.append(f1)

flhb = np.asarray(flhb)
flo3 = np.asarray(flo3)

ef = np.sqrt(eg ** 2 + er ** 2)     # errors of photometric estimated flux

# Obtaining spectral continuum reading
a1, b1 = 4700, 5100
# a1, b1 = 2500, 3000
ctms = sixdfgs_reader(np.ones_like(spec_id)*3, spec_id, z, path + 'bel_0-4_v4.txt', 0)
ctmhb = []
ctmor = []
snr = []
for c1, c2, s1, s2 in zip(ctms[0], ctms[1], ctms[2], ctms[3]):
    if (c1 != 0 and ~np.isnan(c1)) and (c2 != 0 and ~np.isnan(c2)):
        ctmhb.append(linear_interp(c1, c2, a1, b1, L1))
        ctmor.append(linear_interp(c1, c2, a1, b1, L2))
        snr.append(np.mean([s1, s2]))
    elif c1 == 0 or np.isnan(c1):
        ctmhb.append(c2)
        ctmor.append(c2)
        snr.append(s2)
    elif c2 == 0 or np.isnan(c2):
        ctmhb.append(c1)
        ctmor.append(c1)
        snr.append(s1)
ctmhb = np.asarray(ctmhb)
ctmor = np.asarray(ctmor)
snr = np.asarray(snr)

# Converting spectral line flux into estimated line flux from photometry
fehb = flx1 * flhb
feor = flx2 * flo3
# fehb = flx1 / ctmhb * flhb
# feor = flx2 / ctmor * flo3
errhb = fehb * np.sqrt((err1/flx1) ** 2 + (ef/flhb) ** 2)
erro3 = feor * np.sqrt((err2/flx2) ** 2 + (ef/flo3) ** 2)

# De-redenning the lines, errors are the same
a_v = 3.1 * embv
fehb_0 = np.asarray([rm(fp(np.asarray([i]), j), np.asarray(k))[0] for i, j, k in zip(L1*(1+z), a_v, fehb)])
feor_0 = np.asarray([rm(fp(np.asarray([i]), j), np.asarray(k))[0] for i, j, k in zip(L2*(1+z), a_v, feor)])

# Converting de-reddened lines to luminosity
lumhb = np.asarray([flux2L(i, j) for i, j in zip(fehb_0, z)])
lumor = np.asarray([flux2L(i, j) for i, j in zip(feor_0, z)])

elumhb = errhb * lumhb / fehb
elumor = erro3 * lumor / feor
elumhb = np.where(~np.isnan(elumhb), elumhb, 0)
elumor = np.where(~np.isnan(elumor), elumor, 0)

ascii.write(Table([spec_id, snr, flg, flr, ctmhb, ctmor, fehb, feor, errhb, erro3,
                   fehb_0, feor_0, lumhb, lumor, elumhb, elumor]),
            'D:/6dfgs_classfication/0-4/take2/flux_calc_results.txt')
