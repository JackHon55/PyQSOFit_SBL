import numpy as np
import matplotlib.pyplot as plt
from Spectra_handling.AllSpectrum import *
from scipy.stats import median_abs_deviation as mad
from astropy.io import ascii
from tqdm import tqdm
import os

loc = os.getcwd() + '/test_spectra/'


def end_point_cleaner(flux, wave):
    start_flux = flux[0:150]
    end_flux = flux[-150:]
    start_grad = np.gradient(start_flux)
    end_grad = np.gradient(end_flux)
    ids = np.arange(0, 150, 1)
    ide = np.arange(-150, 0, 1)
    start_spike = ids[abs(start_grad) > np.mean(start_grad) + 3*np.var(start_grad)**0.5]
    end_spike = ide[abs(end_grad) > np.mean(end_grad) + 3*np.var(end_grad)**0.5]
    return_flux = np.copy(flux)
    return_wave = np.copy(wave)
    if len(start_spike) != 0:
        return_flux = return_flux[start_spike[-1]+20:]
        return_wave = return_wave[start_spike[-1]+20:]
    if len(end_spike) != 0:
        return_flux = return_flux[0:end_spike[0]-20]
        return_wave = return_wave[0:end_spike[0]-20]
    return return_flux, return_wave


def hb_scorer(file, zshift):
    a = fits.open(loc + ('0000000' + str(file))[-7:] + '_1d.fits')
    rawflux = np.asarray([a[1].data[i][1] for i in range(len(a[1].data))])
    rawwave = np.asarray([a[1].data[i][0] for i in range(len(a[1].data))])
    rawwidth = rawwave[1] - rawwave[0]
    rawflux, rawwave = end_point_cleaner(rawflux, rawwave)
    datamean = np.mean(rawflux)
    dataflux = g_filt(rawflux, 1) / datamean

    spec_a = np.asarray([rawwave, dataflux])
    w, f = blueshifting(spec_a, zshift)
    fc = f - continuum_fitting(w, f, 25)

    hb_region = [4830, 5050]
    ha_region = [6470, 6660]
    fhb = fc[(w > hb_region[0]) & (w < hb_region[1])]
    fhb = fhb[~np.isnan(fhb)]
    hbscore = np.mean(fhb[fhb > np.mean(fhb)])/1.4826/mad(fhb)

    fha = fc[(w > ha_region[0]) & (w < ha_region[1])]
    fha = fha[~np.isnan(fha)]
    hascore = np.mean(fha[fha > np.mean(fha)])/1.4826/mad(fha)

    return hbscore, hascore


test_files = np.genfromtxt(os.getcwd() + '/test_spectra_list.csv', delimiter=',')[1:].T
test_z = test_files[-2]
test_files = test_files[0]

xx = []
xy = []
for xi, xj in tqdm(zip(test_files, test_z)):
    ax, bx = hb_scorer(int(xi), xj)
    xx.append(ax)
    xy.append(bx)

xx = np.asarray(xx)
xy = np.asarray(xy)
ascii.write([test_files, xx, xy], os.getcwd() + '/test_spectra_ranks.txt')
