import numpy as np
# import matplotlib

# matplotlib.use('agg')
import matplotlib.pyplot as plt
from Spectra_handling.AllSpectrum import *
from Spectra_handling.Spectrum_utls import bpt_plotter, bpt_test, continuum_fitting
from PyQSOFit.PyQSOFit_SVL import *
from rebin_spec import rebin_spec
from scipy.signal import medfilt
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

HA = 6562.819
NL = 6548.050
NR = 6583.460
SL = 6716.440
SR = 6730.810

E_LINES = np.asarray([NL, HA, NR])
S_LINES = np.asarray([SL, SR])

HB = 4861.333
OL = 4958.911
OR = 5006.843
B_LINES = np.asarray([HB, OL, OR])

MG = 2799.117

path1 = os.getcwd() + '/PyQSOFit/'


# Opens the spectrum and pre-process to fit
# Pre-process involces removing negative pixels and normalising it to have continuum level ~1
def six_bel_fit(xfilelocs, z_arr, xres_path, path_write=1):
    xresults = []
    loc = os.getcwd() + '/test_spectra/'
    with open(xres_path, 'a') as wr_data:
        if path_write == 1:
            wr_data.write('\n')
        for xfile, xz in tqdm(zip(xfilelocs, z_arr)):
            print(xfile)
            a = fits.open(loc + ('0000000' + str(int(xfile)))[-7:] + '_1d.fits')
            rawflux = np.asarray([a[1].data[i][1] for i in range(len(a[1].data))])
            rawwave = np.asarray([a[1].data[i][0] for i in range(len(a[1].data))])
            rawwave = rawwave[rawflux > 1e-5]
            rawflux = rawflux[rawflux > 1e-5]
            # plt.plot(rawwave, rawflux)
            datamean = np.mean(rawflux)
            dataflux = g_filt(rawflux, 0.5) / datamean

            spec_w, spec_f = blueshifting([rawwave, dataflux], xz)

            # plt.figure(str(xfile))
            # plt.plot(rawwave, g_filt(dataflux, 1.5))

            if path_write == 1:
                # Calling function to fit
                xresults.append(
                    bel_fitting(spec_w, spec_f, xz, os.getcwd() + '/fitting_plots/' + str(int(xfile))))

            if path_write == 1:
                to_save = "\t".join(map(str, xresults[-1]))
                wr_data.write(str(int(xfile)) + '\t' + to_save + '\n')
                print(str(int(xfile)) + '\t' + to_save + '\n')

    return np.asarray(xresults)


# Fitting function, allowing for further control of spectra before calling PyQSOFit
def bel_fitting(w, f, z, fig_path):
    # Spectra Trimming
    f = f[(w > 4000) & (w < 5500)][:]
    w = w[(w > 4000) & (w < 5500)][:]

    # w, f = noise_to_linear([w, f], [6150, 6400]) # Function to mask manually
    p1 = [False, 0]     # Uncomment this for normal fitting
    # p1 = [True, 1]    # Uncomment this for error fitting
    fs = True           # Uncomment this for nomal fitting
    # f, fs = f - continuum_fitting(w, f, 75), False   # Uncomment this to use the continuum fitting tool for tricky spectra
    err = np.ones_like(f)   # These spectra files do not have variance provided. The dominant noise is calculated post-fitting inside PyQSOFit anyways
    # f = medfilt(f, 7)     # Uncomment to apply smoothing to help with visualisation
    q = QSOFit(w, f, err, 0, path=path1)

    start = timeit.default_timer()
    q.Fit(name=None, nsmooth=1, deredden=False, reject_badpix=False, wave_range=None, redshift=False,
          Fe_uv_op=False, poly=fs, CFT=False, CFT_smooth=75, BC=True, initial_guess=None, MC=p1[0], MC_conti=False, decomposition_host=fs,
          PL=fs, n_trails=15, linefit=True, save_result=False, plot_fig=True, save_fig=True, plot_line_name=True,
          plot_legend=True, dustmap_path=None, save_fig_path=fig_path, save_fits_path=None, save_fits_name=None)
    plt.show()
    plt.axvline(6885 / (1 + z))     # Indicates B-band telluric line
    plt.axvline(5577 / (1 + z))     # Indicates the 5577 skyline

    # plt.close()
    end = timeit.default_timer()
    end_flux = np.abs(q.flux-q.f_conti_model-q.gauss_line)
    end_wave = q.wave
    end_flux = end_flux
    print(np.trapz(end_flux, end_wave)/len(end_flux[end_flux > 1.4826*3*mad(end_flux)]))
    noise_area = np.trapz(end_flux, end_wave) / len(end_flux[end_flux > 1.4826 * 3 * mad(end_flux)])
    print('Fitting finished in : ' + str(np.round(end - start)) + 's')
    '''
    a = 0
    for i, j in zip(q.line_result_name, q.line_result):
        print(a, i, j)
        a += 1
    '''
    moe = p1[1]
    a = q.line_result_output('Hb_na')[moe]
    b = q.line_result_output('OIII5007')[moe]
    # a = np.zeroes(6)
    if w[-1] > HA:
        c = q.line_result_output('Ha_na')[moe]
        # d = np.zeros(6)
        d = q.line_result_output('NII6585c')[moe]
        # g1 = q.line_result_output('SII6718', 'narrow')[0]
        # g2 = q.line_result_output('SII6732', 'narrow')[0]
    else:
        c = np.zeros(6)
        d = np.copy(c)

    # Want only the fwhm, peak position, and flux (EW calculation is broken, and skew is not relevant)
    a = np.asarray([a[0], a[2], a[5]])
    b = np.asarray([b[0], b[2], b[5]])
    c = np.asarray([c[0], c[2], c[5]])
    d = np.asarray([d[0], d[2], d[5]])

    return np.concatenate([[noise_area], a, b, c, d])


file_data = np.genfromtxt(os.getcwd() + '/test_spectra_list.csv', delimiter=',')[1:].T

# Mode A - Fitting selected spectrum
wid = np.asarray([
32928
])
name = np.asarray([file_data[0][id_finder(file_data[0], i)] for i in wid])
spec_z = np.asarray([file_data[4][id_finder(file_data[0], i)] for i in wid])

# Part of Mode A, when redshifts require specific tweaks
# spec_z = [0.7681]
# spec_z = np.asarray([4999*(1+spec_z[0])/OR - 1])

# Mode B - Fitting all in the specified file
# name = file_data[0]
# spec_z = file_data[4]

bpt_results = six_bel_fit(name, spec_z, os.getcwd() + '/fitting_results.txt')



