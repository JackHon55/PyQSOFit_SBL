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
    f = f[(w > 4000) & (w < 7000)][:]
    w = w[(w > 4000) & (w < 7000)][:]
    '''if len(w) == 0:
        return np.zeros((5, 6))
    if w[10] > HA or w[-10] < HA:
        return np.zeros((5, 6))'''
    # w, f = noise_to_linear([w, f], [5071, 5164])
    fs = True
    # p1 = [False, 0]
    p1 = [True, 1]
    # f, fs = f - continuum_fitting(w, f, 100), False
    err = np.ones_like(f)

    q = QSOFit(w, f, err, 0, path=path1)

    start = timeit.default_timer()
    q.Fit(decomposition_host=fs, PL=fs, poly=fs, Fe_uv_op=False, BC=True,
          CFT_smooth=75, CFT=False, MC=p1[0], MC_conti=False,
          name=None, nsmooth=1, deredden=False, reject_badpix=False, wave_range=None, redshift=False,
          initial_guess=None, n_trails=5, linefit=True, save_result=False, plot_fig=True, save_fig=False,
          plot_line_name=True, plot_legend=True, dustmap_path=None, save_fig_path=fig_path,
          save_fits_path=None, save_fits_name=None)

    plt.axvline(6885 / (1 + z))
    plt.axvline(5577 / (1 + z))

    # plt.close()
    end = timeit.default_timer()
    end_flux = np.abs(q.flux - q.f_conti_model - np.sum(q.gauss_line, 0))
    end_wave = q.wave
    end_flux = end_flux
    print(np.trapz(end_flux, end_wave) / len(end_flux[end_flux > 1.4826 * 3 * mad(end_flux)]))
    noise_area = np.trapz(end_flux, end_wave) / len(end_flux[end_flux > 1.4826 * 3 * mad(end_flux)])
    end = timeit.default_timer()
    print('Fitting finished in : ' + str(np.round(end - start)) + 's')

    '''a = 0
    for i, j in zip(q.line_result_name, q.line_result):
        print(a, i, j)
        a += 1'''

    moe = p1[1]
    block1 = 0
    block2 = 0
    if block1 == 2:
        a = np.zeros(6)
        b = np.copy(a)
        c = np.copy(a)
    else:
        a = q.line_result_output('Hb_br', True)[moe]
        b = q.line_result_output('Hb_na')[moe]
        # b = np.zeros(6)
        c = q.line_result_output('OIII5007c')[moe]
        '''a1 = q.line_result_output('Hb_br1')[0]
        a2 = q.line_result_output('Hb_br2')[0]
        print(np.average([a1[4], a2[4]], weights=[a1[-1], a2[-1]]))'''

    if w[-1] > HA and block2 == 0:
        d = q.line_result_output('Ha_br')[moe]
        e = q.line_result_output('NII6549c')[moe]
        # e = np.zeros(6)
        # f = q.line_result_output('NII6585', 'narrow')[0]
        # g1 = q.line_result_output('SII6718', 'narrow')[0]
        # g2 = q.line_result_output('SII6732', 'narrow')[0]
        '''a1 = q.line_result_output('Ha_br1')[0]
        a2 = q.line_result_output('Ha_br2')[0]
        print(np.average([a1[4], a2[4]], weights=[a1[-1], a2[-1]]))'''
    else:
        d = np.zeros(len(a))
        e = np.copy(d)

    return np.concatenate([a, b, c, d, e])


file_data = np.genfromtxt(os.getcwd() + '/test_spectra_list.csv', delimiter=',')[1:].T

# Mode A - Fitting selected spectrum
wid = np.asarray([
22878
])
name = np.asarray([file_data[0][id_finder(file_data[0], i)] for i in wid])
spec_z = np.asarray([file_data[2][id_finder(file_data[0], i)] for i in wid])

# Part of Mode A, when redshifts require specific tweaks
# spec_z = [0.7681]
# spec_z = np.asarray([4999*(1+spec_z[0])/OR - 1])

# Mode B - Fitting all in the specified file
# name = file_data[0]
# spec_z = file_data[4]

bpt_results = six_bel_fit(name, spec_z, os.getcwd() + '/fitting_results.txt')



