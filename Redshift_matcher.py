# this file only works on my computer
# To run in outside requires major directory modifications

import numpy as np
import matplotlib.pyplot as plt
from rebin_spec import rebin_spec
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation as mad
from tqdm import tqdm
from Spectra_handling.Spectrum_utls import *


SY_LINES = [4861.333, 4958.911, 5006.843, 6562.819, 6583.460, 6725]
SX_LINES = [3727.092, 4340.471, 4861.333, 4958.911, 5006.843]
SZ_LINES = [2799.117, 3727.092, 4340.471, 4861.333, 4958.911, 5006.843]
QSO_LINES = [1548.187, 1908.734, 2799.117]

HA = 6562.819
NL = 6548.050
NR = 6583.460
SL = 6716.440
SR = 6730.810

HB = 4861.333
OL = 4958.911
OR = 5006.843

MG = 2799.117

SNR_WINDOW = np.asarray([4250, 4750])
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['font.size'] = 16


def seyfert_ratio(line_arr):
    return np.asarray([np.asarray(line_arr)/i for i in line_arr])


E_RATIO = seyfert_ratio(SY_LINES)
B_RATIO = seyfert_ratio(SX_LINES)
C_RATIO = seyfert_ratio(SZ_LINES)


def continuum_fitting(wave, flux, smooth_val=30, thres=2.5, end_clip=100):
    reject = 1
    i = 0
    ctm = np.copy(flux)
    t_flux = np.copy(flux)
    t_wave = np.copy(wave)

    n_flux = np.asarray([t_flux[100 * i:100 * (i + 1)] for i in range(0, int(len(t_flux) / 100))])
    n_flux_r = t_flux[int(len(t_flux) / 100) * 100:]
    m_flux = np.median(n_flux, 1)
    m_diff = np.diff(m_flux)
    mx = np.where(np.abs(m_diff) >= 3.5 * np.var(m_diff) ** 0.5)

    if len(mx[0]) != 0:
        if mx[0][0] != 0:
            for mi in mx[0]:
                f1 = np.concatenate(n_flux[:mi])
                f2 = np.concatenate(n_flux[mi+1:])
                f2 = np.append(f2, n_flux_r)

                if m_flux[mi] - 1 >= m_flux[mi+1] - 1:
                    f3 = np.ones_like(n_flux[mi:mi + 1][0]) * m_flux[mi+1]
                    t_flux = np.concatenate([f1 - m_flux[mi] + m_flux[mi+1], f3, f2])
                    t_flux = t_flux + np.abs(np.amin(t_flux)*2)
                else:
                    f3 = np.ones_like(n_flux[mi:mi + 1][0]) * m_flux[mi]
                    t_flux = np.concatenate([f1, f3, f2 - m_flux[mi+1] + m_flux[mi]])
                    t_flux = t_flux + np.abs(np.amin(t_flux)*2)

    t_end = np.copy(t_flux)
    while reject != 0 or i < 15:
        ctm = g_filt(t_flux, smooth_val)
        std = np.std(t_flux - ctm)

        t2_flux = t_flux[end_clip:-end_clip][np.abs(t_flux-ctm)[end_clip:-end_clip] < thres * std]
        t2_wave = t_wave[end_clip:-end_clip][np.abs(t_flux-ctm)[end_clip:-end_clip] < thres * std]
        reject = len(t_flux[end_clip:-end_clip]) - len(t2_flux)

        t_wave = np.concatenate([t_wave[:end_clip], t2_wave, t_wave[-end_clip:]])
        t_flux = np.concatenate([t_flux[:end_clip], t2_flux, t_flux[-end_clip:]])
        i += 1

    return rebin_spec(t_wave, ctm, wave), t_end


# Cleans up the end points, so the spectra doesnt start/end with infinite gradient or has spikes
def end_point_cleaner(wave, flux):
    start_flux = flux[:200]
    start_wave = wave[:200]
    start_len = len(start_flux) + 1
    while start_len != len(start_flux):
        start_len = len(start_flux)
        start_wave = start_wave[start_flux < np.mean(start_flux) + 3 * np.var(start_flux) ** 0.5]
        start_flux = start_flux[start_flux < np.mean(start_flux) + 3 * np.var(start_flux) ** 0.5]

    end_flux = flux[-200:]
    end_wave = wave[-200:]
    end_len = len(end_flux) + 1
    while end_len != len(end_flux):
        end_len = len(end_flux)
        end_wave = end_wave[end_flux < np.mean(end_flux) + 5 * np.var(end_flux) ** 0.5]
        end_flux = end_flux[end_flux < np.mean(end_flux) + 5 * np.var(end_flux) ** 0.5]

    return_flux = np.concatenate([start_flux, flux[200:-200], end_flux])
    return_wave = np.concatenate([start_wave, wave[200:-200], end_wave])

    return return_wave, return_flux


# If a spike as a negative component, I can detect it separately from the emission lines and mask the area
def spike_masking(wave, flux, zero_flux):
    if np.amin(zero_flux) < -np.abs(np.mean(zero_flux)) and 20 < np.argmin(zero_flux) < len(zero_flux) - 20:
        neg_id = np.argmin(zero_flux)
        return np.concatenate([wave[:neg_id-10], wave[neg_id+10:]]), \
               np.concatenate([flux[:neg_id-10], flux[neg_id+10:]]), \
               np.concatenate([zero_flux[:neg_id-10], zero_flux[neg_id + 10:]]),
    return wave, flux, zero_flux


# sort value with 10% error margin, finds unique, and count the frequency
def sort_10(arr):
    s_arr = np.sort(arr)
    u_arr = s_arr * 1.05
    d_arr = s_arr * 0.95
    n_arr = [s_arr[0]]
    c_arr = []
    xc = 1
    xs = 0
    for i in range(1, len(d_arr)):
        if d_arr[xs] < s_arr[i] < u_arr[xs] or s_arr[i] == s_arr[i-1]:
            n_arr.append(n_arr[i-1])
            xc += 1
        else:
            n_arr.append(s_arr[i])
            c_arr.append(xc)
            xc = 1
            xs = i
    c_arr.append(xc)
    return np.asarray(n_arr), np.asarray(c_arr), np.unique(n_arr)


# step towards and increasing gradient, the first point that is higher than initial point is the peak
def peak_locator(flux, xid, xpeak_id):  # NOT USED
    if flux[xid] != flux[0] and xid < xpeak_id:
        if flux[xid-1] >= flux[xid]:
            a = 0
            while flux[xid - (a + 1)] >= flux[xid - a] and a < len(flux[:xid-2]):
                a += 1
            return xid - a
        else:
            return xid
    elif flux[xid] != flux[-1] and xid > xpeak_id:
        if flux[xid+1] >= flux[xid]:
            a = 0
            while flux[xid + (a + 1)] >= flux[xid - a] and a < len(flux[xid+2:]):
                a += 1
            return xid - a
        else:
            return xid
    else:
        return xid


def feature_picker(wave, flux):
    # plt.figure()
    # plt.plot(wave, flux)
    ctm_flux, flux = continuum_fitting(wave, flux, 50)
    zero_flux = flux - ctm_flux
    # plt.plot(wave, ctm_flux, linestyle='--')
    wave, flux, zero_flux = spike_masking(wave, flux, zero_flux)

    wave_pixel = np.median(np.diff(wave))

    # plt.plot(wave, flux, linewidth=3, alpha=0.5)
    # plt.plot(wave, zero_flux)
    half_id = id_finder(wave, 5571)
    var_flux = np.concatenate([np.ones_like(zero_flux[:half_id])*np.var(zero_flux[100:half_id-100]),
                               np.ones_like(zero_flux[half_id:])*np.var(zero_flux[half_id+100:-100])])
    peaks, _ = find_peaks(zero_flux, height=3 * var_flux ** 0.5,
                          distance=45 / wave_pixel)
    # plt.plot(wave[peaks], zero_flux[peaks], 'x')
    # plt.plot(wave, 3 * var_flux ** 0.5, color='blue', linestyle='-.')

    peak_num = len(peaks)
    a = 0
    b = 0

    if peak_num < 3:
        while (peak_num <= 3) and (peak_num < 20) and b < 15:
            peak_results = ''
            peaks, _ = find_peaks(zero_flux, height=(3 - a / 2) * var_flux ** 0.5,
                                  distance=45 / wave_pixel)
            peak_num = len(peaks)
            for i in peaks:
                peak_results += str(np.round(wave[i], 5)) + ','
            # plt.plot(wave[peaks], zero_flux[peaks], 'x')
            # plt.plot(wave, (3 - a / 2) * var_flux ** 0.5, color='magenta')
            a += 1
            b += 1
    elif peak_num > 19:
        while (peak_num > 19) and (peak_num > 2) and b < 15:
            peak_results = ''
            peaks, _ = find_peaks(zero_flux, height=(3 - a / 2) * var_flux ** 0.5,
                                  distance=45 / wave_pixel)
            peak_num = len(peaks)
            for i in peaks:
                peak_results += str(np.round(wave[i], 5)) + ','
            # plt.plot(wave[peaks], zero_flux[peaks], 'x')
            # plt.plot(wave, (3 - a / 2) * var_flux ** 0.5, color='red')
            a -= 1
            b += 1
    else:
        peak_results = ''
        for i in peaks:
            peak_results += str(np.round(wave[i], 5)) + ','

    if peak_num >= 3:
        return peak_results[:-1]
    else:
        return 'no lines'


# Split with comma to separate into each detected lines
# If a detected line has a '-' it is done, read off the redshift
# For the remaining ones, or each without '-', ratio it with another available line to test what line it is
# Take the median at the end
def redshift_finder(str_res, xz):
    if 'no' in str_res:
        return -1, -1, -1

    guess_z = []
    arr_res = np.sort(np.asarray([float(i) for i in str_res.split(',')]))
    if len(arr_res) == 1:
        return -1, -1, -1

    if HA * (1 + xz) * 0.9 <= arr_res[-1]:
        e_lines = SY_LINES
        e_ratio = E_RATIO
    elif MG * (1 + xz) * 0.9 < arr_res[0] < MG * (1 + xz) * 1.1:
        e_lines = SZ_LINES
        e_ratio = C_RATIO
    else:
        e_lines = SX_LINES
        e_ratio = B_RATIO

    bad_lines = []
    for i in arr_res:
        guess_lines = []
        for j in arr_res:
            if i != j:
                guess_ratio = np.argmin(np.abs(e_ratio - i/j))
                ida = int(guess_ratio / len(e_lines))
                idb = int(guess_ratio % len(e_lines))
                line_a = e_lines[ida]
                line_b = e_lines[idb]
                lines = np.sort([line_a, line_b])
                test_lines = np.sort([i, j])
                line_z = test_lines / lines - 1
                if line_z[0] > -0.001 and line_z[1] > -0.001:
                    line_z = np.sort(np.abs(line_z))
                    # print(test_lines, lines, line_z)
                    if line_z[1] * 0.95 < line_z[0] < line_z[1] * 1.05:
                        if [lines[0], lines[1]] not in guess_lines and [test_lines[0], test_lines[1]] not in bad_lines:
                            # print('match')
                            guess_z.append(line_z[0])
                            guess_lines.append([lines[0], lines[1]])
                else:
                    bad_lines.append([test_lines[0], test_lines[1]])

    if len(guess_z) != 0:
        guess_z, count_z, unique_z = sort_10(guess_z)
        # print(unique_z)
        # print(count_z)
        # [print(i) for i in guess_z]
        return -2, str(unique_z), unique_z[np.argmax(count_z)]
    else:
        return -1, -1, -1


def ha_region_checker(wave, flux, test_lines, to_test=HA):
    ha_rex = []
    x1, x2 = 0, 0
    for xz in test_lines:
        if xz > to_test and id_finder(wave, xz) / len(wave) < 0.99:
            guess_z = xz/to_test - 1
            ha_flux = np.max(flux[id_finder(wave, xz) - 20:id_finder(wave, xz) + 20])
            ha_sig = 1.4826*mad(flux[id_finder(wave, (guess_z + 1)*6000):id_finder(wave, (guess_z + 1)*6200)])
            # print(ha_flux, 3 * ha_sig)
            # plt.plot(wave[id_finder(wave, (guess_z + 1)*6000):id_finder(wave, (guess_z + 1)*6200)],
            #          flux[id_finder(wave, (guess_z + 1)*6000):id_finder(wave, (guess_z + 1)*6200)])
            if ha_flux > 3 * ha_sig:
                ha_rex.append(xz/to_test - 1)
                # x1 = ha_flux
                # x2 = ha_sig
                # plt.axvline(xz, alpha=0.5)
                # plt.axhline(3*ha_sig, alpha=0.5)
    return np.asarray(ha_rex)#, x1, x2


def hb_region_checker(wave, flux, test_lines, to_test=OR):
    ha_rex = []
    x1, x2 = 0, 0
    for xz in test_lines:
        if xz > to_test and id_finder(wave, xz) / len(wave) < 0.99:
            guess_z = xz/to_test - 1
            ha_flux = np.max(flux[np.abs(id_finder(wave, xz) - 20):id_finder(wave, xz) + 20])
            ha_sig = 1.4826*mad(flux[id_finder(wave, (guess_z + 1)*5100):id_finder(wave, (guess_z + 1)*5150)])
            # print(ha_flux, 3 * ha_sig)
            if ha_flux > 3 * ha_sig:
                # plt.plot(wave[id_finder(wave, (guess_z + 1) * 5100):id_finder(wave, (guess_z + 1) * 5150)],
                #          flux[id_finder(wave, (guess_z + 1) * 5100):id_finder(wave, (guess_z + 1) * 5150)])
                ha_rex.append(xz/to_test - 1)
                # x1 = ha_flux
                # x2 = ha_sig
                # plt.axvline(xz, alpha=0.5)
                # plt.axhline(3*ha_sig, alpha=0.5)
    return np.asarray(ha_rex)#, x1, x2


def or_region_checker(wave, flux, test_lines):
    or_res = []
    for xz in test_lines:
        if xz > OR:
            or_z = xz / OR - 1
            ol_idx = id_finder(wave, OL * (1 + or_z))
            ol_ids = np.arange(ol_idx - 20, ol_idx + 20)
            ol_flux = g_filt(flux[ol_ids], 3)
            # plt.plot(wave[ol_ids], ol_flux)
            if np.amax(ol_flux) != ol_flux[0] and np.amax(ol_flux) != ol_flux[-1]:
                ol_diff = np.diff([np.amin(ol_flux[:np.argmax(ol_flux)]), np.amax(ol_flux),
                                   np.amin(ol_flux[np.argmax(ol_flux):])])

                # print(xz, ol_diff, ol_diff[1]/ol_diff[0])
                if 0.6 < np.abs(ol_diff[1])/np.abs(ol_diff[0]) < 1.4:
                    or_res.append(wave[np.argmax(ol_flux) + ol_idx - 20] / OL - 1)
    return np.asarray(or_res)

# Check 21201 4957 21263 122537

dataf = np.genfromtxt('D:/6dfgs_classfication/4-8/ra4-8.csv', delimiter=',')[1:].T
# dataf = np.genfromtxt('D:/6dfgs_classfication/0-4/take2/check.csv', delimiter=',')[1:].T
resultf = []
nametf = []
linesf = []
snrtf = []
comment2 = []
xi = 0
# ai = id_finder(dataf[0], 22135)
for ij in tqdm(range(11783, len(dataf[0]))):
# for ij in tqdm(range(ai, ai + 1)):
    # with fits.open(dataf[ij]) as tmp:
    with fits.open('D:/ALL_6DFGS_Q3-4_60-120/' + ('0000000' + str(int(dataf[0][ij])))[-7:] + '_1d.fits') as tmp:
        rawflux = np.asarray([tmp[1].data[i][1] for i in range(len(tmp[1].data))])
        rawwave = np.asarray([tmp[1].data[i][0] for i in range(len(tmp[1].data))])
        rawwidth = rawwave[1] - rawwave[0]
        cleanwave, cleanflux = end_point_cleaner(rawwave[rawflux > 0], rawflux[rawflux > 0])
        datamean = np.mean(cleanflux)
        dataflux = g_filt(cleanflux, 1) / datamean
    obs_snr_window = SNR_WINDOW * (1 + dataf[1][ij])
    # print(str(int(dataf[0][ij])))
    xresults = feature_picker(cleanwave, dataflux)
    xredshifts = redshift_finder(xresults, dataf[1][ij])
    xpass = 0

    # resultf.append(xredshifts[0])
    if np.abs(xredshifts[2] - dataf[1][ij]) / np.abs(dataf[1][ij]) < 0.1:
        linesf.append(xresults)
        resultf.append(xredshifts[2])
        comment2.append(0)
        nametf.append(str(int(dataf[0][ij])))
        xpass = 1

    if xredshifts[0] == -2 and xpass == 0:
        test_xreds = np.asarray([float(xi) if '.' in xi else -1 for xi in xredshifts[1][1:-1].split(' ')])
        for xr in test_xreds:
            if np.abs(xr - dataf[1][ij]) / np.abs(dataf[1][ij]) < 0.1 and xpass == 0:
                linesf.append(xresults)
                resultf.append(xr)
                comment2.append(-2)
                nametf.append(str(int(dataf[0][ij])))
                xpass = 1

    if xpass == 0:
        datactm = continuum_fitting(cleanwave, dataflux, 50)
        xres = np.asarray([float(xi) for xi in xresults.split(',')])
        # ox_res = or_region_checker(cleanwave, datactm[1] - datactm[0], xres)
        ox_res = hb_region_checker(cleanwave, datactm[1] - datactm[0], xres, OR)
        hb_res = hb_region_checker(cleanwave, datactm[1] - datactm[0], xres, HB)
        # oy_res = or_region_checker(cleanwave, dataflux, xres)
        ha_res = ha_region_checker(cleanwave, datactm[1] - datactm[0], xres)
        xxres = np.concatenate([ox_res, hb_res, ha_res])

        for xr in xxres:
            if np.abs(xr - dataf[1][ij]) / np.abs(dataf[1][ij]) < 0.1 and xpass == 0:
                linesf.append(xresults)
                resultf.append(xr)
                comment2.append(-3)
                nametf.append(str(int(dataf[0][ij])))
                xpass = 1

    if xpass == 0:
        linesf.append(xresults)
        resultf.append(-1)
        comment2.append(-1)
        nametf.append(str(int(dataf[0][ij])))