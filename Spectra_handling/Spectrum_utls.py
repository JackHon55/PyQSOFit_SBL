from rebin_spec import rebin_spec
# if __name__ is "__main__":
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as g_filt
from scipy.special import erf, wofz


def blueshifting(spec, zshift):
    zspec = np.copy(spec)
    wave0 = zspec[0] / (1 + zshift)
    wavenew = np.linspace(wave0[0], wave0[-1], len(wave0))
    fluxnew = rebin_spec(wave0, zspec[1], wavenew)
    return np.asarray([wavenew, fluxnew])


def redshifting(spec, zshift):
    wave0 = spec[0] * (1 + zshift)
    wavenew = np.linspace(wave0[0], wave0[-1], len(wave0))
    fluxnew = rebin_spec(wave0, spec[1], wavenew)
    return np.asarray([wavenew, fluxnew])


def nan_specremove(spec):
    return np.asarray([spec[0][~np.isnan(spec[1])], spec[1][~np.isnan(spec[1])]])


def id_finder(arr, val):
    return np.argmin(np.abs(np.asarray(arr) - val))


# spec in form [wave, flux] wrange in form [min, max]
# returns in spec format
def range_select(spec, wrange):
    min_id = id_finder(spec[0], wrange[0])
    max_id = id_finder(spec[0], wrange[1])
    return np.asarray([spec[0][min_id:max_id], spec[1][min_id:max_id]])


# designed to merge the two wifes spectra
def merge(bspec, rspec, side_bias=0):
    bstop = id_finder(bspec[0], rspec[0][0])
    rstart = id_finder(rspec[0], bspec[0][-1])

    if side_bias == -1:
        rbins = rspec[0][1] - rspec[0][0]
        b_rewave = np.arange(bspec[0][0], bspec[0][-1], rbins)
        b_rebin = rebin_spec(bspec[0], bspec[1], b_rewave)
        w_tmp = np.concatenate((b_rewave, rspec[0][rstart:]))
        f_tmp = np.concatenate((b_rebin, rspec[1][rstart:]))
        w_new = np.arange(bspec[0][0], rspec[0][-1], rbins)
        return np.asarray([w_new, np.interp(w_new, w_tmp, f_tmp)])
    elif side_bias == 1:
        rbins = rspec[0][1] - rspec[0][0]
        b_rewave = np.arange(bspec[0][0], bspec[0][-1], rbins)
        rstart = id_finder(rspec[0], b_rewave[-1])
        b_rebin = rebin_spec(bspec[0], bspec[1], b_rewave)
        w_tmp = np.concatenate((b_rewave[:rstart], rspec[0]))
        f_tmp = np.concatenate((b_rebin[:rstart], rspec[1]))
        w_new = np.arange(bspec[0][0], rspec[0][-1], rbins)
        return np.asarray([w_new, np.interp(w_new, w_tmp, f_tmp)])

    k = -1      # can change this value to trim the red end
    # returns a spec
    roverlap = range_select(rspec, [rspec[0][0], bspec[0][-1]])
    # returns only wavelength
    boverlap = rebin_spec(bspec[0], bspec[1], roverlap[0])
    # create a section of wavelength where they overlap
    broverlap = []
    for i, j in enumerate(boverlap):
        # j is value in boverlap, i is the id
        diff = np.abs(j - roverlap[1][i])
        if (diff > np.abs(0.1 * j)) or (diff > np.abs(0.1 * roverlap[1][i])):
            if i < 0.5 * len(boverlap):
                broverlap.append(j)
            else:
                broverlap.append(roverlap[1][i])
        else:
            broverlap.append((boverlap[i] + roverlap[1][i]) / 2)
    # join the b spec + overlap + r spec
    # overlap wavelength is a part of r spec

    w_tmp = np.concatenate((bspec[0][0:bstop], rspec[0][0:k]))
    f_tmp = np.concatenate((bspec[1][0:bstop], broverlap, rspec[1][rstart:k]))

    return np.asarray([w_tmp, f_tmp])


# define an area of noise, replace with a
# linear line joining the ends
# intended to work with noise spikes
def noise_to_linear(spec, x=(3000, 4000)):
    x1 = id_finder(spec[0], x[0])
    x2 = id_finder(spec[0], x[1])
    grad = (spec[1][x1] - spec[1][x2]) / (spec[0][x1] - spec[0][x2])
    const = spec[1][x1] - grad * spec[0][x1]
    y = grad * spec[0] + const
    return np.asarray([spec[0], np.concatenate([spec[1][0:x1], y[x1:x2], spec[1][x2 - 1:-1]])])


# Extrapolate a linear line with a given reference region
# will fit a linear line in the reference region
# and add onto the spectrum
def extrapolate(spec, xend, ref=(3000, 4000)):
    x1 = id_finder(spec[0], ref[0])
    x2 = id_finder(spec[0], ref[1])
    grad = (spec[1][x1] - spec[1][x2]) / (spec[0][x1] - spec[0][x2])
    const = spec[1][x1] - grad * spec[0][x1]

    xstep = spec[0][1] - spec[0][0]
    end = (xend - spec[0][-1]) / xstep
    x = np.asarray([spec[0][-1] + i * xstep for i in range(0, int(end))])
    y = grad * x + const

    return np.asarray([np.concatenate([spec[0], x]), np.concatenate([spec[1], y])])


def bpt_test(xhb, xha, xor, xnr):
    nrha = np.round(np.log10(xnr/xha), 3)
    if xor == 100:
        # print('WARNING Hbeta < 0. BPT will ignore Hb region')
        if nrha > -0.3:
            return 'X/' + str(nrha) + '/AGN/*log(nr/ha) > -0.3'
        else:
            return 'X/' + str(nrha) + '/Maybe AGN/*'
    orhb = np.round(np.log10(xor/xhb), 3)

    if np.isnan(nrha) or nrha > 1.5 or nrha < -10:
        # print('WARNING HA REGION CANNOT BE FIT')
        if orhb > 0.9:
            return str(orhb) + '/X/AGN/*log(or/hb) > 0.9'
        else:
            return str(orhb) + '/X/Maybe AGN/*'

    if nrha > 0:
        return str(orhb) + '/' + str(nrha) + '/AGN/*'
    bpt_val = orhb - 0.61 / (nrha - 0.05) - 1.3
    bpt_deg = np.round(bpt_val / 5 * 100, 3)
    if bpt_val < 0:
        return str(orhb) + '/' + str(nrha) + '/NOT AGN/' + str(bpt_deg)
    if bpt_val > 0:
        return str(orhb) + '/' + str(nrha) + '/AGN/' + str(bpt_deg)


def bpt_plotter(xorhb, xnrha, xsiiha=None, xerr=None):
    nii_comp_axis = np.arange(-1, 0, 0.05)
    nii_agn_axis = np.arange(-1, 0.4, 0.05)
    sii_agn_axis = np.arange(-1, 0.25, 0.05)
    sii_lin_axis = np.arange(-0.31, 0.5, 0.05)
    nii_comp_line = 0.61 / (nii_comp_axis - 0.05) + 1.3
    nii_agn_line = 0.61 / (nii_agn_axis - 0.47) + 1.19
    sii_agn_line = 0.72 / (sii_agn_axis - 0.32) + 1.3
    sii_lin_line = 1.89 * sii_lin_axis + 0.76
    plt.figure('BPT_plotter', figsize=(14, 6))

    plt.subplot(121)
    plt.ylim(-1.5, 3)
    plt.plot(nii_comp_axis, nii_comp_line, color='b', linestyle='-')
    plt.plot(nii_agn_axis, nii_agn_line, color='b', linestyle='--')
    plt.xlabel(r'log$_{10}$([Nii]/H$\alpha$)')
    plt.ylabel(r'log$_{10}$([Oiii]/H$\beta$)')
    plt.text(-0.4, -1, 'STF')
    plt.text(-0.15, -1, 'STF+AGN')
    plt.text(0.25, -0.4, 'AGN')
    plt.scatter(xnrha, xorhb, marker='.')
    # plt.errorbar(xnrha, xorhb, xorhb*0.07, xnrha*0.12, alpha=0.5, fmt='none')
    if xerr is not None:
        plt.errorbar(xnrha, xorhb, xerr[0], xerr[1], fmt='none')

    plt.subplot(122)
    plt.ylim(-1.5, 3)
    plt.plot(sii_lin_axis, sii_lin_line, color='b', linestyle='-')
    plt.plot(sii_agn_axis, sii_agn_line, color='b', linestyle='-')
    if xsiiha is not None:
        plt.scatter(xsiiha, xorhb, marker='.')
        # plt.errorbar(xsiiha, xorhb, xorhb * 0.07, xsiiha * 0.02, alpha=0.5, fmt='none')
    if xerr is not None:
        plt.errorbar(xsiiha, xorhb, xerr[0], xerr[2], fmt='none')
    plt.xlabel(r'log$_{10}$([Sii]/H$\alpha$)')
    plt.ylabel(r'log$_{10}$([Oii]/H$\beta$)')
    plt.text(-0.4, -1, 'STF')
    plt.text(-0.75, 1.2, 'AGN')
    plt.text(0.25, -0.4, 'LINER')
    plt.tight_layout()
    return


def continuum_fitting(wave, flux, smooth_val=50, thres=2.5, end_clip=100):
    reject = 1
    i = 0
    ctm = np.copy(flux)
    t_flux = np.copy(flux)
    t_wave = np.copy(wave)

    while reject != 0 or i < 15:
        ctm = g_filt(t_flux, smooth_val)
        std = np.std(t_flux - ctm)

        t2_flux = t_flux[end_clip:-end_clip][np.abs(t_flux-ctm)[end_clip:-end_clip] < thres * std]
        t2_wave = t_wave[end_clip:-end_clip][np.abs(t_flux-ctm)[end_clip:-end_clip] < thres * std]
        reject = len(t_flux[end_clip:-end_clip]) - len(t2_flux)

        t_wave = np.concatenate([t_wave[:end_clip], t2_wave, t_wave[-end_clip:]])
        t_flux = np.concatenate([t_flux[:end_clip], t2_flux, t_flux[-end_clip:]])
        i += 1

    return rebin_spec(t_wave, ctm, wave)


def voigt(x, cen=0., sigma=1., gamma=None):
    """1 dimensional voigt function.
    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    if gamma is None:
        gamma = sigma

    z = (x-cen + 1j*gamma) / (sigma*np.sqrt(2))
    return wofz(z).real / (sigma*np.sqrt(2*np.pi))


def skewed_voigt(x, cen=0, sigma=1.0, gamma=None, skew=0.0):
    """Skewed Voigt lineshape, skewed with error function
    useful for ad-hoc Compton scatter profile

    with beta = skew/(sigma*sqrt(2))
    = voigt(x, cen, sigma, gamma)*(1+erf(beta*(x-cen)))

    skew < 0:  tail to low value of centroid
    skew > 0:  tail to high value of centroid

    see http://en.wikipedia.org/wiki/Skew_normal_distribution
    """
    beta = skew/(np.sqrt(2)*sigma)
    return (1 + erf(beta*(x-cen)))*voigt(x, cen=cen, sigma=sigma, gamma=gamma)

