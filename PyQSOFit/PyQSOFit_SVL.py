# A code for quasar spectrum fitting
# Last modified on 3/24/2020
# Auther: Hengxiao Guo AT UIUC
# Email: hengxiaoguo AT gmail DOT com
# Co-Auther Shu Wang, Yue Shen
# version 1.0
# -------------------------------------------------

# fix the error problem, previous error was underestimated by a factor of 1+z


import glob
import os
import sys
import timeit
import matplotlib
import typing
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from typing import Tuple

from Spectra_handling.Spectrum_utls import skewed_voigt
from Spectra_handling.Spectrum_processing import blueshifting
from scipy.stats import median_abs_deviation as mad
from kapteyn import kmpfit
from PyAstronomy import pyasl
from extinction import fitzpatrick99, remove
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.blackbody import blackbody_lambda
from astropy.table import Table
import warnings
from scipy.ndimage import gaussian_filter as g_filt
from rebin_spec import rebin_spec

warnings.filterwarnings("ignore")


def smooth(y, box_pts):
    """Smooth the flux with n pixels"""
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def continuum_fitting(wave, flux, smooth_val=50, thres=2.5, end_clip=100):
    reject = 1
    i = 0
    ctm = np.copy(flux)
    t_flux = np.copy(flux)
    t_wave = np.copy(wave)

    while reject != 0 or i < 15:
        ctm = g_filt(t_flux, smooth_val)
        std = np.std(t_flux - ctm)

        t2_flux = t_flux[end_clip:-end_clip][np.abs(t_flux - ctm)[end_clip:-end_clip] < thres * std]
        t2_wave = t_wave[end_clip:-end_clip][np.abs(t_flux - ctm)[end_clip:-end_clip] < thres * std]
        reject = len(t_flux[end_clip:-end_clip]) - len(t2_flux)

        t_wave = np.concatenate([t_wave[:end_clip], t2_wave, t_wave[-end_clip:]])
        t_flux = np.concatenate([t_flux[:end_clip], t2_flux, t_flux[-end_clip:]])
        i += 1

    return rebin_spec(t_wave, ctm, wave)


def Fe_flux_mgii(xval, pp):
    """Fit the UV Fe compoent on the continuum from 1200 to 3500 A based on the Boroson & Green 1992."""
    yval = np.zeros_like(xval)
    wave_Fe_mgii = 10 ** fe_uv[:, 0]
    flux_Fe_mgii = fe_uv[:, 1]
    Fe_FWHM = pp[1]
    xval_new = xval * (1.0 + pp[2])

    ind = np.where((xval_new > 1200.) & (xval_new < 3500.), True, False)
    if np.sum(ind) > 100:
        if Fe_FWHM < 900.0:
            sig_conv = np.sqrt(910.0 ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))
        else:
            sig_conv = np.sqrt(Fe_FWHM ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))  # in km/s
        # Get sigma in pixel space
        sig_pix = sig_conv / 106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
        khalfsz = np.round(4 * sig_pix + 1, 0)
        xx = np.arange(0, khalfsz * 2, 1) - khalfsz
        kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
        kernel = kernel / np.sum(kernel)

        flux_Fe_conv = np.convolve(flux_Fe_mgii, kernel, 'same')
        tck = interpolate.splrep(wave_Fe_mgii, flux_Fe_conv)
        yval[ind] = pp[0] * interpolate.splev(xval_new[ind], tck)
    return yval


def Fe_flux_balmer(xval, pp, xfe=None):
    """Fit the optical FeII on the continuum from 3686 to 7484 A based on Vestergaard & Wilkes 2001"""
    yval = np.zeros_like(xval)

    if xfe is None:
        wave_Fe_balmer = 10 ** fe_op[0]
        mult_flux = fe_op[1:] * 1e15
    else:
        wave_Fe_balmer = xfe[0]
        mult_flux = xfe[1:]
    mult_scale = np.asarray(pp[2:])
    # mult_flux = fe_op[1:]
    flux_Fe_balmer = np.dot(mult_scale, mult_flux)

    ind = np.where((wave_Fe_balmer > 3686.) & (wave_Fe_balmer < 7484.), True, False)
    wave_Fe_balmer = wave_Fe_balmer[ind]
    flux_Fe_balmer = flux_Fe_balmer[ind]

    Fe_FWHM = pp[0]
    xval_new = xval * (1.0 + pp[1])
    ind = np.where((xval_new > 3686.) & (xval_new < 7484.), True, False)
    if np.sum(ind) > 100:
        if Fe_FWHM < 900.0:
            sig_conv = np.sqrt(910.0 ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))
        else:
            sig_conv = np.sqrt(Fe_FWHM ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))  # in km/s
        # Get sigma in pixel space
        sig_pix = sig_conv / 106.3  # 106.3 km/s is the dispersion for the BG92 FeII template
        khalfsz = np.round(4 * sig_pix + 1, 0)
        xx = np.arange(0, khalfsz * 2, 1) - khalfsz
        kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
        kernel = kernel / np.sum(kernel)
        flux_Fe_conv = np.convolve(flux_Fe_balmer, kernel, 'same')
        tck = interpolate.splrep(wave_Fe_balmer, flux_Fe_conv)

        yval[ind] = interpolate.splev(xval_new[ind], tck)
        # yval[ind] = yval[ind] + interpolate.splev(xval[ind], tck) * 0.3
    return yval


def balmer_conti(xval, pp):
    """Fit the Balmer continuum from the model of Dietrich+02"""
    # xval = input wavelength, in units of A
    # pp=[norm, Te, tau_BE] -- in units of [--, K, --]

    lambda_BE = 3646.  # A
    bbflux = blackbody_lambda(xval, pp[1]).value * 3.14  # in units of ergs/cm2/s/A
    tau = pp[2] * (xval / lambda_BE) ** 3
    result = pp[0] * bbflux * (1. - np.exp(-tau))
    ind = np.where(xval > lambda_BE, True, False)
    if ind.any():
        result[ind] = 0.
    return result


def f_poly_conti(xval, pp):
    """Fit the continuum with a polynomial component account for the dust reddening with a*X+b*X^2+c*X^3"""
    xval2 = xval - 3000.
    yval = 0. * xval2
    for i in range(len(pp)):
        yval = yval + pp[i] * xval2 ** (i + 1)
    return yval


def flux2L(flux, z):
    """Transfer flux to luminoity assuming a flat Universe"""
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DL = cosmo.luminosity_distance(z).value * 10 ** 6 * 3.08 * 10 ** 18  # unit cm
    L = flux * 1.e-17 * 4. * np.pi * DL ** 2  # erg/s/A
    return L


# ----line function------
def onegauss(xval, pp):
    """The single Gaussian model used to fit the emission lines
    Parameter: the scale factor, central wavelength in logwave, line FWHM in logwave, skewness
    """

    yval = pp[0] * skewed_voigt(xval, pp[1], pp[2], pp[4] / 100, pp[3])
    return yval


def manygauss(xval, pp):
    """The multi-Gaussian model used to fit the emission lines, it will call the onegauss function"""
    ngauss = int(pp.shape[0] / 5)
    if ngauss != 0:
        yval = 0.
        for i in range(ngauss):
            yval = yval + onegauss(xval, pp[i * 5:(i + 1) * 5])
        return yval


class LineProperties:
    """Class to hold the PyQSOFit output property for a line"""
    def __init__(self, properties: Tuple = (0, 0, 0, 0, 0, 0)):
        self.fwhm = properties[0]
        self.sigma = properties[1]
        self.skew = properties[2]
        self.ew = properties[3]
        self.peak = properties[4]
        self.area = properties[5]
        self.list = {'fwhm': self.fwhm, 'sigma': self.sigma, 'skew': self.skew,
                     'ew': self.ew, 'peak': self.peak, 'area': self.area}


class SectionParameters:
    """class to hold to information about a fitting section when fitting a section of lines"""
    def __init__(self, wave: np.array, linelist, section_lines: list):
        self.line_indices = np.where(linelist['compname'] == section_lines, True, False)  # get line index
        self.n_line = np.sum(self.line_indices)  # n line in one complex

        linelist_fit = linelist[self.line_indices]
        # read section range from table
        self.section_range = [linelist_fit[0]['minwav'], linelist_fit[0]['maxwav']]
        bool_section_indices = (wave > self.section_range[0]) & (wave < self.section_range[1])
        self.section_indices = np.where(bool_section_indices, True, False)


class InitialProfileParameters:
    """class to translate input profile into input fitting parameters"""
    def __init__(self, inputstring):
        mode_definitions = inputstring
        self.mode_mode = {"v": "", "g": "0", "s": "0"}
        for xmode in mode_definitions.split(";"):
            self.mode_mode[xmode[0]] = xmode.split("[")[1][:-1]

    def process_profile(self, v_pos: float, voffset: float, ini_sig: str):
        """function to call to run all the operations"""
        ini_arr = np.empty(4, dtype=float)
        par_arr = np.empty(4, dtype=dict)
        ini_arr[0], par_arr[0] = self._sigma_defition(ini_sig)
        ini_arr[1], par_arr[1] = self._velocity_offset_definition(v_pos, voffset)
        ini_arr[2], par_arr[2] = self._gamma_defintion()
        ini_arr[3], par_arr[3] = self._skew_definitions()
        return {'ini_sig': ini_arr[0], 'sig_par': par_arr[0],
                'ini_voff': ini_arr[1], 'lam_par': par_arr[1],
                'ini_gam': ini_arr[2], 'gam_par': par_arr[2],
                'ini_skw': ini_arr[3], 'skw_par': par_arr[3]}

    def _velocity_offset_definition(self, v_pos: float, voffset: float):
        """initialise the velocity offset bounds based on vmode"""
        mode = self.mode_mode["v"]
        lambda_low = v_pos - voffset
        lambda_up = v_pos + voffset
        if voffset == 0:  # Fixed
            lam_par = {'fixed': True}
        elif mode == "":  # Free
            lam_par = {'limits': (lambda_low, lambda_up)}
        elif mode == "+":  # positive only
            lam_par = {'limits': (v_pos - voffset * 0.01, lambda_up)}
        elif mode == "-":  # negative only
            lam_par = {'limits': (lambda_low, v_pos + voffset * 0.01)}
        elif mode == ".":  # exact as defined
            v_pos += voffset
            lam_par = {'fixed': True}
        else:
            print("Velocity offset input parameter error")
            sys.exit()
        return v_pos, lam_par

    def _gamma_defintion(self):
        """Turns gamma on or off and sets bounds for Voigt profile"""
        mode = self.mode_mode["g"]
        if mode == "0":  # off
            ini_gam = 0
            gam_par = {'fixed': True}
        elif mode == "":  # free
            ini_gam = 1e-2
            gam_par = {'limits': (0, 1)}
        else:
            try:
                ini_gam = float(mode)  # exact as defined
                gam_par = {'fixed': True}
            except ValueError:
                print("Gamma input parameter error")
                sys.exit()
        return ini_gam, gam_par

    def _skew_definitions(self):
        """Initialises the first guess for skew and sets the bounds for skew"""
        mode = self.mode_mode["s"]
        if mode == "":  # Free
            ini_skw = 0
            skw_par = {}
        elif mode == "0":  # off
            ini_skw = 0
            skw_par = {'fixed': True}
        elif "," in mode:  # given range
            try:
                skw_down, skw_up = np.sort([float(mode.split(',')[0]), float(mode.split(',')[1])])
                ini_skw = np.mean([skw_down, skw_up])
                skw_par = {'limits': (skw_down, skw_up)}
            except ValueError:
                print("Skew input parameter error")
                sys.exit()
        else:
            try:
                ini_skw = float(mode)  # exact as defined
                skw_down, skw_up = np.sort([ini_skw * 1.001, ini_skw * 0.999])
                skw_par = {'limits': (skw_down, skw_up)}
            except ValueError:
                print("Skew input parameter error")
                sys.exit()
        return ini_skw, skw_par

    def _sigma_defition(self, ini_sig: str):
        """initialises the first guess for sigma and sets the bounds for sigma"""
        if len(ini_sig.split(',')) == 1:    # Fixed sigma
            sig_par = {'fixed': True}
            ini_sig = float(ini_sig.split(',')[0])
        else:   # Free sigma
            sig_down, sig_up = np.sort([float(ini_sig.split(',')[0]), float(ini_sig.split(',')[1])])
            ini_sig = np.mean([sig_down, sig_up])
            sig_par = {'limits': (sig_down, sig_up)}
        return ini_sig, sig_par


class LineDef:
    """
            Class to create line profile to add to Section for fitting

            Parameters:
            -----------
            l_name: str
                Name of the line. Must be unique.

            l_center: float
                The rest wavelength for the line.

            fwhm: tuple of two elements in km/s
                  One value input as (500,) will fix the fwhm of the modelling profile at 500.
                  Two values input as (1200, 7000) will have the fwhm range from 1200, 7000.
                  Values are not exact due to conversion to a log scale during modelling.

            profile_link: str, optional
                 A string that tells PyQSOFit which line the profile of this one is linked to.
                 For example, definition for OIII4959, will have 'OIII5007*1' as profile_link.
                 This overwrites fwhm, meaning fwhm can be ignored.

            skew: tuple of two elements
                One value input as (2.3,) will fix the skew of the modelling profile at 2.3.
                Two values input as (-10, 10) will have the skew range from -10 to 10.

            gamma: str, optional
                A string that controls the Voigt profile for the modelling.
                By default, it is an empty string and will turn off Voigt profile and use Gaussian profile instead.

                - If specified 'On', it will have a free gamma value and uses Voigt profile.

                - If the string has 'f0.53', it will fix the gamma at 0.53.

            voffset: float, in km/s
                The range of velocity offset allowed from the centre wavelength.

            vmode: str, optional
                Controls how the voffset range is defined.

                - Default is '', and means the offset is free on both ends.

                - '+' restricts the range such that the velocity offset can only increase in wavelength.

                - '-' restricts the range such that the velocity offset can only decrease in wavelength.

                - '.' sets the velocity offset at the specified voffset value.

            scale: float
                A value that controls the scale of the modelling profile.

            flux_link: str, optional
                Tells PyQSOFit which line the flux of this one is linked to.
                For example, definition for OIII4959, will have 'OIII5007*0.33' as flux_link.
                This overwrites scale, meaning scale can be ignored.

            default_bel: bool, optional.
                If True will set the fwhm to (1200, 7000), voffset to 5e-3, and skew to (-10, 10).

            default_nel. bool, optional.
                If True will set the sig to (50, 1200), voffset to 1e-3, and skew to (0,).

    """
    def __init__(self, fwhm: Tuple = (100, 1000), profile_link: str = "", skew: Tuple = (-10, 10), gamma: str = "",
                 vmode: str = "", voffset: float = 0.0, scale: float = 0.005, flux_link: str = "",
                 default_bel: bool = False, default_nel: bool = False, *, l_name: str = "", l_center: float = 0.0):
        self.l_name = l_name
        self.l_center = l_center
        self.fwhm = fwhm
        self._sig = (0, 0.01)
        self._voffset = voffset
        self._skew = skew
        self.gamma = gamma
        self.vmode = vmode
        self.plink = profile_link
        self._scale = scale
        self.flux_link = flux_link

        if default_nel:
            self.fwhm = (50, 1200)
            self._voffset = 300
            self._skew = (0,)
        if default_bel:
            self.fwhm = (1200, 7000)
            self._voffset = 1500
            self._skew = (-10, 10)

    @property
    def sig(self):
        if len(self.fwhm) == 2:
            self._sig = (self.fwhm_to_sig(self.fwhm[0]), self.fwhm_to_sig(self.fwhm[1]))
            return f"[{self._sig[0]}, {self._sig[1]}]"
        elif len(self._sig) == 1:
            self._sig = (self.fwhm_to_sig(self.fwhm[0]),)
            return f"[{self._sig[0]}]"
        else:
            print("Incorrect number of inputs for line fwhm. Should be 1 (exact) or 2 (range) values")
            print("Default to narrow line")
            return "[1e-5, 0.0017]"

    @property
    def scale(self):
        if self.flux_link == "":
            return str(self._scale)
        elif "*" in self.flux_link:
            return self.flux_link
        else:
            print("Incorrect inputs for line flux. Should be 1 float or 1 Line*multiplier (OIII*3)")
            print("Default to 0.005")
            return "0.005"

    @property
    def skew(self):
        if len(self._skew) == 2:
            return f"{self._skew[0]}, {self._skew[1]}"
        elif len(self._skew) == 1:
            return f"{self._skew[0]}"
        else:
            print("Incorrect number of inputs for line skew. Should be 1 (exact) or 2 (range) values")
            print("Default to no skew")
            return "0"

    @property
    def profile(self):
        l_gamma = ""
        l_voff = ""
        if self.gamma == "On":
            l_gamma = f"g[];"
        elif "f" in self.gamma:
            l_gamma = self.gamma[1:]
        if l_voff != "":
            l_voff = f"v[{self.vmode}];"

        return f"{l_voff}{l_gamma}s[{self.skew}]"

    @property
    def sig_info(self):
        if self.plink == "":
            return self.sig
        else:
            return self.plink

    @staticmethod
    def fwhm_to_sig(xfwhm: float):
        c = 299792.458  # km/s
        return np.round(np.sqrt(np.log(1 + ((xfwhm / c / 2.355) ** 2))), 5)

    @property
    def voffset(self):
        c = 299792.458  # km/s
        wave_offset = self._voffset / 3e5 * self.l_center
        log_offset = np.log(self.l_center + wave_offset) - np.log(self.l_center)
        return log_offset

class Section:
    """
    Class to create sections to add lines into for fitting

    Call line_add to add a single line

    Call add_lines to add a list of lines
    """
    def __init__(self, *, section_name: str = "", start_range: float = 0.0, end_range: float = 1.0):
        self.section_name = section_name
        self.start_range = start_range
        self.end_range = end_range
        self._lines = []
        self.rarray = None

    def line_add(self, new_line: LineDef):
        if self.start_range <= new_line.l_center <= self.end_range:
            self._lines.append(new_line)

    def add_lines(self, lines: list):
        for xline in lines:
            self.line_add(xline)

    @property
    def lines(self):
        if self.rarray is None:
            xtuples = [self.linedef_to_tuple(xline) for xline in self._lines]
            xdtype = np.dtype([('lambda', 'float32'), ('compname', 'U20'), ('minwav', 'float32'), ('maxwav', 'float32'),
                               ('linename', 'U20'), ('sigval', 'U20'), ('voff', 'float32'), ('iniskw', 'U20'),
                               ('fvalue', 'U20')])
            self.rarray = np.rec.array(xtuples, dtype=xdtype)
        return self.rarray

    def linedef_to_tuple(self, xline: LineDef) -> Tuple:
        return (xline.l_center, self.section_name, self.start_range, self.end_range, xline.l_name,
                xline.sig_info, xline.voffset, xline.profile, xline.scale)

    @staticmethod
    def hdu_generate() -> fits.Header:
        hdr = fits.Header()
        hdr['lambda'] = 'Vacuum Wavelength in Ang'
        hdr['minwav'] = 'Lower complex fitting wavelength range'
        hdr['maxwav'] = 'Upper complex fitting wavelength range'
        hdr['ngauss'] = 'Number of Gaussians for the line. Not all functions supported for >1'
        hdr['sigval'] = 'Sigma information in strings, either a single value, [min, max], or name of line to tie to'
        hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
        hdr['iniskw'] = 'Initial guess of lineskew'
        hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'
        return hdr


class QSOFit:

    def __init__(self, lam, flux, err, z, ra=- 999., dec=-999., ebmv=None, plateid=None, mjd=None, fiberid=None,
                 path=None, and_mask=None, or_mask=None):
        """
        Get the input data perpared for the QSO spectral fitting

        Parameters:
        -----------
        lam: 1-D array with Npix
             Observed wavelength in unit of Angstrom

        flux: 1-D array with Npix
             Observed flux density in unit of 10^{-17} erg/s/cm^2/Angstrom

        err: 1-D array with Npix
             1 sigma err with the same unit of flux

        z: float number
            redshift

        ra, dec: float number, optional
            the location of the source, right ascension and declination. The default number is 0

        plateid, mjd, fiberid: integer number, optional
            If the source is SDSS object, they have the plate ID, MJD and Fiber ID in their file herader.

        path: str
            the path of the input data

        and_mask, or_mask: 1-D array with Npix, optional
            the bad pixels defined from SDSS data, which can be got from SDSS datacube.

        """

        self.lam = np.asarray(lam, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.err = np.asarray(err, dtype=np.float64)
        self.z = z
        self.and_mask = and_mask
        self.or_mask = or_mask
        self.ra = ra
        self.dec = dec
        self.plateid = plateid
        self.mjd = mjd
        self.fiberid = fiberid
        self.path = path
        self.ebmv = ebmv
        self.c = 299792.458  # km/s

    def Fit(self, name=None, nsmooth=1, and_or_mask=True, reject_badpix=True, deredden=True,
            decomposition_host=True, BC03=False, Mi=None, npca_gal=5, npca_qso=20,
            Fe_uv_op=True, Conti_window=True, redshift=True,
            poly=False, PL=True, CFT=False, CFT_smooth=75, BC=False, rej_abs=False, initial_guess=None, MC_conti=False,
            MC=True, n_trails=1, linefit=True,
            save_result=True, plot_fig=True, save_fig=True, plot_line_name=True, plot_legend=True, dustmap_path=None,
            save_fig_path=None, save_fits_path=None, save_fits_name=None):

        """
        Fit the QSO spectrum and get different decomposed components and corresponding parameters

        Parameter:
        ----------
        name: str, optinal
            source name, Default is None. If None, it will use plateid+mjd+fiberid as the name. If there are no
            such parameters, it will be empty.

        nsmooth: integer number, optional
            do n-pixel smoothing to the raw input flux and err spectra. The default is set to 1 (no smooth).
            It will return the same array size. We note that smooth the raw data is not suggested,
            this function is in case of some fail-fitted low S/N spectra.

        and_or_mask: bool, optional
            If True, and and_mask or or_mask is not None, it will delete the masked pixels,
            and only return the remained pixels. Default: False

        reject_badpix: bool, optional
            reject 10 most possible outliers by the test of pointDistGESD. One important Caveat here
            is that this process will also delete narrow emission lines in some high SN ratio object e.g., [OIII].
            Only use it when you are definitely clear about what you are doing. It will return the remained pixels.

        deredden: bool, optional
            correct the Galactic extinction only if the RA and Dec are available. It will return the corrected
            flux with the same array size. Default: True.

        decomposition_host:
            bool, optional If True, the host galaxy-QSO decomposition will be applied. If no more than 100 pixels are
            negative, the result will be applied. The Decomposition is based on the PCA method of Yip et al. 2004 (
            AJ, 128, 585) & (128, 2603). Now the template is only available for redshift < 1.16 in specific absolute
            magnitude bins. For galaxy, the global model has 10 PCA components and first 5 will enough to reproduce
            98.37% galaxy spectra. For QSO, the global model has 50, and the first 20 will reproduce 96.89% QSOs. If
            with i-band absolute magnitude, the Luminosity-redshift binned PCA components are available. Then the
            first 10 PCA in each bin is enough to reproduce most QSO spectrum. Default: False

        BC03: bool, optional
            if True, it will use Bruzual1 & Charlot 2003 host model to fit spectrum, high shift host will be low
            resolution R ~ 300, the rest is R ~ 2000. Default: False

        Mi: float, optional
            the absolute magnitude of 'i' band. It only works when decomposition_host is True. If not None,
            the Luminosity redshift binned PCA will be used to decompose the spectrum. Default: None

        npca_gal: int, optional
            the number of galaxy PCA components applied. It only works when decomposition_host is True. The default
            is 5, which is already account for 98.37% galaxies.

        npca_qso: int, optional
            the number of QSO PCA components applied. It only works when decomposition_host is True. The default is 20,
            No matter the global or luminosity-redshift binned PCA is used, it can reproduce > 92% QSOs. The binned PCA
            is better if with Mi information.

        Fe_uv_op: bool, optional
            if True, fit continuum with UV and optical FeII template. Default: True

        Fe_in_line: bool, optional
            if True, improve FeII template along with emission line fitting. Default: False

        poly: bool, optional
            if True, fit continuum with the polynomial component to account for the dust reddening. Default: False

        BC: bool, optional
            if True, fit continuum with Balmer continua from 1000 to 3646A. Default: False

        rej_abs: bool, optional
            if True, it will iterate the continuum fitting for deleting some 3 sigmas out continuum window points
            (< 3500A), which might fall into the broad absorption lines. Default: False

        initial_gauss: 1*14 array, optional
            better initial value will help find a solution faster. Default initial is np.array([0., 3000., 0., 0.,
            3000., 0., 1., -1.5, 0., 15000., 0.5, 0., 0., 0.]). First six parameters are flux scale, FWHM,
            small shift for wavelength for UV and optical FeII template, respectively. The next two parameters are
            the power-law slope and intercept. The next three are the norm, Te, tau_BE in Balmer continuum model in
            Dietrich et al. 2002. the last three parameters are a,b,c in polynomial function a*(x-3000)+b*x^2+c*x^3.

        MC: bool, optional
            if True, do the Monte Carlo simulation based on the input error array to produce the MC error array.
            if False, the code will not save the error produced by kmpfit since it is biased and can not be trusted.
            But it can be still output by in kmpfit attribute. Default: False

        n_trails: int, optional
            the number of trails of the MC process to produce the error array. The conservative number should be
            larger than 20. It only works when MC is True. Default: 20

        linefit: bool, optional
            if True, the emission line will be fitted. Default: True

        save_result: bool, optional
            if True, all the fitting results will be saved to a fits file, Default: True

        plot_fig: bool, optional
            if True, the fitting results will be plotted. Default: True

        save_fig: bool, optional
            if True, the figure will be saved, and the path can be set by "save_fig_path". Default: True

        plot_line_name: bool, optional
            if True, serval main emission lines will be plotted in the first panel of the output figure. Default: False

        plot_legend: bool, optional
            if True, open legend in the first panel of the output figure. Default: False

        dustmap_path: str, optional
            if Deredden is True, the dustmap_path must be set. If None, the default "dustmap_path" is set to "path"

        save_fig_path: str, optional
            the output path of the figure. If None, the default "save_fig_path" is set to "path"

        save_fit_path: str, optional
            the output path of the result fits. If None, the default "save_fits_path" is set to "path"

        save_fit_name: str, optional
            the output name of the result fits. Default: "result.fits"

        Return:
        -----------
        .wave: array
            the rest wavelength, some pixels have been removed.

        .flux: array
            the rest flux. Dereddened and *(1+z) flux.

        .err: array
            the error.

        .wave_prereduced: array
            the wavelength after removing bad pixels, masking, deredden, spectral trim, and smoothing.

        .flux_prereduced: array
            the flux after removing bad pixels, masking, deredden, spectral trim, and smoothing.

        .err_prereduced: array
            the error after removing bad pixels, masking, deredden, spectral trim, and smoothing.

        .host: array
            the model of host galaxy from PCA method

        .qso: array
            the model of a quasar from PCA method.

        .SN_ratio_conti: float
            the mean S/N ratio of 1350, 3000 and 5100A.

        .conti_fit.: structure
            all the kmpfit continuum fitting results, including best-fit parameters and Chisquare, etc. For details,
            see https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html

        .f_conti_model: array
            the continuum model including power-law, polynomial, optical/UV FeII, Balmer continuum.

        .f_bc_model: array
            the Balmer continuum model.

        .f_fe_uv_model: array
            the UV FeII model.

        .f_fe_op_model: array
            the optical FeII model.

        .f_pl_model: array
            the power-law model.

        .f_poly_model: array
            the polynomial model.

        .PL_poly_BC: array
            The combination of Powerlaw, polynomial and Balmer continuum model.

        .line_flux: array
            the emission line flux after subtracting the .f_conti_model.

        .line_fit: structrue
            kmpfit line fitting results for last complexes (From Lya to Ha) , including best-fit parameters,
            errors (kmpfit derived) and Chisquare, etc. For details,
            see https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html

        .gauss_result: array
            3*n Gaussian parameters for all lines in the format of [scale, centerwave, sigma ], n is number of
            Gaussians for all complexes.

        .conti_result: array
            continuum parameters, including widely used continuum parameters and monochromatic flux at 1350,
            3000 and5100 Angstrom, etc. The corresponding names are listed in .conti_result_name. For all continuum
            fitting results, go to .conti_fit.params.

        .conti_result_name: array
            the names for .conti_result.

        .line_result: array
            emission line parameters, including FWHM, sigma, EW, measured from whole model of each main broad
            emission line covered, and fitting parameters of each Gaussian component. The corresponding names are
            listed in .line_result_name.


        .line_result_name: array
            the names for .line_result.

        .uniq_linecomp_sort: array
            the sorted complex names.

        .all_comp_range: array
            the start and end wavelength for each complex. e.g., Hb in [4640.  5100.].

        .linelist: array
            the information listed in the qsopar.fits.
        """

        self.name = name
        self.BC03 = BC03
        self.Mi = Mi
        self.npca_gal = npca_gal
        self.npca_qso = npca_qso
        self.initial_guess = initial_guess
        self.host_decomposition = decomposition_host
        self.Fe_uv_op = Fe_uv_op
        self.Conti_window = Conti_window
        self.poly = poly
        self.CFT = CFT
        self.CFT_smooth = CFT_smooth
        self.PL = PL
        self.BC = BC
        self.rej_abs = rej_abs
        self.MC = MC
        self.MC_conti = MC_conti
        self.n_trails = n_trails
        self.plot_line_name = plot_line_name
        self.plot_legend = plot_legend
        self.save_fig = save_fig

        # get the source name in plate-mjd-fiber, if no then None
        if name is None:
            if np.array([self.plateid, self.mjd, self.fiberid]).any() is not None:
                self.sdss_name = str(self.plateid).zfill(4) + '-' + str(self.mjd) + '-' + str(self.fiberid).zfill(4)
            else:
                if self.plateid is None:
                    self.plateid = 0
                if self.mjd is None:
                    self.mjd = 0
                if self.fiberid is None:
                    self.fiberid = 0
                self.sdss_name = ''
        else:
            self.sdss_name = name

        # set default path for figure and fits
        if save_result and save_fits_path is None:
            save_fits_path = self.path
        if save_fig and save_fig_path is None:
            save_fig_path = self.path
        if save_fits_name is None:
            if self.sdss_name == '':
                save_fits_name = 'result'
            else:
                save_fits_name = self.sdss_name
        else:
            save_fits_name = save_fits_name

        self._remove_bad_error()

        # Apply smoothing if required
        if nsmooth is not None:
            self.flux = smooth(self.flux, nsmooth)
            self.err = smooth(self.err, nsmooth)

        # Handle SDSS masking operation
        if and_or_mask:
            self._MaskSdssAndOr()

        # Reject bad pixels if required
        if reject_badpix:
            self._RejectBadPix()

        # Deredden if ebmv given and required
        if deredden:
            self.flux = self._DeRedden(self.lam, self.flux, self.ebmv)

        # blueshift spectrum if required
        if redshift:
            self.wave, self.flux, self.err = self._RestFrame(self.lam, self.flux, self.err, self.z)
        else:
            self.wave = self.lam

        self.SN_ratio_conti = self._CalculateSN(self.lam, self.flux)
        self._OrignialSpec()

        if CFT:
            self._reset_contipp_cft()
            self._apply_cft()

        # do host decomposition --------------
        if self.z < 1.16 and self.host_decomposition:
            self._DoDecomposition()
        else:
            self.decomposed = False
            if self.z > 1.16 and self.host_decomposition:
                print('redshift larger than 1.16 is not allowed for host decomposion!')

        # fit continuum --------------------
        if not self.Fe_uv_op and not self.poly and not self.BC:
            self.line_flux = self.flux if self.line_flux is None else self.line_flux
            self.conti_fit = np.zeros_like(self.wave)
            self.f_bc_model = np.zeros_like(self.wave)
            self.f_fe_uv_model = np.zeros_like(self.wave)
            self.f_fe_op_model = np.zeros_like(self.wave)
            self.f_pl_model = np.zeros_like(self.wave)
            self.f_poly_model = np.zeros_like(self.wave)
            self.PL_poly_BC = np.zeros_like(self.wave)
            self.tmp_all = np.zeros_like(self.wave)
            self.conti_result = np.zeros_like(self.wave)
            self.f_conti_model = np.zeros_like(self.wave) if self.f_conti_model is None else self.f_conti_model
        else:
            print('Fit Conti')
            self._DoContiFit()
            print('Conti done')
        # fit line
        print('Fit Line')
        if linefit:
            self._DoLineFit()
        print('Line done')
        # save data -------
        if save_result:
            if not linefit:
                self.line_result = np.array([])
                self.line_result_name = np.array([])
            self._SaveResult(self.conti_result, self.conti_result_name, self.fe_op_result, self.fe_op_result_name,
                             self.line_result, self.line_result_name, save_fits_path, save_fits_name)

        # plot fig and save ------
        if plot_fig:
            if not linefit:
                self.gauss_result = np.array([])
                self.all_comp_range = np.array([])
                self.uniq_linecomp_sort = np.array([])
            self._PlotFig(linefit, save_fig_path)

    def _remove_bad_error(self):
        """deal with pixels with error equal 0 or inifity"""
        ind_gooderror = [(self.err != 0) & ~np.isinf(self.err)]
        self.err = self.err[ind_gooderror]
        self.flux = self.flux[ind_gooderror]
        self.lam = self.lam[ind_gooderror]

        if (self.and_mask is not None) & (self.or_mask is not None):
            self.and_mask = self.and_mask[ind_gooderror]
            self.or_mask = self.or_mask[ind_gooderror]

    def _MaskSdssAndOr(self):
        """
        Remove SDSS and_mask and or_mask points are not zero
        Parameter:
        ----------
        lam: wavelength
        flux: flux
        err: 1 sigma error
        and_mask: SDSS flag "and_mask", mask out all non-zero pixels
        or_mask: SDSS flag "or_mask", mask out all npn-zero pixels

        Retrun:
        ---------
        return the same size array of wavelength, flux, error
        """
        if self.and_mask is None or self.or_mask is None:
            return
        ind_and_or = [(self.and_mask == 0) & (self.or_mask == 0)]
        self.lam, self.flux, self.err = self.lam[ind_and_or], self.flux[ind_and_or], self.err[ind_and_or]

    def _RejectBadPix(self):
        """
        Reject 10 most possiable outliers, input wavelength, flux and error. Return a different size wavelength,
        flux, and error.
        """
        # -----remove bad pixels, but not for high SN spectrum------------
        ind_bad = pyasl.pointDistGESD(self.flux, 10)
        wv = np.asarray([i for j, i in enumerate(self.lam) if j not in ind_bad[1]], dtype=np.float64)
        fx = np.asarray([i for j, i in enumerate(self.flux) if j not in ind_bad[1]], dtype=np.float64)
        er = np.asarray([i for j, i in enumerate(self.err) if j not in ind_bad[1]], dtype=np.float64)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = wv, fx, er

    @staticmethod
    def _DeRedden(lam: np.array, flux: np.array, ebmv: float) -> np.array:
        if ebmv is None:
            print('No ebmv value given. No dereddening applied')
            return flux
        """Correct the Galatical extinction"""
        # m = sfdmap.SFDMap(dustmap_path)
        flux_unred = remove(fitzpatrick99(lam, a_v=ebmv * 3.1, r_v=3.1), flux)
        flux = flux_unred
        return flux

    @staticmethod
    def _RestFrame(lam: np.array, flux: np.array, err: np.array, z: float) -> np.array:
        """Move wavelenth and flux to rest frame"""
        wave, flux = blueshifting([lam, flux], z)
        err = err
        return wave, flux, err

    def _OrignialSpec(self):
        """save the orignial spectrum before host galaxy decompsition"""
        self.wave_prereduced = self.wave
        self.flux_prereduced = self.flux
        self.err_prereduced = self.err

    @staticmethod
    def _CalculateSN(wave: np.array, flux: np.array) -> float:
        """calculate the spectral SN ratio for 1350, 3000, 5100A, return the mean value of Three spots"""
        if (wave.min() < 1350. < wave.max()) or (wave.min() < 3000. < wave.max()) or (wave.min() < 5100. < wave.max()):
            ind5100 = np.where((wave > 5080.) & (wave < 5130.), True, False)
            ind3000 = np.where((wave > 3000.) & (wave < 3050.), True, False)
            ind1350 = np.where((wave > 1325.) & (wave < 1375.), True, False)

            tmp_SN = np.array([flux[ind5100].mean() / flux[ind5100].std(), flux[ind3000].mean() / flux[ind3000].std(),
                               flux[ind1350].mean() / flux[ind1350].std()])
            tmp_SN = tmp_SN[~np.isnan(tmp_SN)]
            SN_ratio_conti = tmp_SN.mean()
        else:
            SN_ratio_conti = -1.

        return SN_ratio_conti

    def _reset_contipp_cft(self):
        self.host_decomposition = False
        self.poly = False
        self.BC = False
        self.Fe_uv_op = False
        self.PL = False

    def _apply_cft(self):
        self.f_conti_model = continuum_fitting(self.wave, self.flux, self.CFT_smooth)
        self.line_flux = self.flux - self.f_conti_model

    def _DoDecomposition(self):
        """Decompose the host galaxy from QSO"""
        datacube = self._HostDecompose()

        # for some negtive host templete, we do not do the decomposition
        if np.sum(np.where(datacube[3, :] < 0., True, False)) > 100:
            self.host = np.zeros(len(self.wave))
            self.decomposed = False
            print('Get negtive host galaxy flux larger than 100 pixels, decomposition is not applied!')
            return self.wave, self.flux, self.err

        self.decomposed = True
        del self.wave, self.flux, self.err
        self.wave = datacube[0, :]
        # block OIII, ha,NII,SII,OII,Ha,Hb,Hr,hdelta

        line_mask = np.where((self.wave < 4970.) & (self.wave > 4950.) |
                             (self.wave < 5020.) & (self.wave > 5000.) |
                             (self.wave < 6590.) & (self.wave > 6540.) |
                             (self.wave < 6740.) & (self.wave > 6710.) |
                             (self.wave < 3737.) & (self.wave > 3717.) |
                             (self.wave < 4872.) & (self.wave > 4852.) |
                             (self.wave < 4350.) & (self.wave > 4330.) |
                             (self.wave < 4111.) & (self.wave > 4091.), True, False)

        f = interpolate.interp1d(self.wave[~line_mask], datacube[3, :][~line_mask], bounds_error=False, fill_value=0)
        masked_host = f(self.wave)

        self.flux = datacube[1, :] - masked_host  # ** change back to masked_host for BEL
        self.err = datacube[2, :]
        self.host = datacube[3, :]
        self.qso = datacube[4, :]
        self.host_data = datacube[1, :] - self.qso

        return self.wave, self.flux, self.err

    def _HostDecompose(self):
        """
        core function to do host decomposition
        #Wave is the obs frame wavelength, n_gal and n_qso are the number of eigenspectra used to fit
        #If Mi is None then the qso use the globle ones to fit. If not then use the
        #redshift-luminoisty binded ones to fit
        #See details:
        #Yip, C. W., Connolly, A. J., Szalay, A. S., et al. 2004a, AJ, 128, 585
        #Yip, C. W., Connolly, A. J., Vanden Berk, D. E., et al. 2004b, AJ, 128, 2603
        """

        # read galaxy and qso eigenspectra -----------------------------------
        if not self.BC03:
            galaxy = fits.open(self.path + 'pca/Yip_pca_templates/gal_eigenspec_Yip2004.fits')
            gal = galaxy[1].data
            wave_gal = gal['wave'].flatten()
            flux_gal = gal['pca'].reshape(gal['pca'].shape[1], gal['pca'].shape[2])

        if self.BC03:
            cc = 0
            flux03 = np.array([])
            for i in glob.glob(self.path + '/bc03/*.gz'):
                cc = cc + 1
                gal_temp = np.genfromtxt(i)
                wave_gal = gal_temp[:, 0]
                flux03 = np.concatenate((flux03, gal_temp[:, 1]))
            flux_gal = np.array(flux03).reshape(cc, -1)

        if self.Mi is None:
            quasar = fits.open(self.path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_global.fits')
        else:
            if -24 < self.Mi <= -22 and 0.08 <= self.z < 0.53:
                quasar = fits.open(self.path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_CZBIN1.fits')
            elif -26 < self.Mi <= -24 and 0.08 <= self.z < 0.53:
                quasar = fits.open(self.path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_DZBIN1.fits')
            elif -24 < self.Mi <= -22 and 0.53 <= self.z < 1.16:
                quasar = fits.open(self.path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_BZBIN2.fits')
            elif -26 < self.Mi <= -24 and 0.53 <= self.z < 1.16:
                quasar = fits.open(self.path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_CZBIN2.fits')
            elif -28 < self.Mi <= -26 and 0.53 <= self.z < 1.16:
                quasar = fits.open(self.path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_DZBIN2.fits')
            else:
                raise RuntimeError('Host galaxy template is not available for this redshift and Magnitude!')

        qso = quasar[1].data
        wave_qso = qso['wave'].flatten()
        flux_qso = qso['pca'].reshape(qso['pca'].shape[1], qso['pca'].shape[2])

        # get the shortest wavelength range
        wave_min = max(self.wave.min(), wave_gal.min(), wave_qso.min())
        wave_max = min(self.wave.max(), wave_gal.max(), wave_qso.max())

        ind_data = np.where((self.wave > wave_min) & (self.wave < wave_max), True, False)
        ind_gal = np.where((wave_gal > wave_min - 1.) & (wave_gal < wave_max + 1.), True, False)
        ind_qso = np.where((wave_qso > wave_min - 1.) & (wave_qso < wave_max + 1.), True, False)

        flux_gal_new = np.zeros(flux_gal.shape[0] * self.flux[ind_data].shape[0]).reshape(flux_gal.shape[0],
                                                                                          self.flux[ind_data].shape[0])
        flux_qso_new = np.zeros(flux_qso.shape[0] * self.flux[ind_data].shape[0]).reshape(flux_qso.shape[0],
                                                                                          self.flux[ind_data].shape[0])
        for i in range(flux_gal.shape[0]):
            fgal = interpolate.interp1d(wave_gal[ind_gal], flux_gal[i, ind_gal], bounds_error=False, fill_value=0)
            flux_gal_new[i, :] = fgal(self.wave[ind_data])
        for i in range(flux_qso.shape[0]):
            fqso = interpolate.interp1d(wave_qso[ind_qso], flux_qso[0, ind_qso], bounds_error=False, fill_value=0)
            flux_qso_new[i, :] = fqso(self.wave[ind_data])

        wave_new = self.wave[ind_data]
        flux_new = self.flux[ind_data]
        err_new = self.err[ind_data]

        flux_temp = np.vstack((flux_gal_new[0:self.npca_gal, :], flux_qso_new[0:self.npca_qso, :]))
        res = np.linalg.lstsq(flux_temp.T, flux_new)[0]

        host_flux = np.dot(res[0:self.npca_gal], flux_temp[0:self.npca_gal])
        qso_flux = np.dot(res[self.npca_gal:], flux_temp[self.npca_gal:])

        data_cube = np.vstack((wave_new, flux_new, err_new, host_flux, qso_flux))

        ind_f4200 = [(wave_new > 4160.) & (wave_new < 4210.)]
        frac_host_4200 = np.sum(host_flux[ind_f4200]) / np.sum(flux_new[ind_f4200])
        ind_f5100 = [(wave_new > 5080.) & (wave_new < 5130.)]
        frac_host_5100 = np.sum(host_flux[ind_f5100]) / np.sum(flux_new[ind_f5100])

        return data_cube

    def _DoContiFit(self):
        """Fit the continuum with PL, Polynomial, UV/optical FeII, Balmer continuum"""
        global fe_uv, fe_op, fe_op_name
        fe_uv = np.genfromtxt(self.path + 'feII_UV_Mejia.txt')
        fe_op = np.genfromtxt(self.path + 'fe_optical.txt', delimiter='')[1:].T
        fe_op_name = np.genfromtxt(self.path + 'fe_op_lines.csv', delimiter=',', dtype=str)[0]

        '''
        # do continuum fit--------------------------
        window_all = np.array([[1150., 1170.], [1275., 1290.], [1350., 1360.], [1445., 1465.],
                               [1690., 1705.], [1770., 1810.], [1970., 2400.], [2480., 2675.],
                               [2925., 3400.], [3775., 3832.], [4000., 4050.], [4200., 4230.],
                               [4435., 4640.], [5100., 5535.], [6005., 6035.], [6110., 6250.],
                               [6800., 7000.], [7160., 7180.], [7500., 7800.], [8050., 8150.]])
        '''
        # 2 windows removed to allow for CIV abs line
        if self.Conti_window:
            window_all = np.array([[1150., 1170.], [1275., 1290.],
                                   [1690., 1705.], [1770., 1810.], [1970., 2400.], [2480., 2675.],
                                   [2925., 3300.], [3500., 3650.], [3775., 3832.], [4000., 4050.], [4200., 4270.],
                                   [4470., 4650.], [5050., 5535.], [6005., 6035.], [6110., 6400.], [6700, 6750],
                                   [6800., 7000.], [7170., 7240.], [7500., 7800.], [8050., 8150.],
                                   [8528., 8930.], [9310., 9426.], [9666.24, 9786],
                                   [10400., 10600.], [12000, 12433], [13333, 13360],
                                   [13922, 14490], [16660, 18340], [19000, 19800]])
        else:
            window_all = np.array([[self.wave[0], self.wave[-1]]])

        tmp_all = np.array([np.repeat(False, len(self.wave))]).flatten()
        for jj in range(len(window_all)):
            tmp = np.where((self.wave > window_all[jj, 0]) & (self.wave < window_all[jj, 1]), True, False)
            tmp_all = np.any([tmp_all, tmp], axis=0)

        if self.wave[tmp_all].shape[0] < 10:
            print('Continuum fitting pixel < 10.  ')

        # set initial paramiters for continuum
        if self.initial_guess is not None:
            pp0 = self.initial_guess
        else:
            # ini val
            pp0 = np.array([0.001, 0,  # P-law norm, slope
                            0., 15000., 0.5,  # BC norm, Te, Tau
                            0., 0., 0.])  # Polynomial
            global ppxfe
            global feopxmg
            ppxfe = len(pp0)
            # pp_fe_op = np.concatenate([[3000., 0.], 0.2 * np.ones_like(fe_op_name[1:], dtype=float)])
            pp_fe_op = [1000., 0, 0.0]
            feopxmg = len(pp_fe_op) + ppxfe
            pp_mg_op = np.asarray([0.01, 3000., 0.])
            pp0 = np.concatenate([pp0, pp_fe_op, pp_mg_op])

        conti_fit = kmpfit.Fitter(residuals=self._residuals,
                                  data=(self.wave[tmp_all], self.flux[tmp_all], self.err[tmp_all]))
        tmp_parinfo = [{'limits': (0., 100)}, {'limits': (-5., 3)},
                       {'limits': (0, 100)}, {'limits': (10000., 50000.)}, {'limits': (0.1, 2.)},
                       None, None, None]
        # fe_op_info = [{'limits': (500., 5000.)}, {'limits': (-0.1, 0.1)},
        #               *[{'limits': (0., 10. ** 10)} for i in fe_op_name[1:]]]
        fe_op_info = [{'limits': (500., 5000.)}, {'limits': (-0.1, 0.1)}, {'limits': (0., 1000)}]
        fe_mg_info = [{'limits': (0., 1000)}, {'limits': (1200., 10000.)}, {'limits': (-0.1, 0.1)}]

        conti_fit.parinfo = np.concatenate([tmp_parinfo, fe_op_info, fe_mg_info])
        conti_fit.fit(params0=pp0)

        # Perform one iteration to remove 3sigma pixel below the first continuum fit
        # to avoid the continuum windows falls within a BAL trough
        if self.rej_abs:
            if self.poly:
                tmp_conti = conti_fit.params[0] * (self.wave[tmp_all] / 3000.0) ** conti_fit.params[1] + f_poly_conti(
                    self.wave[tmp_all], conti_fit.params[5:8])
            else:
                tmp_conti = conti_fit.params[0] * (self.wave[tmp_all] / 3000.0) ** conti_fit.params[1]

            ind_BAL = [(self.flux[tmp_all] < tmp_conti - 3. * self.err[tmp_all]) & (self.wave[tmp_all] < 3500.)]
            ind_noBAL = [not b for b in ind_BAL]
            f = kmpfit.Fitter(residuals=self._residuals,
                              data=(self.wave[tmp_all][ind_noBAL],
                                    smooth(self.flux[tmp_all][ind_noBAL], 10),
                                    self.err[tmp_all][ind_noBAL]))

            conti_fit.parinfo = tmp_parinfo
            conti_fit.fit(params0=pp0)

        # calculate continuum luminoisty
        L = self._L_conti(self.wave, conti_fit.params)
        f_4570 = self._fe4570(self.wave, conti_fit.params)

        # calculate MC err
        if self.MC_conti and self.n_trails > 0:
            conti_para_std, all_L_std, all_fe_std = self._conti_mc(self.wave[tmp_all], self.flux[tmp_all],
                                                                   self.err[tmp_all], pp0,
                                                                   conti_fit.parinfo, self.n_trails)

        res_name = np.array(['ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift', 'SN_ratio_conti', 'PL_norm',
                             'PL_slope', 'BC_norm', 'BC_Te', 'BC_Tau', 'POLY_a', 'POLY_b', 'POLY_c',
                             'L1350', 'L3000', 'L5100', 'L0.9', 'L1.2', 'L2.0', 'f4570'])

        fe_name = np.array(['Fe_op_FWHM', 'Fe_op_shift', *fe_op_name[1:], 'Fe_uv_scale', 'Fe_uv_FWHM', 'Fe_uv_shift'])
        # get conti result -----------------------------
        if not self.MC_conti:
            self.conti_result = np.array(
                [self.ra, self.dec, self.plateid, self.mjd, self.fiberid, self.z, self.SN_ratio_conti,
                 *conti_fit.params[:8], *L, f_4570])
            self.conti_result_name = res_name
            self.fe_result = np.array([*conti_fit.params[ppxfe:]])
            self.fe_result_name = fe_name

        else:
            res_err = np.concatenate([[i, j] for i, j in zip(conti_fit.params[:8],
                                                             conti_para_std[:8])])
            L_err = np.concatenate([[i, j] for i, j in zip(L, all_L_std)])
            res_err_name = np.concatenate([[i, i + '_err'] for i in res_name[7:]])

            self.conti_result = np.array(
                [self.ra,self. dec, self.plateid, self.mjd, self.fiberid, self.z, self.SN_ratio_conti,
                 *res_err, *L_err, f_4570, all_fe_std])
            self.conti_result_name = np.array(
                ['ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift', 'SN_ratio_conti', *res_err_name])

            self.fe_result = np.concatenate([[i, j] for i, j in zip(conti_fit.params[ppxfe:],
                                                                    conti_para_std[ppxfe:])])
            self.fe_result_name = np.concatenate([[i, i + '_err'] for i in fe_name])

        self.conti_fit = conti_fit
        self.tmp_all = tmp_all

        # save different models--------------------
        self.f_fe_mgii_model = Fe_flux_mgii(self.wave, conti_fit.params[feopxmg:])
        self.f_fe_balmer_model = Fe_flux_balmer(self.wave, conti_fit.params[ppxfe:feopxmg])
        self.f_pl_model = conti_fit.params[0] * (self.wave / 3000.0) ** conti_fit.params[1]
        self.f_bc_model = balmer_conti(self.wave, conti_fit.params[2:5])
        self.f_poly_model = f_poly_conti(self.wave, conti_fit.params[5:8])
        self.PL_poly_BC = self.f_pl_model + self.f_poly_model + self.f_bc_model
        self.f_conti_model = self.f_fe_mgii_model + self.f_fe_balmer_model + self.PL_poly_BC
        self.line_flux = self.flux - self.f_conti_model

        return

    def _L_conti(self, wave, pp):
        """Calculate continuum Luminoisity at 1350,3000,5100A"""
        conti_flux = pp[0] * (wave / 3000.0) ** pp[1] + f_poly_conti(wave, pp[5:8])

        L = np.array([])
        for LL in [1350., 3000., 5100., 9750., 12300., 19750]:
            if wave.max() > LL > wave.min():
                L_tmp = np.asarray([np.log10(LL * flux2L(conti_flux[np.where(abs(wave - LL) < 5.)].mean(), self.z))])
            else:
                L_tmp = np.array([-1.])
            L = np.concatenate([L, L_tmp])  # save log10(L1350,L3000,L5100)
        return L

    def _fe4570(self, wave, pp):
        fe_flux = Fe_flux_balmer(wave, pp[ppxfe:feopxmg])
        fe_4570 = np.where((wave > 4434) & (wave < 4684), fe_flux, 0)

        return np.trapz(fe_4570, dx=np.diff(self.wave)[0])

    def _f_conti_all(self, xval, pp):
        """
        Continuum components described by 8 + x + y parameters
         pp[0]: norm_factor for continuum f_lambda = (lambda/3000.0)^{-alpha}
         pp[1]: slope for the power-law continuum
         pp[2:5]: norm, Te and Tau_e for the Balmer continuum at <3646 A
         pp[5:8]: polynomial for the continuum
         pp[ppxfe:feopxmg]: x amount of parameters for Fe_optical. First 2 is FWHM and shift, the rest is scaling
         pp[feopxmg]: y amount of parameters for Fe_mg.
        """
        f_Fe_MgII = Fe_flux_mgii(xval, pp[feopxmg:])  # iron flux for MgII line region
        f_Fe_Balmer = Fe_flux_balmer(xval, pp[ppxfe:feopxmg])  # iron flux for balmer line region
        f_pl = pp[0] * (xval / 3000.0) ** pp[1]  # power-law continuum
        f_conti_BC = balmer_conti(xval, pp[2:5])  # Balmer continuum
        f_poly = f_poly_conti(xval, pp[5:8])  # polynormal conponent for reddened spectra

        if not self.PL:
            f_pl = f_pl * 0
        if not self.Fe_uv_op:
            f_Fe_Balmer = f_Fe_Balmer * 0

        if self.Fe_uv_op and not self.poly and not self.BC:
            yval = f_pl + f_Fe_MgII + f_Fe_Balmer
        elif self.Fe_uv_op and self.poly and not self.BC:
            yval = f_pl + f_Fe_MgII + f_Fe_Balmer + f_poly
        elif self.Fe_uv_op and not self.poly and self.BC:
            yval = f_pl + f_Fe_MgII + f_Fe_Balmer + f_conti_BC
        elif not self.Fe_uv_op and self.poly and not self.BC:
            yval = f_pl + f_poly
        elif not self.Fe_uv_op and not self.poly and not self.BC:
            yval = f_pl
        elif not self.Fe_uv_op and not self.poly and self.BC:
            yval = f_pl + f_conti_BC
        elif self.Fe_uv_op and self.poly and self.BC:
            yval = f_pl + f_Fe_MgII + f_Fe_Balmer + f_poly + f_conti_BC
        elif not self.Fe_uv_op and self.poly and self.BC:
            yval = f_pl + f_Fe_Balmer + f_poly + f_conti_BC
        else:
            raise RuntimeError('No this option for Fe_uv_op, poly and BC!')

        return yval

    def _residuals(self, pp, data):
        """Continual residual function used in kmpfit"""
        xval, yval, weight = data
        return (yval - self._f_conti_all(xval, pp)) / weight

    # ---------MC error for continuum parameters-------------------
    def _conti_mc(self, x, y, err, pp0, pp_limits, n_trails):
        """Calculate the continual parameters' Monte carlo errrors"""
        all_para = np.zeros(len(pp0) * n_trails).reshape(len(pp0), n_trails)
        all_L = np.zeros(6 * n_trails).reshape(6, n_trails)
        all_fe = np.zeros(n_trails)
        all_para_std = np.zeros(len(pp0))
        all_L_std = np.zeros(6)
        for tra in range(n_trails):
            flux = y + np.random.randn(len(y)) * err
            conti_fit = kmpfit.Fitter(residuals=self._residuals, data=(x, flux, err), maxiter=50)
            conti_fit.parinfo = pp_limits
            conti_fit.fit(params0=pp0)
            all_para[:, tra] = conti_fit.params
            all_L[:, tra] = np.asarray(self._L_conti(x, conti_fit.params))
            all_fe[tra] = np.asarray(self._fe4570(x, conti_fit.params))

        for st in range(len(pp0)):
            all_para_std[st] = all_para[st, :].std()
        for ll in range(6):
            all_L_std[ll] = all_L[ll, :].std()
        return all_para_std, all_L_std, all_fe.std()

    def _DoLineFit_sectiondefinitions(self) -> list:
        """Reads the Component_definition.py file to obtain the sections to line fit"""
        bool_fitted_section = (self.linelist['lambda'] > self.wave.min()) & (self.linelist['lambda'] < self.wave.max())
        fitted_section_index = np.where(bool_fitted_section, True, False)

        # sort section name with line wavelength
        uniq_linecomp, uniq_ind = np.unique(self.linelist['compname'][fitted_section_index], return_index=True)
        return uniq_linecomp[self.linelist['lambda'][fitted_section_index][uniq_ind].argsort()]

    def _DoLineFit_linefitresult(self, fit: kmpfit.Fitter, all_para_std: np.array, fwhm_lines: np.array,
                                 fwhm_std: np.array, indices: np.array, ) -> list:
        """Translate the results of line fitting into a machine-readable format"""
        comp_result_tmp = np.array([[self.linelist['compname'][indices][0]], [fit.status], [fit.chi2_min],
                                    [fit.rchi2_min], [fit.niter], [fit.dof]]).flatten()

        gauss_val = np.hstack((fwhm_lines.reshape(len(fwhm_lines), 1), np.split(fit.params, len(fwhm_lines)))).flatten()
        if self.MC and self.n_trails > 0:
            gauss_err = np.hstack((fwhm_std.reshape(len(fwhm_std), 1), np.split(all_para_std, len(fwhm_std)))).flatten()
            gauss_result = self.array_interlace(gauss_val, gauss_err)
        else:
            gauss_result = gauss_val

        return [comp_result_tmp, gauss_result]

    def _DoLineFit_lineresultnames(self, section: SectionParameters) -> np.array:
        """Create names for the result of line fitting for machine-readable format"""
        section_number = str(self.ncomp)
        section_property = ['_complex_name', '_line_status', '_line_min_chi2', '_line_red_chi2', '_niter', '_ndof']
        comp_result_name = np.array([section_number + xproperty for xproperty in section_property], dtype='U50')

        gauss_name = np.array([])
        gauss_property = ['_fwhm', '_scale', '_centerwave', '_sigma', '_skewness', '_gamma']
        # for each fitted line
        for n in range(section.n_line):
            line_name = self.linelist['linename'][section.line_indices][n]
            gauss_result_names = np.array([line_name + xproperty for xproperty in gauss_property], dtype='U50')

            if self.MC and self.n_trails > 0:
                gauss_error_names = np.array([xname + "_err" for xname in gauss_result_names], dtype='U50')
                gauss_name = np.append(gauss_name, self.array_interlace(gauss_result_names, gauss_error_names))
            else:
                gauss_name = np.append(gauss_name, gauss_result_names)

        return np.concatenate([comp_result_name, gauss_name.flatten()])

    def _DoLineFit_linefitting(self, section: SectionParameters) -> list:
        """Runs the line fitting for a section and the MC if called"""
        print(self.linelist['compname'][section.line_indices][0])
        all_para_std = np.asarray([])
        fwhm_std = np.asarray([])
        fwhm_lines = []

        # call kmpfit for lines
        line_fit = self._do_line_kmpfit(section)

        # calculate FWMH for each fitted components
        for xx in np.split(line_fit.params, len(self.linelist['compname'][section.line_indices])):
            fwhm_lines.append(self.line_property(xx).fwhm)
        fwhm_lines = np.asarray(fwhm_lines)

        # Perform MC for error calculation
        if self.MC and self.n_trails > 0:
            MCwave = np.log(self.wave[section.section_indices])
            MCflux = self.line_flux[section.section_indices]
            MCerr = self.err[section.section_indices]
            MCargs = (MCwave, MCflux, MCerr, self.line_fit.params, self.line_fit_par, self.n_trails,
                      section.line_indices)
            all_para_std, fwhm_std = self._line_mc(*MCargs)

        return [line_fit, all_para_std, fwhm_lines, fwhm_std]

    # line function-----------
    def _DoLineFit(self):
        """Runs the emission line modelling process"""
        # read line parameter
        linelist = fits.open(self.path + 'qsopar2.fits')[1].data
        self.linelist = linelist

        # define the sections for fitting
        self.uniq_linecomp_sort = self._DoLineFit_sectiondefinitions()
        self.ncomp = len(self.uniq_linecomp_sort)
        if self.ncomp == 0:
            print("No line to fit! Please set Line_fit to FALSE or enlarge wave_range!")
            return np.asarray([]), np.asarray([])

        # define arrays for result taking
        gauss_line = np.empty(self.ncomp, dtype=list)
        line_result = np.empty(self.ncomp, dtype=list)
        line_result_name = np.empty(self.ncomp, dtype=list)
        gauss_result = np.empty(self.ncomp, dtype=list)
        self.all_comp_range = np.empty(self.ncomp, dtype=list)

        # loop over each section and fit n lines simultaneously
        for xsection in range(self.ncomp):
            # component definitions for this section
            xsection_param = SectionParameters(self.wave, self.linelist, self.uniq_linecomp_sort[xsection])
            self.all_comp_range[xsection] = xsection_param.section_range

            # skip section if less than 10 pixels to fit
            if np.sum(xsection_param.section_indices) <= 10:
                print("less than 10 pixels in line fitting!")
                continue

            # Fit lines and obtain results
            section_fit_result = self._DoLineFit_linefitting(xsection_param)
            gauss_line[xsection] = manygauss(np.log(self.wave), section_fit_result[0].params)

            # Turn results into machine-readable format
            tmp = self._DoLineFit_linefitresult(*section_fit_result, indices=xsection_param.line_indices)
            line_result[xsection] = np.concatenate(tmp)
            gauss_result[xsection] = tmp[1]
            line_result_name[xsection] = self._DoLineFit_lineresultnames(xsection_param)

        self.gauss_line = gauss_line
        self.gauss_result = np.concatenate(gauss_result)
        self.line_result = np.concatenate(line_result)
        self.line_result_name = np.concatenate(line_result_name)
        return self.line_result, self.line_result_name

    def _do_line_kmpfit(self, section: SectionParameters):
        ind_line = section.line_indices
        ind_n = section.section_indices
        """The key function to do the line fit with kmpfit"""
        kmpargs = [np.log(self.wave[ind_n]), self.line_flux[ind_n], self.err[ind_n], ind_line]
        line_fit = kmpfit.Fitter(self._residuals_line, data=kmpargs)  # fitting wavelength in ln space
        line_fit_ini = np.array([])
        line_fit_par = np.array([])
        self.f_ratio = np.zeros_like(self.linelist['linename'])

        # initial parameter definition
        for n in range(section.n_line):
            # voffset initial is always at the line wavelength defined
            ini_voff = np.log(self.linelist['lambda'][ind_line][n])

            # -------------- Line Profile definitions -------------
            # Cases with linked lines. Gamma, sigma, skew and voffset are all tied
            if '*' in self.linelist['sigval'][ind_line][n]:
                ini_sig, ini_skw = 0.0018, 0
                sig_par, lam_par, gam_par, skw_par = [{'fixed': True}] * 4
                if 'g' in self.linelist['iniskw'][ind_line][n]:
                    ini_gam = 1e-3
                else:
                    ini_gam = 0
            # Cases without linked lines
            else:
                # profile definitions based on the profile input that looks like this "v[+],g[45],s[-10,10]"
                voffset = self.linelist['voff'][ind_line][n]
                ini_sig = self.linelist['sigval'][ind_line][n].strip('[').strip(']')
                nprofile = InitialProfileParameters(self.linelist['iniskw'][ind_line][n])
                profile_pp = nprofile.process_profile(ini_voff, voffset, ini_sig)
                ini_sig, sig_par = profile_pp["ini_sig"], profile_pp["sig_par"]
                ini_voff, lam_par = profile_pp["ini_voff"], profile_pp["lam_par"]
                ini_skw, skw_par = profile_pp["ini_skw"], profile_pp["skw_par"]
                ini_gam, gam_par = profile_pp["ini_gam"], profile_pp["gam_par"]

            # ----------------- Flux input parameters -----------------
            fluxstring = self.linelist['fvalue'][ind_line][n]
            if '[' in fluxstring:  # fixed
                ini_flx = fluxstring.strip('[').strip(']')
                if ',' in ini_flx:
                    sc_down, sc_up = np.sort([float(ini_flx.split(',')[0]), float(ini_flx.split(',')[1])])
                    ini_flx = np.mean([sc_down, sc_up])
                    sc_par = {'limits': (sc_down, sc_up)}
                else:
                    ini_flx = float(ini_flx)
                    sc_down, sc_up = np.sort([ini_flx * 1.03, ini_flx * 0.97])
                    sc_par = {'limits': (sc_down, sc_up)}

            elif '*' in fluxstring:  # linked
                ini_flx = 0.005
                if '<' in fluxstring or '>' in fluxstring:
                    sc_par = {'limits': (1e-7, None)}
                else:
                    sc_par = {'fixed': True}

            else:  # Free
                if float(fluxstring) >= 0:
                    ini_flx = float(fluxstring)
                    sc_par = {'limits': (1e-7, None)}
                else:
                    ini_flx = float(fluxstring)
                    sc_par = {'limits': (None, -1e-7)}

            line_fit_ini0 = [ini_flx, ini_voff, ini_sig, ini_skw, ini_gam]
            line_fit_ini = np.concatenate([line_fit_ini, line_fit_ini0])
            # set up parameter limits
            line_fit_par0 = [sc_par, lam_par, sig_par, skw_par, gam_par]
            line_fit_par = np.concatenate([line_fit_par, line_fit_par0])

        line_fit.parinfo = line_fit_par
        line_fit.fit(params0=line_fit_ini)
        line_fit.params = self.newpp

        self.line_fit = line_fit
        self.line_fit_ini = line_fit_ini
        self.line_fit_par = line_fit_par
        return line_fit

    # ---------MC error for emission line parameters-------------------
    def _line_mc(self, x, line_to_fit, err, pp0, pp_limits, n_trails, ind_line) -> Tuple[np.array, np.array]:
        """calculate the Monte Carlo error of line parameters"""
        all_pp_1comp = np.zeros(len(pp0) * n_trails).reshape(len(pp0), n_trails)
        all_pp_std = np.zeros(len(pp0))
        all_fwhm = []
        all_peak = []

        rms_line = line_to_fit - manygauss(x, pp0)
        noise = self.noise_calculator(rms_line)

        print(n_trails)
        for trail in range(n_trails):
            mc_start = timeit.default_timer()

            # randomly vary line to fit with noise
            ctm_pts = np.concatenate([np.random.choice(len(noise), int(len(noise) / 24))])
            ctm_noise = np.random.normal(rms_line[ctm_pts], noise[ctm_pts])
            ctm_poly = np.poly1d(np.polyfit(x[ctm_pts], ctm_noise, 3))
            line_to_fit += ctm_poly(x)

            # refit randomly varied line to fit with parameters of the fitted model
            line_fit = kmpfit.Fitter(residuals=self._residuals_line, data=(x, line_to_fit, err, ind_line), maxiter=50)
            line_fit.parinfo = pp_limits
            line_fit.fit(params0=pp0)
            line_fit.params = self.newpp
            all_pp_1comp[:, trail] = line_fit.params
            mc_end = timeit.default_timer()
            print('Fitting ', trail, ' mc in : ' + str(np.round(mc_end - mc_start)) + 's')

            # further line properties
            for xcomponent in np.split(line_fit.params, len(self.linelist['compname'][ind_line])):
                xproperty = self.line_property(xcomponent)
                all_fwhm.append(xproperty.fwhm)
                all_peak.append(xproperty.peak)

        # Calculate standard deviation for each property of each line
        fwhm_std = 1.4826 * mad(np.split(np.asarray(all_fwhm), n_trails), 0)
        peak_std = 1.4826 * mad(np.split(np.asarray(all_peak), n_trails), 0)

        for st in range(len(pp0)):
            all_pp_std[st] = 1.4826 * mad(all_pp_1comp[st, :])
            if (st - 1) % 5 == 0:
                all_pp_std[st] = peak_std[int(st / 5)]

        return all_pp_std, fwhm_std

    def _line_fwhm(self, xx: np.array, yy: np.array, centroid: float) -> float:
        """Creates a generator of the model shifted down in y-axis by half to find FWHM"""
        if np.max(yy) > 0:
            spline = interpolate.UnivariateSpline(xx, yy - np.max(yy) / 2, s=0)
        else:
            spline = interpolate.UnivariateSpline(xx, yy - np.min(yy) / 2, s=0)

        if len(spline.roots()) > 0:
            fwhm_left, fwhm_right = spline.roots().min(), spline.roots().max()
            return abs(fwhm_left - fwhm_right) / centroid * self.c
        else:
            return -999

    def _line_gaussian_pp(self, pp: np.array, n_gauss: int) -> Tuple[np.array, np.array, np.array]:
        """Extracts centroid, sigma, and skew parameters from the input array."""
        cen = np.zeros(n_gauss)  # centroid array of line
        sig = np.zeros(n_gauss)  # sigma array of line
        skw = np.zeros(n_gauss)  # skew array of lines

        for i in range(n_gauss):
            cen[i] = pp[5 * i + 1]
            sig[i] = pp[5 * i + 2]
            skw[i] = pp[5 * i + 3]

        return cen, sig, skw

    def _line_model_compute(self, pp: np.array) -> Tuple[np.array, np.array]:
        """Computes the wavelength and corresponding spectrum values."""
        disp = np.diff(self.wave)[0]
        left = self.wave.min()
        right = self.wave.max()
        xx = np.arange(left, right, disp)
        xlog = np.log(xx)
        yy = manygauss(xlog, pp)
        return xx, yy

    def _line_flux_ew(self, xx: np.array, yy: np.array) -> Tuple[float, float]:
        """Calculates the total broad line flux and equivalent width (EW)."""
        ff = interpolate.interp1d(self.wave, self.PL_poly_BC, bounds_error=False, fill_value=0)
        valid_indices = yy > 0.01 * np.amax(yy)
        area = np.trapz(yy[valid_indices], x=xx[valid_indices])
        ew = area / np.mean(ff(xx[valid_indices]))
        return area, ew

    def _line_sigma(self, pp: np.array, cen: np.array, xx: np.array) -> float:
        """Calculates the line sigma in normal wavelength."""
        disp = np.diff(self.wave)[0]
        xlog = np.log(xx)
        lambda0 = 0.
        lambda1 = 0.
        lambda2 = 0.

        for lm in range(int((xx.max() - xx.min()) / disp)):
            gauss_val = manygauss(xlog[lm], pp)
            lambda0 += gauss_val * disp * xx[lm]
            lambda1 += xx[lm] * gauss_val * disp * xx[lm]
            lambda2 += xx[lm] ** 2 * gauss_val * disp * xx[lm]

        sigma = np.sqrt(lambda2 / lambda0 - (lambda1 / lambda0) ** 2) / np.exp(np.mean(cen)) * self.c
        return sigma

    def _line_peak(self, n_gauss: int, pp: np.array, xx: np.array):
        peaks = np.empty((n_gauss,))
        areas = np.empty((n_gauss,))
        xlog = np.log(xx)

        for xline in range(n_gauss):
            xprofile = onegauss(xlog, np.split(pp, n_gauss)[xline])
            peaks[xline] = xx[np.argmax(abs(xprofile))]
            valid_indices = xprofile > 0.01 * np.amax(xprofile)
            areas[xline] = np.trapz(xprofile[valid_indices], x=xx[valid_indices])

        return np.average(peaks, weights=areas)

    # -----line properties calculation function--------
    def line_property(self, pp: np.array) -> LineProperties:
        """
        Calculate the further results for the broad component in emission lines, e.g., FWHM, sigma, peak, line flux
        """
        pp = pp.astype(float)
        n_gauss = int(len(pp) / 5)

        if n_gauss == 0:
            return LineProperties((0., 0., 0., 0., 0., 0.))

        cen, sig, skw = self._line_gaussian_pp(pp, n_gauss)

        skew = -999 if n_gauss > 1 else np.mean(skw)
        xx, yy = self._line_model_compute(pp)
        fwhm = self._line_fwhm(xx, yy, np.exp(np.mean(cen)))    # Calculate FWHM, -999 if error
        area, ew = self._line_flux_ew(xx, yy)
        sigma = self._line_sigma(pp, cen, xx)
        single_peak = xx[np.argmax(abs(yy))]
        peak = single_peak if n_gauss == 1 else self._line_peak(n_gauss, pp, xx)

        return LineProperties((fwhm, sigma, skew, ew, peak, area))

    def _lineflux_link(self, pp: np.array, line_indicies: np.array) -> None:
        """rescale the height of fitted line if height/flux is linked to another line"""
        for line_index, line_flux in enumerate(self.linelist['fvalue'][line_indicies]):
            if '*' in line_flux:
                input_flux = line_flux.split('*')
                link_index = np.where(self.linelist['linename'][line_indicies] == input_flux[0])[0]
                flux_target = self.linelist['lambda'][line_indicies][link_index]
                flux_now = self.linelist['lambda'][line_indicies][line_index]

                # If component is less than linked component but should be greater, set to boundary
                if '>' in input_flux[1][0] and pp[5 * line_index] < pp[5 * link_index] * float(input_flux[1][1:]):
                    pp[5 * line_index] = pp[5 * link_index] * float(input_flux[1][1:]) / flux_target * flux_now

                # If component is greater than linked component but should be less, set to boundary
                elif '<' in input_flux[1][0] and pp[5 * line_index] > pp[5 * link_index] * float(input_flux[1][1:]):
                    pp[5 * line_index] = pp[5 * link_index] * float(input_flux[1][1:]) / flux_target * flux_now

                # If component set exactly to be a multiplier of the linked component, scale it to multiplier
                elif input_flux[1][0] not in '<>':
                    pp[5 * line_index] = pp[5 * link_index] * float(input_flux[1]) / flux_target * flux_now

    def _lineprofile_link(self, pp: np.array, line_indicies: np.array) -> None:
        """reset the sigma, skew, velocity offset, and gamma of component to linked component"""
        for line_index, line_sigma in enumerate(self.linelist['sigval'][line_indicies]):
            if '*' in line_sigma:
                input_sigma = line_sigma.split('*')
                link_index = np.where(self.linelist['linename'][line_indicies] == input_sigma[0])[0]
                sigma_target = self.linelist['lambda'][line_indicies][link_index]
                sigma_now = self.linelist['lambda'][line_indicies][line_index]

                pp[5 * line_index + 1] = np.log(np.exp(pp[5 * link_index + 1]) / sigma_target * sigma_now)
                pp[5 * line_index + 2] = pp[5 * link_index + 2]
                pp[5 * line_index + 3] = pp[5 * link_index + 3]
                pp[5 * line_index + 4] = pp[5 * link_index + 4]

    def _residuals_line(self, pp: np.array, data: Tuple[np.array, np.array, np.array, np.array]) -> np.array:
        """The line residual function used in kmpfit"""
        xval, yval, weight, ind_line = data

        # Compute line linking prior to residual calculation
        self._lineflux_link(pp, ind_line)
        self._lineprofile_link(pp, ind_line)

        # restore parameters
        self.newpp = pp.copy()

        return (yval - manygauss(xval, pp)) / weight

    @staticmethod
    def _SaveResult(conti_result, conti_result_name, fe_op_result, fe_op_names, line_result,
                    line_result_name, save_fits_path, save_fits_name):
        """Save all data to fits"""
        all_result = np.concatenate([conti_result, line_result])
        all_result_name = np.concatenate([conti_result_name, line_result_name])

        t = Table(all_result, names=all_result_name)
        t.write(save_fits_path + save_fits_name + '.fits', format='fits', overwrite=True)

    def _plotSubPlot_Elines(self, arr_conti: np.array) -> None:
        """Plotting emission lines onto main spectrum figure"""
        if self.MC:
            for p in range(int(len(self.gauss_result) / 12)):
                if self.gauss_result[6 * (p - 1) + 12] < 1200.:
                    color = 'g'
                else:
                    color = 'r'
                gauss_line = manygauss(np.log(self.wave), self.gauss_result[::2][p * 6:(p + 1) * 6][1:])
                plt.plot(self.wave, gauss_line + arr_conti, color=color)
            total_param = np.delete(self.gauss_result[::2], np.arange(0, self.gauss_result[::2].size, 6))

        else:
            for p in range(int(len(self.gauss_result) / 6)):
                if self.gauss_result[6 * (p - 1) + 6] < 1200.:
                    color = 'g'
                else:
                    color = 'r'
                gauss_line = manygauss(np.log(self.wave), self.gauss_result[p * 6:(p + 1) * 6][1:])
                plt.plot(self.wave, gauss_line + arr_conti, color=color)
            total_param = np.delete(self.gauss_result, np.arange(0, self.gauss_result.size, 6))

        total_gauss_line = manygauss(np.log(self.wave), total_param)
        plt.plot(self.wave, total_gauss_line + arr_conti, 'b', label='line', lw=2)

    def _plotSubPlot_Main(self, linefit: bool) -> None:
        """Plotting the main spectrum figure"""
        plt.plot(self.wave_prereduced, self.flux_prereduced, 'k', label='data')

        # Residual strength at each wavelength bin
        if linefit:
            rms_line = self.line_flux - manygauss(np.log(self.wave), self.line_fit.params)
            noise_line = self.noise_calculator(rms_line)
            plt.plot(self.wave, np.random.normal(self.flux, noise_line, len(self.flux)), 'grey', alpha=0.5)

        # Host template if fitted
        if self.decomposed:
            plt.plot(self.wave, self.qso + self.host, 'pink', label='host+qso temp')
            plt.plot(self.wave, self.flux, 'grey', label='data-host')
            plt.plot(self.wave, self.host, 'purple', label='host')

        # Markers for the continuum windows
        if self.Fe_uv_op or self.poly or self.BC:
            conti_window_markers = np.repeat(self.flux_prereduced.max() * 1.05, len(self.wave[self.tmp_all]))
            plt.scatter(self.wave[self.tmp_all], conti_window_markers, color='grey', marker='o', alpha=0.5)

        # Fitted Emission line models
        if linefit:
            self._plotSubPlot_Elines(self.f_conti_model)

        plt.plot([0, 0], [0, 0], 'r', label='line br')
        plt.plot([0, 0], [0, 0], 'g', label='line na')

        # Continuum with Fe emission
        if self.Fe_uv_op:
            plt.plot(self.wave, self.f_conti_model, 'c', lw=2, label='FeII')

        # Balmer Continuum
        if self.BC:
            plt.plot(self.wave, self.PL_poly_BC, 'y', lw=2, label='BC')

        if self.CFT:
            plt.plot(self.wave, self.f_conti_model, color='orange', lw=2, label='conti')
        else:
            plt.plot(self.wave, self.f_pl_model + self.f_poly_model, color='orange', lw=2, label='conti')

        # Framing
        if not self.decomposed:
            self.host = self.flux_prereduced.min()
        plt.ylim(min(self.host.min(), self.flux.min()) * 0.9, self.flux_prereduced.max() * 1.1)
        plt.xlim(self.wave.min(), self.wave.max())

        if self.plot_legend:
            plt.legend(loc='best', frameon=False, fontsize=10)

        # plot line name--------
        if self.plot_line_name:
            line_cen = np.array([6564.60, 6549.85, 6585.27, 6718.29, 6732.66, 4862.68, 5008.24, 4687.02,
                                 4341.68, 3934.78, 3728.47, 3426.84, 2798.75, 1908.72, 1816.97,
                                 1750.26, 1718.55, 1549.06, 1640.42, 1402.06, 1396.76, 1335.30,
                                 1215.67])

            line_name = np.array(['', '', 'Ha+NII', '', 'SII6718,6732', 'Hb', '[OIII]',
                                  'HeII4687', 'Hr', 'CaII3934', 'OII3728', 'NeV3426', 'MgII', 'CIII]',
                                  'SiII1816', 'NIII1750', 'NIV1718', 'CIV', 'HeII1640', '',
                                  'SiIV+OIV', 'CII1335', 'Lya'])

            for ll in range(len(line_cen)):
                if self.wave.min() < line_cen[ll] < self.wave.max():
                    plt.plot([line_cen[ll], line_cen[ll]],
                             [min(self.host.min(), self.flux.min()), self.flux_prereduced.max() * 1.1], 'k:')
                    plt.text(line_cen[ll] + 10, 0.8 * self.flux_prereduced.max(), line_name[ll], rotation=90,
                             fontsize=15)

    def _plotSubPlot_Lines(self) -> None:
        """Plotting the emission line subplot"""
        for c in range(self.ncomp):
            # create subplot axes
            if self.ncomp == 4:
                axn = plt.subplot(2, 12, (12 + 3 * c + 1, 12 + 3 * c + 3))
            elif self.ncomp == 3:
                axn = plt.subplot(2, 12, (12 + 4 * c + 1, 12 + 4 * c + 4))
            elif self.ncomp == 2:
                axn = plt.subplot(2, 12, (12 + 6 * c + 1, 12 + 6 * c + 6))
            elif self.ncomp == 1:
                axn = plt.subplot(2, 12, (13, 24))
            else:
                print("Too many fitted sections to plot")
                return

            # plot lines
            plt.plot(self.wave, self.line_flux, 'k')
            self._plotSubPlot_Elines(np.zeros_like(self.wave))

            # subplot setup
            plt.xlim(self.all_comp_range[c])
            bool_linearea = (self.wave > self.all_comp_range[c][0]) & (self.wave < self.all_comp_range[c][1])
            f_max = self.line_flux[bool_linearea].max()
            f_min = self.line_flux[bool_linearea].min()
            plt.ylim(f_min * 0.9, f_max * 1.1)
            axn.set_xticks([self.all_comp_range[c][0], np.round(np.mean(self.all_comp_range[c]), -1),
                            self.all_comp_range[c][1]])
            plt.text(0.02, 0.9, self.uniq_linecomp_sort[c], fontsize=20, transform=axn.transAxes)

    def noise_calculator(self, rms_line: np.array) -> np.array:
        """Approximates fitting uncertainty with the rms of the residual at each bin"""
        noise_level = []
        for i in range(len(rms_line)):
            xx = np.std(rms_line[i - 20:i + 20])
            if np.isnan(xx):
                noise_level.append(0)
            else:
                noise_level.append(xx)
        return np.asarray(noise_level)

    def _PlotFig(self, linefit: bool, save_fig_path: str) -> None:
        """Plot the results"""

        # Figure Frame Setup
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        # plot the first subplot
        fig = plt.figure(figsize=(15, 8))
        plt.subplots_adjust(wspace=3., hspace=0.2)
        ax = plt.subplot(2, 6, (1, 6))

        if self.ra == -999. or self.dec == -999.:
            plt.title(f"{str(self.sdss_name)}    z = {str(self.z)}", fontsize=20)
        else:
            plt.title(f"ra,dec = ({str(self.ra)},{str(self.dec)})    {str(self.sdss_name)}    z = {str(self.z)}",
                      fontsize=20)

        # Creating the main spectrum figure
        self._plotSubPlot_Main(linefit)

        # Creating the emission line model subplots
        if linefit:
            self._plotSubPlot_Lines()

        if linefit:
            plt.text(0.4, -1.4, r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20, transform=ax.transAxes)
            plt.text(-0.1, -0.7, r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20,
                     transform=ax.transAxes, rotation=90)
        else:
            plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20)
            plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20)

        if self.save_fig:
            plt.savefig(save_fig_path + self.sdss_name + '.png')

    @staticmethod
    def Onegauss(xval: np.array, pp: np.array) -> np.array:
        """The single Gaussian model used to fit the emission lines
        Parameter: the scale factor, central wavelength in logwave, line FWHM in logwave
        """
        yval = pp[0] * skewed_voigt(xval, pp[1], pp[2], pp[4], pp[3])
        return yval

    @staticmethod
    def Manygauss(xval: np.array, pp: np.array) -> np.array:
        """The multi-Gaussian model used to fit the emission lines, it will call the onegauss function"""
        ngauss = int(pp.shape[0] / 5)
        if ngauss != 0:
            yval = 0.
            for i in range(ngauss):
                yval = yval + onegauss(xval, pp[i * 5:(i + 1) * 5])
            return yval

    def line_errordata_read(self, line_name: str) -> Tuple[list, list]:
        """Obtains id of line property within PyQSOFit line result list"""
        name_id = []
        err_id = []
        for i in range(0, len(self.line_result_name)):
            if line_name not in self.line_result_name[i]:
                continue
            elif 'err' in self.line_result_name[i]:
                err_id.append(i)
            elif 'err' not in self.line_result_name[i] and 'fwhm' not in self.line_result_name[i]:
                name_id.append(i)

        return name_id, err_id

    def line_error_calculate(self, name_id: list, err_id: list, property_values: LineProperties) -> LineProperties:
        """Perform basic error analysis calculations of the line"""
        scale = float(self.line_result[name_id[1]])

        if self.conti_result[7] != 0:
            conti_error = self.conti_result[8] / self.conti_result[7]
        else:
            conti_error = 0

        err_peak = float(self.line_result[err_id[2]])
        err_sig = property_values.sigma * float(err_id[0]) / property_values.fwhm
        err_fwhm = float(self.line_result[err_id[0]])
        err_skw = -999 if property_values.skew == -999 else float(self.line_result[err_id[4]])
        err_scale = float(self.line_result[err_id[1]])
        err_area = property_values.area * np.sqrt((err_scale / scale) ** 2 + (err_fwhm / property_values.fwhm) ** 2)
        err_ew = property_values.ew * np.sqrt(conti_error ** 2 + (err_area / property_values.area) ** 2)

        return LineProperties((err_fwhm, err_sig, err_skw, err_ew, err_peak, err_area))

    def line_result_output(self, line_name: str, to_print: bool = False) -> np.array:
        """Compile errors of specified line to output as an array or print neatly"""
        # name_id and err_id follows the name order as in properties list, repeating every 6 element
        properties = ['fwhm', 'sigma', 'skewness', 'EW', 'Peak', 'Area']
        name_id, err_id = self.line_errordata_read(line_name)
        if len(name_id) == 0:
            print(f"{line_name} not fitted")
            return np.asarray([LineProperties().list, LineProperties().list])

        # Calculate values and errors for each property
        property_values = self.line_property(self.line_result[name_id])

        if self.MC:
            property_errors = self.line_error_calculate(name_id, err_id, property_values)
        else:
            property_errors = LineProperties()

        if to_print:
            print('----------------------------------------------')
            print(line_name + ' data')
            for k in range(0, len(properties)):
                print_values = list(property_values.list.values())
                print_errors = list(property_errors.list.values())
                print((properties[k] + '               ')[:15] + ':', '\t', np.round(print_values[k], 5))
                if self.MC:
                    print((properties[k] + '_err           ')[:15] + ':', '\t', np.round(print_errors[k], 5))

        return np.asarray([property_values.list, property_errors.list])

    @staticmethod
    def array_interlace(array1: np.array, array2: np.array) -> np.array:
        """interlaces two 1D numpy arrays"""
        interlaced_array = np.empty(len(array1) + len(array2), dtype=array1.dtype)
        interlaced_array[0::2] = array1
        interlaced_array[1::2] = array2
        return interlaced_array

    @property
    def average_noise_area(self) -> float:
        try:
            tmp = self.gauss_line[0]
        except (IndexError, NameError):
            print("Unable to perform this operation yet. Spectrum has not been fit")
            return 0
        end_flux = np.abs(self.flux - self.f_conti_model - np.sum(self.gauss_line, 0))
        end_wave = self.wave
        area_of_residual = np.trapz(end_flux, end_wave)
        strong_noise = end_flux[end_flux > 1.4826 * 3 * mad(end_flux)]
        return area_of_residual / len(strong_noise)

    def full_result_print(self):
        a = 0
        for i, j in zip(self.line_result_name, self.line_result):
            print(a, i, j)
            a += 1

# Add CIV abs optional in continuum removal
# Change abs output for min = peak
# Add Fe emission scaling output functionality
