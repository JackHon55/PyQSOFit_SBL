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
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
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

        t2_flux = t_flux[end_clip:-end_clip][np.abs(t_flux-ctm)[end_clip:-end_clip] < thres * std]
        t2_wave = t_wave[end_clip:-end_clip][np.abs(t_flux-ctm)[end_clip:-end_clip] < thres * std]
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


class QSOFit:

    def __init__(self, lam, flux, err, z, ra=- 999., dec=-999., ebmv=None, plateid=None, mjd=None, fiberid=None,
                 path=None,
                 and_mask=None, or_mask=None):
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

    def Fit(self, name=None, nsmooth=1, and_or_mask=True, reject_badpix=True, deredden=True, wave_range=None,
            wave_mask=None, empty_mask=None, decomposition_host=True, BC03=False, Mi=None, npca_gal=5, npca_qso=20,
            Fe_uv_op=True, Fe_in_line=False, Conti_window=True, redshift=True,
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

        wave_range: 2-element array, optional
            trim input wavelength (lam) according to the min and max range of the input 2-element array, e.g.,
            np.array([4000.,6000.]) in Rest frame range. Default: None

        wave_mask: 2-D array
            mask some absorption lines or skylines in spectrum, e.g., np.array([[2200.,2300.]]),
            np.array([[5650.,5750.],[5850.,5900.]])

        decomposition_host:
            bool, optional If True, the host galaxy-QSO decomposition will be applied. If no more than 100 pixels are
            negative, the result will be applied. The Decomposition is based on the PCA method of Yip et al. 2004 (
            AJ, 128, 585) & (128, 2603). Now the template is only available for redshift < 1.16 in specific absolute
            magnitude bins. For galaxy, the global model has 10 PCA components and first 5 will enough to reproduce
            98.37% galaxy spectra. For QSO, the global model has 50, and the first 20 will reproduce 96.89% QSOs. If
            with i-band absolute magnitude, the Luminosity-redshift binned PCA components are available. Then the
            first 10 PCA in each bin is enough to reproduce most QSO spectrum. Default: False BC03: bool, optional if
            True, it will use Bruzual1 & Charlot 2003 host model to fit spectrum, high shift host will be low
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
        self.wave_range = wave_range
        self.wave_mask = wave_mask
        self.empty_mask = empty_mask
        self.BC03 = BC03
        self.Mi = Mi
        self.npca_gal = npca_gal
        self.npca_qso = npca_qso
        self.initial_guess = initial_guess
        self.Fe_uv_op = Fe_uv_op
        self.Fe_in_line = Fe_in_line
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

        # deal with pixels with error equal 0 or inifity
        ind_gooderror = np.where((self.err != 0) & ~np.isinf(self.err), True, False)
        err_good = self.err[ind_gooderror]
        flux_good = self.flux[ind_gooderror]
        lam_good = self.lam[ind_gooderror]

        if (self.and_mask is not None) & (self.or_mask is not None):
            and_mask_good = self.and_mask[ind_gooderror]
            or_mask_good = self.or_mask[ind_gooderror]
            del self.and_mask, self.or_mask
            self.and_mask = and_mask_good
            self.or_mask = or_mask_good
        del self.err, self.flux, self.lam
        self.err = err_good
        self.flux = flux_good
        self.lam = lam_good

        if nsmooth is not None:
            self.flux = smooth(self.flux, nsmooth)
            self.err = smooth(self.err, nsmooth)
        if and_or_mask and (self.and_mask is not None or self.or_mask is not None):
            self._MaskSdssAndOr(self.lam, self.flux, self.err, self.and_mask, self.or_mask)
        if reject_badpix:
            self._RejectBadPix(self.lam, self.flux, self.err)
        if wave_range is not None:
            self._WaveTrim(self.lam, self.flux, self.err, self.z)
        if wave_mask is not None:
            self._WaveMsk(self.lam, self.flux, self.err, self.z)
        if deredden and self.ebmv is not None:
            self._DeRedden(self.lam, self.flux, self.ebmv)
        if redshift:
            self._RestFrame(self.lam, self.flux, self.err, self.z)
        else:
            self.wave = self.lam
        self._CalculateSN(self.lam, self.flux)
        self._OrignialSpec(self.wave, self.flux, self.err)

        # do host decomposition --------------
        if self.z < 1.16 and decomposition_host:
            self._DoDecomposition(self.wave, self.path)
        else:
            self.decomposed = False
            if self.z > 1.16 and decomposition_host:
                print('redshift larger than 1.16 is not allowed for host decomposion!')

        # fit continuum --------------------
        if not self.Fe_uv_op and not self.poly and not self.BC:
            self.line_flux = self.flux
            self.conti_fit = np.zeros_like(self.wave)
            self.f_conti_model = np.zeros_like(self.wave)
            self.f_bc_model = np.zeros_like(self.wave)
            self.f_fe_uv_model = np.zeros_like(self.wave)
            self.f_fe_op_model = np.zeros_like(self.wave)
            self.f_pl_model = np.zeros_like(self.wave)
            self.f_poly_model = np.zeros_like(self.wave)
            self.PL_poly_BC = np.zeros_like(self.wave)
            self.tmp_all = np.zeros_like(self.wave)
            self.conti_result = np.zeros_like(self.wave)
        else:
            print('Fit Conti')
            self._DoContiFit(self.wave, self.flux, self.err, self.ra, self.dec, self.plateid, self.mjd, self.fiberid)
            print('Conti done')
        # fit line
        print('Fit Line')
        if linefit:
            self._DoLineFit(self.wave, self.line_flux, self.err, self.conti_fit)
        if Fe_in_line:
            self._DoFullfit()
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
            self._PlotFig(self.ra, self.dec, self.z, self.wave, self.flux, self.err, decomposition_host, linefit,
                          self.tmp_all, self.gauss_result, self.f_conti_model, self.all_comp_range,
                          self.uniq_linecomp_sort, self.line_flux, save_fig_path)

    def _MaskSdssAndOr(self, lam, flux, err, and_mask, or_mask):
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
        ind_and_or = np.where((and_mask == 0) & (or_mask == 0), True, False)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = lam[ind_and_or], flux[ind_and_or], err[ind_and_or]

    def _RejectBadPix(self, lam, flux, err):
        """
        Reject 10 most possiable outliers, input wavelength, flux and error. Return a different size wavelength,
        flux, and error.
        """
        # -----remove bad pixels, but not for high SN spectrum------------
        ind_bad = pyasl.pointDistGESD(flux, 10)
        wv = np.asarray([i for j, i in enumerate(lam) if j not in ind_bad[1]], dtype=np.float64)
        fx = np.asarray([i for j, i in enumerate(flux) if j not in ind_bad[1]], dtype=np.float64)
        er = np.asarray([i for j, i in enumerate(err) if j not in ind_bad[1]], dtype=np.float64)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = wv, fx, er
        return self.lam, self.flux, self.err

    def _WaveTrim(self, lam, flux, err, z):
        """
        Trim spectrum with a range in the rest frame.
        """
        # trim spectrum e.g., local fit emiision lines
        ind_trim = np.where((lam / (1. + z) > self.wave_range[0]) & (lam / (1. + z) < self.wave_range[1]), True, False)
        del self.lam, self.flux, self.err
        self.lam, self.flux, self.err = lam[ind_trim], flux[ind_trim], err[ind_trim]
        if len(self.lam) < 100:
            raise RuntimeError("No enough pixels in the input wave_range!")
        return self.lam, self.flux, self.err

    def _WaveMsk(self, lam, flux, err, z):
        """Block the bad pixels or absorption lines in spectrum."""

        for msk in range(len(self.wave_mask)):
            try:
                ind_not_mask = ~np.where(
                    (lam / (1. + z) > self.wave_mask[msk, 0]) & (lam / (1. + z) < self.wave_mask[msk, 1]), True, False)
            except IndexError:
                raise RuntimeError("Wave_mask should be 2D array, e.g., np.array([[2000,3000],[3100,4000]]).")

            del self.lam, self.flux, self.err
            self.lam, self.flux, self.err = lam[ind_not_mask], flux[ind_not_mask], err[ind_not_mask]
            lam, flux, err = self.lam, self.flux, self.err
        return self.lam, self.flux, self.err

    def _DeRedden(self, lam, flux, ebmv):
        """Correct the Galatical extinction"""
        # m = sfdmap.SFDMap(dustmap_path)
        flux_unred = remove(fitzpatrick99(lam, a_v=ebmv * 3.1, r_v=3.1), flux)
        del self.flux
        self.flux = flux_unred
        return self.flux

    def _RestFrame(self, lam, flux, err, z):
        """Move wavelenth and flux to rest frame"""
        self.wave, self.flux = blueshifting([lam, flux], z)
        self.err = err
        return self.wave, self.flux, self.err

    def _OrignialSpec(self, wave, flux, err):
        """save the orignial spectrum before host galaxy decompsition"""
        self.wave_prereduced = wave
        self.flux_prereduced = flux
        self.err_prereduced = err

    def _CalculateSN(self, wave, flux):
        """calculate the spectral SN ratio for 1350, 3000, 5100A, return the mean value of Three spots"""
        if (wave.min() < 1350. < wave.max()) or (wave.min() < 3000. < wave.max()) or (wave.min() < 5100. < wave.max()):
            ind5100 = np.where((wave > 5080.) & (wave < 5130.), True, False)
            ind3000 = np.where((wave > 3000.) & (wave < 3050.), True, False)
            ind1350 = np.where((wave > 1325.) & (wave < 1375.), True, False)

            tmp_SN = np.array([flux[ind5100].mean() / flux[ind5100].std(), flux[ind3000].mean() / flux[ind3000].std(),
                               flux[ind1350].mean() / flux[ind1350].std()])
            tmp_SN = tmp_SN[~np.isnan(tmp_SN)]
            self.SN_ratio_conti = tmp_SN.mean()
        else:
            self.SN_ratio_conti = -1.

        return self.SN_ratio_conti

    def _DoDecomposition(self, wave, path):
        """Decompose the host galaxy from QSO"""
        datacube = self._HostDecompose(self.wave, self.flux, self.err, self.z, self.Mi, self.npca_gal, self.npca_qso,
                                       path)

        # for some negtive host templete, we do not do the decomposition
        if np.sum(np.where(datacube[3, :] < 0., True, False)) > 100:
            self.host = np.zeros(len(wave))
            self.decomposed = False
            print('Get negtive host galaxy flux larger than 100 pixels, decomposition is not applied!')
        else:
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

            f = interpolate.interp1d(self.wave[~line_mask], datacube[3, :][~line_mask], bounds_error=False,
                                     fill_value=0)
            masked_host = f(self.wave)

            self.flux = datacube[1, :] - masked_host  # ** change back to masked_host for BEL
            self.err = datacube[2, :]
            self.host = datacube[3, :]
            self.qso = datacube[4, :]
            self.host_data = datacube[1, :] - self.qso

        return self.wave, self.flux, self.err

    def _HostDecompose(self, wave, flux, err, z, Mi, npca_gal, npca_qso, path):
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
            galaxy = fits.open(path + 'pca/Yip_pca_templates/gal_eigenspec_Yip2004.fits')
            gal = galaxy[1].data
            wave_gal = gal['wave'].flatten()
            flux_gal = gal['pca'].reshape(gal['pca'].shape[1], gal['pca'].shape[2])
        if self.BC03:
            cc = 0
            flux03 = np.array([])
            for i in glob.glob(path + '/bc03/*.gz'):
                cc = cc + 1
                gal_temp = np.genfromtxt(i)
                wave_gal = gal_temp[:, 0]
                flux03 = np.concatenate((flux03, gal_temp[:, 1]))
            flux_gal = np.array(flux03).reshape(cc, -1)

        if Mi is None:
            quasar = fits.open(path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_global.fits')
        else:
            if -24 < Mi <= -22 and 0.08 <= z < 0.53:
                quasar = fits.open(path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_CZBIN1.fits')
            elif -26 < Mi <= -24 and 0.08 <= z < 0.53:
                quasar = fits.open(path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_DZBIN1.fits')
            elif -24 < Mi <= -22 and 0.53 <= z < 1.16:
                quasar = fits.open(path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_BZBIN2.fits')
            elif -26 < Mi <= -24 and 0.53 <= z < 1.16:
                quasar = fits.open(path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_CZBIN2.fits')
            elif -28 < Mi <= -26 and 0.53 <= z < 1.16:
                quasar = fits.open(path + 'pca/Yip_pca_templates/qso_eigenspec_Yip2004_DZBIN2.fits')
            else:
                raise RuntimeError('Host galaxy template is not available for this redshift and Magnitude!')

        qso = quasar[1].data
        wave_qso = qso['wave'].flatten()
        flux_qso = qso['pca'].reshape(qso['pca'].shape[1], qso['pca'].shape[2])

        # get the shortest wavelength range
        wave_min = max(wave.min(), wave_gal.min(), wave_qso.min())
        wave_max = min(wave.max(), wave_gal.max(), wave_qso.max())

        ind_data = np.where((wave > wave_min) & (wave < wave_max), True, False)
        ind_gal = np.where((wave_gal > wave_min - 1.) & (wave_gal < wave_max + 1.), True, False)
        ind_qso = np.where((wave_qso > wave_min - 1.) & (wave_qso < wave_max + 1.), True, False)

        flux_gal_new = np.zeros(flux_gal.shape[0] * flux[ind_data].shape[0]).reshape(flux_gal.shape[0],
                                                                                     flux[ind_data].shape[0])
        flux_qso_new = np.zeros(flux_qso.shape[0] * flux[ind_data].shape[0]).reshape(flux_qso.shape[0],
                                                                                     flux[ind_data].shape[0])
        for i in range(flux_gal.shape[0]):
            fgal = interpolate.interp1d(wave_gal[ind_gal], flux_gal[i, ind_gal], bounds_error=False, fill_value=0)
            flux_gal_new[i, :] = fgal(wave[ind_data])
        for i in range(flux_qso.shape[0]):
            fqso = interpolate.interp1d(wave_qso[ind_qso], flux_qso[0, ind_qso], bounds_error=False, fill_value=0)
            flux_qso_new[i, :] = fqso(wave[ind_data])

        wave_new = wave[ind_data]
        flux_new = flux[ind_data]
        err_new = err[ind_data]

        flux_temp = np.vstack((flux_gal_new[0:npca_gal, :], flux_qso_new[0:npca_qso, :]))
        res = np.linalg.lstsq(flux_temp.T, flux_new)[0]

        host_flux = np.dot(res[0:npca_gal], flux_temp[0:npca_gal])
        qso_flux = np.dot(res[npca_gal:], flux_temp[npca_gal:])

        data_cube = np.vstack((wave_new, flux_new, err_new, host_flux, qso_flux))

        ind_f4200 = np.where((wave_new > 4160.) & (wave_new < 4210.), True, False)
        frac_host_4200 = np.sum(host_flux[ind_f4200]) / np.sum(flux_new[ind_f4200])
        ind_f5100 = np.where((wave_new > 5080.) & (wave_new < 5130.), True, False)
        frac_host_5100 = np.sum(host_flux[ind_f5100]) / np.sum(flux_new[ind_f5100])

        return data_cube  # ,frac_host_4200,frac_host_5100

    def _DoContiFit(self, wave, flux, err, ra, dec, plateid, mjd, fiberid):
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

        tmp_all = np.array([np.repeat(False, len(wave))]).flatten()
        for jj in range(len(window_all)):
            tmp = np.where((wave > window_all[jj, 0]) & (wave < window_all[jj, 1]), True, False)
            tmp_all = np.any([tmp_all, tmp], axis=0)

        if wave[tmp_all].shape[0] < 10:
            print('Continuum fitting pixel < 10.  ')

        # set initial paramiters for continuum
        if self.initial_guess is not None:
            pp0 = self.initial_guess
        else:
            # ini val
            pp0 = np.array([0.001, 0,           # P-law norm, slope
                            0., 15000., 0.5,     # BC norm, Te, Tau
                            0., 0., 0.])        # Polynomial
            global ppxfe
            global feopxmg
            ppxfe = len(pp0)
            # pp_fe_op = np.concatenate([[3000., 0.], 0.2 * np.ones_like(fe_op_name[1:], dtype=float)])
            pp_fe_op = [1000., 0, 0.0]
            feopxmg = len(pp_fe_op) + ppxfe
            pp_mg_op = np.asarray([0.01, 3000., 0.])
            pp0 = np.concatenate([pp0, pp_fe_op, pp_mg_op])

        conti_fit = kmpfit.Fitter(residuals=self._residuals, data=(wave[tmp_all], flux[tmp_all], err[tmp_all]))
        tmp_parinfo = [{'limits': (0., 100)}, {'limits': (-5., 0)},
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
                tmp_conti = conti_fit.params[0] * (wave[tmp_all] / 3000.0) ** conti_fit.params[1] + f_poly_conti(
                    wave[tmp_all], conti_fit.params[5:8])
            else:
                tmp_conti = conti_fit.params[0] * (wave[tmp_all] / 3000.0) ** conti_fit.params[1]
            ind_noBAL = ~np.where((flux[tmp_all] < tmp_conti - 3. * err[tmp_all]) & (wave[tmp_all] < 3500.), True,
                                  False)
            f = kmpfit.Fitter(residuals=self._residuals, data=(
                wave[tmp_all][ind_noBAL], smooth(flux[tmp_all][ind_noBAL], 10), err[tmp_all][ind_noBAL]))
            conti_fit.parinfo = tmp_parinfo
            conti_fit.fit(params0=pp0)

        # calculate continuum luminoisty
        L = self._L_conti(wave, conti_fit.params)
        f_4570 = self._fe4570(wave, conti_fit.params)

        # calculate MC err
        if self.MC_conti and self.n_trails > 0:
            conti_para_std, all_L_std, all_fe_std = self._conti_mc(self.wave[tmp_all], self.flux[tmp_all], self.err[tmp_all], pp0,
                                                       conti_fit.parinfo, self.n_trails)

        res_name = np.array(['ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift', 'SN_ratio_conti', 'PL_norm',
                             'PL_slope', 'BC_norm', 'BC_Te', 'BC_Tau', 'POLY_a', 'POLY_b', 'POLY_c',
                             'L1350', 'L3000', 'L5100', 'L0.9', 'L1.2', 'L2.0', 'f4570'])

        fe_name = np.array(['Fe_op_FWHM', 'Fe_op_shift', *fe_op_name[1:], 'Fe_uv_scale', 'Fe_uv_FWHM', 'Fe_uv_shift'])
        # get conti result -----------------------------
        if not self.MC_conti:
            self.conti_result = np.array(
                [ra, dec, self.plateid, self.mjd, self.fiberid, self.z, self.SN_ratio_conti,
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
                [ra, dec, plateid, mjd, fiberid, self.z, self.SN_ratio_conti, *res_err, *L_err, f_4570, all_fe_std])
            self.conti_result_name = np.array(
                ['ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift', 'SN_ratio_conti', *res_err_name])

            self.fe_result = np.concatenate([[i, j] for i, j in zip(conti_fit.params[ppxfe:],
                                                                       conti_para_std[ppxfe:])])
            self.fe_result_name = np.concatenate([[i, i + '_err'] for i in fe_name])

        self.conti_fit = conti_fit
        self.tmp_all = tmp_all

        # save different models--------------------
        f_fe_mgii_model = Fe_flux_mgii(wave, conti_fit.params[feopxmg:])
        f_fe_balmer_model = Fe_flux_balmer(wave, conti_fit.params[ppxfe:feopxmg])
        f_pl_model = conti_fit.params[0] * (wave / 3000.0) ** conti_fit.params[1]
        f_bc_model = balmer_conti(wave, conti_fit.params[2:5])
        f_poly_model = f_poly_conti(wave, conti_fit.params[5:8])
        f_conti_model = f_pl_model + f_fe_mgii_model + f_fe_balmer_model + f_poly_model + f_bc_model
        line_flux = flux - f_conti_model

        self.f_conti_model = f_conti_model
        self.f_bc_model = f_bc_model
        self.f_fe_uv_model = f_fe_mgii_model
        self.f_fe_op_model = f_fe_balmer_model
        self.f_pl_model = f_pl_model
        self.f_poly_model = f_poly_model
        self.line_flux = line_flux
        self.PL_poly_BC = f_pl_model + f_poly_model + f_bc_model

        return self.conti_result, self.conti_result_name, self.fe_result, self.fe_result_name

    def _L_conti(self, wave, pp):
        """Calculate continuum Luminoisity at 1350,3000,5100A"""
        conti_flux = pp[0] * (wave / 3000.0) ** pp[1] + f_poly_conti(wave, pp[5:8])

        L = np.array([])
        for LL in [1350., 3000., 5100., 9750., 12300., 19750]:
            if wave.max() > LL > wave.min():
                L_tmp = np.asarray([np.log10(
                    LL * flux2L(conti_flux[np.where(abs(wave - LL) < 5.)].mean(), self.z))])
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

        if self.CFT:
            return np.zeros_like(xval)

        return yval

    def _residuals(self, pp, data):
        """Continual residual function used in kmpfit"""
        xval, yval, weight = data
        if self.CFT:
            yval = yval - continuum_fitting(xval, yval + 100, self.CFT_smooth) + 100
            return yval / weight
        return (yval - self._f_conti_all(xval, pp))/weight

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

    # line function-----------
    def _DoLineFit(self, wave, line_flux, err, f):
        """Fit the emission lines with Gaussian profile """

        # remove abosorbtion line in emission line region
        # remove the pixels below continuum
        '''
        ind_neg_line = ~np.where((((wave > 2700.) & (wave < 2900.)) | ((wave > 1700.) & (wave < 1970.)) | \
                                  ((wave > 1500.) & (wave < 1700.)) | ((wave > 1290.) & (wave < 1450.)) | \
                                  ((wave > 1150.) & (wave < 1290.))) & (line_flux < -err), True, False)
        '''

        # read line parameter
        linepara = fits.open(self.path + 'qsopar2.fits')
        linelist = linepara[1].data
        self.linelist = linelist

        ind_kind_line = np.where((linelist['lambda'] > wave.min()) & (linelist['lambda'] < wave.max()), True, False)
        if ind_kind_line.any():
            # sort complex name with line wavelength
            uniq_linecomp, uniq_ind = np.unique(linelist['compname'][ind_kind_line], return_index=True)
            uniq_linecomp_sort = uniq_linecomp[linelist['lambda'][ind_kind_line][uniq_ind].argsort()]
            ncomp = len(uniq_linecomp_sort)
            compname = linelist['compname']
            allcompcenter = np.sort(linelist['lambda'][ind_kind_line][uniq_ind])

            # loop over each complex and fit n lines simutaneously

            comp_result = np.array([])
            comp_result_name = np.array([])
            gauss_result = np.array([])
            gauss_result_name = np.array([])
            gauss_line = np.array([])
            all_comp_range = np.array([])
            fur_result = np.array([])
            fur_result_name = np.array([])

            for ii in range(ncomp):
                compcenter = allcompcenter[ii]
                ind_line = np.where(linelist['compname'] == uniq_linecomp_sort[ii], True, False)  # get line index
                nline_fit = np.sum(ind_line)  # n line in one complex
                linelist_fit = linelist[ind_line]
                ngauss_fit = np.asarray(linelist_fit['ngauss'], dtype=int)  # n gauss in each line

                # for iitmp in range(nline_fit):   # line fit together
                comp_range = [linelist_fit[0]['minwav'], linelist_fit[0]['maxwav']]  # read complex range from table
                all_comp_range = np.concatenate([all_comp_range, comp_range])

                ind_n = np.where((wave > comp_range[0]) & (wave < comp_range[1]), True, False)

                if np.sum(ind_n) > 10:
                    print(linelist['compname'][ind_line][0])
                    # call kmpfit for lines
                    line_fit = self._do_line_kmpfit(linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit)
                    fwhm_lines, sigma_lines = [], []
                    for xx in np.split(line_fit.params, len(linelist['compname'][ind_line])):
                        x1 = self.line_prop(np.exp(xx[1]), xx)[0]
                        fwhm_lines.append(x1)
                    fwhm_lines = np.asarray(fwhm_lines)
                    # calculate MC err
                    if self.MC and self.n_trails > 0:
                        # self.noise_level = np.asarray([np.std(rms_line[i - 20:i + 20]) for i in range(len(rms_line))])
                        all_para_std, fwhm_std, sigma_std = self._line_mc(np.log(wave[ind_n]), line_flux[ind_n],
                                                                                 err[ind_n], self.line_fit.params,
                                                                                 self.line_fit_par, self.n_trails, ind_line)

                    # ----------------------get line fitting results----------------------
                    # complex parameters
                    comp_result_tmp = np.array(
                        [[linelist['compname'][ind_line][0]], [line_fit.status], [line_fit.chi2_min],
                         [line_fit.rchi2_min], [line_fit.niter], [line_fit.dof]]).flatten()
                    comp_result_name_tmp = np.array(
                        [str(ii + 1) + '_complex_name', str(ii + 1) + '_line_status', str(ii + 1) + '_line_min_chi2',
                         str(ii + 1) + '_line_red_chi2', str(ii + 1) + '_niter', str(ii + 1) + '_ndof'])
                    comp_result = np.concatenate([comp_result, comp_result_tmp])
                    comp_result_name = np.concatenate([comp_result_name, comp_result_name_tmp])

                    # gauss result -------------

                    gauss_tmp = np.array([])
                    gauss_name_tmp = np.array([])

                    gauss_line = np.concatenate([gauss_line, manygauss(np.log(wave), line_fit.params)])
                    for gg in range(len(line_fit.params)):
                        if gg % 5 == 0:
                            gauss_tmp = np.concatenate([gauss_tmp, np.array([fwhm_lines[int(gg / 5)]])])
                            if self.MC:
                                gauss_tmp = np.concatenate([gauss_tmp, np.array([fwhm_std[int(gg / 5)]])])
                        gauss_tmp = np.concatenate([gauss_tmp, np.array([line_fit.params[gg]])])
                        if self.MC:
                            gauss_tmp = np.concatenate([gauss_tmp, np.array([all_para_std[gg]])])
                    gauss_result = np.concatenate([gauss_result, gauss_tmp])

                    # gauss result name -----------------
                    for n in range(nline_fit):
                        for nn in range(int(ngauss_fit[n])):
                            line_name = linelist['linename'][ind_line][n] + '_' + str(nn + 1)
                            if self.MC and self.n_trails > 0:
                                gauss_name_tmp_tmp = [line_name + '_fwhm', line_name + '_fwhm_err',
                                                      line_name + '_scale', line_name + '_scale_err',
                                                      line_name + '_centerwave',
                                                      line_name + '_centerwave_err', line_name + '_sigma',
                                                      line_name + '_sigma_err', line_name + '_skewness',
                                                      line_name + '_skewness_err', line_name + '_gamma',
                                                      line_name + '_gamma_err', ]
                            else:
                                gauss_name_tmp_tmp = [line_name + '_fwhm', line_name + '_scale',
                                                      line_name + '_centerwave', line_name + '_sigma',
                                                      line_name + '_skewness', line_name + '_gamma', ]
                            gauss_name_tmp = np.concatenate([gauss_name_tmp, gauss_name_tmp_tmp])
                    gauss_result_name = np.concatenate([gauss_result_name, gauss_name_tmp])

                else:
                    print("less than 10 pixels in line fitting!")

            line_result = np.concatenate([comp_result, gauss_result])
            line_result_name = np.concatenate([comp_result_name, gauss_result_name])

        else:
            line_result = np.array([])
            line_result_name = np.array([])
            print("No line to fit! Pleasse set Line_fit to FALSE or enlarge wave_range!")

        self.gauss_result = gauss_result
        self.line_result = line_result
        self.line_result_name = line_result_name
        self.ncomp = ncomp
        self.line_flux = line_flux
        self.gauss_line = gauss_line
        self.all_comp_range = all_comp_range
        self.uniq_linecomp_sort = uniq_linecomp_sort
        return self.line_result, self.line_result_name

    def _do_line_kmpfit(self, linelist, line_flux, ind_line, ind_n, nline_fit, ngauss_fit):
        """The key function to do the line fit with kmpfit"""
        line_fit = kmpfit.Fitter(self._residuals_line, data=(
            np.log(self.wave[ind_n]), line_flux[ind_n], self.err[ind_n], ind_line))  # fitting wavelength in ln space
        line_fit_ini = np.array([])
        line_fit_par = np.array([])
        self.f_ratio = np.zeros_like(self.linelist['linename'])
        for n in range(nline_fit):
            for nn in range(ngauss_fit[n]):
                # voffset initial is always at the line wavelength defined
                ini_voff = np.log(linelist['lambda'][ind_line][n])
                # Cases with tied lines. Gamma, sigma, skew and voffset are all tied
                if '*' in linelist['sigval'][ind_line][n]:
                    ini_sig, ini_skw = 0.0018, 0
                    sig_par, lam_par, gam_par, skw_par = [{'fixed': True}] * 4
                    if 'g' in linelist['iniskw'][ind_line][n]:
                        ini_gam = 1e-3
                    else:
                        ini_gam = 0
                # Cases without tied lines
                else:
                    # voffset is as defined
                    if linelist['voff'][ind_line][n] != 0:
                        lambda_low = ini_voff - linelist['voff'][ind_line][n]
                        lambda_up = ini_voff + linelist['voff'][ind_line][n]
                        if linelist['iniskw'][ind_line][n][0] in '+':
                            lam_par = {'limits': (ini_voff - linelist['voff'][ind_line][n]*0.01, lambda_up)}
                            skw_data = linelist['iniskw'][ind_line][n][1:]
                        elif linelist['iniskw'][ind_line][n][0] in '-':
                            lam_par = {'limits': (lambda_low, ini_voff + linelist['voff'][ind_line][n]*0.01)}
                            skw_data = linelist['iniskw'][ind_line][n][1:]
                        elif linelist['iniskw'][ind_line][n][0] in '.':
                            ini_voff = ini_voff + linelist['voff'][ind_line][n]
                            lam_par = {'fixed': True}
                            skw_data = linelist['iniskw'][ind_line][n][1:]
                        else:
                            lam_par = {'limits': (lambda_low, lambda_up)}
                            skw_data = linelist['iniskw'][ind_line][n]
                    else:
                        lam_par = {'fixed': True}
                        skw_data = linelist['iniskw'][ind_line][n]

                    # Sigma is simply the range defined
                    ini_sig = linelist['sigval'][ind_line][n].strip('[').strip(']')
                    if len(ini_sig.split(',')) == 1:
                        sig_par = {'fixed': True}
                        ini_sig = float(ini_sig.split(',')[0])
                    else:
                        sig_down, sig_up = np.sort([float(ini_sig.split(',')[0]), float(ini_sig.split(',')[1])])
                        ini_sig = np.mean([sig_down, sig_up])
                        sig_par = {'limits': (sig_down, sig_up)}

                    # Gamma is either on or off
                    if 'g' in skw_data:
                        if 'y' in skw_data:
                            ini_gam, skw_data = skw_data.split('y')
                            ini_gam = float(ini_gam[1:])
                            gam_par = {'fixed': True}
                        else:
                            ini_gam = 1e-2
                            gam_par = {'limits': (0, 1)}
                            skw_data = skw_data[1:]
                    else:
                        ini_gam = 0
                        gam_par = {'fixed': True}
                        skw_data = skw_data
                    # Skew depends on a range, or single value
                    if '[' in skw_data:
                        ini_skw = skw_data.strip('[').strip(']')
                        if ',' in ini_skw:
                            skw_down, skw_up = np.sort([float(ini_skw.split(',')[0]), float(ini_skw.split(',')[1])])
                            ini_skw = np.mean([skw_down, skw_up])
                            skw_par = {'limits': (skw_down, skw_up)}
                        else:
                            ini_skw = float(ini_skw)
                            if ini_skw != 0:
                                skw_down, skw_up = np.sort([ini_skw * 1.001, ini_skw * 0.999])
                                skw_par = {'limits': (skw_down, skw_up)}
                            else:
                                skw_par = {'fixed': True}
                    else:
                        ini_skw = float(skw_data)
                        skw_par = {}

                if '[' in linelist['fvalue'][ind_line][n]:
                    ini_flx = linelist['fvalue'][ind_line][n].strip('[').strip(']')
                    if ',' in ini_flx:
                        sc_down, sc_up = np.sort([float(ini_flx.split(',')[0]), float(ini_flx.split(',')[1])])
                        ini_flx = np.mean([sc_down, sc_up])
                        sc_par = {'limits': (sc_down, sc_up)}
                    else:
                        ini_flx = float(ini_flx)
                        sc_down, sc_up = np.sort([ini_flx * 1.03, ini_flx * 0.97])
                        sc_par = {'limits': (sc_down, sc_up)}
                elif '*' in linelist['fvalue'][ind_line][n]:
                    ini_flx = 0.005
                    if '<' in linelist['fvalue'][ind_line][n] or '>' in linelist['fvalue'][ind_line][n]:
                        sc_par = {'limits': (1e-7, None)}
                    else:
                        sc_par = {'fixed': True}
                else:
                    if float(linelist['fvalue'][ind_line][n]) >= 0:
                        ini_flx = float(linelist['fvalue'][ind_line][n])
                        sc_par = {'limits': (1e-7, None)}
                    else:
                        ini_flx = float(linelist['fvalue'][ind_line][n])
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
    def _line_mc(self, x, y, err, pp0, pp_limits, n_trails, ind_line):
        """calculate the Monte Carlo errror of line parameters"""
        all_para_1comp = np.zeros(len(pp0) * n_trails).reshape(len(pp0), n_trails)
        all_para_std = np.zeros(len(pp0))
        all_fwhm = []
        all_sigma = []
        all_peak = []

        rms_line = y - manygauss(x, pp0)
        noise = []
        for i in range(len(rms_line)):
            xx = np.std(rms_line[i - 20:i + 20])
            if np.isnan(xx):
                noise.append(0)
            else:
                noise.append(xx)
        noise = np.asarray(noise)
        # print(int(len(noise)/24))
        # plt.figure()

        print(n_trails)
        for tra in range(n_trails):
            mc_s = timeit.default_timer()
            ctm_pts = np.concatenate([np.random.choice(len(noise), int(len(noise)/24))])
            ctm_noise = np.random.normal(rms_line[ctm_pts], noise[ctm_pts])
            ctm_poly = np.poly1d(np.polyfit(x[ctm_pts], ctm_noise, 3))
            flux = y + ctm_poly(x)
            # plt.plot(x, flux, alpha=0.2)
            line_fit = kmpfit.Fitter(residuals=self._residuals_line, data=(x, flux, err, ind_line), maxiter=50)
            line_fit.parinfo = pp_limits
            line_fit.fit(params0=pp0)
            line_fit.params = self.newpp
            all_para_1comp[:, tra] = line_fit.params
            mc_e = timeit.default_timer()
            # plt.plot(np.exp(x), manygauss(x, line_fit.params), alpha=0.2)
            print('Fitting ', tra, ' mc in : ' + str(np.round(mc_e - mc_s)) + 's')
            # further line properties
            for xx in np.split(line_fit.params, len(self.linelist['compname'][ind_line])):
                x1 = self.line_prop(np.exp(xx[1]), xx)
                all_fwhm.append(x1[0])
                all_sigma.append(x1[1])
                all_peak.append(x1[4])

        all_fwhm = np.asarray(all_fwhm)
        all_sigma = np.asarray(all_sigma)
        all_peak = np.asarray(all_peak)
        print(all_fwhm)
        fwhm_std = 1.4826*mad(np.split(all_fwhm, n_trails), 0)
        sigma_std = 1.4826*mad(np.split(all_sigma, n_trails), 0)
        peak_std = 1.4826*mad(np.split(all_peak, n_trails), 0)
        for st in range(len(pp0)):
            all_para_std[st] = 1.4826*mad(all_para_1comp[st, :])
            if (st - 1) % 5 == 0:
                all_para_std[st] = peak_std[int(st/5)]
        return all_para_std, fwhm_std, sigma_std

    # -----line properties calculation function--------
    def line_prop(self, compcenter, pp):
        """
        Calculate the further results for the broad component in emission lines, e.g., FWHM, sigma, peak, line flux
        The compcenter is the theortical vacuum wavelength for the broad compoenet.
        """
        pp = pp.astype(float)

        c = 299792.458  # km/s
        n_gauss = int(len(pp) / 5)
        if n_gauss == 0:
            fwhm, sigma, skew, ew, peak, area = 0., 0., 0., 0., 0., 0.
        else:
            cen = np.zeros(n_gauss)
            sig = np.zeros(n_gauss)
            skw = np.zeros(n_gauss)

            for i in range(n_gauss):
                cen[i] = pp[5 * i + 1]
                sig[i] = pp[5 * i + 2]
                skw[i] = pp[5 * i + 3]

            skew = np.mean(skw)
            disp = np.diff(np.log(self.wave))[0]
            left = np.mean(cen) - 1000 * disp
            right = np.mean(cen) + 1000 * disp
            xx = np.arange(left, right, disp)
            yy = manygauss(xx, pp)
            ff = interpolate.interp1d(np.log(self.wave), self.PL_poly_BC, bounds_error=False, fill_value=0)
            contiflux = ff(xx)
            # plt.plot(xx, yy)

            if n_gauss > 3:
                if np.max(manygauss(xx, pp[0:20])) > 0:
                    spline = interpolate.UnivariateSpline(xx, manygauss(xx, pp[0:20]) -
                                                          np.max(manygauss(xx, pp[0:20])) / 2, s=0)
                else:
                    spline = interpolate.UnivariateSpline(xx, manygauss(xx, pp[0:20]) -
                                                          np.min(manygauss(xx, pp[0:20])) / 2, s=0)
            else:
                if np.max(yy) > 0:
                    spline = interpolate.UnivariateSpline(xx, yy - np.max(yy) / 2, s=0)
                else:
                    spline = interpolate.UnivariateSpline(xx, yy - np.min(yy) / 2, s=0)

            if len(spline.roots()) > 0:
                fwhm_left, fwhm_right = spline.roots().min(), spline.roots().max()
                fwhm = abs(np.exp(fwhm_left) - np.exp(fwhm_right)) / np.exp(np.mean(cen)) * c

                # calculate the total broad line flux
                area = np.trapz(yy[yy > 0.01*np.amax(yy)], x=np.exp(xx)[yy > 0.01*np.amax(yy)])

                # calculate the line sigma and EW in normal wavelength
                lambda0 = 0.
                lambda1 = 0.
                lambda2 = 0.
                ew = 0

                for lm in range(int((right - left) / disp)):
                    lambda0 = lambda0 + manygauss(xx[lm], pp) * disp * np.exp(xx[lm])
                    lambda1 = lambda1 + np.exp(xx[lm]) * manygauss(xx[lm], pp) * disp * np.exp(xx[lm])
                    lambda2 = lambda2 + np.exp(xx[lm]) ** 2 * manygauss(xx[lm], pp) * disp * np.exp(xx[lm])

                    ew = ew + abs(manygauss(xx[lm], pp) / contiflux[lm]) * disp * np.exp(xx[lm])

                sigma = np.sqrt(lambda2 / lambda0 - (lambda1 / lambda0) ** 2) / np.exp(np.mean(cen)) * c
            else:
                fwhm, sigma, skew, ew, peak, area = 0., 0., 0., 0., 0., 0.
                # find the line peak location

            if area > 0:
                ypeak_ind = np.argmax(yy)
                peak = np.exp(xx[ypeak_ind])
            else:
                ypeak_ind = np.argmin(yy)
                peak = np.exp(xx[ypeak_ind])

        return fwhm, sigma, skew, ew, peak, area

    def _residuals_line(self, pp, data):
        """The line residual function used in kmpfit"""
        xval, yval, weight, ind_line = data
        for xind, xflx in enumerate(self.linelist['fvalue'][ind_line]):
            if '*' in xflx:
                ini_flx = xflx.split('*')
                ind_tie = np.where(self.linelist['linename'][ind_line] == ini_flx[0])[0]
                vrat = self.linelist['lambda'][ind_line][ind_tie]
                vio = self.linelist['lambda'][ind_line][xind]
                if '>' in ini_flx[1][0] and pp[5 * xind] < pp[5 * ind_tie] * float(ini_flx[1][1:]):
                    pp[5 * xind] = pp[5 * ind_tie] * float(ini_flx[1][1:]) / vrat * vio
                elif '<' in ini_flx[1][0] and pp[5 * xind] > pp[5 * ind_tie] * float(ini_flx[1][1:]):
                    pp[5 * xind] = pp[5 * ind_tie] * float(ini_flx[1][1:]) / vrat * vio
                elif ini_flx[1][0] not in '<>':
                    pp[5 * xind] = pp[5 * ind_tie] * float(ini_flx[1]) / vrat * vio

        for xind, xsig in enumerate(self.linelist['sigval'][ind_line]):
            if '*' in xsig:
                ini_sig = xsig.split('*')
                ind_tie = np.where(self.linelist['linename'][ind_line] == ini_sig[0])[0]
                vrat = self.linelist['lambda'][ind_line][ind_tie]
                vio = self.linelist['lambda'][ind_line][xind]

                pp[5 * xind + 1] = np.log(np.exp(pp[5 * ind_tie + 1]) / vrat * vio)
                pp[5 * xind + 2] = pp[5 * ind_tie + 2]
                pp[5 * xind + 3] = pp[5 * ind_tie + 3]
                pp[5 * xind + 4] = pp[5 * ind_tie + 4]

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

    def _PlotFig(self, ra, dec, z, wave, flux, err, decomposition_host, linefit, tmp_all, gauss_result, f_conti_model,
                 all_comp_range, uniq_linecomp_sort, line_flux, save_fig_path):
        """Plot the results"""

        self.PL_poly = self.f_pl_model + self.f_poly_model

        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        # plot the first subplot
        fig = plt.figure(figsize=(15, 8))
        plt.subplots_adjust(wspace=3., hspace=0.2)
        ax = plt.subplot(2, 6, (1, 6))

        ycft = np.zeros_like(wave)
        if self.CFT:
            ycft = continuum_fitting(wave, flux + 100, self.CFT_smooth) - 100

        if self.ra == -999. or self.dec == -999.:
            plt.title(str(self.sdss_name) + '   z = ' + str(z), fontsize=20)
        else:
            plt.title('ra,dec = (' + str(ra) + ',' + str(dec) + ')   ' + str(self.sdss_name) + '   z = ' + str(z),
                      fontsize=20)

        plt.plot(self.wave_prereduced, self.flux_prereduced, 'k', label='data')
        if linefit:
            rms_line = line_flux - manygauss(np.log(wave), self.line_fit.params)
            noise_level = []
            for i in range(len(rms_line)):
                xx = np.std(rms_line[i - 20:i + 20])
                if np.isnan(xx):
                    noise_level.append(0)
                else:
                    noise_level.append(xx)
            noise_level = np.asarray(noise_level)
            plt.plot(wave, np.random.normal(flux, noise_level, len(flux)), 'grey', alpha=0.5)

        if decomposition_host and self.decomposed:
            plt.plot(wave, self.qso + self.host, 'pink', label='host+qso temp')
            plt.plot(wave, flux, 'grey', label='data-host')
            plt.plot(wave, self.host, 'purple', label='host')
        else:
            host = self.flux_prereduced.min()

        if self.Fe_uv_op or self.poly or self.BC:
            plt.scatter(wave[tmp_all], np.repeat(self.flux_prereduced.max() * 1.05, len(wave[tmp_all])), color='grey',
                        marker='o', alpha=0.5)  # plot continuum region

        if linefit:
            if self.MC:
                for p in range(int(len(gauss_result) / 12)):
                    if gauss_result[6 * (p-1) + 12] < 1200.:
                        color = 'g'
                    else:
                        color = 'r'
                    plt.plot(wave, ycft + manygauss(np.log(wave), gauss_result[::2][p * 6:(p + 1) * 6][1:]) + f_conti_model,
                             color=color)
                plt.plot(wave, ycft + manygauss(np.log(wave),
                                         np.delete(gauss_result[::2],
                                                   np.arange(0, gauss_result[::2].size, 6))) + f_conti_model,
                         'b', label='line', lw=2)
            else:
                for p in range(int(len(gauss_result) / 6)):
                    if gauss_result[6 * (p - 1) + 6] < 1200.:
                        color = 'g'
                    else:
                        color = 'r'
                    plt.plot(wave, ycft + manygauss(np.log(wave), gauss_result[p * 6:(p + 1) * 6][1:]) + f_conti_model,
                             color=color)
                plt.plot(wave, ycft + manygauss(np.log(wave),
                                         np.delete(gauss_result, np.arange(0, gauss_result.size, 6))) + f_conti_model,
                         'b', label='line', lw=2)

        plt.plot([0, 0], [0, 0], 'r', label='line br')
        plt.plot([0, 0], [0, 0], 'g', label='line na')

        if self.Fe_uv_op:
            plt.plot(wave, f_conti_model, 'c', lw=2, label='FeII')

        if self.BC:
            plt.plot(wave, self.PL_poly_BC + ycft, 'y', lw=2, label='BC')
        plt.plot(wave, self.PL_poly + ycft, color='orange', lw=2, label='conti')

        if not self.decomposed:
            self.host = self.flux_prereduced.min()
        plt.ylim(min(self.host.min(), flux.min()) * 0.9, self.flux_prereduced.max() * 1.1)

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
                if wave.min() < line_cen[ll] < wave.max():
                    plt.plot([line_cen[ll], line_cen[ll]],
                             [min(self.host.min(), flux.min()), self.flux_prereduced.max() * 1.1], 'k:')
                    plt.text(line_cen[ll] + 10, 1. * self.flux_prereduced.max(), line_name[ll], rotation=90,
                             fontsize=15)

        plt.xlim(wave.min(), wave.max())

        if linefit:
            # plot subplot from 2 to N
            for c in range(self.ncomp):
                if self.ncomp == 4:
                    axn = plt.subplot(2, 12, (12 + 3 * c + 1, 12 + 3 * c + 3))
                if self.ncomp == 3:
                    axn = plt.subplot(2, 12, (12 + 4 * c + 1, 12 + 4 * c + 4))
                if self.ncomp == 2:
                    axn = plt.subplot(2, 12, (12 + 6 * c + 1, 12 + 6 * c + 6))
                if self.ncomp == 1:
                    axn = plt.subplot(2, 12, (13, 24))
                plt.plot(wave, self.line_flux - ycft, 'k')

                if self.MC:
                    for p in range(int(len(gauss_result) / 12)):
                        if gauss_result[6 * (p-1) + 12] < 1200.:
                            color = 'g'
                        else:
                            color = 'r'
                        plt.plot(wave, manygauss(np.log(wave), gauss_result[::2][p * 6:(p + 1) * 6][1:]), color=color)
                    plt.plot(wave, manygauss(np.log(wave),
                                             np.delete(gauss_result[::2],
                                                       np.arange(0, gauss_result[::2].size, 6))),
                             'b', label='line', lw=2)
                else:
                    for p in range(int(len(gauss_result) / 6)):
                        if gauss_result[6 * (p - 1) + 6] < 1200.:
                            color = 'g'
                        else:
                            color = 'r'
                        plt.plot(wave, onegauss(np.log(wave), gauss_result[p * 6:(p + 1) * 6][1:]), color=color)
                    plt.plot(wave, manygauss(np.log(wave),
                                             np.delete(gauss_result, np.arange(0, gauss_result.size, 6))),
                             'b', label='line', lw=2)
                plt.xlim(all_comp_range[2 * c:2 * c + 2])
                f_max = line_flux[
                    np.where((wave > all_comp_range[2 * c]) & (wave < all_comp_range[2 * c + 1]), True, False)].max()
                f_min = line_flux[
                    np.where((wave > all_comp_range[2 * c]) & (wave < all_comp_range[2 * c + 1]), True, False)].min()
                plt.ylim(f_min * 0.9, f_max * 1.1)
                axn.set_xticks(
                    [all_comp_range[2 * c], np.round((all_comp_range[2 * c] + all_comp_range[2 * c + 1]) / 2, -1),
                     all_comp_range[2 * c + 1]])
                plt.text(0.02, 0.9, uniq_linecomp_sort[c], fontsize=20, transform=axn.transAxes)

        if linefit:
            plt.text(0.4, -1.4, r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20, transform=ax.transAxes)
            plt.text(-0.1, 0.5, r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20,
                     transform=ax.transAxes, rotation=90)
        else:
            plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20)
            plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20)

        if self.save_fig:
            plt.savefig(save_fig_path + self.sdss_name + '.png')

    @staticmethod
    def CalFWHM(logsigma):
        """transfer the logFWHM to normal frame"""
        return 2 * np.sqrt(2 * np.log(2)) * (np.exp(logsigma) - 1) * 300000.

    @staticmethod
    def Onegauss(xval, pp):
        """The single Gaussian model used to fit the emission lines
        Parameter: the scale factor, central wavelength in logwave, line FWHM in logwave
        """
        yval = pp[0] * skewed_voigt(xval, pp[1], pp[2], pp[4], pp[3])
        return yval

    @staticmethod
    def Manygauss(xval, pp):
        """The multi-Gaussian model used to fit the emission lines, it will call the onegauss function"""
        ngauss = int(pp.shape[0] / 5)
        if ngauss != 0:
            yval = 0.
            for i in range(ngauss):
                yval = yval + onegauss(xval, pp[i * 5:(i + 1) * 5])
            return yval

    def line_result_output(self, xname, to_print=0):
        # compute area error from sigma + scale
        line_res_name = ['fwhm', 'sigma', 'skewness', 'EW', 'Peak', 'Area']
        # err_name_1 = ['fwhm', 'scale', 'centerwave', 'sigma', 'skewness']
        xname_id = []
        xerr_id_1 = []
        xfwhm_err = []
        for i in range(0, len(self.line_result_name)):
            if xname in self.line_result_name[i] and 'err' not in self.line_result_name[i] and 'fwhm' not in self.line_result_name[i]:
                xname_id.append(i)
            if xname in self.line_result_name[i] and 'fwhm_err' in self.line_result_name[i]:
                xfwhm_err.append(self.line_result[i])
            elif xname in self.line_result_name[i] and 'err' in self.line_result_name[i]:
                xerr_id_1.append(i)

        xval = []
        for i in range(0, len(self.linelist)):
            if xname in self.linelist[i][4]:
                xval.append(self.linelist[i][0])
        xres = self.line_prop(xval, self.line_result[xname_id])

        scl = float(self.line_result[xname_id[1]])
        if self.conti_result[7] != 0:
            perr_conti = self.conti_result[8] / self.conti_result[7]
        else:
            perr_conti = 0

        if self.MC:
            xerr_peak = float(self.line_result[xerr_id_1[1]])
            xerr_sig = xres[1] * float(xfwhm_err[0]) / xres[0]
            xerr_fwhm = float(xfwhm_err[0])
            xerr_skw = float(self.line_result[xerr_id_1[3]])
            xerr_scl = float(self.line_result[xerr_id_1[0]])
            xerr_area = xres[5] * np.sqrt((xerr_scl / scl) ** 2 + (xerr_fwhm / xres[0]) ** 2)
            xerr_ew = xres[3] * np.sqrt(perr_conti ** 2 + (xerr_area / xres[5]) ** 2)

            xerr_val = [xerr_fwhm, xerr_sig, xerr_skw, xerr_ew, xerr_peak, xerr_area]

        else:
            xerr_val = np.zeros_like(xres)

        if to_print == 1:
            print('----------------------------------------------')
            print(xname + ' data')
            for k in range(0, len(line_res_name)):
                print((line_res_name[k] + '               ')[:15] + ':', '\t', np.round(xres[k], 5))
                if self.MC:
                    print((line_res_name[k] + '_err           ')[:15] + ':', '\t', np.round(xerr_val[k], 5))
        return np.asarray([xres, xerr_val])

# Add CIV abs optional in continuum removal
# Change abs output for min = peak
# Add Fe emission scaling output functionality
