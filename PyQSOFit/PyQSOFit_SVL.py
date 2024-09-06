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

from Spectra_handling.Spectrum_utls import skewed_voigt, blueshifting
from PyQSOFit.Continuum import ContiFit
from PyQSOFit.Lines import LineFit, LineProperties
from scipy.stats import median_abs_deviation as mad
from lmfit import minimize, Parameters, report_fit
from extinction import fitzpatrick99, remove

from PyAstronomy import pyasl
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from scipy.ndimage import gaussian_filter as g_filt
from rebin_spec import rebin_spec

import warnings

warnings.filterwarnings("ignore")


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
                 path=None, config=None, and_mask=None, or_mask=None):
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
        self.param_file_path = config
        self.ebmv = ebmv
        self.c = 299792.458  # km/s

    def Fit(self, name=None, nsmooth=1, and_or_mask=True, reject_badpix=True, deredden=True,
            decomposition_host=True, BC03=False, Mi=None, npca_gal=5, npca_qso=20,
            Fe_uv_op=True, redshift=True, poly=False, PL=True, CFT=False, CFT_smooth=75, BC=False,
            MC_conti=False, MC=True, n_trails=1, linefit=True,
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
        self.host_decomposition = decomposition_host
        self.PL = PL
        self.MC = MC
        self.n_trails = n_trails
        self.plot_line_name = plot_line_name
        self.plot_legend = plot_legend
        self.save_fig = save_fig
        self.gauss_line = None

        self.contiobj = ContiFit(path=self.path, Fe_uv_op=Fe_uv_op, poly=poly, BC=BC, CFT=CFT,
                                 MC_conti=MC_conti, n_trails=n_trails)
        self.lineobj = None
        if linefit:
            self.lineobj = LineFit(MC=MC, n_trails=n_trails)

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
            self.flux = self.smooth(self.flux, nsmooth)
            self.err = self.smooth(self.err, nsmooth)

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
            self.host_decomposition = False
            self._apply_cft(CFT_smooth)

        # do host decomposition --------------
        if self.z < 1.16 and self.host_decomposition:
            self._DoDecomposition()
        else:
            self.decomposed = False
            if self.z > 1.16 and self.host_decomposition:
                print('redshift larger than 1.16 is not allowed for host decomposion!')

        # fit continuum --------------------
        '''if not self.Fe_uv_op and not self.poly and not self.BC and not self.PL:
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
            print('Conti done')'''
        self._DoContiFit()
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
        ind_gooderror = (self.err != 0) & ~np.isinf(self.err)
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

    def _apply_cft(self, smooth):
        self.f_conti_model = continuum_fitting(self.wave, self.flux, smooth)
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

        wave_new, flux_new, err_new = self.wave[ind_data], self.flux[ind_data], self.err[ind_data]

        flux_temp = np.vstack((flux_gal_new[0:self.npca_gal, :], flux_qso_new[0:self.npca_qso, :]))
        res = np.linalg.lstsq(flux_temp.T, flux_new)[0]

        host_flux = np.dot(res[0:self.npca_gal], flux_temp[0:self.npca_gal])
        qso_flux = np.dot(res[self.npca_gal:], flux_temp[self.npca_gal:])

        data_cube = np.vstack((wave_new, flux_new, err_new, host_flux, qso_flux))

        ind_f4200 = (wave_new > 4160.) & (wave_new < 4210.)
        frac_host_4200 = np.sum(host_flux[ind_f4200]) / np.sum(flux_new[ind_f4200])
        ind_f5100 = (wave_new > 5080.) & (wave_new < 5130.)
        frac_host_5100 = np.sum(host_flux[ind_f5100]) / np.sum(flux_new[ind_f5100])

        return data_cube

    def _DoContiFit(self):
        """Fit the continuum with PL, Polynomial, UV/optical FeII, Balmer continuum"""
        contilist, window_all = self.read_conti_params()
        self.contiobj.fit(self.wave, self.flux, self.err, contilist, window_all)

        # calculate continuum luminoisty
        self.contiobj.calc_L(self.z)
        self.contiobj.calc_fe4570()

        res_name = np.array(['ra', 'dec', 'plateid', 'MJD', 'fiberid', 'redshift', 'SN_ratio_conti'])
        self.conti_result = np.asarray([self.ra, self.dec, self.plateid, self.mjd, self.fiberid,
                                        self.z, self.SN_ratio_conti])
        # get conti result -----------------------------
        self.conti_result_name = np.append(res_name, self.contiobj.conti_par_name)
        self.conti_result = np.append(self.conti_result, self.contiobj.conti_par_values)

        # save different models--------------------
        self.PL_poly_BC = self.contiobj.f_pl_model + self.contiobj.f_poly_model + self.contiobj.f_bc_model
        self.f_conti_model = self.contiobj.f_fe_mgii_model + self.contiobj.f_fe_balmer_model + self.PL_poly_BC
        self.line_flux = self.flux - self.f_conti_model

        return None

    def _DoLineFit(self):
        linelist = fits.open(self.path + 'qsopar2.fits')[1].data

        self.lineobj.initialise(linelist, self.wave)

        # define arrays for result taking
        self.all_comp_range = self.lineobj.all_comp_range
        self.lineobj.fit_all(self.line_flux, self.err, self.f_conti_model)

        self.gauss_result = np.concatenate(self.lineobj.gauss_result)
        self.gauss_line = manygauss(np.log(self.wave), self.gauss_result)
        self.line_result = np.concatenate(self.lineobj.line_result).flatten()
        self.line_result_name = np.concatenate(self.lineobj.line_result_name).flatten()

        return self.line_result, self.line_result_name

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
        for xsec in self.lineobj.arr_section:
            for xfwhm, xpp in zip(xsec.fwhms, np.split(xsec.fparams, xsec.n_line)):
                color = 'g' if xfwhm < 1200 else 'r'
                gauss_line = manygauss(np.log(self.wave), xpp)
                plt.plot(self.wave, gauss_line + arr_conti, color=color)


        plt.plot(self.wave, self.gauss_line + arr_conti, 'b', label='line', lw=2)

    def _plotSubPlot_Main(self, linefit: bool) -> None:
        """Plotting the main spectrum figure"""
        plt.plot(self.wave_prereduced, self.flux_prereduced, 'k', label='data')

        # Residual strength at each wavelength bin
        if linefit:
            rms_line = self.line_flux - self.gauss_line
            noise_line = self.noise_calculator(rms_line)
            plt.plot(self.wave, np.random.normal(self.flux, noise_line, len(self.flux)), 'grey', alpha=0.5)

        # Host template if fitted
        if self.decomposed:
            plt.plot(self.wave, self.qso + self.host, 'pink', label='host+qso temp')
            plt.plot(self.wave, self.flux, 'grey', label='data-host')
            plt.plot(self.wave, self.host, 'purple', label='host')

        # Markers for the continuum windows
        if self.contiobj.Fe_uv_op or self.contiobj.poly or self.contiobj.BC:
            conti_window_markers = np.repeat(self.flux_prereduced.max() * 1.05, len(self.contiobj.twave))
            plt.scatter(self.contiobj.twave, conti_window_markers, color='grey', marker='o', alpha=0.5)

        # Fitted Emission line models
        if linefit:
            self._plotSubPlot_Elines(self.f_conti_model)

        plt.plot([0, 0], [0, 0], 'r', label='line br')
        plt.plot([0, 0], [0, 0], 'g', label='line na')

        # Continuum with Fe emission
        if self.contiobj.Fe_uv_op:
            plt.plot(self.wave, self.f_conti_model, 'c', lw=2, label='FeII')

        # Balmer Continuum
        if self.contiobj.BC:
            plt.plot(self.wave, self.PL_poly_BC, 'y', lw=2, label='BC')

        if self.contiobj.CFT:
            plt.plot(self.wave, self.f_conti_model, color='orange', lw=2, label='conti')
        else:
            plt.plot(self.wave, self.contiobj.f_pl_model + self.contiobj.f_poly_model,
                     color='orange', lw=2, label='conti')

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
        for c in range(self.lineobj.ncomp):
            # create subplot axes
            if self.lineobj.ncomp == 4:
                axn = plt.subplot(2, 12, (12 + 3 * c + 1, 12 + 3 * c + 3))
            elif self.lineobj.ncomp == 3:
                axn = plt.subplot(2, 12, (12 + 4 * c + 1, 12 + 4 * c + 4))
            elif self.lineobj.ncomp == 2:
                axn = plt.subplot(2, 12, (12 + 6 * c + 1, 12 + 6 * c + 6))
            elif self.lineobj.ncomp == 1:
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
            plt.text(0.02, 0.9, self.lineobj.uniq_linecomp_sort[c], fontsize=20, transform=axn.transAxes)

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
            plt.savefig(save_fig_path + '.png')

    @staticmethod
    def smooth(y, box_pts):
        """Smooth the flux with n pixels"""
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

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

    def line_result_output(self, line_name: str, to_print: bool = False) -> np.array:
        """Compile errors of specified line to output as an array or print neatly"""
        # name_id and err_id follows the name order as in properties list, repeating every 6 element
        properties = ['fwhm', 'sigma', 'skewness', 'EW', 'Peak', 'Area']
        name_id, err_id = self.line_errordata_read(line_name)
        if len(name_id) == 0:
            print(f"{line_name} not fitted")
            return np.zeros((2, len(properties)))

        # Calculate values and errors for each property
        property_values = LineProperties(self.wave, self.line_result[name_id], self.f_conti_model)

        if self.MC:
            property_values.line_error_calculate(name_id, err_id, self.line_result, self.conti_result)

        if to_print:
            print('----------------------------------------------')
            print(line_name + ' data')
            for k in range(0, len(properties)):
                print_values = property_values.list
                print((properties[k] + '               ')[:15] + ':', '\t', np.round(print_values[k], 5))
                if self.MC:
                    print_errors = property_values.err_list
                    print((properties[k] + '_err           ')[:15] + ':', '\t', np.round(print_errors[k], 5))

        return np.asarray([property_values.dict, property_values.err_dict])

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

    def read_conti_params(self):
        # read line parameter
        hdul = fits.open(self.param_file_path)

        conti_windows = np.vstack([np.array(t) for t in hdul[2].data])
        data = hdul[3].data

        return data, conti_windows
