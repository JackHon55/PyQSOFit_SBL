import numpy as np
from lmfit import minimize, Parameters
from typing import Tuple
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.modeling.physical_models import BlackBody
from scipy import interpolate
from scipy.stats import median_abs_deviation as mad


def noise_calculator(rms_line: np.array) -> np.array:
    """Approximates fitting uncertainty with the rms of the residual at each bin"""
    noise_level = []
    for i in range(len(rms_line)):
        xx = np.std(rms_line[i - 20:i + 20])
        if np.isnan(xx):
            noise_level.append(0)
        else:
            noise_level.append(xx)
    return np.asarray(noise_level)


def flux2L(flux, z):
    """Transfer flux to luminoity assuming a flat Universe"""
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    DL = cosmo.luminosity_distance(z).value * 10 ** 6 * 3.08 * 10 ** 18  # unit cm
    L = flux * 1.e-17 * 4. * np.pi * DL ** 2  # erg/s/A
    return L


def array_interlace(array1: np.array, array2: np.array) -> np.array:
    """interlaces two 1D numpy arrays"""
    interlaced_array = np.empty(len(array1) + len(array2), dtype=array2.dtype)
    interlaced_array[0::2] = array1
    interlaced_array[1::2] = array2
    return interlaced_array


class ContiFit:
    def __init__(self, path: str, Fe_uv_op: bool, poly: bool, BC: bool, CFT: bool, MC_conti: bool,
                 n_trails: int = 0):
        self.path = path
        self.iparams = None
        self.par_names = None
        self.fparams = None
        self.fparams_samples = None
        self.lum = None
        self.lum_samples = None
        self.wave, self.flux, self.err = None, None, None
        self.twave, self.tflux, self.terr = None, None, None
        self.tmp_all = None
        self.lum_locs = [1350., 3000., 5100., 9750., 12300., 19750]
        self.fe4570 = 0
        self.fe4570_error = 0

        if CFT:
            self.Fe_uv_op = False
            self.poly = False
            self.BC = False
            self.PL = False
            self.fe_uv = None
            self.fe_op = None
        else:
            self.Fe_uv_op = Fe_uv_op
            self.poly = poly
            self.BC = BC
            self.PL = True
            self.fe_uv = np.genfromtxt(self.path + 'feII_UV_Mejia.txt')
            self.fe_op = np.genfromtxt(self.path + 'fe_optical.txt', delimiter='')[1:].T

        self.MC_conti = MC_conti
        self.n_trails = n_trails

        self. _conti_par_names = None
        self._conti_par_values = None

    def _construct_ContiWindow(self, window_all: np.array):
        tmp_all = np.zeros(len(self.wave), dtype=bool)

        # Iterate over each window range in window_all
        for start, end in window_all:
            # Update tmp_all where wave values are within the current window
            tmp_all |= (self.wave > start) & (self.wave < end)

        if self.wave[tmp_all].shape[0] < 10:
            print('Continuum fitting pixel < 10.  ')
        if self.wave[tmp_all].shape[0] == 0:
            print('No pixels in the continuum windows to be fit.')

        self.tmp_all = tmp_all

    def _construct_ContiParam(self, contilist: np.array):
        fit_params = Parameters()

        for c in contilist[:11]:
            fit_params.add(name=c['parname'], value=c['initial'], min=c['min'], max=c['max'], vary=bool(c['vary']))

        for c in contilist[11:]:
            fit_params.add(c['parname'], value=c['initial'], min=None, max=None, vary=bool(c['vary']))

        self.iparams = fit_params
        self._finalise_ContiParam()

    def _finalise_ContiParam(self):
        # Check if we will attempt to fit the UV FeII continuum region
        ind_uv = ((self.twave > 1200) & (self.twave < 3500))
        if not self.Fe_uv_op or (np.sum(ind_uv) <= 100):
            self.iparams['Fe_uv_norm'].value = 0
            self.iparams['Fe_uv_norm'].vary = False
            self.iparams['Fe_uv_FWHM'].vary = False
            self.iparams['Fe_uv_shift'].vary = False

        # Check if we will attempt to fit the optical FeII continuum region
        ind_opt = ((self.twave > 3686.) & (self.twave < 7484.))
        if (not self.Fe_uv_op and not self.BC) or np.sum(ind_opt) <= 100:
            self.iparams['Fe_op_norm'].value = 0
            self.iparams['Fe_op_norm'].vary = False
            self.iparams['Fe_op_FWHM'].vary = False
            self.iparams['Fe_op_shift'].vary = False

        # Check if we will attempt to fit the Balmer continuum region
        ind_BC = self.twave < 3646
        if not self.BC or np.sum(ind_BC) <= 100:
            self.iparams['Balmer_norm'].value = 0
            self.iparams['Balmer_norm'].vary = False
            self.iparams['Balmer_Te'].vary = False
            self.iparams['Balmer_Tau'].vary = False

        # Check if we will fit the polynomial component
        if not self.poly:
            self.iparams['conti_a_0'].value = 0
            self.iparams['conti_a_1'].value = 0
            self.iparams['conti_a_2'].value = 0
            self.iparams['conti_a_0'].vary = False
            self.iparams['conti_a_1'].vary = False
            self.iparams['conti_a_2'].vary = False

    def fit(self, wave: np.array, flux: np.array, err: np.array, contilist: np.array, window_all: np.array):
        self.wave, self.flux, self.err = wave, flux, err

        # set initial parameter for continuum
        self._construct_ContiWindow(window_all)

        self.twave, self.tflux, self.terr = self.wave[self.tmp_all], self.flux[self.tmp_all], self.err[self.tmp_all]
        self._construct_ContiParam(contilist)

        conti_fit = minimize(self._residuals, self.iparams, args=(self.twave, self.tflux, self.terr), calc_covar=False)

        params_dict = conti_fit.params.valuesdict()
        self.par_names = list(params_dict.keys())
        self.fparams = list(params_dict.values())

        if self.MC_conti and self.n_trails > 0:
            self.mc_fit(conti_fit)

    def calc_L(self, z: float) -> Tuple[np.array, np.array]:
        self.lum = self._L_conti(self.fparams, z)

        if self.fparams_samples is None:
            self.lum_samples = None
        else:
            Ls = np.empty((self.n_trails, len(self.lum_locs)))
            # Samples loop
            for k, s in enumerate(self.fparams_samples):
                Ls[k] = self._L_conti(s, z)

            self.lum_samples = Ls

        return self.lum, self.lum_error

    @property
    def lum_error(self) -> np.array:
        if self.lum_samples is None:
            return 0
        return self.lum_samples.std(0)

    def calc_fe4570(self) -> Tuple[float, float]:
        fe_flux = self.Fe_flux_balmer(self.wave, self.fparams[3:6])
        fe_4570 = np.where((self.wave > 4434) & (self.wave < 4684), fe_flux, 0)

        self.fe4570 = np.trapz(fe_4570, dx=float(np.diff(self.wave)[0]))

        if self.fparams_samples is not None:
            fs = np.zeros(self.n_trails)
            for k, s in enumerate(self.fparams_samples):
                _flux = self.Fe_flux_balmer(self.wave, s[3:6])
                _fe = np.where((self.wave > 4434) & (self.wave < 4684), _flux, 0)
                fs[k] = np.trapz(_fe, dx=float(np.diff(self.wave)[0]))

            self.fe4570_error = fs.std()

        return self.fe4570, self.fe4570_error

    def _residuals(self, p, xval, yval, weight):
        """Continual residual function used in lmpfit"""
        pp = list(p.valuesdict().values())
        return (yval - self._f_conti_all(xval, pp)) / weight

    def _f_conti_all(self, xval, pp):
        """
        Continuum components described by 14 parameters
         pp[0]: norm_factor for the MgII Fe_template
         pp[1]: FWHM for the MgII Fe_template
         pp[2]: small shift of wavelength for the MgII Fe template
         pp[3:5]: same as pp[0:2] but for the Hbeta/Halpha Fe template
         pp[6]: norm_factor for continuum f_lambda = (lambda/3000.0)^{-alpha}
         pp[7]: slope for the power-law continuum
         pp[8:10]: norm, Te and Tau_e for the Balmer continuum at <3646 A
         pp[11:13]: polynomial for the continuum
        """

        # There is always powerlaw
        _conti_model = self.powerlaw(xval, pp[6:8])

        if self.Fe_uv_op:
            _conti_model += self.Fe_flux_mgii(xval, pp[0:3]) + self.Fe_flux_balmer(xval, pp[3:6])

        if self.poly:
            _conti_model += self.f_poly_conti(xval, pp[11:])

        if self.BC:
            _conti_model += self.balmer_conti(xval, pp[8:11])

        return _conti_model

    def mc_fit(self, ini_fit):
        samples = np.zeros((self.n_trails, len(self.iparams)))
        rms_line = self.tflux - self._f_conti_all(self.twave, self.fparams)
        noise = noise_calculator(rms_line)

        for k in range(self.n_trails):
            ctm_pts = np.concatenate([np.random.choice(len(noise), int(len(noise) / 24))])
            ctm_noise = np.random.normal(rms_line[ctm_pts], noise[ctm_pts])
            ctm_poly = np.poly1d(np.polyfit(self.twave[ctm_pts], ctm_noise, 3))
            flux_resampled = self.tflux + ctm_poly(self.twave)

            conti_fit = minimize(self._residuals, ini_fit.params,
                                 args=(self.twave, flux_resampled, self.terr), calc_covar=False)
            params_dict = conti_fit.params.valuesdict()
            samples[k] = list(params_dict.values())

        self.fparams_samples = samples

    def _L_conti(self, pp, z):
        """Calculate continuum Luminoisity at 1350,3000,5100A"""
        conti_flux = self.powerlaw(self.wave, pp[6:8]) + self.f_poly_conti(self.wave, pp[11:])
        waves = self.lum_locs
        L = -1 * np.ones_like(waves)

        for i in range(len(waves)):
            if self.wave.max() > waves[i] > self.wave.min():
                mean_flux = conti_flux[abs(self.wave - waves[i]) < 5.].mean()
                L[i] = np.log10(waves[i] * flux2L(mean_flux, z))
        return L

    @property
    def fparams_error(self):
        if self.fparams_samples is None:
            return 0
        return 1.4826 * mad(self.fparams_samples, 0)

    @staticmethod
    def powerlaw(xval, pp):
        f_pl = pp[0] * (xval / 3000.0) ** pp[1]
        return f_pl

    @staticmethod
    def balmer_conti(xval, pp):
        """Fit the Balmer continuum from the model of Dietrich+02"""
        # xval = input wavelength, in units of A
        # pp=[norm, Te, tau_BE] -- in units of [--, K, --]
        xval = xval * u.AA
        lambda_BE = 3646.  # A
        bb_lam = BlackBody(pp[1] * u.K, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
        bbflux = bb_lam(xval).value * 3.14  # in units of ergs/cm2/s/A
        tau = pp[2] * (xval.value / lambda_BE) ** 3
        result = pp[0] * bbflux * (1 - np.exp(-tau))
        ind = np.where(xval.value > lambda_BE, True, False)
        if ind.any():
            result[ind] = 0
        return result

    @staticmethod
    def f_poly_conti(xval, pp):
        """Fit the continuum with a polynomial component account for the dust reddening with a*X+b*X^2+c*X^3"""
        xval2 = xval - 3000.
        yval = 0. * xval2
        for i in range(len(pp)):
            yval = yval + pp[i] * xval2 ** (i + 1)
        return yval

    def Fe_flux_mgii(self, xval, pp):
        """Fit the UV Fe compoent on the continuum from 1200 to 3500 A based on the Boroson & Green 1992."""
        yval = np.zeros_like(xval)
        wave_Fe_mgii = 10 ** self.fe_uv[:, 0]
        flux_Fe_mgii = self.fe_uv[:, 1]
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

    def Fe_flux_balmer(self, xval, pp, xfe=None):
        """Fit the optical FeII on the continuum from 3686 to 7484 A based on Vestergaard & Wilkes 2001"""
        yval = np.zeros_like(xval)

        if xfe is None:
            wave_Fe_balmer = 10 ** self.fe_op[0]
            mult_flux = self.fe_op[1:] * 1e15
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

    @property
    def f_fe_mgii_model(self) -> np.array:
        if self.Fe_uv_op:
            return self.Fe_flux_mgii(self.wave, self.fparams[0:3])
        return np.zeros_like(self.wave)

    @property
    def f_fe_balmer_model(self) -> np.array:
        if self.Fe_uv_op:
            return self.Fe_flux_balmer(self.wave, self.fparams[3:6])
        return np.zeros_like(self.wave)

    @property
    def f_pl_model(self) -> np.array:
        if self.PL:
            return self.powerlaw(self.wave, self.fparams[6:8])
        return np.zeros_like(self.wave)

    @property
    def f_bc_model(self) -> np.array:
        if self.BC:
            return self.balmer_conti(self.wave, self.fparams[8:11])
        return np.zeros_like(self.wave)

    @property
    def f_poly_model(self) -> np.array:
        if self.poly:
            return self.f_poly_conti(self.wave, self.fparams[11:])
        return np.zeros_like(self.wave)

    @property
    def conti_par_name(self) -> np.array:
        if self._conti_par_names is not None:
            return self._conti_par_names

        lum_names = [f"L{i}" for i in self.lum_locs]
        names = np.concatenate([self.par_names, lum_names, ['fe_4570AA']])
        self._conti_par_names = names

        if self.MC_conti and self.n_trails > 0:
            names_err = np.array([f"{n}_err" for n in names], dtype=f'<U50')
            self._conti_par_names = array_interlace(names, names_err)

        return self._conti_par_names

    @property
    def conti_par_values(self) -> np.array:
        if self._conti_par_values is not None:
            return self._conti_par_values

        values = np.concatenate([self.fparams, self.lum, [self.fe4570]])
        self._conti_par_values = values

        if self.MC_conti and self.n_trails > 0:
            errors = np.concatenate([self.fparams_error, self.lum_error, [self.fe4570_error]])
            self._conti_par_values = array_interlace(values, errors)

        return self._conti_par_values
