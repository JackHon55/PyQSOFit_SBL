import numpy as np
from lmfit import minimize, Parameters
from typing import Tuple
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const
from astropy.modeling.physical_models import BlackBody
from scipy import interpolate
from scipy.stats import median_abs_deviation as mad
from Spectra_handling.Spectrum_utls import skewed_voigt
import sys

cspeed = const.c.to(u.km / u.s).value  # km/s


def array_interlace(array1: np.array, array2: np.array) -> np.array:
    """interlaces two 1D numpy arrays"""
    interlaced_array = np.empty(len(array1) + len(array2), dtype=array2.dtype)
    interlaced_array[0::2] = array1
    interlaced_array[1::2] = array2
    return interlaced_array


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


class InitialProfileParameters:
    """class to translate input profile into input fitting parameters"""

    def __init__(self, inputstring):
        mode_definitions = inputstring
        self.mode_mode = {"v": "", "g": "0", "s": "0"}
        for xmode in mode_definitions.split(";"):
            self.mode_mode[xmode[0]] = xmode.split("[")[1][:-1]

    def velocity_offset_definition(self, v_pos: float, voffset: float):
        """initialise the velocity offset bounds based on vmode"""
        mode = self.mode_mode["v"]
        lambda_low = v_pos - voffset
        lambda_up = v_pos + voffset
        if voffset == 0:  # Fixed
            vdown, vup, vary = None, None, False
        elif mode == "":  # Free
            vdown, vup, vary = lambda_low, lambda_up, True
        elif mode == "+":  # positive only
            vdown, vup, vary = v_pos - voffset * 0.01, lambda_up, True
        elif mode == "-":  # negative only
            vdown, vup, vary = lambda_low, v_pos + voffset * 0.01, True
        elif mode == ".":  # exact as defined
            v_pos = voffset
            vdown, vup, vary = None, None, False
        else:
            print("Velocity offset input parameter error")
            sys.exit()

        return {'value': v_pos, 'min': vdown, 'max': vup, 'vary': vary}

    def gamma_defintion(self):
        """Turns gamma on or off and sets bounds for Voigt profile"""
        mode = self.mode_mode["g"]
        if mode == "0":  # off
            ini_gam = 0
            gam_down, gam_up, vary = None, None, False
        elif mode == "":  # free
            ini_gam = 1e-2
            gam_down, gam_up, vary = 0, 1, True
        else:
            try:
                ini_gam = float(mode)  # exact as defined
                gam_down, gam_up, vary = None, None, False
            except ValueError:
                print("Gamma input parameter error")
                sys.exit()
        return {'value': ini_gam, 'min': gam_down, 'max': gam_up, 'vary': vary}

    def skew_definitions(self):
        """Initialises the first guess for skew and sets the bounds for skew"""
        mode = self.mode_mode["s"]
        if mode == "":  # Free
            ini_skw = 0
            skw_down, skw_up, vary = None, None, True
        elif mode == "0":  # off
            ini_skw = 0
            skw_down, skw_up, vary = None, None, False
        elif "," in mode:  # given range
            try:
                skw_down, skw_up = np.sort([float(mode.split(',')[0]), float(mode.split(',')[1])])
                ini_skw = np.mean([skw_down, skw_up])
                vary = True
            except ValueError:
                print("Skew input parameter error")
                sys.exit()
        else:
            try:
                ini_skw = float(mode)  # exact as defined
                skw_down, skw_up, vary = None, None, False
            except ValueError:
                print("Skew input parameter error")
                sys.exit()
        return {'value': ini_skw, 'min': skw_down, 'max': skw_up, 'vary': vary}

    def sigma_defition(self, ini_sig: str):
        """initialises the first guess for sigma and sets the bounds for sigma"""
        if len(ini_sig.split(',')) == 1:  # Fixed sigma
            sig_down, sig_up, vary = None, None, False
            ini_sig = float(ini_sig.split(',')[0])
        else:  # Free sigma
            sig_down, sig_up = np.sort([float(ini_sig.split(',')[0]), float(ini_sig.split(',')[1])])
            ini_sig = np.mean([sig_down, sig_up])
            vary = True
        return {'value': ini_sig, 'min': sig_down, 'max': sig_up, 'vary': vary}


class LineFit:
    def __init__(self, MC: bool = False, n_trails: int = 0):
        self.gauss_line = None
        self.linelist = None
        self.wave, self.flux, self.err = None, None, None
        self.ncomp = 0
        self.uniq_linecomp_sort = None
        self.arr_section = None
        self.arr_fitted = None
        self.MC = MC
        self.n_trails = n_trails

        self.gauss_line = None
        self.line_result_name = None
        self.line_result = None

    def _DoLineFit_sectiondefinitions(self):
        """Reads the Component_definition.py file to obtain the sections to line fit"""
        bool_fitted_section = (self.linelist['lambda'] > self.wave.min()) & (self.linelist['lambda'] < self.wave.max())
        fitted_section_index = np.where(bool_fitted_section, True, False)

        # sort section name with line wavelength
        uniq_linecomp, uniq_ind = np.unique(self.linelist['compname'][fitted_section_index], return_index=True)
        self.uniq_linecomp_sort = uniq_linecomp[self.linelist['lambda'][fitted_section_index][uniq_ind].argsort()]

    def initialise(self, linelist):
        self.linelist = linelist
        self._DoLineFit_sectiondefinitions()
        self.ncomp = len(self.uniq_linecomp_sort)
        if self.ncomp == 0:
            print("No line to fit! Please set Line_fit to FALSE or enlarge wave_range!")
            return None

        self.arr_section = np.empty(self.ncomp)
        self.arr_fitted = np.empty(self.ncomp)
        for xsec in range(self.ncomp):
            self.arr_section[xsec] = self.SectionFit(self.wave, self.linelist,
                                                     self.uniq_linecomp_sort[xsec], self.MC, self.n_trails)
            self.arr_fitted[xsec] = False

        self.gauss_line = np.empty(self.ncomp, dtype=list)
        self.line_result = np.empty(self.ncomp, dtype=list)
        self.line_result_name = np.empty(self.ncomp, dtype=list)

    def fit_all(self, wave, flux, err, conti):
        self.wave, self.flux, self.err = wave, flux, err
        if self.arr_section is None:
            return None

        for i, xsec in enumerate(self.arr_section):
            if np.sum(xsec.section_indices) <= 10:
                print("less than 10 pixels in line fitting!")
                continue

            xsec.fit(flux, err, conti)
            self.arr_fitted[i] = True
            self.gauss_line[i] = xsec.gauss_line
            self.line_result[i] =

    @property
    def all_comp_range(self):
        return np.array([i.section_range for i in self.arr_section])

    class SectionFit:
        def __init__(self, wave: np.array, linelist, section_lines: list,
                     MC: bool, n_trails: int = 0):
            self.line_indices = (linelist['compname'] == section_lines)  # get line index
            self.n_line = np.sum(self.line_indices)  # n line in one complex
            self.n_trails = n_trails
            self.MC = MC

            self.tlinelist = linelist[self.line_indices]
            # read section range from table
            self.section_range = [self.tlinelist[0]['minwav'], self.tlinelist[0]['maxwav']]
            self.section_indices = (wave > self.section_range[0]) & (wave < self.section_range[1])

            self.line_props = None
            self.gauss_line = None

            self.iparams = None
            self.tmpparams = None
            self.fparams = None
            self.fparams_samples = None

            self.wave, self.line_flux, self.err = wave, None, None
            self.twave, self.tline_flux, self.terr = wave[self.section_indices], None, None

            self._fwhms = np.zeros(self.n_line)
            self._fwhms_error = np.zeros(self.n_line)
            self._peaks_error = np.zeros(self.n_line)

        def fit(self, flux, err, conti):
            print(self.tlinelist['compname'][0])

            # call kmpfit for lines
            self._construct_LineParam()
            self.line_flux, self.err = flux, err
            self.tline_flux, self.terr = flux[self.section_indices], err[self.section_indices]
            """The key function to do the line fit with kmpfit"""
            # initial parameter definition

            lmpargs = [np.log(self.twave), self.tline_flux, self.terr]
            line_fit = minimize(self.residuals_line, self.iparams, args=lmpargs, calc_covar=False)
            self.fparams = self.tmpparams
            self.gauss_line = LineProperties(self.twave, self.fparams).gauss_line

            if self.MC and self.n_trails > 0:
                self.mc_fit(line_fit, conti)

        def _construct_LineParam(self):
            fit_params = Parameters()
            for n in range(self.n_line):
                linename = self.tlinelist['linename'][n]
                # voffset initial is always at the line wavelength defined
                ini_voff = np.log(self.tlinelist['lambda'][n])

                # -------------- Line Profile definitions -------------
                # Cases with linked lines. Gamma, sigma, skew and voffset are all tied
                if '*' in self.tlinelist['sigval'][n]:
                    lam_par = {'value': ini_voff, 'min': None, 'max': None, 'vary': False}
                    sig_par = {'value': 0.0018, 'min': None, 'max': None, 'vary': False}
                    skw_par = {'value': 0, 'min': None, 'max': None, 'vary': False}
                    if 'g' in self.tlinelist['iniskw'][n]:
                        gam_par = {'value': 1e-3, 'min': None, 'max': None, 'vary': True}
                    else:
                        gam_par = {'value': 0, 'min': None, 'max': None, 'vary': False}
                # Cases without linked lines
                else:
                    # profile definitions based on the profile input that looks like this "v[+],g[45],s[-10,10]"
                    voffset = self.tlinelist['voff'][n]
                    ini_sig = self.tlinelist['sigval'][n].strip('[').strip(']')
                    nprofile = InitialProfileParameters(self.tlinelist['iniskw'][n])
                    sig_par = nprofile.sigma_defition(ini_sig)
                    lam_par = nprofile.velocity_offset_definition(ini_voff, voffset)
                    skw_par = nprofile.skew_definitions()
                    gam_par = nprofile.gamma_defintion()

                # ----------------- Flux input parameters -----------------
                fluxstring = self.tlinelist['fvalue'][n]
                if '[' in fluxstring:  # fixed
                    ini_flx = fluxstring.strip('[').strip(']')
                    flx_vary = True
                    if ',' in ini_flx:
                        sc_down, sc_up = np.sort([float(ini_flx.split(',')[0]), float(ini_flx.split(',')[1])])
                        ini_flx = np.mean([sc_down, sc_up])
                    else:
                        ini_flx = float(ini_flx)
                        sc_down, sc_up = np.sort([ini_flx * 1.03, ini_flx * 0.97])

                elif '*' in fluxstring:  # linked
                    ini_flx = 0.005
                    if '<' in fluxstring or '>' in fluxstring:
                        sc_down, sc_up, flx_vary = 1e-7, None, True
                    else:
                        sc_down, sc_up, flx_vary = None, None, False

                else:  # Free
                    if float(fluxstring) >= 0:
                        ini_flx = float(fluxstring)
                        sc_down, sc_up, flx_vary = 1e-7, None, True
                    else:
                        ini_flx = float(fluxstring)
                        sc_down, sc_up, flx_vary = None, -1e-7, True

                fit_params.add(f'{linename}_scale', value=ini_flx, min=sc_down, max=sc_up, vary=flx_vary)
                fit_params.add(f'{linename}_voff', **lam_par)
                fit_params.add(f'{linename}_sigma', **sig_par)
                fit_params.add(f'{linename}_skew', **skw_par)
                fit_params.add(f'{linename}_gamma', **gam_par)

            self.iparams = fit_params

        def residuals_line(self, pp: np.array, data: Tuple[np.array, np.array, np.array]) -> np.array:
            """The line residual function used in kmpfit"""
            xval, yval, weight = data

            # Compute line linking prior to residual calculation
            self._lineflux_link(pp)
            self._lineprofile_link(pp)

            self.tmpparams = pp

            return (yval - manygauss(xval, pp)) / weight

        def _lineflux_link(self, pp: np.array) -> None:
            """rescale the height of fitted line if height/flux is linked to another line"""
            for line_index, line_flux in enumerate(self.tlinelist['fvalue']):
                if '*' in line_flux:
                    input_flux = line_flux.split('*')
                    link_index = np.where(self.tlinelist['linename'] == input_flux[0])[0]
                    flux_target = self.tlinelist['lambda'][link_index]
                    flux_now = self.tlinelist['lambda'][line_index]

                    # If component is less than linked component but should be greater, set to boundary
                    if '>' in input_flux[1][0] and pp[5 * line_index] < pp[5 * link_index] * float(input_flux[1][1:]):
                        pp[5 * line_index] = pp[5 * link_index] * float(input_flux[1][1:]) / flux_target * flux_now

                    # If component is greater than linked component but should be less, set to boundary
                    elif '<' in input_flux[1][0] and pp[5 * line_index] > pp[5 * link_index] * float(input_flux[1][1:]):
                        pp[5 * line_index] = pp[5 * link_index] * float(input_flux[1][1:]) / flux_target * flux_now

                    # If component set exactly to be a multiplier of the linked component, scale it to multiplier
                    elif input_flux[1][0] not in '<>':
                        pp[5 * line_index] = pp[5 * link_index] * float(input_flux[1]) / flux_target * flux_now

        def _lineprofile_link(self, pp: np.array) -> None:
            """reset the sigma, skew, velocity offset, and gamma of component to linked component"""
            for line_index, line_sigma in enumerate(self.tlinelist['sigval']):
                if '*' in line_sigma:
                    input_sigma = line_sigma.split('*')
                    link_index = np.where(self.tlinelist['linename'] == input_sigma[0])[0]
                    sigma_target = self.tlinelist['lambda'][link_index]
                    sigma_now = self.tlinelist['lambda'][line_index]

                    pp[5 * line_index + 1] = np.log(np.exp(pp[5 * link_index + 1]) / sigma_target * sigma_now)
                    pp[5 * line_index + 2] = pp[5 * link_index + 2]
                    pp[5 * line_index + 3] = pp[5 * link_index + 3]
                    pp[5 * line_index + 4] = pp[5 * link_index + 4]

        def _calculate_FWHM_peak(self, pp: np.array) -> Tuple[np.array, np.array]:
            tmp_fwhm = np.zeros(self.n_line)
            tmp_peak = np.zeros(self.n_line)
            pps = np.split(pp, len(self.tlinelist['compname']))
            for i, xx in enumerate(pps):
                tmp_fwhm[i] = LineProperties(self.twave, xx).fwhm
                tmp_fwhm[i] = LineProperties(self.twave, xx).peak
            return tmp_fwhm, tmp_peak

        def mc_fit(self, ini_fit, conti):
            samples = np.zeros((self.n_trails, len(self.iparams)))
            rms_line = self.tline_flux - manygauss(self.twave, self.fparams)
            noise = noise_calculator(rms_line)

            for k in range(self.n_trails):
                ctm_pts = np.concatenate([np.random.choice(len(noise), int(len(noise) / 24))])
                ctm_noise = np.random.normal(rms_line[ctm_pts], noise[ctm_pts])
                ctm_poly = np.poly1d(np.polyfit(self.twave[ctm_pts], ctm_noise, 3))
                flux_resampled = self.tline_flux + ctm_poly(self.twave)

                conti_fit = minimize(self.residuals_line, ini_fit.params,
                                     args=(self.twave, flux_resampled, self.terr), calc_covar=False)

                samples[k] = self.tmpparams

            self.fparams_samples = samples

        @property
        def fwhms(self) -> np.array:
            if self._fwhms is None:
                self._fwhms, peak = self._calculate_FWHM_peak(self.fparams)
            return self._fwhms

        @property
        def fwhms_error(self) -> np.array:
            if self._fwhms_error is None:
                fwhm_samples = np.empty(self.n_trails)
                peak_samples = np.empty(self.n_trails)
                for i, pp in enumerate(self.fparams_samples):
                    fwhm_samples[i], peak_samples[i] = self._calculate_FWHM_peak(pp)
                self._fwhms_error = 1.4826 * mad(fwhm_samples, 0)
                self._peaks_error = 1.4826 * mad(peak_samples, 0)
            return self._fwhms_error

        @property
        def peaks_error(self) -> np.array:
            if self._peaks_error is None:
                fwhm_samples = np.empty(self.n_trails)
                peak_samples = np.empty(self.n_trails)
                for i, pp in enumerate(self.fparams_samples):
                    fwhm_samples[i], peak_samples[i] = self._calculate_FWHM_peak(pp)
                self._fwhms_error = 1.4826 * mad(fwhm_samples, 0)
                self._peaks_error = 1.4826 * mad(peak_samples, 0)
            return self._peaks_error

        @property
        def fparams_errors(self) -> np.array:
            all_pp_1comp = np.zeros(len(self.iparams) * self.n_trails).reshape(len(self.iparams), self.n_trails)
            all_pp_std = np.zeros(len(self.iparams))
            for st in range(len(self.iparams)):
                all_pp_std[st] = 1.4826 * mad(all_pp_1comp[st, :])
                if (st - 1) % 5 == 0:
                    all_pp_std[st] = self.peaks_error[int(st / 5)]
            return all_pp_std

        @property
        def fitting_result(self):
            values = np.asarray([np.concatenate([[self.fwhms[i]], j]) for i, j in enumerate(self.fparams)])
            errors = np.asarray([np.concatenate([[self.fwhms_error[i]], j]) for i, j in enumerate(self.fparams_errors)])

class LineProperties:
    def __init__(self, wave: np.array, pp: np.array, conti: np.array = None):
        self.pp = pp.astype(float)
        self.n_gauss = int(len(pp) / 5)
        self.conti = np.ones_like(wave) if conti is None else conti

        self.cen, self.sig, self.skw = None, None, None
        self.line_gaussian_pp()
        self.wave, self.gauss_line = self.line_model_compute(wave)

        self._area, self._ew = None, None

    def line_gaussian_pp(self):
        """Extracts centroid, sigma, and skew parameters from the input array."""
        cen = np.zeros(self.n_gauss)  # centroid array of line
        sig = np.zeros(self.n_gauss)  # sigma array of line
        skw = np.zeros(self.n_gauss)  # skew array of lines

        for i in range(self.n_gauss):
            cen[i] = self.pp[5 * i + 1]
            sig[i] = self.pp[5 * i + 2]
            skw[i] = self.pp[5 * i + 3]

        self.cen, self.sig, self.skw = cen, sig, skw

    def line_model_compute(self, wave: np.array) -> Tuple[np.array, np.array]:
        """Computes the wavelength and corresponding spectrum values."""
        disp = np.diff(wave)[0]
        left = wave.min()
        right = wave.max()
        xx = np.arange(left, right, disp)
        xlog = np.log(xx)
        yy = manygauss(xlog, self.pp)
        return xx, yy

    def line_peak(self):
        xx = self.wave
        peaks = np.empty((self.n_gauss,))
        areas = np.empty((self.n_gauss,))
        xlog = np.log(xx)

        for xline in range(self.n_gauss):
            xprofile = onegauss(xlog, np.split(self.pp, self.n_gauss)[xline])
            peaks[xline] = xx[np.argmax(abs(xprofile))]
            valid_indices = xprofile > 0.01 * np.amax(xprofile)
            areas[xline] = np.trapz(xprofile[valid_indices], x=xx[valid_indices])

        return np.average(peaks, weights=areas)

    def line_flux_ew(self) -> Tuple[float, float]:
        """Calculates the total broad line flux and equivalent width (EW)."""
        xx, yy = self.wave, self.gauss_line
        ff = interpolate.interp1d(xx, self.conti, bounds_error=False, fill_value=0)
        valid_indices = yy > 0.01 * np.amax(yy)
        area = np.trapz(yy[valid_indices], x=xx[valid_indices])
        ew = area / np.mean(ff(xx[valid_indices]))
        return area, ew

    @property
    def skew(self):
        if self.n_gauss == 0:
            return 0
        return -999 if self.n_gauss > 1 else np.mean(self.skw)

    @property
    def fwhm(self):
        if self.n_gauss == 0:
            return 0
        xx, yy = self.wave, self.gauss_line
        cen = np.exp(np.mean(self.cen))
        if np.max(yy) > 0:
            spline = interpolate.UnivariateSpline(xx, yy - np.max(yy) / 2, s=0)
        else:
            spline = interpolate.UnivariateSpline(xx, yy - np.min(yy) / 2, s=0)

        if len(spline.roots()) > 0:
            fwhm_left, fwhm_right = spline.roots().min(), spline.roots().max()
            return abs(fwhm_left - fwhm_right) / cen * cspeed
        else:
            return -999

    @property
    def peak(self):
        if self.n_gauss == 0:
            return 0
        xx, yy = self.wave, self.gauss_line
        single_peak = xx[np.argmax(abs(yy))]
        peak = single_peak if self.n_gauss == 1 else self.line_peak()
        return peak

    @property
    def sigma(self):
        if self.n_gauss == 0:
            return 0
        xx = self.wave
        cen = np.exp(np.mean(self.cen))
        disp = np.diff(xx)[0]
        xlog = np.log(xx)
        lambda0 = 0.
        lambda1 = 0.
        lambda2 = 0.

        for lm in range(int((xx.max() - xx.min()) / disp)):
            gauss_val = manygauss(xlog[lm], self.pp)
            lambda0 += gauss_val * disp * xx[lm]
            lambda1 += xx[lm] * gauss_val * disp * xx[lm]
            lambda2 += xx[lm] ** 2 * gauss_val * disp * xx[lm]

        sigma = np.sqrt(lambda2 / lambda0 - (lambda1 / lambda0) ** 2) / cen * cspeed
        return sigma

    @property
    def area(self):
        if self.n_gauss == 0:
            return 0
        if self._area is None:
            self._area, self._ew = self.line_flux_ew()
        return self._area

    @property
    def ew(self):
        if self.n_gauss == 0:
            return 0
        if self._ew is None:
            self._area, self._ew = self.line_flux_ew()
        return self._ew
