import numpy as np
from lmfit import minimize, Parameters
from typing import Tuple
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.modeling.physical_models import BlackBody
from scipy import interpolate
from scipy.stats import median_abs_deviation as mad
from Spectra_handling.Spectrum_utls import skewed_voigt
import sys


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
    def __init__(self):
        self.gauss_line = None
        self.linelist = None
        self.wave, self.flux, self.err = None, None, None
        self.ncomp = 0
        self.uniq_linecomp_sort = None
        self.arr_section = None

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
        for xsec in range(self.ncomp):
            self.arr_section[xsec] = self.SectionFit(self.wave, self.linelist, self.uniq_linecomp_sort[xsec])

    def fit_all(self, wave, flux, err):
        self.wave, self.flux, self.err = wave, flux, err
        if self.arr_section is None:
            return None

        for xsec in self.arr_section:
            xsec.fit(flux, err)

    @property
    def all_comp_range(self):
        return np.array([i.section_range for i in self.arr_section])

    def residuals_line(self, pp: np.array, data: Tuple[np.array, np.array, np.array, np.array]) -> np.array:
        """The line residual function used in kmpfit"""
        xval, yval, weight, ind_line = data

        # Compute line linking prior to residual calculation
        self._lineflux_link(pp, ind_line)
        self._lineprofile_link(pp, ind_line)

        # restore parameters
        self.newpp = pp.copy()

        return (yval - self.manygauss(xval, pp)) / weight

    @staticmethod
    def onegauss(xval, pp):
        """The single Gaussian model used to fit the emission lines
        Parameter: the scale factor, central wavelength in logwave, line FWHM in logwave, skewness
        """

        yval = pp[0] * skewed_voigt(xval, pp[1], pp[2], pp[4] / 100, pp[3])
        return yval

    def manygauss(self, xval, pp):
        """The multi-Gaussian model used to fit the emission lines, it will call the onegauss function"""
        ngauss = int(pp.shape[0] / 5)
        if ngauss != 0:
            yval = 0.
            for i in range(ngauss):
                yval = yval + self.onegauss(xval, pp[i * 5:(i + 1) * 5])
            return yval

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

    class SectionFit:
        def __init__(self, wave: np.array, linelist, section_lines: list):
            self.linelist = linelist
            self.line_indices = (self.linelist['compname'] == section_lines)  # get line index
            self.n_line = np.sum(self.line_indices)  # n line in one complex

            linelist_fit = linelist[self.line_indices]
            # read section range from table
            self.section_range = [linelist_fit[0]['minwav'], linelist_fit[0]['maxwav']]
            self.section_indices = (wave > self.section_range[0]) & (wave < self.section_range[1])

            self.gauss_line = None
            self.line_result = None
            self.line_result_name = None
            self.gauss_result = None

            self.iparams = None
            self.fparams = None
            self.fparams_samples = None

            self.wave, self.line_flux, self.err = wave, None, None
            self.twave, self.tline_flux, self.terr = wave[self.section_indices], None, None

        def fit(self, flux, err):
            print(self.linelist['compname'][self.line_indices][0])
            all_para_std = np.asarray([])
            fwhm_std = np.asarray([])
            fwhm_lines = []

            # call kmpfit for lines
            self._construct_LineParam()
            self.line_flux, self.err = flux, err
            self.tline_flux, self.terr = flux[self.section_indices], err[self.section_indices]
            """The key function to do the line fit with kmpfit"""
            # initial parameter definition

            lmpargs = [np.log(self.twave), self.tline_flux, self.terr, self.line_indices]
            line_fit = minimize(LineFit.residuals_line, self.iparams, args=lmpargs, calc_covar=False)

            params_dict = line_fit.params.valuesdict()
            params = list(params_dict.values())
            self.line_fit = line_fit
            self.line_fit_par = params

            # calculate FWMH for each fitted components
            for xx in np.split(line_fit.params, len(self.linelist['compname'][self.line_indices])):
                fwhm_lines.append(self.line_property(xx).fwhm)
            fwhm_lines = np.asarray(fwhm_lines)

        def _construct_LineParam(self):
            ind_line = self.line_indices
            fit_params = Parameters()
            for n in range(self.n_line):
                linename = self.linelist['linename'][ind_line][n]
                # voffset initial is always at the line wavelength defined
                ini_voff = np.log(self.linelist['lambda'][ind_line][n])

                # -------------- Line Profile definitions -------------
                # Cases with linked lines. Gamma, sigma, skew and voffset are all tied
                if '*' in self.linelist['sigval'][ind_line][n]:
                    lam_par = {'value': ini_voff, 'min': None, 'max': None, 'vary': False}
                    sig_par = {'value': 0.0018, 'min': None, 'max': None, 'vary': False}
                    skw_par = {'value': 0, 'min': None, 'max': None, 'vary': False}
                    if 'g' in self.linelist['iniskw'][ind_line][n]:
                        gam_par = {'value': 1e-3, 'min': None, 'max': None, 'vary': True}
                    else:
                        gam_par = {'value': 0, 'min': None, 'max': None, 'vary': False}
                # Cases without linked lines
                else:
                    # profile definitions based on the profile input that looks like this "v[+],g[45],s[-10,10]"
                    voffset = self.linelist['voff'][ind_line][n]
                    ini_sig = self.linelist['sigval'][ind_line][n].strip('[').strip(']')
                    nprofile = InitialProfileParameters(self.linelist['iniskw'][ind_line][n])
                    sig_par = nprofile.sigma_defition(ini_sig)
                    lam_par = nprofile.velocity_offset_definition(ini_voff, voffset)
                    skw_par = nprofile.skew_definitions()
                    gam_par = nprofile.gamma_defintion()

                # ----------------- Flux input parameters -----------------
                fluxstring = self.linelist['fvalue'][ind_line][n]
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
