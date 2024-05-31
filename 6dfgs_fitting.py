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


class SixDFGSFitter:
    def __init__(self, *, file_path: str, z: float):
        self.file = file_path
        self.z = z
        a = fits.open(self.file)
        self._input_wave = a[1].data['WAVE']
        self._input_flux = a[1].data['FLUX']
        self._output_wave = np.empty((len(self._input_wave),))
        self._output_flux = np.empty((len(self._input_flux),))

    @property
    def input_spectrum(self):
        return np.array([self._input_wave, self._input_flux])

    @property
    def output_spectrum(self):
        if len(self._output_wave) == 0:
            self.reset_output_spectrum()
        return np.array([self._output_wave, self._output_flux])

    def clean_input_spec(self, nsmooth: float = 0.5) -> Tuple[np.array, np.array]:
        clean_bool = self._input_flux > 1e-5
        rawwave = self._input_wave[clean_bool]
        rawflux = self._input_flux[clean_bool]
        rawflux = g_filt(rawflux, nsmooth)
        return rawwave, rawflux

    @staticmethod
    def normalise_input_spec(flux: np.array) -> np.array:
        return flux / np.mean(flux)

    @staticmethod
    def blueshift_spec(wave: np.array, flux: np.array, z: float) -> Tuple[np.array, np.array]:
        spec_w, spec_f = blueshifting([wave, flux], z)
        return spec_w, spec_f

    def reset_output_spectrum(self):
        wave_tmp, flux_tmp = self.clean_input_spec()
        flux_tmp = self.normalise_input_spec(flux_tmp)
        self._output_wave, self._output_flux = self.blueshift_spec(wave_tmp, flux_tmp, self.z)

    def trim_spec(self, wave_range: Tuple, start_pixel_rm: int = 0, end_pixel_rm: int = 0):
        trim_bool = (self._output_wave > wave_range[0]) & (self._output_wave < wave_range[1])
        self._output_wave = self._output_wave[trim_bool]
        if end_pixel_rm == 0:
            end_pixel_rm = len(self._output_wave)
        self._output_wave = self._output_wave[start_pixel_rm:end_pixel_rm]
        self._output_flux = self._output_flux[trim_bool][start_pixel_rm:end_pixel_rm]

    def noise_remove(self, noise_start: float, noise_end: float):
        self._output_wave, self._output_flux = noise_to_linear([self._output_wave, self._output_flux],
                                                               [noise_start, noise_end])

    def flux_use_CFT(self, nsmooth: float = 100):
        self._output_flux -= continuum_fitting(self._output_wave, self._output_flux, smooth_val=nsmooth)

    @property
    def err(self):
        return np.ones_like(self._output_flux)

    @staticmethod
    def save_result(spec_id: int, results: list, save_properties: list = None, save_error: bool = False,
                    save_path: str = ""):
        """
        Call function to write to file or print.

        Parameters:

        spec_id: int
            6dfgs id for the row entry

        results: list
            A from pyqsofit line_result_output

        save_properties: list, optional
            If None, generates header with fwhm, sigma, skew, ew, peak, area
            A list of the six measured properties can be given to selectively save those values only

        save_error: bool
            If True, saves the measured values of lines. If False, saves the errors only.

        save_path: str, optional
            If provided, will create, if file does not exist, and append results.
        """
        r_index = 1 if save_error else 0
        tmp_results = [i[r_index] for i in results]
        if save_properties is None:
            save_properties = ['fwhm', 'sigma', 'skew', 'ew', 'peak', 'area']

        tmp_results = [[i[xkey] for xkey in save_properties if xkey in i] for i in tmp_results]

        to_save = np.round(tmp_results, 3).flatten()
        to_save = "\t".join(map(str, to_save))
        if save_path != "":
            with open(save_path, 'a') as save_file:
                save_file.seek(0, 2)

                if save_file.tell() == 0:
                    heading = "\t".join(map(str, save_properties*len(results)))
                    save_file.write(f'spec_id\t{heading}\n')

                save_file.write(f'{int(spec_id)}\t{to_save}\n')
        print(f'{int(spec_id)}\t{to_save}\n')


file_data = np.genfromtxt(os.getcwd() + '/test_spectra_list.csv', delimiter=',')[1:].T

# Mode A - Fitting selected spectrum
name = np.asarray([
22878
])
spec_z = np.asarray([file_data[2][id_finder(file_data[0], i)] for i in name])

# Part of Mode A, when redshifts require specific tweaks
# spec_z = [0.7681]
# spec_z = np.asarray([4999*(1+spec_z[0])/OR - 1])

# Mode B - Fitting all in the specified file
# name = file_data[0]
# spec_z = file_data[2]

for xname, xz in zip(name, spec_z):
    loc = os.getcwd() + '/test_spectra/'
    a_path = loc + ('0000000' + str(int(xname)))[-7:] + '_1d.fits'
    a_spec = SixDFGSFitter(file_path=a_path, z=xz)
    a_spec.reset_output_spectrum()
    a_spec.trim_spec((4000, 5500))

    q = QSOFit(*a_spec.output_spectrum, a_spec.err, z=0, path=path1)

    start = timeit.default_timer()
    q.Fit(decomposition_host=True, PL=True, poly=True, Fe_uv_op=False, BC=True,
          CFT_smooth=75, CFT=False, MC=False, MC_conti=False,
          nsmooth=1, deredden=False, reject_badpix=False, wave_range=None, redshift=False,
          initial_guess=None, n_trails=5, linefit=True, save_result=False, plot_fig=True, save_fig=False,
          plot_line_name=True, plot_legend=True, dustmap_path=None, save_fig_path=None,
          save_fits_path=None, save_fits_name=None)

    plt.axvline(6885 / (1 + xz))
    plt.axvline(5577 / (1 + xz))

    # plt.close()
    end = timeit.default_timer()
    print('Fitting finished in : ' + str(np.round(end - start)) + 's')

    print(f"Noise Area = {q.average_noise_area}")
    # q.full_result_print()

    a = q.line_result_output('Hb_br')
    b = q.line_result_output('Hb_na')
    c = q.line_result_output('OIII5007c')

    d = q.line_result_output('Ha_br')
    e = q.line_result_output('NII6549c')

    a_spec.save_result(xname, [a, b, c, d, e], save_properties=['fwhm', 'skew', 'peak', 'area'],
                       save_path="", save_error=False)
