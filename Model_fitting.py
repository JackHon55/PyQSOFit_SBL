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
fig_path = os.getcwd() + '/fitting_plots/'

file_data = SixDFGS(os.getcwd() + '/test_spectra/g1639001-012857.fits')     # open file using supported 6dfgs function

f = file_data.pspec.rest()[1]
w = file_data.pspec.rest()[0]
z = file_data.redshift

f = f[(w > 4000) & (w < 5500)][:]
w = w[(w > 4000) & (w < 5500)][:]

# w, f = noise_to_linear([w, f], [6150, 6400]) # Function to mask manually
# f, fs = f - continuum_fitting(w, f, 75), False   # Uncomment this to use the continuum fitting tool for tricky spectra
err = np.ones_like(f)  # These spectra files do not have variance provided. The dominant noise is calculated post-fitting inside PyQSOFit anyways

q = QSOFit(w, f, err, 0, path=path1)

start = timeit.default_timer()
q.Fit(name=None, nsmooth=1, deredden=False, reject_badpix=False, wave_range=None, redshift=False,
      Fe_uv_op=True, poly=True, CFT=False, CFT_smooth=75, BC=True, initial_guess=None, MC=False, MC_conti=False,
      decomposition_host=True, PL=True, n_trails=15, linefit=True, save_result=False, plot_fig=True, save_fig=True,
      plot_line_name=True, plot_legend=True, dustmap_path=None, save_fig_path=fig_path, save_fits_path=None,
      save_fits_name=None)
plt.show()
plt.axvline(6885 / (1 + z))  # Indicates B-band telluric line
plt.axvline(5577 / (1 + z))  # Indicates the 5577 skyline

# plt.close()
end = timeit.default_timer()
print('Fitting finished in : ' + str(np.round(end - start)) + 's')

moe = 0
a = q.line_result_output('Hb_na')[moe]
b = q.line_result_output('Hb_br')[moe]
ab = q.line_result_output('Hb_')[moe]
c = q.line_result_output('OIII5007')[moe]
if w[-1] > HA:
    d = q.line_result_output('Ha_na')[moe]
    e = q.line_result_output('NII6585c')[moe]
else:
    d = np.zeros(6)
    e = np.copy(c)

# Want only the fwhm, peak position, and flux (EW calculation is broken, and skew is not relevant)
a = np.asarray([a[0], a[4], a[5]])
ab = np.asarray([ab[0], ab[4], ab[5]])
c = np.asarray([c[0], c[4], c[5]])
d = np.asarray([d[0], d[4], d[5]])
