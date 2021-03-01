import glob, os, sys, timeit
from Spectra_handling.AllSpectrum import *
from Spectra_handling.Spectrum_utls import *
from rebin_spec import rebin_spec
import matplotlib
from PyQSO.PyQSOFit import *
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skewnorm


rest_norm_range = np.asarray([5200, 6200])
warnings.filterwarnings("ignore")

path = 'F:/PyQSOtests/'

newdata = np.rec.array([(6564.61, 'Ha', 6400., 6800., 'Ha_br', 1, 5e-3, 0.004, 0.05, 0.015, 0, 0.05),
                        (6564.61, 'Ha', 6400., 6800., 'Ha_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 0, 0.002),
                        (6549.85, 'Ha', 6400., 6800., 'NII6549', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 0, 0.001),
                        (6585.28, 'Ha', 6400., 6800., 'NII6585', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 0, 0.003),
                        (6718.29, 'Ha', 6400., 6800., 'SII6718', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 0, 0.001),
                        (6732.67, 'Ha', 6400., 6800., 'SII6732', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 0, 0.001),

                        (4862.68, 'Hb', 4640., 5100., 'Hb_br', 1, 5e-3, 0.004, 0.05, 0.01, 0, 0.01),
                        (4862.68, 'Hb', 4640., 5100., 'Hb_na', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 0, 0.002),
                        (4960.30, 'Hb', 4640., 5100., 'OIII4959c', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 0, 0.002),
                        (5008.24, 'Hb', 4640., 5100., 'OIII5007c', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 0, 0.004),
                        ],
                       formats='float32,a20,float32,float32,a20,float32,float32,float32,float32,'
                               'float32,float32,float32,',
                       names='lambda,compname,minwav,maxwav,linename,ngauss,inisig,minsig,'
                             'maxsig,voff,iniskw,fvalue')
# ------header-----------------
hdr = fits.Header()
hdr['lambda'] = 'Vacuum Wavelength in Ang'
hdr['minwav'] = 'Lower complex fitting wavelength range'
hdr['maxwav'] = 'Upper complex fitting wavelength range'
hdr['ngauss'] = 'Number of Gaussians for the line'
hdr['inisig'] = 'Initial guess of linesigma [in lnlambda]'
hdr['minsig'] = 'Lower range of line sigma [lnlambda]'
hdr['maxsig'] = 'Upper range of line sigma [lnlambda]'
hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr['iniskw'] = 'Initial guess of lineskew'
hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'
# ------save line info-----------
hdu = fits.BinTableHDU(data=newdata, header=hdr, name='data')
hdu.writeto(path + 'qsopar2.fits', overwrite=True)


xfile = 13456
path1 = path  # the path of the source code file and qsopar.fits
path2 = path + 'results/' + str(xfile)  # path of fitting results
path3 = path + 'QA_other/' + str(xfile)  # path of figure
path4 = 'F:/CLagn compilation/My CLAGN data/J1949137/'

spec_data = Wifes(path4 + 'sp_35b.fits', path4 + 'sp_35r.fits')
# err_data = Wifes(path4 + 'ep_35b.fits', path4 + 'ep_35r.fits')

z = 0.017303
wave = spec_data.spectrum[0]
flux = spec_data.spectrum[1]
# err = err_data.spectrum[1] * 1e17
err = spec_data.error[1]
norm_range = rest_norm_range * (1 + z)
xnorm = np.mean(range_select([wave, flux], norm_range)[1])
wave = wave[~np.isnan(flux)]
err = err[~np.isnan(flux)]/xnorm
flux = flux[~np.isnan(flux)]/xnorm
wavenew = np.arange(wave[0], wave[-1], 10 ** 1e-4)
flux = rebin_spec(wave, flux, wavenew)

# err = 1 / g_filt((np.sqrt(rebin_spec(wave, err, wavenew)) + 1), 5)
lam = wavenew

q = QSOFit(lam, flux, np.ones_like(lam), z, path=path1)

start = timeit.default_timer()
# do the fitting
q.Fit(name=None, nsmooth=1, and_or_mask=False, deredden=False, reject_badpix=False, wave_range=None,
      wave_mask=None, decomposition_host=False, Mi=None, npca_gal=5, npca_qso=20,
      Fe_uv_op=True, poly=True, BC=False, rej_abs=False, initial_guess=None, MC=False,
      n_trails=5, linefit=True, tie_lambda=True, tie_width=True, tie_flux_1=True, tie_flux_2=True,
      save_result=True, plot_fig=True, save_fig=True, plot_line_name=True, plot_legend=True,
      dustmap_path=None, save_fig_path=path3, save_fits_path=path2, save_fits_name=None)

end = timeit.default_timer()
print('Fitting finished in : ' + str(np.round(end - start)) + 's')

