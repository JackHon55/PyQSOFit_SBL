import glob, os, sys, timeit
from PyQSO.rebin_spec import rebin_spec
import matplotlib
from PyQSO.PyQSOFit import *
import matplotlib.pyplot as plt
import warnings
from astropy.io import ascii


warnings.filterwarnings("ignore")

newdata = np.rec.array([(6564.61, 'Ha', 6400., 6800., 'Ha_br', 2, 5e-3, 0.004, 0.05, 0.015, 0, 0.05),
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

path = os.getcwd() + '/PyQSO/example_wifes/'
path1 = os.getcwd() + '/PyQSO/'  # the path of the source code file and qsopar.fits
path2 = path + '/J1949137'  # path of fitting results
path3 = path + '/J1949137'  # path of figure

hdu.writeto(path1 + 'qsopar2.fits', overwrite=True)

# Load in your spectrum however you like, define the redshift, wavelength and flux
# Note that this code runs for SDSS so flux has to be in 1e-17.
# This means for some spectrum you need to scale it by * 1e17.
# In my example, it is already scaled
# err array is optional. If you have the variance spectrum, load it in and scale it accordingly.
# Else an empty array like mine is fine. (MC=True in q.Fit if you want to fit errors)
spec_data = np.genfromtxt(path + 'J1949137.txt')[1:].T
z = 0.081129
wave = spec_data[0]
flux = spec_data[1]
err = np.ones_like(flux)

# These steps here is how you turn any spectrum into a sdss spectrum
# rebin_spec is a very handy function

wavenew = np.arange(wave[0], wave[-1], 10 ** 1e-4)
flux = rebin_spec(wave, flux, wavenew)
err = rebin_spec(wave, err, wavenew)
lam = wavenew

q = QSOFit(lam, flux, err, z, path=path1)  # note path here has to be where you saved qsopar.fits and PyQSO

start = timeit.default_timer()
q.Fit(name=None, nsmooth=1, and_or_mask=False, deredden=False, reject_badpix=False, wave_range=None,
      wave_mask=None, decomposition_host=False, Mi=None, npca_gal=5, npca_qso=20,
      Fe_uv_op=True, poly=True, BC=False, rej_abs=False, initial_guess=None, MC=False,
      n_trails=5, linefit=True, save_result=True, plot_fig=True, save_fig=True, plot_line_name=True, plot_legend=True,
      dustmap_path=None, save_fig_path=path3, save_fits_path=path2, save_fits_name=None)

end = timeit.default_timer()
print('Fitting finished in : ' + str(np.round(end - start)) + 's')
