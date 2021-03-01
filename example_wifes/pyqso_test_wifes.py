import glob, os, sys, timeit
from Spectra_handling.AllSpectrum import *
from Spectra_handling.Spectrum_utls import *
from rebin_spec import rebin_spec
import matplotlib
from PyQSO.PyQSOFit_SBL import *
import matplotlib.pyplot as plt
import warnings

# Ignore this file and this version
warnings.filterwarnings("ignore")

path = 'F:/PyQSOtests/'

# These are the default parameters for Ha an Hb complex. They work really well so i kept them as is
# If you need MgII or CIII or example, you will need to add a similar line, or copy the line from the example code
# The starting parameters are a bit tricky for me to explain, so I would just refer to example code
# In the end, the code is meant to smartly fit the lines, so the starting parameter will just lower the room for error
# and fitting time
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

# Up until here is to create the initial parameter file to tell the code what to fit.
# You can techinically run this section separately just like in the example code with jupiter notebook


# The paths here can be redefined as you wish. I just have mine setup like this for my convenience.
path1 = path                          # the path of PyQSO and qsopar.fits
path2 = path + 'results/'             # path of fitting results
path3 = path + 'QA_other/'            # path of figure
path4 = 'F:/CLagn compilation/My CLAGN data/J1949137/'          # path of the wifes files

# I still haven't combine the red and blues, so I am carrying 2 files and using my code to open them WiFeS()
# You can open your r+b files if you know how to with astropy.io.fits.open() for example
spec_data = Wifes(path4 + 'sp_35b.fits', path4 + 'sp_35r.fits')
z = 0.081129
wave = spec_data.spectrum[0]
flux = spec_data.spectrum[1] * 1e17
# Define the redshift, define the wavelength array and the flux array. Note that this code runs for SDSS so flux
# has to be in 1e-17. This means for WiFeS spectrum you need to scale it by * 1e17.

# These steps here is how you turn our wifes spectrum into a sdss spectrum
# rebin_spec is a very handy function, I will send it to you as well
# the 'err' array means nothing here
wavenew = np.arange(wave[0], wave[-1], 10 ** 1e-4)
flux = rebin_spec(wave, flux, wavenew)
err = np.ones_like(flux)
lam = wavenew

q = QSOFit(lam, flux, err, z, path=path1)  # note path here has to be where you saved qsopar.fits and PyQSO

start = timeit.default_timer()
'''
There are so many inputs here, I will mention the important ones. Note you turn them on/off with True/False
'decomposition_host' not important. It will attempt to fit the host galaxy and remove it, but can sometimes fail and
give errors. 

'MC' means to use the 'err' and estimate errors or not. Turning it on will make the fitting process go 4 times longer

'dustmap_path' if you want to dereden your spectra. But I have to turn it off because I am missing the files. Not sure
where to find 

'save_fig_path' put where you want the plots to go here

'save_fits_path' put where you want the results to go here. But the output results is pretty confusing and hard to use
'''
q.Fit(name=None, nsmooth=1, and_or_mask=False, deredden=False, reject_badpix=False, wave_range=None,
      wave_mask=None, decomposition_host=False, Mi=None, npca_gal=5, npca_qso=20,
      Fe_uv_op=True, poly=True, BC=False, rej_abs=False, initial_guess=None, MC=False,
      n_trails=5, linefit=True, tie_lambda=True, tie_width=True, tie_flux_1=True, tie_flux_2=True,
      save_result=True, plot_fig=True, save_fig=True, plot_line_name=True, plot_legend=True,
      dustmap_path=None, save_fig_path=path3, save_fits_path=path2, save_fits_name=None)

end = timeit.default_timer()
print('Fitting finished in : ' + str(np.round(end - start)) + 's')
