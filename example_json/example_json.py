import glob, os, sys, timeit
from PyQSO.rebin_spec import rebin_spec
import matplotlib
from PyQSO.PyQSOFit_SBL import *
import matplotlib.pyplot as plt
import warnings
import json

warnings.filterwarnings("ignore")

newdata = np.rec.array([  # (6564.61, 'Ha', 6400., 6800., 'Ha_br', 3, 5e-3, 0.004, 0.05, 0.015, 0, 0.05),
    # (6564.61, 'Ha', 6400., 6800., 'Ha_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 0, 0.002),
    # (6549.85, 'Ha', 6400., 6800., 'NII6549', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 0, 0.001),
    # (6585.28, 'Ha', 6400., 6800., 'NII6585', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 0, 0.003),
    # (6718.29, 'Ha', 6400., 6800., 'SII6718', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 0, 0.001),
    # (6732.67, 'Ha', 6400., 6800., 'SII6732', 1, 1e-3, 2.3e-4, 0.0017, 5e-3, 0, 0.001),

    # (4862.68, 'Hb', 4640., 5100., 'Hb_br', 1, 5e-3, 0.004, 0.05, 0.01, 0, 0.01),
    # (4862.68, 'Hb', 4640., 5100., 'Hb_na', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 0, 0.002),
    # (4960.30, 'Hb', 4640., 5100., 'OIII4959c', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 0, 0.002),
    # (5008.24, 'Hb', 4640., 5100., 'OIII5007c', 1, 1e-3, 2.3e-4, 0.0017, 0.01, 0, 0.004),
    # (4960.30,'Hb',4640.,5100.,'OIII4959w',1,3e-3,2.3e-4,0.004,0.01,0,0.001),
    # (5008.24,'Hb',4640.,5100.,'OIII5007w',1,3e-3,2.3e-4,0.004,0.01,0,0.002),
    # (4687.02,'Hb',4640.,5100.,'HeII4687_br',1,5e-3,0.004,0.05,0.005,0,0.001),
    # (4687.02,'Hb',4640.,5100.,'HeII4687_na',1,1e-3,2.3e-4,0.0017,0.005,0,0.001),

    # (3934.78,'CaII',3900.,3960.,'CaII3934',2,1e-3,3.333e-4,0.0017,0.01,0,-0.001),

    # (3728.48,'OII',3650.,3800.,'OII3728',1,1e-3,3.333e-4,0.0017,0.01,0,0.001),

    # (3426.84,'NeV',3380.,3480.,'NeV3426',1,1e-3,3.333e-4,0.0017,0.01,0,0.001),
    # (3426.84,'NeV',3380.,3480.,'NeV3426_br',1,5e-3,0.0025,0.02,0.01,0,0.001),

    (2798.75, 'MgII', 2700., 2900., 'MgII_br', 1, 5e-3, 0.004, 0.05, 0.0017, 0, 0.05),
    # (2798.75, 'MgII', 2700., 2900., 'MgII_na', 2, 1e-3, 5e-4, 0.0017, 0.01, 0, 0.002),

    (1908.73, 'CIII', 1700., 1970., 'CIII_br', 1, 5e-3, 0.004, 0.05, 0.015, 0, 0.01),
    # (1908.73, 'CIII', 1700., 1970., 'CIII_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 0, 0.002),
    # (1892.03,'CIII',1700.,1970.,'SiIII1892',1,2e-3,0.001,0.015,0.003,0,0.005),
    # (1857.40,'CIII',1700.,1970.,'AlIII1857',1,2e-3,0.001,0.015,0.003,0,0.005),
    # (1816.98,'CIII',1700.,1970.,'SiII1816',1,2e-3,0.001,0.015,0.01,0,0.0002),
    # (1786.7,'CIII',1700.,1970.,'FeII1787',1,2e-3,0.001,0.015,0.01,0,0.0002),
    # (1750.26,'CIII',1700.,1970.,'NIII1750',1,2e-3,0.001,0.015,0.01,0,0.001),
    # (1718.55,'CIII',1700.,1900.,'NIV1718',1,2e-3,0.001,0.015,0.01,0,0.001),

    (1549.06, 'CIV', 1400., 1700., 'CIV_br', 1, 5e-3, 0.004, 0.05, 0.015, 0, 0.05),
    (1549.06, 'CIV', 1400., 1700., 'CIV_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 0, 0.002),
    (1640.42, 'CIV', 1400., 1700., 'HeII1640', 1, 1e-3, 5e-4, 0.0017, 0.008, 0, 0.002),
    # (1663.48,'CIV',1500.,1700.,'OIII1663',1,1e-3,5e-4,0.0017,0.008,0,0.002),
    (1640.42, 'CIV', 1400., 1700., 'HeII1640_br', 1, 5e-3, 0.0025, 0.02, 0.008, 0, 0.002),
    # (1663.48,'CIV',1500.,1700.,'OIII1663_br',1,5e-3,0.0025,0.02,0.008,0,0.002),

    # (1436.06, 'CIV', 1400., 1700., 'CIV_abs', 1, 5e-3, 0.004, 0.05, 0.015, 0, -0.05),

    (1402.06, 'SiIV', 1290., 1450., 'SiIV_OIV1', 1, 5e-3, 0.002, 0.05, 0.015, 0, 0.05),
    # (1396.76, 'SiIV', 1290., 1450., 'SiIV_OIV2', 1, 5e-3, 0.002, 0.05, 0.015, 0, 0.05),
    # (1335.30, 'SiIV', 1290., 1450., 'CII1335', 1, 2e-3, 0.001, 0.015, 0.01, 0, 0.001),
    # (1304.35,'SiIV',1290.,1450.,'OI1304',1,2e-3,0.001,0.015,0.01,0,0.001),

    (1215.67, 'Lya', 1150., 1290., 'Lya_br', 1, 5e-3, 0.004, 0.05, 0.02, 0, 0.05),
    (1215.67, 'Lya', 1150., 1290., 'Lya_na', 1, 1e-3, 5e-4, 0.0017, 0.01, 0, 0.002)
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

path = os.getcwd() + '/PyQSO/example_json/'
path1 = os.getcwd() + '/PyQSO/'  # the path of the source code file and qsopar.fits
path2 = path + '/'  # path of fitting results
path3 = path + '/'  # path of figure

hdu.writeto(path1 + 'qsopar2.fits', overwrite=True)

# Load in your spectrum however you like, define the redshift, wavelength and flux
# Note that this code runs for SDSS so flux has to be in 1e-17.
# This means for some spectrum you need to scale it by * 1e17.
# In my example, it is already scaled
# err array is optional. If you have the variance spectrum, load it in and scale it accordingly.
# Else an empty array like mine is fine. (MC=True in q.Fit if you want to fit errors)
with open(path + 'sample_spectrum_as_json.json') as f:
    spec_data = json.load(f)
z = 2
data_flux = spec_data['intensity']
wave = []
err = []
flux = []

for ff in range(len(data_flux)):
    if data_flux[ff] is not None:
        flux.append(spec_data['intensity'][ff])
        wave.append(spec_data['wavelength'][ff])
        err.append(spec_data['variance'][ff])

wave = np.asarray(wave)
flux = np.asarray(flux)
err = np.asarray(err)
# These steps here is how you turn any spectrum into a sdss spectrum
# rebin_spec is a very handy function

wavenew = np.arange(wave[0], wave[-1], 10 ** 1e-4)
flux = rebin_spec(wave, flux, wavenew)
err = rebin_spec(wave, err, wavenew)
lam = wavenew

# note path here has to be where you saved qsopar.fits and PyQSO
q = QSOFit(lam[1500:2500], flux[1500:2500], err[1500:2500], z, path=path1)

start = timeit.default_timer()
q.Fit(name=None, nsmooth=1, and_or_mask=False, deredden=False, reject_badpix=False, wave_range=None,
      wave_mask=None, decomposition_host=False, Mi=None, npca_gal=5, npca_qso=20,
      Fe_uv_op=True, poly=True, BC=False, rej_abs=False, initial_guess=None, MC=False,
      n_trails=5, linefit=True, save_result=True, plot_fig=True, save_fig=True, plot_line_name=True, plot_legend=True,
      dustmap_path=None, save_fig_path=path3, save_fits_path=path2, save_fits_name=None)

end = timeit.default_timer()
print('Fitting finished in : ' + str(np.round(end - start)) + 's')
