import glob, os, sys, timeit
import matplotlib
from PyQSOFit_SBL.PyQSOFit_SBL import *
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skewnorm

rest_norm_range = np.asarray([5200, 6200])
warnings.filterwarnings("ignore")

# Note the differece in parameters.
newdata = np.rec.array([
    (2798.75, 'MgII', 2700., 2900., 'MgII_br', 1, '[0.004, 0.05]', 0.015, '0', '0.05'),
    (2798.75, 'MgII', 2700., 2900., 'MgII_na', 2, '[0.0005, 0.0017]', 0.01, '0', '0.002'),

    (1908.73, 'CIII', 1700., 1970., 'CIII_br', 1, '[0.004, 0.05]', 0.015, '0', '0.05'),
    # (1908.73, 'CIII', 1700., 1970., 'CIII_na', 1, '[0.0005, 0.0017]', 0.01, '0', '0.002'),
    # (1892.03,'CIII',1700.,1970.,'SiIII1892',1,'[0.001, 0.015]', 0.003, '0', '0.005'),
    # (1857.40,'CIII',1700.,1970.,'AlIII1857',1,'[0.001, 0.015]', 0.003, '0', '0.005'),
    # (1816.98,'CIII',1700.,1970.,'SiII1816',1,'[0.001, 0.015]', 0.01, '0', '0.003'),
    # (1786.7,'CIII',1700.,1970.,'FeII1787',1,'[0.001, 0.015]', 0.01, '0', '0.003'),
    # (1750.26,'CIII',1700.,1970.,'NIII1750',1,'[0.001, 0.015]', 0.01, '0', '0.001'),
    # (1718.55,'CIII',1700.,1900.,'NIV1718',1,'[0.001, 0.015]', 0.01, '0', '0.001'),

    (1549.06, 'CIV', 1400., 1700., 'CIV_br', 1, '[0.004, 0.05]', 0.015, '0', '0.05'),
    (1549.06, 'CIV', 1400., 1700., 'CIV_na', 1, '[0.0005, 0.0017]', 0.01, '0', '0.002'),
    (1640.42, 'CIV', 1400., 1700., 'HeII1640', 1, '[0.0005, 0.0017]', 0.008, '0', '0.002'),
    # (1663.48,'CIV',1500.,1700.,'OIII1663',1,'[0.0005, 0.0017]', 0.008, '0', '0.002'),
    (1640.42, 'CIV', 1400., 1700., 'HeII1640_br', 1, '[0.0025, 0.02]', 0.008, '0', '0.002'),
    # (1663.48,'CIV',1500.,1700.,'OIII1663_br',1,'[0.0025, 0.02]', 0.008, '0', '0.002'),

    # These are extra lines to demonstrate absorption
    (1436.06, 'CIV', 1400., 1700., 'CIV_abs', 1, '[0.004, 0.05]', 0.015, '0', '-0.05'),
    (1556.06, 'CIV', 1400., 1700., 'CIV_na_r', 1, '[5e-4, 0.0017]', 0.01, '0', '-0.02'),
    (1541.06, 'CIV', 1400., 1700., 'CIV_na_b', 1, '[5e-4, 0.0017]', 0.01, '0', '-0.02'),

    # (1402.06, 'SiIV', 1290., 1450., 'SiIV_OIV1', 1, '[0.002, 0.05]', 0.015, '0', '0.05'),
    # (1396.76, 'SiIV', 1290., 1450., 'SiIV_OIV2', 1, '[0.002, 0.05]', 0.015, '0', '0.05'),
    # (1335.30, 'SiIV', 1290., 1450., 'CII1335', 1, '[0.001, 0.015]', 0.01, '0', '0.001'),
    # (1304.35,'SiIV',1290.,1450.,'OI1304',1,'[0.001, 0.015]', 0.01, '0', '0.001'),

    (1215.67, 'Lya', 1150., 1290., 'Lya_br', 1, '[0.004, 0.05]', 0.02, '0', '0.05'),
    (1215.67, 'Lya', 1150., 1290., 'Lya_na', 1, '[0.0005, 0.0017]', 0.01, '0', '0.002'),
],
    formats='float32,a20,float32,float32,a20,float32,a20,'
            'float32,a20,a20,',
    names='lambda,compname,minwav,maxwav,linename,ngauss,sigval,'
          'voff,iniskw,fvalue')

# ------header-----------------
hdr = fits.Header()
hdr['lambda'] = 'Vacuum Wavelength in Ang'
hdr['minwav'] = 'Lower complex fitting wavelength range'
hdr['maxwav'] = 'Upper complex fitting wavelength range'
hdr['ngauss'] = 'Number of Gaussians for the line'
hdr['sigval'] = 'Sigma information in strings, either a single value, [min, max], or name of line to tie to'
hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr['iniskw'] = 'Initial guess of lineskew'
hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'
# ------save line info-----------
hdu = fits.BinTableHDU(data=newdata, header=hdr, name='data')

path = os.getcwd() + '/PyQSO/example_sdss/'
path1 = os.getcwd() + '/PyQSO/'  # the path of the source code file and qsopar.fits
path2 = path + '/spec-2751-54243-0269-pole-on'  # path of fitting results
path3 = path + '/spec-2751-54243-0269-pole-on'  # path of figure

hdu.writeto(path1 + 'qsopar2.fits', overwrite=True)

data = fits.open(path + 'spec-2751-54243-0269-pole-on.fits')
lam = 10 ** data[1].data['loglam']  # OBS wavelength [A]
flux = data[1].data['flux']  # OBS flux [erg/s/cm^2/A]
err = 1. / np.sqrt(data[1].data['ivar'])  # 1 sigma error
z = data[2].data['z'][0]  # Redshift

q = QSOFit(lam, flux, err, z, path=path1)

start = timeit.default_timer()
# do the fitting
q.Fit(name=None, nsmooth=1, and_or_mask=False, deredden=False, reject_badpix=False, wave_range=None,
      wave_mask=None, decomposition_host=False, Mi=None, npca_gal=5, npca_qso=20,
      Fe_uv_op=True, poly=True, BC=False, rej_abs=False, initial_guess=None, MC=False,
      n_trails=5, linefit=True, save_result=True, plot_fig=True, save_fig=True, plot_line_name=True, plot_legend=True,
      dustmap_path=None, save_fig_path=path3, save_fits_path=path2, save_fits_name=None)

end = timeit.default_timer()
print('Fitting finished in : ' + str(np.round(end - start)) + 's')

'''
a = 0
for i, j in zip(q.line_result_name, q.line_result):
    print(a, i, j)
    a += 1
'''
