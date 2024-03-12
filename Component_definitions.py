import os
import warnings
import numpy as np
from astropy.io import fits

warnings.filterwarnings("ignore")

# Narrow emission lines fitting for Sy2
# The format for these inputs are
# (Mean of Gaussian, Name of fitting region, Boundaries for fitting region,
#               Name of fitted component, number of gaussian for this component, FWHM definition, Max velocity offset,
#               Skew definition, Flux definition)
# see details for individual components in the notes
newdata = np.rec.array([(4861.33, 'O3', 4500, 6700, 'Hb_na', 1, '[1e-4, 0.001]', 1e-3, '[0]', '0.002'),
                        (4861.33, 'O3', 4500, 6700, 'Hb_br', 1, '[0.0017, 0.01]', 5e-3, '[-10, 10]', '0.02'),
                        (5006.843, 'O3', 4500, 6700, 'OIII5007c', 1, '[1e-4, 0.0017]', 1e-3, '[0]', '0.003'),
                        (4958.91, 'O3', 4500, 6700, 'OIII4959c', 1, 'OIII5007c*1', 1e-3, '[0]', 'OIII5007c*0.33'),
                        # (4958.91, 'O3', 4500, 6700, 'OIII4959d', 1, 'OIII5007d*1', 1e-3, '[0]', 'OIII5007d*0.33'),
                        # (5006.843, 'O3', 4500, 6700, 'OIII5007d', 1, '[0.0017, 0.005]', 5e-3, '[0]', '0.005'),
                        # (6446, 'O3', 4500, 6700, 'spka', 1, '[1e-5, 0.0005]', 1e-3, '0', '0.002'),
                        # (6500, 'O3', 4500, 6700, 'spka', 1, '[1e-5, 0.0005]', 1e-3, '0', '-0.002'),
                        # (4875, 'O3', 4500, 6700, 'spkb', 1, '[1e-5, 0.0005]', 1e-3, '0', '-0.002'),

                        (6562.82, 'O3', 4500, 6700, 'Ha_na', 1, 'Hb_na*1', 1e-3, '[0]', '0.002'),
                        (6583.46, 'O3', 4500, 6700, 'NII6585c', 1, '[1e-4, 0.001]', 1e-3, '[0]', '0.002'),
                        (6548.05, 'O3', 4500, 6700, 'NII6549c', 1, 'NII6585c*1', 1e-3, '[0]', 'NII6585c*0.33'),
                        # (6583.46, 'O3', 4500, 6700, 'NII6585d', 1, '[0.0017, 0.005]', 1e-3, '[0]', '0.005'),
                        # (6548.05, 'O3', 4500, 6700, 'NII6549d', 1, 'NII6585d*1', 1e-3, '[0]', 'NII6585d*0.33'),
                        # (6716.44, 'O3', 4500, 6700, 'SIIL', 1, 'SIIR*1', 1e-3, '[0]', '0.002'),
                        # (6730.81, 'O3', 4500, 6700, 'SIIR', 1, '[1e-5, 0.001]', 1e-3, '[0]', '0.002'),

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
hdr['ngauss'] = 'Number of Gaussians for the line. Not all functions supported for >1'
hdr['sigval'] = 'Sigma information in strings, either a single value, [min, max], or name of line to tie to'
hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr['iniskw'] = 'Initial guess of lineskew'
hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'
# ------save line info-----------
hdu = fits.BinTableHDU(data=newdata, header=hdr, name='data')

# path definitions to save the component definitions file
path1 = os.getcwd() + '/PyQSOFit/'  # the path of the source code file and qsopar.fits

hdu.writeto(path1 + 'qsopar2.fits', overwrite=True)     # dont change this

