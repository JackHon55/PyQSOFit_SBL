import os
import warnings
import numpy as np
from astropy.io import fits
from PyQSOFit.Components import LineDef, Section

warnings.filterwarnings("ignore")

# create a header
hdr0 = fits.Header()
hdr0['Author'] = 'Mol the Hrafn'
primary_hdu = fits.PrimaryHDU(header=hdr0)

hbo3_section = Section(section_name='O3', start_range=4500, end_range=5300)
hanl_section = Section(section_name='HA', start_range=6000, end_range=7000)

line_hb_br1 = LineDef(l_name='Hb_br1', l_center=4861.33, scale=0.005, fwhm=(1200, 5000), default_bel=True)
line_hb_br2 = LineDef(l_name='Hb_br2', l_center=4861.33, scale=0.005, fwhm=(1200, 5000), voffset=1500, vmode="-")
line_hb_na = LineDef(l_name='Hb_na', l_center=4861.33, scale=0.002, fwhm=(50, 700), voffset=300, skew=(0,))

line_or = LineDef(l_name='OIII5007c', l_center=5006.843, scale=0.003, default_nel=True)
line_ol = LineDef(l_name='OIII4959c', l_center=4958.91, flux_link='OIII5007c*0.33', profile_link='OIII5007c*1')

hbo3_section.add_lines([line_hb_br1, line_hb_na, line_or, line_ol])

line_ha_br1 = LineDef(l_name='Ha_br1', l_center=6562.82, scale=0.005, default_bel=True)
line_ha_na = LineDef(l_name='Ha_na', l_center=6562.82, scale=0.002, fwhm=(50, 700), voffset=300, skew=(0,))

line_nr = LineDef(l_name='NII6585c', l_center=6583.46, scale=0.002, default_nel=True)
line_nl = LineDef(l_name='NII6549c', l_center=6548.05, flux_link='NII6585c*0.33', profile_link='NII6585c*1')

hanl_section.add_lines([line_ha_br1, line_ha_na, line_nr, line_nl])

noise_a = LineDef(l_name='noiA', l_center=4977, scale=-0.002, fwhm=(10, 350), voffset=300, skew=(0,))

newdata = np.concatenate([hbo3_section.lines, hanl_section.lines])
hdu1 = fits.BinTableHDU(data=newdata, header=Section.hdu_generate(), name='line_priors')

conti_windows = np.rec.array([
    (1150., 1170.),
    (1275., 1290.),
    (1350., 1360.),
    (1445., 1465.),
    (1690., 1705.),
    (1770., 1810.),
    (1970., 2400.),
    (2480., 2675.),
    (2925., 3400.),
    (3775., 3832.),
    (4000., 4050.),
    (4200., 4230.),
    (4435., 4640.),
    (5100., 5535.),
    (6005., 6035.),
    (6110., 6250.),
    (6800., 7000.),
    (7160., 7180.),
    (7500., 7800.),
    (8050., 8150.), # Continuum fitting windows (to avoid emission line, etc.)  [AA]
    ],
    formats = 'float32,  float32',
    names =    'min,     max')

hdu2 = fits.BinTableHDU(data=conti_windows, name='conti_windows')

conti_priors = np.rec.array([
    ('Fe_uv_norm',  0.0,   0.0,   1000,  1), # Normalization of the MgII Fe template [flux]
    ('Fe_uv_FWHM',  3000,  1200,  5000, 1), # FWHM of the MgII Fe template [AA]
    ('Fe_uv_shift', 0.0,   -0.01, 0.01,  1), # Wavelength shift of the MgII Fe template [lnlambda]
    ('Fe_op_norm',  0.0,   0.0,   1000,  1), # Normalization of the Hbeta/Halpha Fe template [flux]
    ('Fe_op_FWHM',  3000,  1200,  5000, 1), # FWHM of the Hbeta/Halpha Fe template [AA]
    ('Fe_op_shift', 0.0,   -0.01, 0.01,  1), # Wavelength shift of the Hbeta/Halpha Fe template [lnlambda]
    ('PL_norm',     0.001,   0.0,   100,  1), # Normalization of the power-law (PL) continuum f_lambda = (lambda/3000)^-alpha
    ('PL_slope',    0,  -5.0,  3.0,   1), # Slope of the power-law (PL) continuum
    ('Balmer_norm', 0.0,   0.0,   1000,  1), # Normalization of the Balmer continuum at < 3646 AA [flux] (Dietrich et al. 2002)
    ('Balmer_Te',   15000, 10000, 50000, 1), # Te of the Balmer continuum at < 3646 AA [K?]
    ('Balmer_Tau',  0.5,   0.1,   2.0,   1), # Tau of the Balmer continuum at < 3646 AA
    ('conti_a_0',   0.0,   None,  None,  1), # 1st coefficient of the polynomial continuum
    ('conti_a_1',   0.0,   None,  None,  1), # 2nd coefficient of the polynomial continuum
    ('conti_a_2',   0.0,   None,  None,  1), # 3rd coefficient of the polynomial continuum
    # Note: The min/max bounds on the conti_a_0 coefficients are ignored by the code,
    # so they can be determined automatically for numerical stability.
    ],

    formats = 'a20,  float32, float32, float32, int32',
    names = 'parname, initial,   min,     max,     vary')

hdr3 = fits.Header()
hdr3['ini'] = 'Initial guess of line scale [flux]'
hdr3['min'] = 'FWHM of the MgII Fe template'
hdr3['max'] = 'Wavelength shift of the MgII Fe template'

hdr3['vary'] = 'Whether or not to vary the parameter (set to 0 to fix the continuum parameter to initial values)'


hdu3 = fits.BinTableHDU(data=conti_priors, header=hdr3, name='conti_priors')

# path definitions to save the component definitions file
path1 = os.getcwd() + '/PyQSOFit/'  # the path of the source code file and qsopar.fits
hdu_list = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
hdu_list.writeto(path1 + 'qsopar2.fits', overwrite=True)



