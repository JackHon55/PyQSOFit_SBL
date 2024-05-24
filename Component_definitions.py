import os
import warnings
import numpy as np
from astropy.io import fits
from PyQSOFit.PyQSOFit_SVL import LineDef, Section

warnings.filterwarnings("ignore")

hbo3_section = Section(section_name='O3', start_range=4500, end_range=5300)
hanl_section = Section(section_name='HA', start_range=6000, end_range=7000)

line_hb_br1 = LineDef(l_name='Hb_br1', l_center=4861.33, scale=0.005, default_bel=True)
line_hb_na = LineDef(l_name='Hb_na', l_center=4861.33, scale=0.002, fwhm=(50, 700), voffset=1e-3, skew=(0,))

line_or = LineDef(l_name='OIII5007c', l_center=5006.843, scale=0.003, default_nel=True)
line_ol = LineDef(l_name='OIII4959c', l_center=4958.91, flux_link='OIII5007c*0.33', profile_link='OIII5007c*1')

hbo3_section.add_lines([line_hb_br1, line_hb_na, line_or, line_ol])

line_ha_br1 = LineDef(l_name='Ha_br1', l_center=6562.82, scale=0.005, default_bel=True)
line_ha_na = LineDef(l_name='Ha_na', l_center=6562.82, scale=0.002, fwhm=(50, 700), voffset=1e-3, skew=(0,))

line_nr = LineDef(l_name='NII6585c', l_center=6583.46, scale=0.002, default_nel=True)
line_nl = LineDef(l_name='NII6549c', l_center=6548.05, flux_link='NII6585c*0.33', profile_link='NII6585c*1')

hanl_section.add_lines([line_ha_br1, line_ha_na, line_nr, line_nl])

noise_a = LineDef(l_name='noiA', l_center=4977, scale=-0.002, fwhm=(10, 350), voffset=1e-3, skew=(0,))

newdata = np.concatenate([hbo3_section.lines, hanl_section.lines])
hdu = fits.BinTableHDU(data=newdata, header=Section.hdu_generate(), name='data')

# path definitions to save the component definitions file
path1 = os.getcwd() + '/PyQSOFit/'  # the path of the source code file and qsopar.fits
hdu.writeto(path1 + 'qsopar2.fits', overwrite=True)

