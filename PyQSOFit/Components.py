from typing import Tuple
from astropy.io import fits
import numpy as np


class LineDef:
    """
            Class to create line profile to add to Section for fitting

            Parameters:
            -----------
            l_name: str
                Name of the line. Must be unique.

            l_center: float
                The rest wavelength for the line.

            fwhm: tuple of two elements in km/s
                  One value input as (500,) will fix the fwhm of the modelling profile at 500.
                  Two values input as (1200, 7000) will have the fwhm range from 1200, 7000.
                  Values are not exact due to conversion to a log scale during modelling.

            profile_link: str, optional
                 A string that tells PyQSOFit which line the profile of this one is linked to.
                 For example, definition for OIII4959, will have 'OIII5007*1' as profile_link.
                 This overwrites fwhm, meaning fwhm can be ignored.

            skew: tuple of two elements
                One value input as (2.3,) will fix the skew of the modelling profile at 2.3.
                Two values input as (-10, 10) will have the skew range from -10 to 10.

            gamma: str, optional
                A string that controls the Voigt profile for the modelling.
                By default, it is an empty string and will turn off Voigt profile and use Gaussian profile instead.

                - If specified 'On', it will have a free gamma value and uses Voigt profile.

                - If the string has 'f0.53', it will fix the gamma at 0.53.

            voffset: float, in km/s
                The range of velocity offset allowed from the centre wavelength.

            vmode: str, optional
                Controls how the voffset range is defined.

                - Default is '', and means the offset is free on both ends.

                - '+' restricts the range such that the velocity offset can only increase in wavelength.

                - '-' restricts the range such that the velocity offset can only decrease in wavelength.

                - '.' sets the velocity offset at the specified voffset value.

            scale: float
                A value that controls the scale of the modelling profile.

            flux_link: str, optional
                Tells PyQSOFit which line the flux of this one is linked to.
                For example, definition for OIII4959, will have 'OIII5007*0.33' as flux_link.
                This overwrites scale, meaning scale can be ignored.

            default_bel: bool, optional.
                If True will set the fwhm to (1200, 7000), voffset to 5e-3, and skew to (-10, 10).

            default_nel. bool, optional.
                If True will set the sig to (50, 1200), voffset to 1e-3, and skew to (0,).

    """

    def __init__(self, fwhm: Tuple = (100, 1000), profile_link: str = "", skew: Tuple = (-10, 10), gamma: str = "",
                 vmode: str = "", voffset: float = 0.0, scale: float = 0.005, flux_link: str = "",
                 default_bel: bool = False, default_nel: bool = False, *, l_name: str = "", l_center: float = 0.0):
        self.l_name = l_name
        self.l_center = l_center
        self.fwhm = fwhm
        self._sig = (0, 0.01)
        self._voffset = voffset
        self._skew = skew
        self.gamma = gamma
        self.vmode = vmode
        self.plink = profile_link
        self._scale = scale
        self.flux_link = flux_link

        if default_nel:
            self.fwhm = (50, 1200)
            self._voffset = 300
            self._skew = (0,)
        if default_bel:
            self.fwhm = (1200, 7000)
            self._voffset = 1500
            self._skew = (-10, 10)

    @property
    def sig(self):
        if len(self.fwhm) == 2:
            self._sig = (self.fwhm_to_sig(self.fwhm[0]), self.fwhm_to_sig(self.fwhm[1]))
            return f"[{self._sig[0]}, {self._sig[1]}]"
        elif len(self._sig) == 1:
            self._sig = (self.fwhm_to_sig(self.fwhm[0]),)
            return f"[{self._sig[0]}]"
        else:
            print("Incorrect number of inputs for line fwhm. Should be 1 (exact) or 2 (range) values")
            print("Default to narrow line")
            return "[1e-5, 0.0017]"

    @property
    def scale(self):
        if self.flux_link == "":
            return str(self._scale)
        elif "*" in self.flux_link:
            return self.flux_link
        else:
            print("Incorrect inputs for line flux. Should be 1 float or 1 Line*multiplier (OIII*3)")
            print("Default to 0.005")
            return "0.005"

    @property
    def skew(self):
        if len(self._skew) == 2:
            return f"{self._skew[0]}, {self._skew[1]}"
        elif len(self._skew) == 1:
            return f"{self._skew[0]}"
        else:
            print("Incorrect number of inputs for line skew. Should be 1 (exact) or 2 (range) values")
            print("Default to no skew")
            return "0"

    @property
    def profile(self):
        l_gamma = ""
        l_voff = ""
        if self.gamma == "On":
            l_gamma = f"g[];"
        elif "f" in self.gamma:
            l_gamma = self.gamma[1:]
        if l_voff != "":
            l_voff = f"v[{self.vmode}];"

        return f"{l_voff}{l_gamma}s[{self.skew}]"

    @property
    def sig_info(self):
        if self.plink == "":
            return self.sig
        else:
            return self.plink

    @staticmethod
    def fwhm_to_sig(xfwhm: float):
        c = 299792.458  # km/s
        return np.round(np.sqrt(np.log(1 + ((xfwhm / c / 2.355) ** 2))), 5)

    @property
    def voffset(self):
        c = 299792.458  # km/s
        wave_offset = self._voffset / 3e5 * self.l_center
        log_offset = np.log(self.l_center + wave_offset) - np.log(self.l_center)
        return log_offset


class Section:
    """
    Class to create sections to add lines into for fitting

    Call line_add to add a single line

    Call add_lines to add a list of lines
    """

    def __init__(self, *, section_name: str = "", start_range: float = 0.0, end_range: float = 1.0):
        self.section_name = section_name
        self.start_range = start_range
        self.end_range = end_range
        self._lines = []
        self.rarray = None

    def line_add(self, new_line: LineDef):
        if self.start_range <= new_line.l_center <= self.end_range:
            self._lines.append(new_line)

    def add_lines(self, lines: list):
        for xline in lines:
            self.line_add(xline)

    @property
    def lines(self):
        if self.rarray is None:
            xtuples = [self.linedef_to_tuple(xline) for xline in self._lines]
            xdtype = np.dtype([('lambda', 'float32'), ('compname', 'U20'), ('minwav', 'float32'), ('maxwav', 'float32'),
                               ('linename', 'U20'), ('sigval', 'U20'), ('voff', 'float32'), ('iniskw', 'U20'),
                               ('fvalue', 'U20')])
            self.rarray = np.rec.array(xtuples, dtype=xdtype)
        return self.rarray

    def linedef_to_tuple(self, xline: LineDef) -> Tuple:
        return (xline.l_center, self.section_name, self.start_range, self.end_range, xline.l_name,
                xline.sig_info, xline.voffset, xline.profile, xline.scale)

    @staticmethod
    def hdu_generate() -> fits.Header:
        hdr = fits.Header()
        hdr['lambda'] = 'Vacuum Wavelength in Ang'
        hdr['minwav'] = 'Lower complex fitting wavelength range'
        hdr['maxwav'] = 'Upper complex fitting wavelength range'
        hdr['ngauss'] = 'Number of Gaussians for the line. Not all functions supported for >1'
        hdr['sigval'] = 'Sigma information in strings, either a single value, [min, max], or name of line to tie to'
        hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
        hdr['iniskw'] = 'Initial guess of lineskew'
        hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'
        return hdr