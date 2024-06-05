import numpy as np
from typing import Tuple

if __name__ == "__main__":
    # import numpy as np
    import matplotlib.pyplot as plt
from astropy.io import fits
from Spectra_handling.Spectrum_utls import *
from coord_format import Coord
from date_format import Datefs


class Spectrum(object):
    """ The superclass with all the basic properties a spectrum should be associated with """

    def __init__(self):
        self.spectrum: np.array = None
        self.spectrum_name: str = ""
        self._p_spectrum: np.array = None
        self.ra = None
        self.dec = None
        self._time = None
        self.mjd = None
        self.coord_mode = 'DEG'

    def rebin(self, smoothing: int = 0, wave_in: np.array = None, persist: bool = True) -> np.array:
        """ Function to rebin spectrum to different wavelength array """
        xspec = self.p_spectrum if persist else self.spectrum
        if wave_in is None:
            print("Only smoothing, rebin operation requires a wave_in")
            return np.asarray([xspec[0], g_filt(xspec[1], smoothing)])
        tmp = rebin_spec(*xspec, wave_in)
        self.p_spectrum = np.asarray([tmp[0], g_filt(tmp[1], smoothing)])
        return self.p_spectrum

    def rest(self, smoothing: int = 0, persist: bool = True) -> np.array:
        """ Function to remove redshifting provided redshift exist"""
        xspec = self.p_spectrum if persist else self.spectrum
        tmp = blueshifting(xspec, self.redshift)
        self.p_spectrum = np.asarray([tmp[0], g_filt(tmp[1], smoothing)])
        return self.p_spectrum

    # Just normal smoothing
    def smooth(self, smoothing: int = 0, persist: bool = True) -> np.array:
        """ Function to smooth spectrum """
        xspec = self.p_spectrum if persist else self.spectrum
        self.p_spectrum = np.asarray([xspec[0], g_filt(xspec[1], smoothing)])
        return self.p_spectrum

    def clean_sky(self):
        """ Function to remove 5577AA skyline"""
        pix_size = np.diff(self.spectrum[0])[3]
        self.spectrum = noise_to_linear(self.spectrum, x=(5578 - pix_size * 15, 5578 + pix_size * 15))

    @property
    def p_spectrum(self):
        if self._p_spectrum is None and self.spectrum is not None:
            self._p_spectrum = self.spectrum
        return self._p_spectrum

    @p_spectrum.setter
    def p_spectrum(self, xspec):
        self._p_spectrum = xspec


class SDSSspec(Spectrum):
    def __init__(self, filename: str):
        """
        The class for calling SDSS spectrum

        Parameters:
        ------------
        filename: str
                File path of the sdss .fits file
        """
        super(SDSSspec, self).__init__()
        self.plate = None
        self.fiber = None
        self.best_fit: np.array = None
        self.redshift: float = -999
        # For extraction of the wanted properties
        # ra, dec, plate, mjd, fiber in meta
        # wave, flux in spec
        # redshift in line
        with fits.open(filename) as data:
            self.read_sdss_meta(data[0])
            self.read_sdss_redshift(data[3].data)
            self.read_sdss_spectrum(data[1].data)

    def read_sdss_meta(self, meta_data):
        xcoord = Coord(meta_data.header['RA'], meta_data.header['DEC'], self.coord_mode)
        self.ra = xcoord.ra
        self.dec = xcoord.dec

        self._time = Datefs(meta_data.header['MJD'], 'mjd')
        self.mjd = self._time.mjd

        self.plate = meta_data.header['PLATEID']
        self.fiber = meta_data.header['FIBERID']
        # plate-mjd-fiber for SDSS sites
        self.spectrum_name = f"{self.plate}-{self.mjd}-{('00000' + self.fiber)[-5:]}"

    def read_sdss_redshift(self, line_data):
        self.redshift = np.mean([j for j in [i[5] for i in line_data] if j != 0])

    def read_sdss_spectrum(self, spec_data):
        wave = np.asarray([10 ** i[1] for i in spec_data])
        flux = np.asarray([i[0] for i in spec_data])

        self.spectrum = np.asarray([wave, flux])
        self.best_fit = np.asarray([wave, np.asarray([i[7] for i in spec_data])])


# 6dfgs subclass
class SixDFGS(Spectrum):
    def __init__(self, filename: str):
        """
        The class for calling 6dFGS official .fits spectrum

        Parameters:
        ---------------
        filename: str
            File path of the 6dfgs .fits file
        """
        super(SixDFGS, self).__init__()
        self.data = fits.open(filename)
        self.hdr_id = 7
        self.error = None
        self.variance = None
        self._specv = None
        self._specr = None
        self.redshift: float = -999

        self.read_6dfgs_spectrum()
        self.read_6dfgs_meta()

    def read_6dfgs_meta(self):
        self.spectrum_name = self.data[7].header['TARGET']
        self.redshift = self.data[7].header['Z']

        xcoord = Coord(self.data[7].header['OBSRA'], self.data[7].header['OBSDEC'], self.coord_mode)
        self.ra = xcoord.ra
        self.dec = xcoord.dec

        self._time = Datefs(self.data[5].header['MJDOBS_V'], 'mjd')
        self.mjd = self._time.mjd

    def read_6dfgs_spectrum(self, hdr_id: int = 7, clean_sky: bool = True):
        """
        Function calling to read the spectrum from the .fits file

        Parameters:
        -------------
            hdr_id: int, Optional, default 7
                Default value is the first instance of 6dFGS spectrum in file. Refer to headers
            clean_sky: bool, Default True
                If True, will mask the 5577AA skyline
        """
        self.hdr_id = hdr_id
        xflux, xerr, xvar, xwave = self.data[hdr_id].data[0:4]
        bool_notnan = ~np.isnan(xflux)
        xwave = xwave[bool_notnan]
        self.spectrum = np.asarray([xwave, xflux[bool_notnan]])
        self.error = np.asarray([xwave, xerr[bool_notnan]])

        xvar = xvar[bool_notnan]
        xvar_empty = len(xvar[xvar > 0])
        xvar_empty_size = len(xvar) - xvar_empty
        xvar = np.concatenate([xvar[:xvar_empty], xvar[:xvar_empty_size]])
        self.variance = np.asarray([xwave, xvar])

        if clean_sky:
            self.clean_sky()

    @property
    def specv(self):
        if self._specv is None:
            x_id = 5
            xflux = self.data[5].data[0]
            xwave = np.asarray([self.data[x_id].header['CRVAL1'] + i * self.data[x_id].header['CDELT1']
                                for i in range(len(xflux))])
            xwave = xwave[~np.isnan(xflux)]
            xflux = xflux[~np.isnan(xflux)]

            self._specv = np.asarray([xwave, xflux])

        return self._specv

    @property
    def specr(self):
        if self._specr is None:
            x_id = 6
            xflux = self.data[6].data[0]
            xwave = np.asarray([self.data[x_id].header['CRVAL1'] + i * self.data[x_id].header['CDELT1']
                                for i in range(len(xflux))])
            xwave = xwave[~np.isnan(xflux)]
            xflux = xflux[~np.isnan(xflux)]

            self._specr = np.asarray([xwave, xflux])

        return self._specr


# wifes subclass. A lot messier to get the spec out due to
# it being two spectra, red and blue side
class Wifes(Spectrum):
    def __init__(self, filename_b: str, filename_r: str):
        """
        The class for calling WiFeS .p11 spectrum after processed by PyWiFeS

        Parameters:
        ---------------
        filename_b: str
            File path of the blue CCD spectrum. use blank string "" if not exist
        filename_r: str
            File path of the red CCD spectrum. use blank string "" if not exist
        """
        super(Wifes, self).__init__()
        self.rflux: np.array = None
        self.bflux: np.array = None
        self.rwave: np.array = None
        self.bwave: np.array = None
        self.coord_mode = 'DMS'
        self._redshift = None

        # Extract the red
        if filename_b != "" and filename_r != "":
            with fits.open(filename_r) as rdata:
                self.read_wifes_meta(rdata[0])
                self.rwave, self.rflux = self.read_wifes_halfspec(rdata)

            # Extract the blue
            with fits.open(filename_b) as bdata:
                self.bwave, self.bflux = self.read_wifes_halfspec(bdata)

            self.create_spectrum()

        elif filename_b != "":
            with fits.open(filename_b) as bdata:
                self.read_wifes_meta(bdata[0])
                self.bwave, self.bflux = self.read_wifes_halfspec(bdata)
            self.spectrum = np.asarray([self.bwave, self.bflux])

        elif filename_r != "":
            with fits.open(filename_r) as rdata:
                self.read_wifes_meta(rdata[0])
                self.rwave, self.rflux = self.read_wifes_halfspec(rdata)
            self.spectrum = np.asarray([self.rwave, self.rflux])

    def read_wifes_meta(self, hdr_data: fits.header):
        self.mjd = hdr_data.header['MJD-OBS']
        self.ra = hdr_data.header['RA']
        self.dec = hdr_data.header['DEC']
        self.spectrum_name = hdr_data.header['OBJECT']

    @staticmethod
    def read_wifes_halfspec(hdr_data: fits.hdu) -> Tuple[np.array, np.array]:
        xflux = np.asarray(hdr_data[0].data)
        start_wave = hdr_data[0].header["CRVAL1"]
        pix_wave = hdr_data[0].header["CDELT1"]
        len_wave = hdr_data[0].header["NAXIS1"]
        end_wave = start_wave + (len_wave - 1) * pix_wave
        xwave = np.linspace(start_wave, end_wave, len_wave)
        return xwave, xflux

    @property
    def redshift(self):
        if self._redshift is None:
            print('Wifes has no redshift property.'
                  'Please specify from 6dFGS or SDSS data')
            pass
        else:
            return self._redshift

    @redshift.setter
    def redshift(self, redshift):
        self._redshift = redshift

    def create_spectrum(self, norm_side: str = 'R', norm_range: Tuple = (-20, 60), clean_sky: bool = True):
        """
        Main function to call for spectrum generation

        Parameters:
        ----------------
        norm_side: str, Default "R"
            If "R", still set the red CCD spectrum as the reference and shift the blue CCD spectrum to match.
            If "B", vice versa
        norm_range: tuple of two int, Default (-20, 60)
            The range of numpy array ids to sample for the average flux to use for scaling/normalising the
            blue or red CCD spectrum
        clean_sky: bool, Default True
            If True, removes the 5577AA skyline
        """
        # find the mid-point
        br_mid = np.mean([self.rwave[0], self.bwave[-1]])
        r_id = id_finder(self.rwave, br_mid)
        b_id = id_finder(self.bwave, br_mid)

        # Average across the norm_range to find a flux level that will be used as the normalisation
        rnorm = np.mean(self.rflux[r_id - norm_range[0]:r_id + norm_range[1]])
        bnorm = np.mean(self.bflux[b_id - norm_range[0]:b_id + norm_range[1]])

        # Shifting one side to the other depending on Norm side
        if norm_side in 'Rr':
            bflux = self.bflux / bnorm * rnorm
            rflux = self.rflux
        elif norm_side in 'Bb':
            rflux = self.rflux / rnorm * bnorm
            bflux = self.bflux
        else:
            print("Incorrect norm_side argument, default to using redside (R)")
            bflux = self.bflux / bnorm * rnorm
            rflux = self.rflux

        self.spectrum = merge([self.bwave, bflux], [self.rwave, rflux])

        if clean_sky == 1:
            self.clean_sky()
