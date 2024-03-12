if __name__ == "__main__":    
    # import numpy as np
    import matplotlib.pyplot as plt
from astropy.io import fits
from Spectra_handling.Spectrum_utls import *
from Spectra_handling.Spectrum_processing import *
from coord_format import Coord
from date_format import Datefs


# Spectrum super class
class Spectrum(object):
    def __init__(self):
        self._spectrum = None
        self._redshift = None
        self._coord = None
        self._ra = None
        self._dec = None
        self._time = None
        self._spectrum_name = None
        self._pspec = None
        self._fluxmult = None

    @property
    def pspec(self):
        if self._pspec is None:
            self._pspec = ProcessedSpec(self.spectrum, self.redshift)
        return self._pspec


# SDSS subclass of spectrum
class SDSSspec(Spectrum):
    def __init__(self, filename):
        super(SDSSspec, self).__init__()
        self._plate = None
        self._fiber = None
        self._best_fit = None
        self.coord_mode = 'DEG'
        self._coord = None
        # For extraction of the wanted properties
        # ra, dec, plate, mjd, fiber in meta
        # wave, flux in spec
        # redshift in line
        with fits.open(filename) as data:
            self.meta = data[0]
            self.spec = data[1].data
            self.line = data[3].data

    @property
    def ra(self):
        if self._coord is None:
            self._coord = Coord(self.meta.header['RA'], self.meta.header['DEC'], self.coord_mode)
        if self._ra is None:
            self._ra = self._coord.ra
        return self._ra

    @property
    def dec(self):
        if self._coord is None:
            self._coord = Coord(self.meta.header['RA'], self.meta.header['DEC'], self.coord_mode)
        if self._dec is None:
            self._dec = self._coord.dec
        return self._dec

    @ra.setter
    def ra(self, newra):
        self._coord = Coord(newra, self.meta.header['DEC'], self.coord_mode)
        self._ra = None

    @dec.setter
    def dec(self, newdec):
        self._coord = Coord(self.meta.header['RA'], newdec, self.coord_mode)
        self._dec = None

    @property
    def plate(self):
        if self._plate is None:
            self._plate = self.meta.header['PLATEID']
        return self._plate

    @property
    def time(self):
        if self._time is None:
            self._time = Datefs(self.meta.header['MJD'], 'mjd')
        return self._time

    @property
    def mjd(self):
        return self.time.mjd

    @property
    def fiber(self):
        if self._fiber is None:
            self._fiber = self.meta.header['FIBERID']
        return self._fiber

    @property
    def spectrum(self):
        if self._spectrum is None:
            wave = np.asarray([10 ** i[1] for i in self.spec])
            flux = np.asarray([i[0] for i in self.spec])
            self._spectrum = np.asarray([wave, flux])
        return self._spectrum

    @property
    def redshift(self):
        if self._redshift is None:
            self._redshift = np.mean([j for j in [i[5] for i in self.line] if j != 0])
        return self._redshift

    @property
    def best_fit(self):
        if self._best_fit is None:
            bflux = np.asarray([i[7] for i in self.spec])
            self._spectrum = np.asarray([self.spectrum[0], bflux])
        return self._best_fit

        # This method allows printing the spectrum name as
        # plate-mjd-xxxx
        # This allows easy integration with the sdss online browser service
        # as well as conversion into urls

    @property
    def spectrum_name(self):
        if self._spectrum_name is None:
            fiber_string = str(self.fiber)
            while len(fiber_string) != 4:
                fiber_string = '0' + fiber_string
            self._spectrum_name = str(self.plate) + '-' + str(self.mjd) + '-' + fiber_string
        return self._spectrum_name


# 6dfgs subclass
class SixDFGS(Spectrum):
    def __init__(self, filename, clean_sky=1):
        super(SixDFGS, self).__init__()
        self.data = fits.open(filename)
        self.coord_mode = 'DEG'
        self._spec_id = 7
        self._variance = None
        self.clean = clean_sky
        self._specv = None
        self._specr = None

    @property
    def spectrum(self):
        if self._spectrum is None:
            x_id = self._spec_id
            xwave = self.data[x_id].data[3]
            xflux = self.data[x_id].data[0]
            xwave = xwave[~np.isnan(xflux)]
            xflux = xflux[~np.isnan(xflux)]

            if self.clean == 1:
                sky_id = id_finder(xwave, 5578)
                xbridge = np.concatenate([xflux[sky_id-30:sky_id-15], xflux[sky_id+15:sky_id+30]])
                xflux = np.concatenate([xflux[:sky_id-15], g_filt(xbridge, 5), xflux[sky_id+15:]])
            self._spectrum = np.asarray([xwave, xflux])

        return self._spectrum

    @property
    def specv(self):
        if self._specv is None:
            x_id = 5
            xflux = self.data[x_id].data[0]
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
            xflux = self.data[x_id].data[0]
            xwave = np.asarray([self.data[x_id].header['CRVAL1'] + i * self.data[x_id].header['CDELT1']
                                for i in range(len(xflux))])
            xwave = xwave[~np.isnan(xflux)]
            xflux = xflux[~np.isnan(xflux)]

            self._specr = np.asarray([xwave, xflux])

        return self._specr

    @property
    def variance(self):
        if self._variance is None:
            x_id = self._spec_id
            xwave = self.data[x_id].data[3]
            xvar = self.data[x_id].data[2]
            xflux = self.data[x_id].data[0]
            xwave = xwave[~np.isnan(xflux)]
            xvar = xvar[~np.isnan(xflux)]
            xvar_empty = len(xvar[xvar > 0])
            xvar_empty_size = len(xvar) - xvar_empty
            xvar = np.concatenate([xvar[:xvar_empty], xvar[:xvar_empty_size]])
            self._variance = np.asarray([xwave, xvar])
        return self._variance

    @property
    def error(self):
        x_id = self._spec_id
        return np.asarray([self.data[x_id].data[3], self.data[x_id].data[1]])

    @property
    def time(self):
        if self._time is None:
            self._time = Datefs(self.data[5].header['MJDOBS_V'], 'mjd')
        return self._time

    @property
    def mjd(self):
        return self.time.mjd

    @property
    def redshift(self):
        if self._redshift is None:
            self._redshift = self.data[7].header['Z']
        return self._redshift

    @property
    def ra(self):
        if self._coord is None:
            self._coord = Coord(self.data[7].header['OBSRA'], self.data[7].header['OBSDEC'], self.coord_mode)
        if self._ra is None:
            self._ra = self._coord.ra
        return self._ra

    @property
    def dec(self):
        if self._coord is None:
            self._coord = Coord(self.data[7].header['OBSRA'], self.data[7].header['OBSDEC'], self.coord_mode)
        if self._dec is None:
            self._dec = self._coord.dec
        return self._dec

    @ra.setter
    def ra(self, newra):
        self._coord = Coord(newra, self.data[7].header['OBSDEC'], self.coord_mode)
        self._ra = None

    @dec.setter
    def dec(self, newdec):
        self._coord = Coord(self.data[7].header['OBSRA'], newdec, self.coord_mode)
        self._dec = None

    @property
    def spec_id(self):
        return self._spec_id

    @spec_id.setter
    def spec_id(self, new_id):
        self._spec_id = new_id
        self._spectrum = None

    @property
    def spectrum_name(self):
        if self._spectrum_name is None:
            self._spectrum_name = self.data[7].header['TARGET']
        return self._spectrum_name


# wifes subclass. A lot messier to get the spec out due to
# it being two spectra, red and blue side
class Wifes(Spectrum):
    def __init__(self, filename_b, filename_r, clean_sky=1):
        super(Wifes, self).__init__()

        # Extract the red
        with fits.open(filename_r) as rdata:
            # self._mjd_tmp = rdata[0].header['MJD-OBS']
            # self.ra_tmp = rdata[0].header['RA']
            # self.dec_tmp = rdata[0].header['DEC']
            # self._spectrum_name = rdata[0].header['OBJECT']
            self.rflux = rdata[0].data
            self.rwave = np.linspace(5400, 9500, len(self.rflux))
            self.clean = clean_sky

        # Extract the blue
        with fits.open(filename_b) as bdata:
            self.bflux = bdata[0].data
            self.bwave = np.linspace(3500, 5700, len(self.bflux))

        # These are properties to tweak the resulting spectrum
        # Norm side determines which side to fix while we match-
        # -the other to fit it
        # Norm range is the range from the overlap center point-
        # -we consider for normalisation
        self._norm_side = 'R'
        self._norm_range = [-20, 60]
        self.spectrum_name = self._spectrum_name
        self.coord_mode = 'DMS'

    @property
    def time(self):
        if self._time is None:
            self._time = Datefs(self._mjd_tmp, 'mjd')
        return self._time

    @property
    def mjd(self):
        return self.time.mjd

    @property
    def ra(self):
        if self._coord is None:
            self._coord = Coord(self.ra_tmp, self.dec_tmp, self.coord_mode)
        if self._ra is None:
            self._ra = self._coord.coord_hms[0]
        return self._ra

    @property
    def dec(self):
        if self._coord is None:
            self._coord = Coord(self.ra_tmp, self.dec_tmp, self.coord_mode)
        if self._dec is None:
            self._dec = self._coord.coord_hms[1]
        return self._dec

    @ra.setter
    def ra(self, newra):
        self._coord = Coord(newra, self.dec_tmp, self.coord_mode)
        self._ra = None

    @dec.setter
    def dec(self, newdec):
        self._coord = Coord(self.ra_tmp, newdec, self.coord_mode)
        self._dec = None

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

    @property
    def fluxmult(self):
        if self._fluxmult is None:
            self._fluxmult = 1
            return self._fluxmult
        else:
            return self._fluxmult

    @fluxmult.setter
    def fluxmult(self, fm):
        self._fluxmult = fm
        self._pspec = None

    @property
    def spectrum(self):
        if self._spectrum is None:
            # 5650 is the mid point for overlap in the spectra
            r_id = id_finder(self.rwave, 5650)
            b_id = id_finder(self.bwave, 5650)
            # Average across the range to find a flux level
            rnorm = np.mean(
                self.rflux[r_id - self._norm_range[0]:r_id + self._norm_range[1]])
            bnorm = np.mean(
                self.bflux[b_id - self._norm_range[0]:b_id + self._norm_range[1]])

            # Shifting one side to the other depending on Norm side
            if self._norm_side in 'Rr':
                bflux = self.bflux / bnorm * rnorm
                rflux = self.rflux
            elif self._norm_side in 'Bb':
                rflux = self.rflux / rnorm * bnorm
                bflux = self.bflux

            if self.clean == 1:
                sky_id = id_finder(self.bwave, 5578)
                xbridge = np.concatenate([bflux[sky_id-50:sky_id-25], bflux[sky_id+25:sky_id+50]])
                bflux = np.concatenate([bflux[:sky_id-25], g_filt(xbridge, 5), bflux[sky_id+25:]])

                sky_id = id_finder(self.rwave, 5578)
                xbridge = np.concatenate([rflux[sky_id-50:sky_id-25], rflux[sky_id+25:sky_id+50]])
                rflux = np.concatenate([rflux[:sky_id-25], g_filt(xbridge, 5), rflux[sky_id+25:]])

            self._spectrum = merge([self.bwave, bflux], [self.rwave, rflux])
        return self._spectrum

    @property
    def norm_side(self):
        return self._norm_side

    @norm_side.setter
    def norm_side(self, side='R'):
        assert len(side) == 1 and side in 'RBrb', 'invalid norm side, only use R or B'
        self._norm_side = side
        self._spectrum = None

    @property
    def norm_range(self):
        return self._norm_range

    @norm_range.setter
    def norm_range(self, newrange):
        assert len(newrange) == 2 and newrange[1] > newrange[0], 'Invalid Norm Range. Ensure [min, max]'
        self._norm_range = newrange
        self._spectrum = None


# wifes subclass. A lot messier to get the spec out due to
# it being two spectra, red and blue side
class WifesC(Spectrum):
    def __init__(self, filename_b, filename_r, clean_sky=1):
        super(WifesC, self).__init__()

        # Extract the red
        with fits.open(filename_r) as rdata:
            # self._mjd_tmp = rdata[0].header['MJD-OBS']
            # self.ra_tmp = rdata[0].header['RA']
            # self.dec_tmp = rdata[0].header['DEC']
            # self._spectrum_name = rdata[0].header['OBJECT']
            self.rflux = rdata[0].data
            self.rwave = np.linspace(5000, 9566, len(self.rflux))
            self.clean = clean_sky

        # Extract the blue
        with fits.open(filename_b) as bdata:
            self.bflux = bdata[0].data
            if 'x' in filename_b:
                self.bwave = np.linspace(3500, 5700, len(self.bflux))
            else:
                self.bwave = np.linspace(3000, 5700, len(self.bflux))

        # These are properties to tweak the resulting spectrum
        # Norm side determines which side to fix while we match-
        # -the other to fit it
        # Norm range is the range from the overlap center point-
        # -we consider for normalisation
        self._norm_side = 'R'
        self._norm_range = [-20, 60]
        self.spectrum_name = self._spectrum_name
        self.coord_mode = 'DMS'

    @property
    def time(self):
        if self._time is None:
            self._time = Datefs(self._mjd_tmp, 'mjd')
        return self._time

    @property
    def mjd(self):
        return self.time.mjd

    @property
    def ra(self):
        if self._coord is None:
            self._coord = Coord(self.ra_tmp, self.dec_tmp, self.coord_mode)
        if self._ra is None:
            self._ra = self._coord.coord_hms[0]
        return self._ra

    @property
    def dec(self):
        if self._coord is None:
            self._coord = Coord(self.ra_tmp, self.dec_tmp, self.coord_mode)
        if self._dec is None:
            self._dec = self._coord.coord_hms[1]
        return self._dec

    @ra.setter
    def ra(self, newra):
        self._coord = Coord(newra, self.dec_tmp, self.coord_mode)
        self._ra = None

    @dec.setter
    def dec(self, newdec):
        self._coord = Coord(self.ra_tmp, newdec, self.coord_mode)
        self._dec = None

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

    @property
    def fluxmult(self):
        if self._fluxmult is None:
            self._fluxmult = 1
            return self._fluxmult
        else:
            return self._fluxmult

    @fluxmult.setter
    def fluxmult(self, fm):
        self._fluxmult = fm
        self._pspec = None

    @property
    def spectrum(self):
        if self._spectrum is None:
            # 5650 is the mid point for overlap in the spectra
            r_id = id_finder(self.rwave, 5650)
            b_id = id_finder(self.bwave, 5650)
            # Average across the range to find a flux level
            rnorm = np.mean(
                self.rflux[r_id - self._norm_range[0]:r_id + self._norm_range[1]])
            bnorm = np.mean(
                self.bflux[b_id - self._norm_range[0]:b_id + self._norm_range[1]])

            # Shifting one side to the other depending on Norm side
            if self._norm_side in 'Rr':
                bflux = self.bflux / bnorm * rnorm
                rflux = self.rflux
            elif self._norm_side in 'Bb':
                rflux = self.rflux / rnorm * bnorm
                bflux = self.bflux

            if self.clean == 1:
                sky_id = id_finder(self.bwave, 5578)
                xbridge = np.concatenate([bflux[sky_id-50:sky_id-25], bflux[sky_id+25:sky_id+50]])
                bflux = np.concatenate([bflux[:sky_id-25], g_filt(xbridge, 5), bflux[sky_id+25:]])

                sky_id = id_finder(self.rwave, 5578)
                xbridge = np.concatenate([rflux[sky_id-50:sky_id-25], rflux[sky_id+25:sky_id+50]])
                rflux = np.concatenate([rflux[:sky_id-25], g_filt(xbridge, 5), rflux[sky_id+25:]])

            self._spectrum = merge([self.bwave, bflux], [self.rwave, rflux], -1)
        return self._spectrum

    @property
    def norm_side(self):
        return self._norm_side

    @norm_side.setter
    def norm_side(self, side='R'):
        assert len(side) == 1 and side in 'RBrb', 'invalid norm side, only use R or B'
        self._norm_side = side
        self._spectrum = None

    @property
    def norm_range(self):
        return self._norm_range

    @norm_range.setter
    def norm_range(self, newrange):
        assert len(newrange) == 2 and newrange[1] > newrange[0], 'Invalid Norm Range. Ensure [min, max]'
        self._norm_range = newrange
        self._spectrum = None


class Wifes460(Spectrum):
    def __init__(self, filename_b, filename_r):
        super(Wifes460, self).__init__()

        # Extract the red
        with fits.open(filename_r) as rdata:
            # self._mjd_tmp = rdata[0].header['MJD-OBS']
            # self.ra_tmp = rdata[0].header['RA']
            # self.dec_tmp = rdata[0].header['DEC']
            # self._spectrum_name = rdata[0].header['OBJECT']
            self.rflux = rdata[0].data
            self.rwave = np.linspace(5000, 9565, len(self.rflux))

        # Extract the blue
        with fits.open(filename_b) as bdata:
            self.bflux = bdata[0].data
            self.bwave = np.linspace(3000, 5700, len(self.bflux))

        # These are properties to tweak the resulting spectrum
        # Norm side determines which side to fix while we match-
        # -the other to fit it
        # Norm range is the range from the overlap center point-
        # -we consider for normalisation
        self._norm_side = 'R'
        self._norm_range = [-20, 60]
        self.spectrum_name = self._spectrum_name
        self.coord_mode = 'DMS'

    @property
    def time(self):
        if self._time is None:
            self._time = Datefs(self._mjd_tmp, 'mjd')
        return self._time

    @property
    def mjd(self):
        return self.time.mjd

    @property
    def ra(self):
        if self._coord is None:
            self._coord = Coord(self.ra_tmp, self.dec_tmp, self.coord_mode)
        if self._ra is None:
            self._ra = self._coord.coord_hms[0]
        return self._ra

    @property
    def dec(self):
        if self._coord is None:
            self._coord = Coord(self.ra_tmp, self.dec_tmp, self.coord_mode)
        if self._dec is None:
            self._dec = self._coord.coord_hms[1]
        return self._dec

    @ra.setter
    def ra(self, newra):
        self._coord = Coord(newra, self.dec_tmp, self.coord_mode)
        self._ra = None

    @dec.setter
    def dec(self, newdec):
        self._coord = Coord(self.ra_tmp, newdec, self.coord_mode)
        self._dec = None

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

    @property
    def fluxmult(self):
        if self._fluxmult is None:
            self._fluxmult = 1
            return self._fluxmult
        else:
            return self._fluxmult

    @fluxmult.setter
    def fluxmult(self, fm):
        self._fluxmult = fm
        self._pspec = None

    @property
    def spectrum(self):
        if self._spectrum is None:
            # 5650 is the mid point for overlap in the spectra
            r_id = id_finder(self.rwave, 5350)
            b_id = id_finder(self.bwave, 5350)
            # Average across the range to find a flux level
            rnorm = np.mean(
                self.rflux[r_id - self._norm_range[0]:r_id + self._norm_range[1]])
            bnorm = np.mean(
                self.bflux[b_id - self._norm_range[0]:b_id + self._norm_range[1]])

            # Shifting one side to the other depending on Norm side
            if self._norm_side in 'Rr':
                bflux = self.bflux / bnorm * rnorm
                rflux = self.rflux
            elif self._norm_side in 'Bb':
                rflux = self.rflux / rnorm * bnorm
                bflux = self.bflux

            self._spectrum = merge([self.bwave, bflux], [self.rwave, rflux])
        return self._spectrum

    @property
    def norm_side(self):
        return self._norm_side

    @norm_side.setter
    def norm_side(self, side='R'):
        assert len(side) == 1 and side in 'RBrb', 'invalid norm side, only use R or B'
        self._norm_side = side
        self._spectrum = None

    @property
    def norm_range(self):
        return self._norm_range

    @norm_range.setter
    def norm_range(self, newrange):
        assert len(newrange) == 2 and newrange[1] > newrange[0], 'Invalid Norm Range. Ensure [min, max]'
        self._norm_range = newrange
        self._spectrum = None


class WifesR(Spectrum):
    def __init__(self, filename_r):
        super(WifesR, self).__init__()

        # Extract the red
        with fits.open(filename_r) as rdata:
            self.rflux = rdata[0].data
            self.rwave = np.linspace(5400, 9500, len(self.rflux))

        # These are properties to tweak the resulting spectrum
        # Norm side determines which side to fix while we match-
        # -the other to fit it
        self.spectrum_name = self._spectrum_name
        self.coord_mode = 'DMS'

    @property
    def time(self):
        if self._time is None:
            self._time = Datefs(self._mjd_tmp, 'mjd')
        return self._time

    @property
    def mjd(self):
        return self.time.mjd

    @property
    def ra(self):
        if self._coord is None:
            self._coord = Coord(self.ra_tmp, self.dec_tmp, self.coord_mode)
        if self._ra is None:
            self._ra = self._coord.coord_hms[0]
        return self._ra

    @property
    def dec(self):
        if self._coord is None:
            self._coord = Coord(self.ra_tmp, self.dec_tmp, self.coord_mode)
        if self._dec is None:
            self._dec = self._coord.coord_hms[1]
        return self._dec

    @ra.setter
    def ra(self, newra):
        self._coord = Coord(newra, self.dec_tmp, self.coord_mode)
        self._ra = None

    @dec.setter
    def dec(self, newdec):
        self._coord = Coord(self.ra_tmp, newdec, self.coord_mode)
        self._dec = None

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

    @property
    def fluxmult(self):
        if self._fluxmult is None:
            self._fluxmult = 1
            return self._fluxmult
        else:
            return self._fluxmult

    @fluxmult.setter
    def fluxmult(self, fm):
        self._fluxmult = fm
        self._pspec = None

    @property
    def spectrum(self):
        if self._spectrum is None:
            self._spectrum = [self.rwave, self.rflux]
        return self._spectrum


'''
ss = SDSSspec('Data/spec-0698-52203-0114.fits')
fgs = SixDFGS('Data/g0040392-371317.fits')
wfs = Wifes('Data/sp_33b.fits', 'Data/sp_33r.fits')
plt.plot(*wfs.spectrum)
plt.ion()
'''
