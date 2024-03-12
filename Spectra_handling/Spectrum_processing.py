import sys
from rebin_spec import rebin_spec
from scipy.ndimage import gaussian_filter as g_filt
# from extinction import fitzpatrick99 as fp99, apply
import numpy as np
import matplotlib.pyplot as plt


def blueshifting(spec, zshift):
    zspec = np.copy(spec)
    wave0 = zspec[0] / (1 + zshift)
    wavenew = np.linspace(wave0[0], wave0[-1], len(wave0))
    fluxnew = rebin_spec(wave0, zspec[1], wavenew)
    return np.asarray([wavenew, fluxnew])


def redshifting(spec, zshift):
    wave0 = spec[0] * (1 + zshift)
    wavenew = np.linspace(wave0[0], wave0[-1], len(wave0))
    fluxnew = rebin_spec(wave0, spec[1], wavenew)
    return np.asarray([wavenew, fluxnew])


class ProcessedSpec(object):
    def __init__(self, spectrum, redshift, fluxmultiplier=1):
        self.wavelength = spectrum[0]
        self.flux = spectrum[1]*fluxmultiplier
        self.redshift = redshift

    # Only rebinning
    def rebin(self, smoothing=0, wavein=None):
        if wavein is None:
            wavein = self.wavelength
        tmp = rebin_spec(self.wavelength, self.flux, wavein)
        if smoothing != 0:
            return np.asarray([wavein, g_filt(tmp, smoothing)])
        else:
            return np.asarray([wavein, tmp])

    # Redshifts the spectrum, then rebins it have consistent flux
    def rest(self, smoothing=0, redshift=None):
        if redshift is None:
            redshift = self.redshift
        tmp = blueshifting([self.wavelength, self.flux], redshift)
        if smoothing != 0:
            return np.asarray([tmp[0], g_filt(tmp[1], smoothing)])
        else:
            return tmp

    # Just normal smoothing
    def smooth(self, smoothing=0):
        if smoothing != 0:
            return np.asarray([self.wavelength, g_filt(self.flux, smoothing)])
        else:
            return np.asarray([self.wavelength, self.flux[:]])


class SpecAxis(object):
    def __init__(self):
        self._xlab = None
        self._ylab = None
        self._title = None
        self._figx = None
        self._fontsize = None

        # Emission line stuff
        self._linefile = None
        self._lineredshift = None
        self._lineshow = None
        self._linecolor = None
        self._lineargs = None

    def fplot(self):
        plt.figure(figsize=self.figx)
        plt.ylabel(self.ylab)
        plt.xlabel(self.xlab)
        plt.title(self.title)
        self.lines(**self.lineargs)
        plt.tight_layout()

    def replot(self):
        plt.close()
        self.fplot()

    @property
    def figx(self):
        if self._figx is None:
            self._figx = (5, 5)
        return self._figx

    @figx.setter
    def figx(self, newfigx):
        self._figx = newfigx
        self.replot()

    @property
    def xlab(self):
        if self._xlab is None:
            self._xlab = 'Wavelength (Angstrom)'
        return self._xlab

    @xlab.setter
    def xlab(self, newxlab):
        self._xlab = newxlab
        self.replot()

    @property
    def ylab(self):
        if self._ylab is None:
            self._ylab = 'Flux (ergs/cm/cm/s/Ang)'
        return self._ylab

    @ylab.setter
    def ylab(self, newylab):
        self._ylab = newylab
        self.replot()

    @property
    def title(self):
        if self._title is None:
            self._title = ''
        return self._title

    @title.setter
    def title(self, newtitle):
        self._title = newtitle
        self.replot()

    @property
    def fontsize(self):
        if self._fontsize is None:
            self._fontsize = 16
            plt.rcParams.update({'font.size': 16})
        return self._fontsize

    @fontsize.setter
    def fontsize(self, newfontsize):
        self._fontsize = newfontsize
        plt.rcParams.update({'font.size': newfontsize})
        self.replot()

    @property
    def linefile(self):
        if self._linefile is None:
            self._linefile = 'Spectra_handling/Data/line_names.csv'
        return self._linefile

    @linefile.setter
    def linefile(self, newlinefile):
        self._linefile = newlinefile
        self.replot()

    @property
    def lineredshift(self):
        if self._lineredshift is None:
            self._lineredshift = 0
        return self._lineredshift

    @lineredshift.setter
    def lineredshift(self, newlineredshift):
        self._lineredshift = newlineredshift
        self.replot()

    @property
    def linecolor(self):
        return self._linecolor

    @linecolor.setter
    def linecolor(self, newlinecolor):
        self._linecolor = newlinecolor
        self.replot()

    @property
    def lineshow(self):
        return self._lineshow

    @lineshow.setter
    def lineshow(self, newlineshow):
        self._lineshow = newlineshow
        self.replot()

    @property
    def lineargs(self):
        if self._lineargs is None:
            self._lineargs = {"linestyle": '-.', "alpha": 0.5, "linewidth": 0.5,
                              "addtext": '', "yadd": 0, "xadd": 0, "linefontsize": 15}
        return self._lineargs

    @lineargs.setter
    def lineargs(self, newlineargs):
        self._lineargs = self.lineargsgen(**newlineargs)
        self.replot()

    @staticmethod
    def lineargsgen(linestyle='-.', alpha=0.5, linewidth=0.5,
                    addtext='', yadd=0, xadd=0, linefontsize=15):
        return {"linestyle": linestyle, "alpha": alpha, "linewidth": linewidth,
                "addtext": addtext, "yadd": yadd, "xadd": xadd, "linefontsize": linefontsize}

    def lines(self, addtext='', yadd=0, xadd=0, linefontsize=15, **vlineargs):
        ypos = yadd
        linetmp = np.genfromtxt(self.linefile, dtype=str, delimiter=',')
        if self.lineshow is None:
            line = linetmp
        else:
            line = np.asarray([linetmp[i] for i in self.lineshow])
        zline = np.asarray([float(i[1]) for i in line]) * (1 + self.lineredshift)
        zname = np.asarray([i[0] + addtext for i in line])
        if self.linecolor is None:
            zcolor = np.asarray([i[2] for i in line])
        elif type(self.linecolor) is str and len(self.linecolor) == 1:
            zcolor = np.asarray([self.linecolor for i in line])
        elif type(self.linecolor) is np.ndarray or type(self.linecolor) is list:
            if len(self.linecolor) == len(self.lineshow):
                zcolor = self.linecolor
            else:
                print("Incorrect length for lineshow and linecolor")
                sys.exit(1)
        else:
            print("Incorrect format for linecolor")
            sys.exit(1)
        for i in range(0, len(zline)):
            plt.axvline(x=zline[i], C=zcolor[i], **vlineargs)
            if i % 2 == 0:
                plt.text(zline[i] + xadd, ypos, zname[i], fontsize=linefontsize)
            else:
                plt.text(zline[i] + xadd, ypos * 0.95, zname[i], fontsize=linefontsize)


'''
t = SpecAxis()
t.fontsize = 18
t.ylab = 'Normalised Flux'
t.lineredshift = a.redshift
t.lineshow = [4, 6, 7, 8, 9, 10, 11]
t.figx = (15, 5)
t.title = ''
'''
