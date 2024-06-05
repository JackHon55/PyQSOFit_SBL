import sys
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


class SpecAxis(object):
    def __init__(self, figsize: Tuple = (10, 5)):
        self._xlab = 'Wavelength (Angstrom)'
        self._ylab = 'Flux (ergs/cm/cm/s/Ang)'
        self._title = ''
        self._figx = figsize
        self._fontsize = 16
        plt.rcParams.update({'font.size': 16})
        self.frame: plt.figure = None

        # Spectrum stuff
        self.list_spec: list = []
        self.list_color: list = []
        self.spec_args = {"linestyle": '-', "linewidth": 1}

        # Emission line stuff
        self._linefile = 'Spectra_handling/Data/line_names.csv'
        self._lineredshift = 0
        self._lineshow = None
        self._linecolor = None
        self._lineargs = {"linestyle": '-.', "alpha": 0.5, "linewidth": 0.5,
                          "addtext": '', "yadd": 0, "xadd": 0, "linefontsize": 15}

    def fplot(self):
        """ Call function to bring up the frame and plot lines """
        self.frame = plt.figure(figsize=self.figsize)
        plt.ylabel(self.ylab)
        plt.xlabel(self.xlab)
        plt.title(self.title)
        for xspec, xcol in zip(self.list_spec, self.list_color):
            plt.plot(*xspec, color=xcol, **self.spec_args)
        self.lines(**self.lineargs)
        plt.tight_layout()

    def replot(self):
        """ Call function to update the plot """
        plt.close(self.frame)
        self.fplot()

    def add_spec(self, xspec, xcol: str = "k"):
        """ Call function to add lines to plot """
        self.list_spec.append(xspec)
        self.list_color.append(xcol)
        self.replot()

    def clear_allspec(self):
        self.list_spec = []
        self.list_color = []
        self.replot()

    @property
    def figsize(self):
        return self._figx

    @figsize.setter
    def figsize(self, newfigx):
        self._figx = newfigx
        self.replot()

    @property
    def xlab(self):
        return self._xlab

    @xlab.setter
    def xlab(self, newxlab):
        self._xlab = newxlab
        self.replot()

    @property
    def ylab(self):
        return self._ylab

    @ylab.setter
    def ylab(self, newylab):
        self._ylab = newylab
        self.replot()

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, newtitle):
        self._title = newtitle
        self.replot()

    @property
    def fontsize(self):
        return self._fontsize

    @fontsize.setter
    def fontsize(self, newfontsize):
        self._fontsize = newfontsize
        plt.rcParams.update({'font.size': newfontsize})
        self.replot()

    @property
    def linefile(self):
        return self._linefile

    @linefile.setter
    def linefile(self, newlinefile):
        self._linefile = newlinefile
        self.replot()

    @property
    def lineredshift(self):
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
        return self._lineargs

    @lineargs.setter
    def lineargs(self, newlineargs):
        self._lineargs = self.lineargsgen(**newlineargs)
        self.replot()

    @staticmethod
    def lineargsgen(linestyle='-.', alpha=0.5, linewidth=0.5, addtext='', yadd=0, xadd=0, linefontsize=15):
        """
        Refer to matplotlib pyplot **kwargs document

        Parameters:
        ---------
        addtext: str, Default ""
            Adds suffix to emission line texts
        yadd: float, Default 0
            Controls the y-offset of the emission line texts
        xadd: float, Default 0
            Controls the x-offset of the emission line texts
        linefontsize: float, Default 15
            Controls the font size of emission line texts
        """
        return {"linestyle": linestyle, "alpha": alpha, "linewidth": linewidth,
                "addtext": addtext, "yadd": yadd, "xadd": xadd, "linefontsize": linefontsize}

    def load_lines(self) -> Tuple[np.array, np.array, np.array]:
        line = np.genfromtxt(self.linefile, dtype=str, delimiter=',')
        if self.lineshow is not None:
            line = np.asarray([line[i] for i in self.lineshow])
        return line.T

    def set_linecolor(self, arr_linecolor: np.array) -> np.array:
        if self.linecolor is None:
            zcolor = arr_linecolor
        elif type(self.linecolor) is str and len(self.linecolor) == 1:
            zcolor = np.asarray([self.linecolor] * len(arr_linecolor))
        elif type(self.linecolor) is np.ndarray or type(self.linecolor) is list:
            if len(self.linecolor) != len(self.lineshow):
                print("Incorrect length for lineshow and linecolor, use default")
                zcolor = arr_linecolor
            else:
                zcolor = self.linecolor
        else:
            print("Incorrect format for linecolor, use default")
            zcolor = arr_linecolor
        return zcolor

    def set_lineypos(self, yadd: float = 0) -> float:
        if len(self.list_spec) == 0:
            return yadd
        else:
            min_flux = plt.ylim()[1] * 0.8
            return min_flux + yadd

    def lines(self, addtext='', yadd=0, xadd=0, linefontsize=15, **vlineargs):
        arr_linename, arr_linewave, arr_linecolor = self.load_lines()

        zline = arr_linewave.astype(float) * (1 + self.lineredshift)
        zname = np.char.add(arr_linename, addtext)
        zcolor = self.set_linecolor(arr_linecolor)

        ypos = self.set_lineypos(yadd)

        for i in range(0, len(zline)):
            plt.axvline(x=zline[i], C=zcolor[i], **vlineargs)
            if i % 2 == 0:
                plt.text(zline[i] + xadd, ypos, zname[i], fontsize=linefontsize)
            else:
                plt.text(zline[i] + xadd, ypos * 0.95, zname[i], fontsize=linefontsize)
