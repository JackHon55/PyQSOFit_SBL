import argparse
import sys
import numpy as np

'''parser = argparse.ArgumentParser(description="This is a script that converts RA,DEC formats",
                                 usage="name_of_script -c <ra,dec> -i <format_in> -o <format_out>")

# argument: files to process
parser.add_argument("-c", "--coord",
                    default="10.1622, 10.1622",
                    type=str,
                    help="Input the coord in any format. hms,dms should be delimited with ':', '-', ' '")

parser.add_argument("-i", "--informat",
                    default="DEG",
                    type=str,
                    help="The format of the input coord")

parser.add_argument("-o", "--outformat",
                    default="DMS",
                    type=str,
                    help="The format of the output coord")

args = parser.parse_args()'''


class RA(object):
    def __init__(self, ra_deg, ra_hms):
        self.deg = ra_deg
        self.hms = ra_hms


class Coord(object):
    def __init__(self, ra, dec, inmode):
        self._ra = None
        self._dec = None
        self._coord_deg = None
        self._coord_hms = None

        self.coord_tmp = [ra, dec]
        self.inmode = inmode

    def digit_setter(self, x):
        if len(x.split('.')[0]) < 2:
            return '0' + x
        else:
            return x

    # For Ra conversion
    def deg_to_hms(self, coord):
        h = coord * 24 / 360
        m = (h - int(h)) * 60
        s = (m - int(m)) * 60
        hms = [self.digit_setter(str(int(h))), self.digit_setter(str(int(m))),
               self.digit_setter((str(round(float(s), 2))))]
        return f'{hms[0]}:{hms[1]}:{hms[2]}'

    # For DEC conversion
    def deg_to_dms(self, coord):
        if coord < 0:
                sign = '-'
        else:
                sign = '+'
        h = abs(coord)
        m = (h - int(h)) * 60
        s = (m - int(m)) * 60
        hms = [self.digit_setter(str(int(h))), self.digit_setter(str(int(m))),
               self.digit_setter((str(round(float(s), 2))))]
        return f'{sign}{hms[0]}:{hms[1]}:{hms[2]}'

    # For Ra conversion
    def hms_to_deg(self, coord):
        dlimits = [':', ' ', '-']
        i = 0
        while len(coord.split(dlimits[i])) != 3:
            i += 1
        h, m, s = coord.split(dlimits[i])
        return round((int(h) + (int(m) + float(s) / 60) / 60) * 360 / 24, 4)

    # For DEC conversion
    def dms_to_deg(self, coord):
        dlimits = [':', ' ', '-']
        i = 0
        while len(coord.split(dlimits[i])) != 3:
            i += 1
        h, m, s = coord.split(dlimits[i])
        if int(h) < 0:
                sign = -1
        else:
                sign = 1
        return round(abs(int(h)) + (int(m) + float(s) / 60) / 60, 4)*sign

    def gen_coord(self):
        if self.inmode == 'DEG':
            self._coord_deg = self.coord_tmp
            self._coord_hms = [self.deg_to_hms(self.coord_tmp[0]), self.deg_to_dms(self.coord_tmp[1])]
        elif self.inmode == 'DMS' or self.inmode == 'HMS':
            self._coord_deg = [self.hms_to_deg(self.coord_tmp[0]), self.dms_to_deg(self.coord_tmp[1])]
            self._coord_hms = self.coord_tmp
        else:
            print('Input format error, use only DEG or DMS')
            sys.exit(1)
        pass

    @property
    def ra(self):
        if self._coord_deg is None:
            self.gen_coord()
        if self._ra is None:
            self._ra = RA(self._coord_deg[0], self._coord_hms[0])
        return self._ra

    @property
    def dec(self):
        if self._coord_deg is None:
            self.gen_coord()
        if self._dec is None:
            self._dec = RA(self._coord_deg[1], self._coord_hms[1])
        return self._dec

    @property
    def coord_deg(self):
        if self._coord_deg is None:
            self.gen_coord()
        return self._coord_deg

    @property
    def coord_hms(self):
        if self._coord_hms is None:
            self.gen_coord()
        return self._coord_hms


'''if __name__ == "__main__":
    tmp = args.coord.split(',')
    if (':' in tmp[0] or
            ' ' in tmp[0] or
            '-' in tmp[0]):
        ra, dec = tmp
    else:
        ra, dec = float(tmp[0]), float(tmp[1])
    coord_x = Coord(ra, dec, args.informat)
    if args.outformat == 'DEG':
        print(f'{coord_x.coord_deg[0]}, {coord_x.coord_deg[1]}')
    elif args.outformat == 'DMS' or args.outformat == 'HMS':
        print(f'{coord_x.coord_hms[0]}, {coord_x.coord_hms[1]}')
    else:
        print('Outformat invalid, use only DEG or DMS')'''


def pos_offset(r1, d1, r2, d2):
    off_east = (r2 - r1) * np.cos(np.radians(np.mean([d1, d2])))
    off_north = d2 - d1

    off_east_sec = Coord(off_east, off_east, 'DEG').coord_hms[1]
    off_north_sec = Coord(off_north, off_north, 'DEG').coord_hms[1]

    off_east_sec = float(off_east_sec.split(":")[-1])
    off_north_sec = float(off_north_sec.split(":")[-1])

    return np.sqrt(off_east_sec ** 2 + off_north_sec ** 2)
