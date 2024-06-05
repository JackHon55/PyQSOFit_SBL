import julian
import datetime
import argparse
from math import floor


class Datefs(object):
    def __init__(self, date=57205, informat='mjd'):
        self._mjd = None
        self._jd = None
        self._date = None
        self._decyear = None

        self.format = informat
        self.indate = date

    def mjd_to_jd(self, x):
        return x + 2400000

    def jd_to_mjd(self, x):
        return x - 2400000

    def jd_to_date(self, x):
        dt = julian.from_jd(x, fmt='jd')
        return f"{dt.year}-{dt.month}-{dt.day}"

    def date_to_jd(self, x):
        [y, m, d] = x.split('-')
        return julian.to_jd(datetime.datetime(int(y), int(m), int(d)), fmt='jd')

    def date_to_dec(self, x):
        [y, m, d] = x.split('-')
        return int(y) + (int(m) + int(d) / 30) / 12

    def dec_to_date(self, x):
        y = floor(x)
        m = floor((x - y) * 12)
        d = floor((((x - y) * 12) - m) * 30)
        return f"{y}-{m}-{d}"

    def gen_dates(self, x):
        if self.format == 'mjd':
            jd = self.mjd_to_jd(x)
            dt = self.jd_to_date(jd)
            dy = self.date_to_dec(dt)
            return [x, jd, dt, dy]
        elif self.format == 'jd':
            mjd = self.jd_to_mjd(x)
            dt = self.jd_to_date(x)
            dy = self.date_to_dec(dt)
            return [mjd, x, dt, dy]
        elif self.format == 'date':
            jd = self.date_to_jd(x)
            mjd = self.jd_to_mjd(jd)
            dy = self.date_to_dec(x)
            return [mjd, jd, x, dy]
        elif self.format == 'decyr':
            dt = self.dec_to_date(x)
            jd = self.date_to_jd(dt)
            mjd = self.jd_to_mjd(jd)
            return [mjd, jd, dt, x]

    @property
    def mjd(self):
        if self.format == 'mjd':
            self._mjd = self.indate
        else:
            self._mjd = self.gen_dates(self.indate)[0]
        return self._mjd

    @property
    def jd(self):
        if self.format == 'jd':
            self._jd = self.indate
        else:
            self._jd = self.gen_dates(self.indate)[1]
        return self._jd

    @property
    def date(self):
        if self.format == 'date':
            self._date = self.indate
        else:
            self._date = self.gen_dates(self.indate)[2]
        return self._date

    @property
    def decyear(self):
        if self.format == 'decyr':
            self._decyear = self.indate
        else:
            self._decyear = self.gen_dates(self.indate)[3]
        return self._decyear


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This is a script that converts date formats",
                                     usage="name_of_script -d <date> -i <format_in> -o <format_out>")

    # argument: files to process
    parser.add_argument("-d", "--date",
                        default="57205",
                        type=str,
                        help="Input the date in any format. Dates should be strings delimited with '-'")

    parser.add_argument("-i", "--informat",
                        default="mjd",
                        type=str,
                        help="The format of the input date")

    parser.add_argument("-o", "--outformat",
                        default="decyr",
                        type=str,
                        help="The format of the output date")

    args = parser.parse_args()

    if "-" not in args.date:
        date_input = float(args.date)
    else:
        date_input = args.date

    format_check = 0
    if args.informat == 'mjd':
        format_check = 1
        assert len(str(int(date_input))) == 5, 'MJD has incorrect number of digits before decimal. Ensure it is 5XXXX'
    if args.informat == 'jd':
        format_check = 1
        assert len(str(int(date_input))) == 7, 'JD has incorrect number of digits before deciman. Ensure it is 24XXXXX'
    if args.informat == 'date':
        format_check = 1
        assert 8 <= len(date_input) <= 10, 'Date has too much digits. Ensure yyyy-mm-dd'
    if args.informat == 'decyr':
        format_check = 1
        assert len(str(int(date_input))) == 4, 'Dec Year has incorrect number of digits before decimal. Ensure it is XXXX.xxxx'
    assert format_check == 1, 'Date in format specification error. Only use mjd, jd, date, decyr'
    date_x = Datefs(date_input, args.informat)
    if args.outformat == 'mjd':
        print(date_x.mjd)
    elif args.outformat == 'jd':
        print(date_x.jd)
    elif args.outformat == 'date':
        print(date_x.date)
    elif args.outformat == 'decyr':
        print(date_x.decyear)
    else:
        print('Date out format specification error. Only use mjd, jd, date, decyr')
