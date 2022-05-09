# PyQSOFit_SBL
## A code to fit the spectrum of quasar with skewed gaussians, Modified from the original code PyQSOFit cited below 
### Go to https://github.com/legolason/PyQSOFit and checkout their demo code 'example.ipynb' for a quick start !!!

See example_sdss.py for SDSS examples

See example_wifes.py for non-SDSS examples

The code takes an input spectrum (observed-frame wavelength, flux density and error arrays) and the redshift as input parameters, performs the fitting in the restframe, and outputs the best-fit parameters and quality-checking plots to the paths specified by the user. 

The code uses an input line-fitting parameter list to specify the fitting range and parameter constraints of the individual emission line components. An example of such a file is provided in the example.ipynb. Within the code, the user can switch on/off components to fit to the pseudo-continuum. For example, for some objects the UV/optical Fe II emission cannot be well constrained and the user may want to exclude this component in the continuum fit. The code is highly flexible and can be modified to meet the specific needs of the user.

Main changes in SBL:
- 1 more parameter for Gaussians was added to allow of skewness
- Input fitting parameters uses more strings than float values. The strings are in '1234', '[1234]', '[1234, 5678]', or 'Line*12' format
- sigma, skew, scale of the initial parameters uses these strings.
- '1234' means this is the initial guess, the lower limit is >0 and upper limit is infinite. Use this to explore the fitting space
- '[1234]' means the initial guess is close to the final value. Use this is you don't want the value to vary too much
- '[1234, 5678]' provides the minimum and maximum values, the initial guess is taken as the mean of the two.
- 'Line*12' means copy this parameter value from the specified line and multiply by a specific value. For example OIII4959 can be tie to OIII5007 by 'OIII5007*0.5'
- Negative flux scale is allowed for absorption lines


## Cite to this code (As the code is still >99% the original code, please cite the orignal creators)

> The preferred citation for this code is Guo, Shen & Wang (2018), ascl:1809:008\
> @misc{2018ascl.soft09008G,\
> author = {{Guo}, H. and {Shen}, Y. and {Wang}, S.},\
> title = "{PyQSOFit: Python code to fit the spectrum of quasars}",\
> keywords = {Software },\
> howpublished = {Astrophysics Source Code Library},\
> year = 2018,\
> month = sep,\
> archivePrefix = "ascl",\
> eprint = {1809.008},\
> adsurl = {[http://adsabs.harvard.edu/abs/2018ascl.soft09008G}](http://adsabs.harvard.edu/abs/2018ascl.soft09008G%7D),\
> adsnote = {Provided by the SAO/NASA Astrophysics Data System}\
> }
