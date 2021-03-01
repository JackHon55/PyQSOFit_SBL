# PyQSOFit_SBL
## A code to fit the spectrum of quasar with skewed gaussians, Modified from the original code PyQSOFit cited below 
### Go to https://github.com/legolason/PyQSOFit and checkout their demo code 'example.ipynb' for a quick start !!!

See example_sdss.py for SDSS examples

See example_wifes.py for non-SDSS examples

The code takes an input spectrum (observed-frame wavelength, flux density and error arrays) and the redshift as input parameters, performs the fitting in the restframe, and outputs the best-fit parameters and quality-checking plots to the paths specified by the user. 

The code uses an input line-fitting parameter list to specify the fitting range and parameter constraints of the individual emission line components. An example of such a file is provided in the example.ipynb. Within the code, the user can switch on/off components to fit to the pseudo-continuum. For example, for some objects the UV/optical Fe II emission cannot be well constrained and the user may want to exclude this component in the continuum fit. The code is highly flexible and can be modified to meet the specific needs of the user.

Main changes in SBL:
- Tie_lines functionality has been removed. The line fit algorithm with tie lines is not needed for our group's purposes
- 1 more parameter for Gaussians was added to allow of skewness
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
