from __future__ import division

import matplotlib

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys

import astropy.io.fits as pyfits
import numpy as np

import webbpsf, webbpsf_ext
from webbpsf_ext import synphot_ext as S

from pyklip.klip import rotate as nanrotate
from scipy.ndimage import rotate
from scipy.ndimage import shift as spline_shift
from scipy.optimize import minimize

from tqdm.auto import tqdm

from webbpsf_ext import NIRCam_ext, MIRI_ext
from webbpsf_ext.utils import siaf_nrc, siaf_mir

from webbpsf_ext.coords import rtheta_to_xy
from webbpsf_ext.image_manip import fourier_imshift, frebin, pad_or_cut_to_size
from webbpsf_ext.image_manip import add_ipc, add_ppc
from webbpsf_ext.imreg_tools import get_coron_apname as gen_nrc_coron_apname
from webbpsf_ext.imreg_tools import crop_image, apply_pixel_diffusion

from webbpsf_ext.logging_utils import setup_logging
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

class JWST_PSF():
    """ Create coronagraphic PSFs for JWST NIRCam and MIRI
    
    This object provides the ability to generate a synthetic NIRCam coronagraphic
    PSF using webbpsf and webbpsf_ext at an arbitrary location relative to the 
    occulting mask, taking into account mask attenuation near the IWA.
    
    There are multiple ways to estimate these PSFs:
      - extrapolation from the theoretical occulting mask transmission (fastest)
      - `webbpsf_ext` PSF coefficients (intermediate speed)
      - on-the-fly calculations with `webbpsf` (slowest, but most accurate).
    
    Includes the ability to use date-specific OPD maps as generated by the JWST
    wavefront sensing group. Simply set `use_coeff=False`, and supply a date in ISO 
    format.
    
    NOTE: By default, resulting PSFs were normalized such that their total intensity 
    is 1.0 at the telescope entrance pupil (e.g., `normalize='first'`). So, the final 
    intensity of these PSFs include throughput attenuation at intermediate optics 
    such as the NIRCam Lyot stops and occulting masks. During PSF generation, set
    `normalize='exit_pupil'` for PSFs that have are normalized to 1.0 when summed
    out to infinity. Only works for `quick=False`.
    """
    
    def __init__(self, apername, filt, date=None, fov_pix=65, oversample=2, 
                 sp=None, use_coeff=False, **kwargs): 
        """ Initialize the JWST_PSF class
        
        Parameters
        ----------
        inst : str
            Instrument name either 'NIRCAM' or 'MIRI'.
        filter : str
            NIRCam filter (e.g., F335M)
        image_mask : str
            NIRCam coronagraphic occulting mask (e.g., MASK335R)
        fov_pix : int
            PSF pixel size. Suggest odd values for centering PSF in middle of a pixel
            rather than pixel corner / boundaries.
        oversample : int
            Size of oversampling.
        sp : :class:`webbpsf_ext.synphot_ext.Spectrum`
            Spectrum to use for PSF wavelength weighting. If None, then default is G2V.
        use_coeff : bool
            Generate PSFs from webbpsf_ext coefficient library. If set to False, then
            will use webbpsf to generate PSFs on-the-fly, opening up the ability to
            use date-specific OPD files via the `date` keyword.
        date : str or None
            Date time in UTC as ISO-format string, a la 2022-07-01T07:20:00.
            If not set, then default webbpsf OPD is used (e.g., RevAA).
        
        Returns
        -------
        None.
        
        """

        if apername in siaf_nrc.apernames:
            inst = 'NIRCAM'
            self.inst_ext = NIRCam_ext
        elif apername in siaf_mir.apernames:
            inst = 'MIRI'
            self.inst_ext = MIRI_ext
        else:
            raise ValueError("apername not found in NIRCam or MIRI SIAF lists")
                
        # Determine image mask based on aperture name
        if inst=='NIRCAM':
            if 'FULL_WEDGE' in apername:
                raise ValueError("NIRCam FULL_WEDGE not supported. Use more specific aperture.")

            ND_acq = False
            if 'MASK' not in apername:
                image_mask = None
            else:
                ap_str_arr = apername.split('_')
                for s in ap_str_arr:
                    if 'MASK' in s:
                        image_mask = s
                        break
                # Special case for TA apertures
                if 'TA' in image_mask:
                    # Set no mask for TA apertures
                    image_mask = None
                    # ND acquisitions use the ND mask
                    ND_acq = False if 'FS' in image_mask else True

            # Choose Lyot stop based on coronagraphic mask input
            if image_mask is None:
                pupil_mask = None
            elif image_mask[-1] == 'R':
                pupil_mask = 'CIRCLYOT'
            elif image_mask[-1] == 'B':
                pupil_mask = 'WEDGELYOT'
        elif inst=='MIRI':
            if '1065' in apername:
                image_mask = 'FQPM1065'
                pupil_mask = 'MASKFQPM'
            elif '1140' in apername:
                image_mask = 'FQPM1140'
                pupil_mask = 'MASKFQPM'
            elif '1550' in apername:
                image_mask = 'FQPM1550'
                pupil_mask = 'MASKFQPM'
            elif 'LYOT' in apername:
                image_mask = 'LYOT2300'
                pupil_mask = 'MASKLYOT'
            else:
                image_mask = pupil_mask = None
        else:
            raise ValueError(f"Instrument {inst} not supported by JWST_PSF class")
        
        # On-axis
        setup_logging('WARN', verbose=False)
        inst_on = self.inst_ext(filter=filt, image_mask=image_mask, pupil_mask=pupil_mask,
                                fov_pix=fov_pix, oversample=oversample, **kwargs)
        inst_on.siaf_ap = inst_on.siaf[apername]
        # Is this a TA aperture on the ND mask?
        if inst=='NIRCAM':
            inst_on.ND_acq = ND_acq

        # Off-axis
        if image_mask is None:
            inst_off = inst_on
        else:        
            inst_off = self.inst_ext(filter=filt, image_mask=None, pupil_mask=pupil_mask,
                                     fov_pix=fov_pix, oversample=oversample, **kwargs)
        
        # Jitter values
        inst_on.options['jitter'] = 'gaussian'
        inst_on.options['jitter_sigma'] = 0.001 #1mas jitter from commissioning
        inst_off.options['jitter'] = 'gaussian'
        inst_off.options['jitter_sigma'] = 0.001 #1mas jitter from commissioning
        
        # Generating initial PSFs...
        # print('Generating initial PSFs...')
        if use_coeff:
            inst_on.gen_psf_coeff()
            if image_mask is not None:
                inst_off.gen_psf_coeff()
                inst_on.gen_wfemask_coeff(large_grid=True)
            func_on = inst_on.calc_psf_from_coeff
            func_off = inst_off.calc_psf_from_coeff
            func_off = inst_off.calc_psf_from_coeff
            inst_on.gen_wfemask_coeff(large_grid=True)
            func_off = inst_off.calc_psf_from_coeff                
            inst_on.gen_wfemask_coeff(large_grid=True)
        else:
            func_on = inst_on.calc_psf
            func_off = inst_off.calc_psf

        # Load date-specific OPD files?
        if date is not None:
            inst_on.load_wss_opd_by_date(date=date, choice='closest', verbose=False, plot=False)
            inst_off.load_wss_opd_by_date(date=date, choice='closest', verbose=False, plot=False)
        
        # Renormalize spectrum to have 1 e-/sec within bandpass to obtain normalized PSFs
        if sp is not None:
            sp = _sp_to_spext(sp, **kwargs)
            sp = sp.renorm(1, 'counts', inst_on.bandpass)
        
        # On axis PSF
        log.info('Generating on-axis and off-axis PSFs...')
        if image_mask[-1] == 'B':
            # Information for bar offsetting (in arcsec)
            bar_offset = inst_on.get_bar_offset(ignore_options=True)
            bar_offset = 0 if bar_offset is None else bar_offset

            # Need an array of PSFs along bar center
            xvals = np.linspace(-8,8,9) - bar_offset
            self.psf_bar_xvals = xvals
            
            psf_bar_arr = []
            for xv in tqdm(xvals, desc='Bar PSFs', leave=False):
                psf = func_on(sp=sp, return_oversample=True, return_hdul=False, 
                              coord_vals=(xv,0), coord_frame='idl')
                psf_bar_arr.append(psf)
            self.psf_on = np.array(psf_bar_arr)
        else:
            self.psf_on = func_on(sp=sp, return_oversample=True, return_hdul=False)
        
        # Off axis PSF
        self.psf_off = func_off(sp=sp, return_oversample=True, return_hdul=False)
        log.info('  Done.')
        
        # Center PSFs
        self._recenter_psfs()
        
        # Store instrument classes
        self.inst_on  = inst_on
        self.inst_off = inst_off
        
        # PSF generation functions for later use
        self._use_coeff = use_coeff
        self._func_on  = func_on
        self._func_off = func_off
        
        self.sp = sp
    
    @property
    def fov_pix(self):
        return self.inst_on.fov_pix
    @property
    def oversample(self):
        return self.inst_on.oversample
    @property
    def filter(self):
        return self.inst_on.filter
    @property
    def image_mask(self):
        return self.inst_on.image_mask
    @property
    def pupil_mask(self):
        return self.inst_on.pupil_mask
    @property
    def use_coeff(self):
        return self._use_coeff
    @property
    def bandpass(self):
        return self.inst_on.bandpass
    @property
    def name(self):
        return self.inst_on.name
    
    def _calc_psf_off_shift(self, xysub=10):
        """
        Calculate oversampled pixel shifts using off-axis PSF and Gaussian
        centroiding.
        
        Returns
        -------
        None.
        
        """
        
        from astropy.modeling import models, fitting
        
        xv = yv = np.arange(xysub)
        xgrid, ygrid = np.meshgrid(xv, yv)
        xc, yc = (xv.mean(), yv.mean())
        
        psf_template = pad_or_cut_to_size(self.psf_off, xysub+10)
        
        xoff = 0
        yoff = 0
        for ii in range(2):
            psf_off = pad_or_cut_to_size(psf_template, xysub)
            
            # Fit the data using astropy.modeling
            p_init = models.Gaussian2D(amplitude=psf_off.max(), x_mean=xc, y_mean=yc, x_stddev=1, y_stddev=2)
            fit_p = fitting.LevMarLSQFitter()
            
            pfit = fit_p(p_init, xgrid, ygrid, psf_off)
            xcen_psf = xc - pfit.x_mean.value
            ycen_psf = yc - pfit.y_mean.value
            
            # Accumulate offsets
            xoff += xcen_psf
            yoff += ycen_psf
            
            # Update initial PSF location
            psf_template = fourier_imshift(psf_template, xcen_psf, ycen_psf, pad=True)
        
        # Save to attribute
        self._xy_off_to_cen_osamp = (xoff, yoff)

    def _recenter_psfs(self, **kwargs):
        """
        Recenter PSFs by centroiding on off-axis PSF and shifting both by same
        amount.
        
        Returns
        -------
        None.
        
        """
        
        # Calculate shift
        self._calc_psf_off_shift(**kwargs)
        xoff, yoff = self._xy_off_to_cen_osamp
        
        # Perform recentering
        self.psf_on = fourier_imshift(self.psf_on, xoff, yoff, pad=True)
        self.psf_off = fourier_imshift(self.psf_off, xoff, yoff, pad=True)
    
    def _shift_psfs(self, shifts):
        """
        Shift the on-axis and off-axis psfs by the desired amount.
        
        Parameters
        ----------
        shifts : list of floats
            The x and y offsets you want to apply [x,y].
        
        Returns
        -------
        None.
        
        """
        
        xoff,yoff = shifts
        
        # Perform the shift
        self.psf_on = fourier_imshift(self.psf_on, xoff, yoff, pad=True)
        self.psf_off = fourier_imshift(self.psf_off, xoff, yoff, pad=True)
        self.xoff = xoff
        self.yoff = yoff
    
    def rth_to_xy(self, r, th, PA_V3=0, frame_out='idl', addV3Yidl=True):
        """
        Convert (r,th) location to (x,y) in idl coords.
        
        Assume (r,th) in coordinate system with North up East to the left.
        Then convert to NIRCam detector orientation (idl coord frame).
        Units assumed to be in arcsec.
        
        Parameters
        ----------
        r : float or ndarray
            Radial offst from mask center.
        th : float or ndarray
            Position angle (positive angles East of North) in degrees.
            Can also be an array; must match size of `r`.
        PA_V3 : float
            V3 PA of ref point N over E (e.g. 'ROLL_REF').
        frame_out : str
            Coordinate frame of output. Default is 'idl'.
                * 'tel': arcsecs V2,V3
                * 'sci': pixels, in DMS axes orientation; aperture-dependent
                * 'det': pixels, in raw detector read out axes orientation
                * 'idl': arcsecs relative to aperture reference location.
        
        Returns
        -------
        None.
        
        """
        
        # Convert to aperture PA
        if addV3Yidl == True:
            PA_ap = PA_V3 + self.inst_on.siaf_ap.V3IdlYAngle
        else:
            PA_ap = PA_V3
        # Get theta relative to detector orientation (idl frame)
        th_fin = th - PA_ap
        # Return (x,y) in idl frame
        xidl, yidl = rtheta_to_xy(r, th_fin)
        
        if frame_out=='idl':
            return (xidl, yidl)
        else:
            return self.inst_on.siaf_ap.convert(xidl, yidl, 'idl', frame_out)
    
    def gen_psf_idl(self, coord_vals, coord_frame='idl', quick=True, sp=None,
                    return_oversample=False, do_shift=False, normalize='first'):
        """
        Generate offset PSF in detector frame.
        
        Generate a PSF with some (x,y) position in some coordinate frame (default idl).
        
        Parameters
        ----------
        coord_vals : tuple or None
            Coordinates (in arcsec or pixels) to calculate field-dependent PSF.
            If multiple values, then this should be an array ([xvals], [yvals]).
        coord_frame : str
            Type of input coordinates. Default is 'idl'.
                * 'tel': arcsecs V2,V3
                * 'sci': pixels, in DMS axes orientation; aperture-dependent
                * 'det': pixels, in raw detector read out axes orientation
                * 'idl': arcsecs relative to aperture reference location.
        quick : bool
            Use linear combination of on-axis and off-axis PSFs to generate
            PSF as a function of corongraphic mask throughput. Typically takes
            10s of msec, compared to standard calculations using coefficients 
            (~1 sec) or on-the-fly calcs w/ webbpsf (10s of sec).
            Only applicable for NIRCam.
        sp : :class:`webbpsf_ext.synphot_ext.Spectrum`
            Manually specify spectrum to get a desired wavelength weighting. 
            Only applicable if ``quick=False``. If not set, defaults to ``self.sp``.
        return_oversample : bool
            Return the oversampled version of the PSF?
        do_shift : bool
            If True, will return the PSF offset from center in 'idl' coords.
            Otherwise, returns PSF in center of image.
        normalize : str
            How to normalize the PSF. Options are:
                * 'first': Normalize to 1.0 at entrance pupil
                * 'exit_pupil': Normalize to 1.0 at exit pupil
            Only works for `quick=False`.
        
        Returns
        -------
        None.
        
        """
        
        from scipy.interpolate import interp1d
        
        # Work with oversampled pixels and downsample at end
        siaf_ap = self.inst_on.siaf_ap
        osamp = self.inst_on.oversample
        ny = nx = self.fov_pix * osamp
        
        # Renormalize spectrum to have 1 e-/sec within bandpass to obtain normalized PSFs
        if sp is not None:
            sp = _sp_to_spext(sp)
            sp = sp.renorm(1, 'counts', self.bandpass)
        
        if self.name.upper()=='NIRCAM' and quick:
            inst_on = self.inst_on

            # Information for bar offsetting (in arcsec)
            bar_offset = inst_on.get_bar_offset(ignore_options=True)
            bar_offset = 0 if bar_offset is None else bar_offset

            # cx and cy are transformed coordinate relative to center of mask in arcsec
            trans, cx, cy = inst_on.gen_mask_transmission_map(coord_vals, coord_frame, return_more=True)
            cx_idl = cx - bar_offset

            # Linear combination of min/max to determine PSF
            # Get a and b values for each position
            avals = trans
            bvals = 1 - avals
            
            if self.image_mask[-1]=='B':
                # Interpolation function
                xvals = self.psf_bar_xvals
                psf_arr = self.psf_on
                finterp = interp1d(xvals, psf_arr, kind='linear', fill_value='extrapolate', axis=0)
                psf_on = finterp(cx_idl)
            else:
                psf_on = self.psf_on
            psf_off = self.psf_off
            
            psfs = avals.reshape([-1,1,1]) * psf_off.reshape([1,ny,nx]) \
                 + bvals.reshape([-1,1,1]) * psf_on.reshape([1,ny,nx])
        
        else:
            calc_psf = self._func_on
            sp = self.sp if sp is None else sp
            psfs = calc_psf(sp=sp, coord_vals=coord_vals, coord_frame=coord_frame,
                            return_oversample=True, return_hdul=False, normalize=normalize)
            
            # Ensure 3D cube
            psfs = psfs.reshape([-1,ny,nx])
            
            # Perform shift to center
            # Already done for quick case
            xoff, yoff = self._xy_off_to_cen_osamp
            psfs = fourier_imshift(psfs, xoff, yoff, pad=True)
        
        if do_shift:
            # Get offset in idl frame
            if coord_frame=='idl':
                xidl, yidl = coord_vals
            else:
                xidl, yidl = siaf_ap.convert(coord_vals[0], coord_vals[1], coord_frame, 'idl')
            
            # Convert to pixels for shifting
            dx_pix = np.array([osamp * xidl / siaf_ap.XSciScale]).ravel()
            dy_pix = np.array([osamp * yidl / siaf_ap.YSciScale]).ravel()
            
            psfs_sh = []
            for i, im in enumerate(psfs):
                psf = fourier_imshift(im, dx_pix[i], dy_pix[i], pad=True)
                psfs_sh.append(psf)
            psfs = np.asarray(psfs_sh)
        
        # Resample to detector pixels?
        if not return_oversample:
            psfs = frebin(psfs, scale=1/osamp)
        
        return psfs.squeeze()
    
    def gen_psf(self, loc, mode='xy', PA_V3=0, return_oversample=False, 
                do_shift=True, addV3Yidl=True, normalize='first', **kwargs):
        """
        Generate offset PSF rotated by PA to N-E orientation.
        
        Generate a PSF for some (x,y) detector position in N-E sky orientation.
        
        Parameters
        ----------
        loc : float or ndarray
            (x,y) or (r,th) location (in arcsec) offset from center of mask.
        PA_V3 : float
            V3 PA of ref point N over E (e.g. 'ROLL_REF'). Will add 'V3IdlYAngle'.
        return_oversample : bool
            Return the oversampled version of the PSF?
        do_shift : bool
            If True, will offset PSF by some amount from center. Otherwise,
            returns PSF in center of image.
        addV3Yidl : bool
            Add V3IdlYAngle to PA_V3 when converting (r,th) to (x,y) in idl coords?
            This assumes that (r,th) are not already in idl coords, but are instead
            relative to North / East sky coordinates.
        normalize : str
            How to normalize the PSF. Options are:
                * 'first': Normalize to 1.0 at entrance pupil
                * 'exit_pupil': Normalize to 1.0 at exit pupil
            Only works for `quick=False`.
                        
        Keyword Args
        ------------
        quick : bool
            Use linear combination of on-axis and off-axis PSFs to generate
            PSF as a function of corongraphic mask throughput. Typically takes
            10s of msec, compared to standard calculations using coefficients 
            (~1 sec) or on-the-fly calcs w/ webbpsf (10s of sec).
            Only applicable for NIRCam.
        sp : :class:`webbpsf_ext.synphot_ext.Spectrum`
            Manually specify spectrum to get a desired wavelength weighting. 
            Only applicable if ``quick=False``. If not set, defaults to ``self.sp``.
        
        Returns
        -------
        None.
        
        """
        
        # Work with oversampled pixels and downsample at end
        siaf_ap = self.inst_on.siaf_ap
        osamp = self.inst_on.oversample
        ny = nx = self.fov_pix * osamp
        
        # Locations in aperture ideal frame to produce PSFs
        if mode == 'rth':
            r, th = loc
            xidl, yidl = self.rth_to_xy(r, th, PA_V3=PA_V3, frame_out='idl', 
                                        addV3Yidl=addV3Yidl)
        elif mode == 'xy':
            xidl, yidl = loc
        
        # Perform shift in idl frame then rotate to sky coords
        psf = self.gen_psf_idl((xidl, yidl), coord_frame='idl', do_shift=do_shift, 
                                return_oversample=True, normalize=normalize, **kwargs)
        
        if do_shift:
            # Shifting PSF, means rotate such that North is up
            psf = psf.reshape([-1,ny,nx])
            # Get aperture position angle
            PA_ap = PA_V3 + siaf_ap.V3IdlYAngle
            psf = rotate(psf, -PA_ap, reshape=False, mode='constant', cval=0, axes=(-1,-2))
        
        # Resample to detector pixels?
        if not return_oversample:
            psf = frebin(psf, scale=1/osamp)
        
        psf = psf.squeeze()
        
        return psf


def _sp_to_spext(sp, **kwargs):
    """Check if input spectrum is a synphot spectrum and convert to webbpsf_ext spectrum"""

    try:
        wave = sp.wave
        flux = sp.flux
    except AttributeError:
        # Assume it's a synphot spectrum
        wave = sp.waveset
        flux = sp(sp.wave)
        wunit = wave.unit.to_string()
        funit = flux.unit.to_string()
        sp = S.ArraySpectrum(wave.value, flux.value, waveunits=wunit, fluxunits=funit, 
                             name=sp.meta['name'], **kwargs)

    return sp


def recenter_jens(image):
    """
    Find the shift that centers a PSF on its nearest pixel by maximizing its
    peak count.
    
    Parameters
    ----------
    image : 2D-array
        Input image to be recentered.
    
    Returns
    -------
    shift : 1D-array
        X- and y-shift that centers the PSF.
    
    """

    from .utils import recenterlsq
    
    # Find the shift that recenters the image.
    p0 = np.array([0., 0.])
    shift = minimize(recenterlsq,
                     p0,
                     args=(image))['x']
    
    return shift

def get_offsetpsf(obs,
                  recenter=True,
                  derotate=True):
    """
    Compute a derotated and integration time weighted average of a WebbPSF
    model PSF.
    
    Parameters
    ----------
    obs : astropy table
        Concatenation of a spaceKLIP observations database for which the
        derotated and integration time weighted average of the model PSF shall
        be computed.
    recenter : bool, optional
        Recenter the model PSF? The offset PSF from WebbPSF is not properly
        centered because the wedge mirror that folds the light onto the
        coronagraphic subarrays introduces a chromatic shift. The default is
        True.
    derotate : bool, optional
        Derotate and integration time weigh the model PSF? The default is True.
    
    Returns
    -------
    totpsf : 2D-array
        Derotated and integration time weighted average of the model PSF.
    
    """

    from .utils import imshift
    
    # Generate an unocculted model PSF using WebbPSF.
    offsetpsf = gen_offsetpsf(obs)
    
    # Recenter the offset PSF.
    if recenter:
        shift = recenter_jens(offsetpsf)
        offsetpsf = imshift(offsetpsf, shift, pad=True)
        ww_max = np.unravel_index(np.argmax(offsetpsf), offsetpsf.shape)
        if ww_max != (32, 32):
            dx, dy = 32 - ww_max[1], 32 - ww_max[0]
            offsetpsf = np.roll(np.roll(offsetpsf, dx, axis=1), dy, axis=0)
    
    # Find the science target observations.
    ww_sci = np.where(obs['TYPE'] == 'SCI')[0]
    
    # Derotate the offset PSF and coadd it weighted by the integration time of
    # the different rolls. Scipy.ndimage.rotate rotates around the image
    # center, i.e., (32, 32) for an image of size (65, 65).
    if derotate:
        totpsf = []
        totexp = 0.  # s
        for j in ww_sci:
            totint = obs['NINTS'][j] * obs['EFFINTTM'][j]  # s
            totpsf += [totint * rotate(offsetpsf.copy(), -obs['ROLL_REF'][j], reshape=False, mode='constant', cval=0.)]
            totexp += totint  # s
        totpsf = np.array(totpsf)
        totpsf = np.sum(totpsf, axis=0) / totexp
    else:
        totpsf = offsetpsf
    
    return totpsf

def gen_offsetpsf(obs,
                  xyoff=None,
                  date=None,
                  source=None):
    """
    Generate a WebbPSF model PSF. The total intensity will be normalized to 1.
    
    Parameters
    ----------
    obs : astropy table
        Concatenation of a spaceKLIP observations database for which the model
        PSF shall be computed.
    xyoff : tuple, optional
        Offset (arcsec) from coronagraphic mask center in detector coordinates
        to generate position-dependent PSF. The default is None.
    date : str, optional
        Observation date in the format 'YYYY-MM-DDTHH:MM:SS.MMM'. Will query
        for the wavefront measurement closest in time *before* the given date.
        If None, then the default WebbPSF OPD is used (RevAA). The default is
        None.
    source : synphot.spectrum.SourceSpectrum, optional
        Defaults to a 5700 K blackbody. The default is None.
    
    Returns
    -------
    offsetpsf : 2D-array
        WebbPSF model PSF.
    
    """
    
    # Find the science target observations.
    ww_sci = np.where(obs['TYPE'] == 'SCI')[0]
    
    # JWST.
    if obs['TELESCOP'][ww_sci[0]] == 'JWST':
        
        # NIRCam.
        if obs['INSTRUME'][ww_sci[0]] == 'NIRCAM':
            nircam = webbpsf.NIRCam()
            
            # Apply the correct pupil mask, but no image mask (unocculted PSF).
            # if obs['CORONMSK'][ww_sci[0]] in ['MASKA210R', 'MASKA335R', 'MASKA430R']:
            #     nircam.pupil_mask = 'MASKRND'
            # elif obs['CORONMSK'][ww_sci[0]] in ['MASKASWB']:
            #     nircam.pupil_mask = 'MASKSWB'
            # elif obs['CORONMSK'][ww_sci[0]] in ['MASKALWB']:
            #     nircam.pupil_mask = 'MASKLWB'
            if obs['PUPIL'][ww_sci[0]] != 'NONE':
                if obs['PUPIL'][ww_sci[0]] == 'MASKBAR':
                    if 'LWB' in obs['CORONMSK'][ww_sci[0]]:
                        nircam.pupil_mask = 'MASKLWB'
                    else:
                        nircam.pupil_mask = 'MASKSWB'
                else:
                    nircam.pupil_mask = obs['PUPIL'][ww_sci[0]]
            else:
                nircam.pupil_mask = None
            if xyoff is not None:  # assume that if an offset is applied, the PSF should be relative to the coronagraphic mask center
                nircam.image_mask = obs['CORONMSK'][ww_sci[0]]
            else:
                nircam.image_mask = None
            webbpsf_inst = nircam
        
        # NIRISS.
        elif obs['INSTRUME'][ww_sci[0]] == 'NIRISS':
            niriss = webbpsf.NIRISS()
            
            # Apply the correct pupil mask, but no image mask (unocculted PSF).
            if obs['PUPIL'][ww_sci[0]] != 'NONE':
                niriss.pupil_mask = obs['PUPIL'][ww_sci[0]]
            else:
                niriss.pupil_mask = None
            niriss.image_mask = None
            webbpsf_inst = niriss
        
        # MIRI.
        elif obs['INSTRUME'][ww_sci[0]] == 'MIRI':
            miri = webbpsf.MIRI()
            
            # Apply the correct pupil mask, but no image mask (unocculted PSF).
            if '4QPM' in obs['CORONMSK'][ww_sci[0]]:
                miri.pupil_mask = 'MASKFQPM'  # F not 4 for WebbPSF
            elif 'LYOT' in obs['CORONMSK'][ww_sci[0]]:
                miri.pupil_mask = 'MASKLYOT'
            if xyoff is not None:  # assume that if an offset is applied, the PSF should be relative to the coronagraphic mask center
                miri.image_mask = obs['CORONMSK'][ww_sci[0]].replace('4QPM_', 'FQPM')
            else:
                miri.image_mask = None
            webbpsf_inst = miri
        
        # Otherwise.
        else:
            raise UserWarning('Data originates from unknown JWST instrument')
    
    # Otherwise.
    else:
        raise UserWarning('Data originates from unknown telescope')
    
    # Apply the correct filter.
    webbpsf_inst.filter = obs['FILTER'][ww_sci[0]]
    
    # If an offset is applied, the PSF should be relative to the coronagraphic
    # mask center.
    if xyoff is not None:
        webbpsf_inst.options['source_offset_x'] = xyoff[0]
        webbpsf_inst.options['source_offset_y'] = xyoff[1]
    
    # If a date is provided, use date-specific OPD files.
    log.info('  --> Generating WebbPSF model')
    if date is not None:
        log.info('  --> Using date-specific OPD files')
        webbpsf_inst.load_wss_opd_by_date(date=date, choice='before', verbose=False, plot=False)
    
    # Generate offset PSF.
    hdul = webbpsf_inst.calc_psf(oversample=1, fov_pixels=65, normalize='exit_pupil', source=source)
    offsetpsf = hdul[0].data
    
    return offsetpsf

def get_transmission(obs):
    """
    Compute a derotated and integration time weighted average of the
    transmission mask.
    
    Parameters
    ----------
    obs : astropy table
        Concatenation of a spaceKLIP observations database for which the
        derotated and integration time weighted average of the transmission
        mask shall be computed.
    
    Returns
    -------
    totmsk : 2D-array
        Derotated and integration time weighted average of the transmission
        mask.
    
    """
    
    # Find the science target observations.
    ww_sci = np.where(obs['TYPE'] == 'SCI')[0]
    
    # Derotate the transmission mask and coadd it weighted by the integration
    # time of the different rolls.
    totmsk = []
    totexp = 0.  # s
    for j in ww_sci:
        
        # If there is no transmission mask for any of the rolls, return None.
        if obs['MASKFILE'][j] == 'NONE':
            return None
        
        # Else compute and return the transmission mask.
        hdul = pyfits.open(obs['MASKFILE'][j])
        mask = hdul['SCI'].data
        hdul.close()
        totint = obs['NINTS'][j] * obs['EFFINTTM'][j]  # s
        center = [obs['CRPIX1'][j] - 1., obs['CRPIX2'][j] - 1.]  # pix (0-indexed)
        new_center = [mask.shape[1] // 2, mask.shape[0] // 2]  # pix (0-indexed)
        totmsk += [totint * nanrotate(mask.copy(), obs['ROLL_REF'][j], center=center, new_center=new_center)]
        totexp += totint  # s
    totmsk = np.array(totmsk)
    
    # Correctly handle nans.
    ww = np.isnan(totmsk)
    ww = np.sum(ww, axis=0) == ww.shape[0]
    totmsk = np.nansum(totmsk, axis=0) / totexp
    totmsk[ww] = np.nan
    
    return totmsk
