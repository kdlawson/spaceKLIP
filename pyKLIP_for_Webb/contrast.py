import sys, re, os

import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

import pyklip.klip as klip

from . import utils

rad2mas = 180./np.pi*3600.*1000.

def raw_contrast_curve(params, obs): # vegamag
    """
    Compute the raw contrast curves. Known companions and the location of
    the bar mask in both rolls can be masked out.
    
    Note: currently masks a circle with a radius of 12 FWHM = ~12 lambda/D
          around known companions. This was found to be a sufficient but
          necessary value based on tests with simulated data.
    
    TODO: tries to convert MJy/sr to contrast using the host star
          magnitude, but the results are wrong by a factor of ~5.
    
    TODO: currently uses WebbPSF to compute a theoretical offset PSF. It
          should be possible to use PSF stamps extracted from the
          astrometric confirmation images to determine the pupil mask
          throughput.
    
    Parameters
    ----------
    mstar: dict of float
        Host star magnitude in each filter. Must contain one entry for
        each filter used in the data in the input directory.
    ra_off: list of float
        RA offset of the known companions.
    de_off: list of float
        DEC offset of the known companions.
    pa_ranges_bar: list of tuple of float
        List of tuples defining the pizza slices that shall be considered
        when computing the contrast curves for the bar masks.
    """
    verbose = params.verbose
    
    if (verbose == True):
        print('--> Computing raw contrast curve...')
    
    # Loop through all modes, numbers of annuli, and numbers of
    # subsections.
    Nscenarios = len(params.mode)*len(params.annuli)*len(params.subsections)
    counter = 1
    for mode in params.mode:
        for annuli in params.annuli:
            for subsections in params.subsections:
                
                if (verbose == True):
                    sys.stdout.write('\r--> Mode = '+mode+', annuli = %.0f, subsections = %.0f, scenario %.0f of %.0f' % (annuli, subsections, counter, Nscenarios))
                    sys.stdout.flush()
                
                # Define the input and output directories for each set of
                # pyKLIP parameters.
                idir = params.odir+mode+'_annu%.0f_subs%.0f/FITS/' % (annuli, subsections)
                odir = params.odir+mode+'_annu%.0f_subs%.0f/CONS/' % (annuli, subsections)
                if (not os.path.exists(odir)):
                    os.makedirs(odir)
                
                # Loop through all sets of observing parameters.
                for i, key in enumerate(obs.keys()):
                    hdul = pyfits.open(idir+key+'-KLmodes-all.fits')
                    data = hdul[0].data
                    pxsc = obs[key]['PIXSCALE'][0] # mas
                    cent = (hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']) # pix
                    temp = [s.start() for s in re.finditer('_', key)]
                    filt = key[temp[1]+1:temp[2]]
                    mask = key[temp[3]+1:temp[4]]
                    subarr = key[temp[4]+1:]
                    wave = params.wave[filt] # m
                    fwhm = wave/params.diam*rad2mas/pxsc # pix
                    hdr = hdul[0].header
                    hdul.close()
                    
                    # Mask out known companions and the location of the
                    # bar mask in both rolls.
                    data_masked = mask_companions(data, pxsc, cent, 12.*fwhm, params.ra_off, params.de_off)
                    if (mask in ['MASKASWB', 'MASKALWB']):
                        data_masked = mask_bar(data_masked, cent, params.pa_ranges_bar)
                    
                    # Plot.
                    extl = (data.shape[1]+1.)/2.*pxsc/1000. # arcsec
                    extr = (data.shape[1]-1.)/2.*pxsc/1000. # arcsec
                    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
                    ax[0].imshow(np.log10(np.abs(data[-1])), origin='lower', cmap='inferno', extent=(extl, -extr, -extl, extr))
                    for i in range(len(params.ra_off)):
                        cc = plt.Circle((params.ra_off[i]/1000., params.de_off[i]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
                        ax[0].add_artist(cc)
                    ax[0].set_xlabel('$\Delta$RA [arcsec]')
                    ax[0].set_ylabel('$\Delta$DEC [arcsec]')
                    ax[0].set_title('KLIP-subtracted')
                    ax[1].imshow(np.log10(np.abs(data_masked[-1])), origin='lower', cmap='inferno', extent=(extl, -extr, -extl, extr))
                    ax[1].set_xlabel('$\Delta$RA [arcsec]')
                    ax[1].set_ylabel('$\Delta$DEC [arcsec]')
                    if (mask in ['MASKASWB', 'MASKALWB']):
                        ax[1].set_title('Companions & bar masked')
                    else:
                        ax[1].set_title('Companions masked')
                    plt.tight_layout()
                    plt.savefig(odir+key+'-mask.pdf')
                    plt.close()
                    
                    # Convert the units and compute the contrast. Use an
                    # unocculted offset PSF from WebbPSF to obtain the
                    # peak count of the host star. The unocculted offset
                    # PSF from WebbPSF is normalized to an integrated
                    # source of 1 and takes into account the throughput
                    # of the pupil mask.
                    offsetpsf = utils.get_offsetpsf(params.offsetpsfdir, obs, filt, mask, key)
                    Fstar = params.F0[filt]/10.**(params.mstar[filt]/2.5)/1e6*np.max(offsetpsf) # MJy; convert the host star brightness from vegamag to MJy
                    Fdata = data_masked*pxsc**2/(180./np.pi*3600.*1000.)**2 # MJy; convert the data from MJy/sr to MJy
                    seps = [] # arcsec
                    cons = []
                    for i in range(Fdata.shape[0]):
                        sep, con = klip.meas_contrast(dat=Fdata[i]/Fstar, iwa=params.iwa, owa=params.owa, resolution=2.*fwhm, center=cent, low_pass_filter=False)
                        seps += [sep*pxsc/1000.] # arcsec
                        cons += [con]
                    seps = np.array(seps) # arcsec
                    cons = np.array(cons)
                    np.save(odir+key+'-seps.npy', seps) # arcsec
                    np.save(odir+key+'-cons.npy', cons)
                    
                    # Plot.
                    plt.figure(figsize=(6.4, 4.8))
                    ax = plt.gca()
                    for i in range(Fdata.shape[0]):
                        ax.plot(seps[i], cons[i], label=str(hdr['KLMODE{}'.format(i)])+' KL')
                    ax.set_yscale('log')
                    ax.set_xlim([0., np.max(seps)]) # arcsec
                    ax.grid(axis='y')
                    ax.set_xlabel('Separation [arcsec]')
                    ax.set_ylabel('Contrast [5$\sigma$]')
                    ax.set_title('Raw contrast curve')
                    ax.legend(loc='upper right')
                    plt.tight_layout()
                    plt.savefig(odir+key+'-cons_raw.pdf')
                    plt.close()
                counter += 1
    
    if (verbose == True):
        print('')
    
    return None

def cal_contrast_curve(self,
                       mstar, # vegamag
                       ra_off=[], # mas
                       de_off=[], # mas
                       seps_inject_rnd=[], # pix
                       pas_inject_rnd=[], # deg
                       seps_inject_bar=[], # pix
                       pas_inject_bar=[], # deg
                       KL=-1,
                       overwrite=False):
    """
    Compute the calibrated contrast curves. Injection and recovery tests
    are performed to estimate the algo & coronmsk throughput.
    
    Note: do not inject fake planets on top of the bar mask in either of
          the rolls!
    
    Note: currently forces any fake companions to be injected at least 10
          FWHM = ~10 lambda/D away from other fake companions or known
          companions. This was found to be a sufficient but necessary
          value based on tests with simulated data.
    
    TODO: use a position dependent offset PSF from pyNRC instead of the
          completely unocculted offset PSF from WebbPSF.
    
    Parameters
    ----------
    mstar: dict of float
        Host star magnitude in each filter. Must contain one entry for
        each filter used in the data in the input directory.
    ra_off: list of float
        RA offset of the known companions.
    de_off: list of float
        DEC offset of the known companions.
    seps_inject_rnd: list of float
        List of separations at which fake planets shall be injected to
        compute the calibrated contrast curve for the round masks.
    pas_inject_rnd: list of float
        List of position angles at which fake planets shall be injected to
        compute the calibrated contrast curve for the round masks.
    seps_inject_bar: list of float
        List of separations at which fake planets shall be injected to
        compute the calibrated contrast curve for the bar masks.
    pas_inject_bar: list of float
        List of position angles at which fake planets shall be injected to
        compute the calibrated contrast curve for the bar masks.
    KL: int
        Index of the KL component for which the calibrated contrast curve
        and the companion properties shall be computed.
    overwrite: bool
        If true overwrite existing data.
    """
    
    if (self.verbose == True):
        print('--> Computing calibrated contrast curve...')
    
    # Make inputs arrays.
    seps_inject_rnd = np.array(seps_inject_rnd)
    pas_inject_rnd = np.array(pas_inject_rnd)
    seps_inject_bar = np.array(seps_inject_bar)
    pas_inject_bar = np.array(pas_inject_bar)
    
    # Loop through all modes, numbers of annuli, and numbers of
    # subsections.
    Nscenarios = len(self.mode)*len(self.annuli)*len(self.subsections)
    counter = 1
    for mode in self.mode:
        for annuli in self.annuli:
            for subsections in self.subsections:
                
                if (self.verbose == True):
                    sys.stdout.write('\r--> Mode = '+mode+', annuli = %.0f, subsections = %.0f, scenario %.0f of %.0f' % (annuli, subsections, counter, Nscenarios))
                    sys.stdout.flush()
                
                # Define the input and output directories for each set of
                # pyKLIP parameters.
                idir = self.odir+mode+'_annu%.0f_subs%.0f/FITS/' % (annuli, subsections)
                odir = self.odir+mode+'_annu%.0f_subs%.0f/CONS/' % (annuli, subsections)
                if (not os.path.exists(odir)):
                    os.makedirs(odir)
                
                # Loop through all sets of observing parameters.
                for i, key in enumerate(self.obs.keys()):
                    ww_sci = np.where(self.obs[key]['TYP'] == 'SCI')[0]
                    filepaths = np.array(self.obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
                    ww_cal = np.where(self.obs[key]['TYP'] == 'CAL')[0]
                    psflib_filepaths = np.array(self.obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
                    hdul = pyfits.open(idir+key+'-KLmodes-all.fits')
                    data = hdul[0].data
                    pxsc = self.obs[key]['PIXSCALE'][0] # mas
                    cent = (hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']) # pix
                    temp = [s.start() for s in re.finditer('_', key)]
                    filt = key[temp[1]+1:temp[2]]
                    mask = key[temp[3]+1:temp[4]]
                    subarr = key[temp[4]+1:]
                    wave = self.wave[filt] # m
                    fwhm = wave/self.diam*rad2mas/pxsc # pix
                    hdul.close()
                    
                    # Load raw contrast curves. If overwrite is false,
                    # check whether the calibrated contrast curves have
                    # been computed already.
                    seps = np.load(odir+key+'-seps.npy')[KL] # arcsec
                    cons = np.load(odir+key+'-cons.npy')[KL]
                    if (overwrite == False):
                        try:
                            flux_all = np.load(odir+key+'-flux_all.npy') # MJy/sr
                            seps_all = np.load(odir+key+'-seps_all.npy') # pix
                            pas_all = np.load(odir+key+'-pas_all.npy') # deg
                            flux_retr_all = np.load(odir+key+'-flux_retr_all.npy') # MJy/sr
                            todo = False
                        except:
                            todo = True
                    else:
                        todo = True
                    
                    # 2D map of the total throughput, i.e., an integration
                    # time weighted average of the coronmsk transmission
                    # over the rolls.
                    tottp = self.get_transmission(pxsc, filt, mask, subarr, odir, key)
                    
                    # The calibrated contrast curves have not been
                    # computed already.
                    if (todo == True):
                        
                        # Offset PSF from WebbPSF, i.e., an integration
                        # time weighted average of the unocculted offset
                        # PSF over the rolls (does account for pupil mask
                        # throughput).
                        offsetpsf = self.get_offsetpsf(filt, mask, key)
                        
                        # Convert the units and compute the injected
                        # fluxes. They need to be in the units of the data
                        # which is MJy/sr.
                        Fstar = self.F0[filt]/10.**(mstar[filt]/2.5)/1e6*np.max(offsetpsf) # MJy; convert the host star brightness from vegamag to MJy
                        if (mask in ['MASKASWB', 'MASKALWB']):
                            cons_inject = np.interp(seps_inject_bar*pxsc/1000., seps, cons)*5. # inject at 5 times 5 sigma
                        else:
                            cons_inject = np.interp(seps_inject_rnd*pxsc/1000., seps, cons)*5. # inject at 5 times 5 sigma
                        flux_inject = cons_inject*Fstar*(180./np.pi*3600.*1000.)**2/pxsc**2 # MJy/sr; convert the injected flux from contrast to MJy/sr
                        
                        # If the separation is too small the
                        # interpolation of the contrasts returns nans,
                        # these need to be filtered out before feeding the
                        # fluxes into the injection and recovery routine.
                        good = np.isnan(flux_inject) == False
                        if (mask in ['MASKASWB', 'MASKALWB']):
                            flux_all, seps_all, pas_all, flux_retr_all = self.inject_recover(filepaths, psflib_filepaths, mode, odir, key, annuli, subsections, pxsc, filt, mask, 10.*fwhm, flux_inject[good], seps_inject_bar[good], pas_inject_bar, KL, ra_off, de_off)
                        else:
                            flux_all, seps_all, pas_all, flux_retr_all = self.inject_recover(filepaths, psflib_filepaths, mode, odir, key, annuli, subsections, pxsc, filt, mask, 10.*fwhm, flux_inject[good], seps_inject_rnd[good], pas_inject_rnd, KL, ra_off, de_off)
                        np.save(odir+key+'-flux_all.npy', flux_all) # MJy/sr
                        np.save(odir+key+'-seps_all.npy', seps_all) # pix
                        np.save(odir+key+'-pas_all.npy', pas_all) # deg
                        np.save(odir+key+'-flux_retr_all.npy', flux_retr_all) # MJy/sr
                    
                    # Group the injection and recovery results by
                    # separation and take the median, then fit a logistic
                    # growth function to them.
                    res = Table([flux_all, seps_all, pas_all, flux_retr_all], names=('flux', 'seps', 'pas', 'flux_retr'))
                    res['tps'] = res['flux_retr']/res['flux']
                    med_res = res.group_by('seps')
                    med_res = med_res.groups.aggregate(np.nanmedian)
                    p0 = np.array([1., 0., 1., 0.2, 15.])
                    pp = least_squares(self.func_lnprob, p0, args=(med_res['seps'], med_res['tps']))
                    # p0 = np.array([0., 1., 1.])
                    # pp = least_squares(self.growth_lnprob, p0, args=(med_res['seps']*pxsc/1000., med_res['tps']))
                    corr_cons = cons/self.func(pp['x'], seps*1000./pxsc)
                    np.save(odir+key+'-pp.npy', pp['x'])
                    
                    # Plot.
                    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
                    extl = (data.shape[1]+1.)/2.*pxsc/1000. # arcsec
                    extr = (data.shape[1]-1.)/2.*pxsc/1000. # arcsec
                    ax[0].imshow(np.log10(np.abs(data[KL])), origin='lower', cmap='inferno', extent=(extl, -extr, -extl, extr))
                    for i in range(len(ra_off)):
                        cc = plt.Circle((ra_off[i]/1000., de_off[i]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
                        ax[0].add_artist(cc)
                    for i in range(len(flux_all)):
                        ra = seps_all[i]*pxsc*np.sin(np.deg2rad(pas_all[i])) # mas
                        de = seps_all[i]*pxsc*np.cos(np.deg2rad(pas_all[i])) # mas
                        cc = plt.Circle((ra/1000., de/1000.), 10.*pxsc/1000., fill=False, edgecolor='red', linewidth=3)
                        ax[0].add_artist(cc)
                    ax[0].set_xlim([5., -5.])
                    ax[0].set_ylim([-5., 5.])
                    ax[0].set_xlabel('$\Delta$RA [arcsec]')
                    ax[0].set_ylabel('$\Delta$DEC [arcsec]')
                    ax[0].set_title('KLIP-subtracted')
                    extl = tottp.shape[1]/2.*pxsc/1000. # arcsec
                    extr = tottp.shape[1]/2.*pxsc/1000. # arcsec
                    p1 = ax[1].imshow(tottp, origin='lower', cmap='viridis', vmin=0., vmax=1., extent=(extl, -extr, -extl, extr))
                    c1 = plt.colorbar(p1, ax=ax[1])
                    c1.set_label('Transmission', rotation=270, labelpad=20)
                    for i in range(len(ra_off)):
                        cc = plt.Circle((ra_off[i]/1000., de_off[i]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
                        ax[1].add_artist(cc)
                    for i in range(len(flux_all)):
                        ra = seps_all[i]*pxsc*np.sin(np.deg2rad(pas_all[i])) # mas
                        de = seps_all[i]*pxsc*np.cos(np.deg2rad(pas_all[i])) # mas
                        cc = plt.Circle((ra/1000., de/1000.), 10.*pxsc/1000., fill=False, edgecolor='red', linewidth=3)
                        ax[1].add_artist(cc)
                    ax[1].set_xlim([5., -5.])
                    ax[1].set_ylim([-5., 5.])
                    ax[1].set_xlabel('$\Delta$RA [arcsec]')
                    ax[1].set_ylabel('$\Delta$DEC [arcsec]')
                    ax[1].set_title('Transmission')
                    plt.tight_layout()
                    plt.savefig(odir+key+'-cons_inj.pdf')
                    plt.close()
                    
                    # Plot.
                    if (mask in ['MASKASWB', 'MASKALWB']):
                        xx = np.linspace(seps_inject_bar[0], seps_inject_bar[-1], 100)
                    else:
                        xx = np.linspace(seps_inject_rnd[0], seps_inject_rnd[-1], 100)
                    f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
                    ax[0].plot(med_res['seps'], med_res['tps'], color='mediumaquamarine', label='Median throughput')
                    ax[0].scatter(res['seps'], res['tps'], s=75, color='mediumaquamarine', alpha=0.5)
                    ax[0].plot(xx, self.func(pp['x'], xx), color='teal', label='Best fit model')
                    ax[0].set_xlim([xx[0], xx[-1]])
                    ax[0].set_ylim([0.0, 1.2])
                    ax[0].grid(axis='y')
                    ax[0].set_xlabel('Separation [pix]')
                    ax[0].set_ylabel('Throughput')
                    ax[0].set_title('Algo & coronmsk throughput')
                    ax[0].legend(loc='lower right')
                    ax[1].plot(seps, cons, color='mediumaquamarine', label='Raw contrast')
                    ax[1].plot(seps, corr_cons, color='teal', label='Calibrated contrast')
                    ax[1].set_yscale('log')
                    ax[1].set_xlim([0., np.max(seps)]) # arcsec
                    ax[1].grid(axis='y')
                    ax[1].set_xlabel('Separation [arcsec]')
                    ax[1].set_ylabel('Contrast [5$\sigma$]')
                    ax[1].set_title('Calibrated contrast curve')
                    ax[1].legend(loc='upper right')
                    plt.tight_layout()
                    plt.savefig(odir+key+'-cons_cal.pdf')
                    plt.close()
                counter += 1
    
    if (self.verbose == True):
        print('')
    
    return None


def mask_companions(data,
                    pxsc, # mas
                    cent, # pix
                    mrad, # pix
                    ra_off=[], # mas
                    de_off=[]): # mas
    """
    Mask out known companions.
    
    Parameters
    ----------
    data: array
        Data cube of shape (Nframes, Npix, Npix) in which the companions
        shall be masked out.
    pxsc: float
        Pixel scale of the data.
    cent: tuple of float
        Center of the data (i.e., center of the host star PSF).
    mrad: float
        Radius of the mask that is used to mask out the companions.
    ra_off: list of float
        RA offset of the companions that shall be masked out.
    de_off: list of float
        DEC offset of the companions that shall be masked out.
    
    Returns
    -------
    data_masked: array
        Data cube of shape (Nframes, Npix, Npix) in which the companions
        are masked out.
    """
    
    # Sanitize inputs.
    if (len(data.shape) != 3):
        raise UserWarning('Data has invalid shape')
    
    # Mask out known companions.
    data_masked = data.copy()
    yy, xx = np.indices(data.shape[1:]) # pix
    for ra, de in zip(ra_off, de_off):
        dist = np.sqrt((xx-cent[0]+ra/pxsc)**2+(yy-cent[1]-de/pxsc)**2) # pix
        data_masked[:, dist <= mrad] = np.nan
    
    return data_masked


def mask_bar(data,
             cent, # pix
             pa_ranges_bar=[]): # deg
    """
    Mask out bar mask occulter. This is done by specifying pizza slices
    that shall be considered when computing the contrast curves.
    
    Note: make sure to mask out the bar mask in either of the rolls!
    
    Parameters
    ----------
    data: array
        Data cube of shape (Nframes, Npix, Npix) in which the bar mask
        occulter shall be masked out.
    cent: tuple of float
        Center of the data (i.e., center of the host star PSF).
    pa_ranges_bar: list of tuple of float
        List of tuples defining the pizza slices that shall be considered
        when computing the contrast curves for the bar masks.
    
    Returns
    -------
    data_masked: array
        Data cube of shape (Nframes, Npix, Npix) in which everything but
        the specified pizza slices is masked out.
    """
    
    # Sanitize inputs.
    if (len(data.shape) != 3):
        raise UserWarning('Data has invalid shape')
    
    # Mask out bar mask occulter.
    data_masked = data.copy()
    yy, xx = np.indices(data.shape[1:]) # pix
    tt = np.rad2deg(-1.*np.arctan2((xx-cent[0]), (yy-cent[1]))) % 360. # deg
    for i in range(len(pa_ranges_bar)):
        if (i == 0):
            data_masked[:] = np.nan
        mask = (tt >= pa_ranges_bar[i][0]) & (tt <= pa_ranges_bar[i][1])
        data_masked[:, mask] = data[:, mask]
    
    return data_masked
