import os, glob

import numpy as np
from datetime import datetime

from jwst.pipeline.calwebb_detector1 import Detector1Pipeline
from jwst.datamodels import dqflags, RampModel

from webbpsf_ext import robust

# Define logging
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Coron1Pipeline(Detector1Pipeline):
    """ Coron1Pipeline
    
    Apply all calibration steps to raw JWST ramps to produce 
    a 2-D slope product. Custom sub-class of ``Detector1Pipeline`` 
    with modifications for coronagraphic data.
    
    Included steps are: group_scale, dq_init, saturation, ipc, 
    superbias, refpix, rscd, firstframe, lastframe, linearity, 
    dark_current, reset, persistence, jump detection, ramp_fit, 
    and gain_scale. 
    """

    class_alias = "calwebb_coron1"

    spec = """
        nrow_ref           = integer(default=20)    # Number of rows for pseudo-ref correction
        grow_diagonal      = boolean(default=False) # Grow saturation along diagonal pixels?
        save_intermediates = boolean(default=False) # Save all intermediate step results
    """
    
    # start the actual processing
    def process(self, input):
        
    
        log.info(f'Starting {self.class_alias} ...')

        # open the input as a RampModel
        input = RampModel(input)

        instrument = input.meta.instrument.name
        if instrument == 'MIRI':

            # process MIRI exposures;
            # the steps are in a different order than NIR
            log.debug('Processing a MIRI exposure')

            input = self.run_step(self.group_scale, input)
            input = self.run_step(self.dq_init, input)
            input = self.run_step(self.saturation, input)
            input = self.run_step(self.ipc, input)
            input = self.run_step(self.firstframe, input)
            input = self.run_step(self.lastframe, input)
            input = self.run_step(self.reset, input)
            input = self.run_step(self.linearity, input)
            input = self.run_step(self.rscd, input)
            input = self.run_step(self.dark_current, input)
            input = self.run_step(self.refpix, input)

            # skip until MIRI team has figured out an algorithm
            # input = self.persistence(input)

        else:

            # process Near-IR exposures
            log.debug('Processing a Near-IR exposure')

            input = self.run_step(self.group_scale, input)
            input = self.run_step(self.dq_init, input)
            input = self.do_saturation(input)
            input = self.run_step(self.ipc, input)
            input = self.run_step(self.superbias, input)
            input = self.do_nircam_refpix(input)
            input = self.run_step(self.linearity, input)
            input = self.run_step(self.persistence, input)
            input = self.run_step(self.dark_current, input)

        # apply the jump step
        input = self.run_step(self.jump, input, save_results=False)

        # save the corrected ramp data, if requested
        if self.save_calibrated_ramp or self.save_intermediates:
            self.save_model(input, suffix='ramp')

        # Apply the ramp_fit step
        # This explicit test on self.ramp_fit.skip is a temporary workaround
        # to fix the problem that the ramp_fit step ordinarily returns two
        # objects, but when the step is skipped due to `skip = True`,
        # only the input is returned when the step is invoked.
        # Don't save results here, as it's basically the same as rate and rateints.
        res = self.run_step(self.ramp_fit, input, save_results=False)
        input, ints_model = (res, None) if self.ramp_fit.skip else res

        if input is None:
            log.info("NoneType returned from ramp_fit.  Gain Scale step skipped.")                
        else:
            # apply the gain_scale step to the exposure-level product
            self.gain_scale.suffix = 'rate'
            # Don't save here. Generally saved on output of run() or call().
            input = self.run_step(self.gain_scale, input, save_results=False)

        # apply the gain scale step to the multi-integration product,
        # if it exists, and then save it
        if ints_model is not None:
            self.gain_scale.suffix = 'rateints'
            ints_model = self.run_step(self.gain_scale, ints_model, save_results=False)
            if self.save_results or self.save_intermediates:
                self.save_model(ints_model, suffix='rateints')

        # setup output_file for saving
        self.setup_output(input)

        log.info(f'... ending {self.class_alias}')

        return input


    def run_step(self, step_obj, input, save_results=None, **kwargs):
        """Run a Step with option to save results. 
        
        Saving results tends to increase memory usage due to a memory leak
        in the datamodels.
        """

        # Determine if we're saving results for real
        if step_obj.skip:
            # Skip save if step is skipped
            really_save_results = False
        elif (save_results is not None):
            # Use keyword specifications
            really_save_results = save_results
        elif self.save_intermediates:
            # Use save_intermediates attribute
            really_save_results = True
        elif step_obj.save_results:
            # Use step attribute
            really_save_results = True
        else:
            # Saving is unspecified
            really_save_results = False

        # Run step. Don't save yet.
        step_save_orig = step_obj.save_results
        step_obj.save_results = False
        res = step_obj(input)
        step_obj.save_results = step_save_orig
        
        # Check if certain steps were skipped
        if step_obj is self.group_scale:
            if res.meta.cal_step.group_scale == 'SKIPPED':
                really_save_results = False
        elif step_obj is self.gain_scale:
            if res.meta.cal_step.gain_scale == 'SKIPPED':
                really_save_results = False
        
        # Now save results if asked
        if really_save_results:
            step_obj.output_dir = self.output_dir
            if isinstance(res, (tuple)):
                self.save_model(res[0], suffix=step_obj.suffix+'0')
                self.save_model(res[1], suffix=step_obj.suffix+'1')
            else:
                self.save_model(res, suffix=step_obj.suffix)

        return res
    
    def do_nircam_refpix(self, input, **kwargs):
        """ Pseudo-Refpix for NIRCam subarrays
        
        If full frame or MIRI, will still run RefPix Step by default.
        
        Parameters
        ==========
        input : 
            Data model or FITS file name.
        
        Keyword Args
        ============
        save_results : bool
            Explictly specify whether or not to save results.
        """
        
        # Slight modifications if NIRCam subarray
        instrument = input.meta.instrument.name
        subsize = input.meta.subarray.xsize
        
        # Perform normal operations if not set
        nrow_ref = self.nrow_ref
        if not (instrument.lower()=='nircam' and subsize<2048 and nrow_ref>0):
            return self.run_step(self.refpix, input, **kwargs)
        
        # Update pixel DQ mask to manually set reference pixels
        log.info(f'Flagging {nrow_ref} references rows at top and bottom of array')
        input.pixeldq[0:nrow_ref,:] = input.pixeldq[0:nrow_ref,:] | dqflags.pixel['REFERENCE_PIXEL']
        input.pixeldq[-nrow_ref:,:] = input.pixeldq[-nrow_ref:,:] | dqflags.pixel['REFERENCE_PIXEL']

        res = self.run_step(self.refpix, input, **kwargs)

        # Return pixel DQ back to original using bitwise AND of inverted flag
        log.info(f'Removing reference pixel flags')
        res.pixeldq[0:nrow_ref,:] = res.pixeldq[0:nrow_ref,:] & ~dqflags.pixel['REFERENCE_PIXEL']
        res.pixeldq[-nrow_ref:,:] = res.pixeldq[-nrow_ref:,:] & ~dqflags.pixel['REFERENCE_PIXEL']
        
        return res
    
    def do_saturation(self, input, **kwargs):
        """Peform custom saturation flagging"""
        
        # Check current setting
        npix_grow = self.saturation.n_pix_grow_sat
        # Just return normal if growing diagonal or set to 0
        if self.grow_diagonal or npix_grow==0:
            return self.run_step(self.saturation, input, **kwargs)
        
        # Run with 1 less pixel growth than specified
        self.saturation.n_pix_grow_sat = npix_grow - 1
        res = self.run_step(self.saturation, input, **kwargs)
            
        # Update saturation dq flags to grow in vertical and horizontal directions
        # Performs a bitwise & in order to find flagged the saturation bits
        mask_sat = (res.groupdq & dqflags.pixel['SATURATED']) > 0
        
        # Create a bunch of shifted masks
        mask_vp1 = np.roll(mask_sat, +1, axis=-1)
        mask_vm1 = np.roll(mask_sat, -1, axis=-1)
        mask_hp1 = np.roll(mask_sat, +1, axis=-2)
        mask_hm1 = np.roll(mask_sat, -1, axis=-2)
        
        # Zero out rows and columns that rolled over to the other side
        mask_vp1[:,:, 0,:] = 0
        mask_vm1[:,:,-1,:] = 0
        mask_hp1[:,:,:, 0] = 0
        mask_hm1[:,:,:,-1] = 0
        
        # Combine saturation masks
        mask_sat = mask_sat | mask_vp1 | mask_vm1 | mask_hp1 | mask_hm1

        # Do a bitwise OR of new mask with groupdq to flip saturation bit
        res.groupdq = res.groupdq | (mask_sat * dqflags.pixel['SATURATED'])

        # Clean up unused arrays
        del mask_vp1, mask_vm1, mask_hp1, mask_hm1

        return res

def run_ramp_fitting(meta, idir, osubdir):
    

    search = '*' + meta.ramp_ext
    # Get all of the files in the input directory
    files = glob.glob(idir+search)
    if len(files) == 0:
        raise ValueError('Unable to locate any {} files in directory {}'.format(search, idir))
    # Run the pipeline on every file in a directory
    for file in files:
        # Set up pipeline
        pipeline = Coron1Pipeline()

        # Skip certain steps?
        if hasattr(meta, 'skip_jump'):
            pipeline.jump.skip = meta.skip_jump
        if hasattr(meta, 'skip_dark_current'):
            pipeline.dark_current.skip = meta.skip_dark_current
        if hasattr(meta, 'skip_ipc'):
            pipeline.ipc.skip = meta.skip_ipc
        # Skip persistence step for now since it doesn't do anything
        pipeline.persistence.skip = True

        # Set some Step parameters
        if hasattr(meta, 'jump_threshold'):
            pipeline.jump.rejection_threshold = meta.jump_threshold
        if hasattr(meta, 'ramp_fit_max_cores'):
            pipeline.ramp_fit.maximum_cores = meta.ramp_fit_max_cores
        if hasattr(meta, 'nrow_ref'):
            pipeline.nrow_ref = meta.nrow_ref
        if hasattr(meta, 'grow_diagonal'):
            pipeline.nrow_ref = meta.grow_diagonal
        if hasattr(meta, 'sat_boundary'):
            pipeline.saturation.n_pix_grow_sat

        # Options for saving intermediate results
        if hasattr(meta, 'save_intermediates'):
            pipeline.save_intermediates = meta.save_intermediates
        pipeline.save_results = True

        # Set up directory to save into
        pipeline.output_dir = meta.odir + osubdir
        if os.path.exists(pipeline.output_dir) == False:
            os.makedirs(pipeline.output_dir)

        # Create log file output
        file_base = os.path.basename(file).replace('uncal.fits', '')
        date_str = datetime.now().isoformat()
        fname = f'{file_base}_detector1_{date_str}.log'
        log_file = os.path.join(pipeline.output_dir, fname)
        # Create empty file
        with open(log_file, 'w') as f:
            pass

        # Add file stream handler to also append log messages to file
        logger = logging.getLogger()
        fh = logging.FileHandler(log_file, 'a')
        fmt = logging.Formatter('%(asctime)s [%(name)s:%(levelname)s] %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Run pipeline, raise exception on error, and close log file handler
        try:
            pipeline.run(file)
        except Exception as e:
            raise RuntimeError(
                'Caugh exception during pipeline processing.'
                '\nException: {}'.format(e)
            )
        finally:
            logger.removeHandler(fh)
            fh.close()

    return

def stsci_ramp_fitting(meta):
    """
    Use the JWST pipeline to process *uncal.fits files to *rateints.fits files
    """
    if meta.rampfit_idir:
        run_ramp_fitting(meta, meta.idir, 'RAMPFIT/SCI+REF/')
    if meta.rampfit_bgdirs:
        if meta.bg_sci_dir != 'None':
            run_ramp_fitting(meta, meta.bg_sci_dir, 'RAMPFIT/BGSCI/')
        if meta.bg_ref_dir != 'None':
            run_ramp_fitting(meta, meta.bg_ref_dir, 'RAMPFIT/BGREF/')

    return