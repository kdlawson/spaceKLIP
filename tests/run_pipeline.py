import os, sys
sys.path.append('..')
from spaceKLIP.engine import JWST

config_file = os.path.dirname(__file__)+'/HIP65426_F1140.yaml'#'/HD114174-round.yaml'
if __name__ == '__main__':
	pipe = JWST(config_file)
	pipe.run_all(skip_ramp=True,
				 skip_imgproc=True,
				 skip_sub=True,
				 skip_rawcon=True,
				 skip_calcon=True,
				 skip_comps=False)
