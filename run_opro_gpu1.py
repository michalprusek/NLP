#!/usr/bin/env python3
"""Run OPRO on GPU 1."""
import os
# MUST be set before any torch/cuda imports
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import sys
sys.argv = ['run_opro.py', '--model', 'qwen', '--budget', '200000',
            '--minibatch-size', '261', '--num-candidates', '8']

# Now import and run
exec(open('run_opro.py').read())
