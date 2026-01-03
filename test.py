#!/usr/bin/env python3

import numpy as np
import utils

filename="/Users/antoniosmecca/Documents/Physics/pdoc_RomaTre/PiPi/Code/DATA/out_smeared/ph0_sm0_Sr0_sm0_ph0_av_all"

corr = utils.read_corr(filename)
print(corr[0])
