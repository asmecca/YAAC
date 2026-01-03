#!/usr/bin/env python3

import numpy as np
import YAAC.utils as yaac

filename="/Users/antoniosmecca/Documents/Physics/pdoc_RomaTre/PiPi/Code/DATA/out_smeared/ph0_sm0_Sr0_sm0_ph0_av_all"

jk_corr = yaac.read_corr(filename)
yaac.plot_corr(jk_corr,"$at$","$C(t)$")
