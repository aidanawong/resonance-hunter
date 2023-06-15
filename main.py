#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on April 28, 2023 at 5:26 PM

Author: Aidan Wong

Description: Run the code from here to find resonant frequencies. Select the boundary, wave velocity, and other preferences. 
"""

import numpy as np
from scipy import constants

import domain
from resonance_hunter import WaveEquationResonsanceHunter

# Domain Coordinates
x, y = domain.rect()

# Speed of sound in dry air
v = lambda T: np.sqrt(1.4 * constants.Boltzmann * (T + constants.zero_Celsius) / (28.96 * constants.u))
temperature = 22.4

WaveEquationResonsanceHunter(
    x, y, 
    v(temperature), 
    N=30, 
    spline_degree=3 
    ).play(
            bc="neumann",
            eigmode=3,
            draw_format="3D"
            )