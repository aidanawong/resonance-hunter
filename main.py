import numpy as np
from scipy import constants
import domain_contours
from resonance_hunter import WaveEquationResonsanceHunter

x, y = domain_contours.circle()

v = lambda T: np.sqrt(1.4 * constants.Boltzmann * (T + 273.15) / (28.96 * constants.u))
temp = 22.4

WaveEquationResonsanceHunter(
    x, y, v(temp), 30
    ).play(
            bc="dirichlet",
            eigmode=7,
            draw_format="3D"
            )