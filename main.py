import numpy as np
from scipy import constants
import domain
from resonance_hunter import WaveEquationResonsanceHunter

x, y = domain.circle()

v = lambda T: np.sqrt(1.4 * constants.Boltzmann * (T + constants.zero_Celsius) / (28.96 * constants.u))
temperature = 22.4

WaveEquationResonsanceHunter(
    x, y, 
    v(temperature), 
    N=30, spline_degree=3
    ).play(
            bc="dirichlet",
            eigmode=1,
            draw_format="3D"
            )