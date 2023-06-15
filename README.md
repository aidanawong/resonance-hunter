# Resonance Hunter
Solves the 2D standing wave equation on any simple closed contour.

Based on the work by ComputationalScientist [here](https://youtu.be/Le4_LZmkZgs).

<img src="figures/wave_map_num_santiana_7.png" width="600">

## How to Use

```
{
  WaveEquationResonsanceHunter(
    x, y, 
    v(temperature), 
    N=30, 
    spline_degree=3 
    ).play(
            bc="dirichlet",
            eigmode=1,
            draw_format="3D"
            )
}
```

<img src="figures/circle_eigmode-1.png" width="600">

<img src="figures/circle_eigmode-1_cmap.png" width="600">