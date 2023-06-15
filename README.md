# Resonance Hunter

Solves the 2D standing wave equation on any simple closed contour.

Based on the work by ComputationalScientist [here](https://youtu.be/Le4_LZmkZgs).

<img src="figures/wave_map_num_santiana_7.png" width="600">

## How to Use

### Creating a 2D Domain

#### Rectangle

For a rectangular domain, the input is simple. Insert `x = width` and `y = length` and the domain will be a rectangle with the prescribed width and length. Below is an example at the eigenmode 3.

<img src="figures/rect_eigmode-3.png" width="600">

#### Contours

A 2D domain is created according to the coordinates of a simple closed contour. The coordinates must end at the same place at the beginning. Below is an example that creates a wacky domain.

```
X = np.array([-3, -2.5, -2, -1, 0, 1, 2, 1.5, 1, 0, -2, -3])
Y = np.array([0, 1, 4, 3, 4, 1, 0.5, -4, -2, -5, -0.3, 0])
```

A less explicit example would be a circle. 

```
Theta = np.linspace(0, 2 * np.pi, 100)
X = np.sin(Theta)
Y = np.cos(Theta)
```

The coordinates can reach any value, positive or negative. However, the entire domain will be moved to the first Cartesian quadrant such that all values are positive. All distances are retained.


### Running the Program

Go to `main.py`, and you can choose your preferences, including the wave speed, resolution, degree of splines in the interpolation. The wave speed can be arbitrary, but `main.py` includes the speed of sound in dry air. In this example we will continue with the unit circle.

```
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
```

You can select a boundary condition, either `"dirichlet"` or `"neumann"`, a particular eigenmode, and the output format. Usually, the higher the eigenmode, the greater the number of nodes and antinodes. For a 3D projection, you can input `"3D"`. Anything else will output a heatmap.

The code will first return a heatmap showing the grid on which the equation will run on.

<img src="figures/circle_grid.png" width="600">

Then a graph will be shown with the chosen resonance.  The eigenvalue and wave frequency will be returned as well. Below you can see a 3D plot, and a heatmap.

<img src="figures/circle_eigmode-1.png" width="600">

<img src="figures/circle_eigmode-1_cmap.png" width="600">

**Happy Hunting!**