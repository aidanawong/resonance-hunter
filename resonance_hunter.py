#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, constants

class WaveEquationResonsanceHunter():
    def __init__(self, input_X, input_Y, phase_velocity=343, N=30, fontsize=16):

        def error_msg(): 
            print("Sorry, your input coordinates are not valid.\nPlease input two arrays for a contour or two numbers for a rectangle.")
            quit()

        self.input_X = input_X
        self.input_Y = input_Y
        self.v = phase_velocity
        self.N = N
        self.fontsize = fontsize
        
        if type(input_X) == np.ndarray and type(input_Y) == np.ndarray:
            self.shape_type = "contour"
            self.interp_x, self.interp_y = self.process_boundary()  
        elif type(input_X) == int or type(input_X) == float:
            if type(input_Y) == int or type(input_Y) == float:
                self.input_X, self.input_Y = np.abs(self.input_X), np.abs(self.input_Y)
                self.shape_type = "rect"
                self.grid_length = np.max([self.input_X, self.input_Y])
            else:
                error_msg()
        else:
            error_msg()
    
    def process_boundary(self):

        def length(u): return np.abs(np.max(u) - np.min(u)) 
        def shift(u, ref): return 0.5 * (length(ref) - length(u)) - np.min(u)
        
        X, Y = np.array(self.input_X), np.array(self.input_Y) 
     
        tck, u = interpolate.splprep([X, Y], s=0, per=True)
        xx, yy = interpolate.splev(np.linspace(0, 1, 5 * self.N), tck)

        if length(xx) >= length(yy) :
            scale = 1/(np.max(xx) - np.min(xx))
            x_shift = np.abs(np.min(xx))
            y_shift = shift(yy, xx)
            self.grid_length = length(X)
        else:
            scale = 1/(np.max(yy) - np.min(yy))
            x_shift = shift(xx, yy)
            y_shift = np.abs(np.min(yy))
            self.grid_length = length(Y)

        xx = scale * (xx + x_shift)
        yy = scale * (yy + y_shift)
        return (xx, yy)
    
    def make_rect_grid(self):
        xx, yy, L = self.input_X, self.input_Y, self.grid_length
        grid = np.zeros([self.N,self.N])
        X_mg, Y_mg = np.meshgrid(np.linspace(-L,L,self.N),np.linspace(-L,L,self.N))
        grid[(-xx<X_mg)&(X_mg<xx)&(-yy<Y_mg)&(Y_mg<yy)] = 1
        return grid
    
    def make_contour_grid(self):
        def fill_contour(arr):
            A = np.maximum.accumulate(arr,1)
            B = np.fliplr(np.maximum.accumulate(np.fliplr(arr),1))
            C = np.flipud(np.maximum.accumulate(np.flipud(arr), 0))
            D = np.maximum.accumulate(arr, 0)
            E = np.logical_and(np.logical_and(A,B),
                                np.logical_and(C,D)
                                ).astype(int)
            return E
        
        N_cut = self.N - 2
        grid = np.zeros([N_cut,N_cut])
        x, y = self.interp_x, self.interp_y
        x = np.ceil(x * N_cut - 1) 
        y = np.ceil(y * N_cut - 1) 
        
        coords = np.c_[y,x]
        
        for coord in coords:
            xi, yi = int(coord[0]), int(coord[1])
            grid[xi, yi] = 1
        
        grid = np.pad(fill_contour(grid), pad_width=1, mode='constant', constant_values=0)
        return grid

    def find_choose_eig(self, M):

        eigval, eigvect = np.linalg.eig(M)
        eigval = np.real(eigval)
        eigvect = np.real(eigvect)

        not_zero = (eigval!=0)
        eigval = eigval[not_zero]
        eigvect = eigvect[: , not_zero]

        idx = np.argsort(np.abs(eigval))
        eigval = eigval[idx]
        eigvect = eigvect[:, idx]

        return [eigval, eigvect, M]

    def solve_wave_eqn_dirichlet(self, domain, Lx=1, Ly=1):
        Nx, Ny = domain.shape
        M = np.zeros([Nx * Ny, Nx * Ny])

        dX = Lx / (Nx - 1)
        dY = Ly / (Ny - 1)

        for i in range(Nx):
            for j in range(Ny):
                if domain[i, j] != 0:
                    index = i * Ny + j
                    M[index, index] = (-2 / dX ** 2 - 2 / dY ** 2)
                    if 0 < i:
                        M[index, (i - 1) * Nx + (j)] = 1 / dX ** 2
                    if 0 < j:
                        M[index, (i) * Ny + (j - 1)] = 1 / dY ** 2
                    if i < Nx - 1:
                        M[index, (i + 1) * Nx + (j)] = 1 / dX ** 2
                    if j < Ny - 1:
                        M[index, (i) * Ny + (j + 1)] = 1 / dY ** 2

        [eigval, eigvect, M] = self.find_choose_eig(M)
        
        return [eigval, eigvect]
    
    def solve_wave_eqn_neumann(self, domain, Lx=1, Ly=1):
        Nx, Ny = domain.shape
        M = np.zeros([Nx * Ny, Nx * Ny])

        dX = Lx / (Nx - 1)
        dY = Ly / (Ny - 1)

        for i in range(Nx):
            for j in range(Ny):
                if domain[i, j] != 0:
                    index = i * Ny + j
                    M[index, index] = (-2 / dX ** 2 - 2 / dY ** 2)
                    if 0 < i:
                        if domain[i-1, j] != 0:
                            M[index, (i - 1) * Nx + (j)] = 1 / dX ** 2
                        else:
                            M[index, index] += 1 / dX ** 2
                    if 0 < j:
                        if domain[i, j-1] != 0:
                            M[index, (i) * Ny + (j - 1)] = 1 / dY ** 2
                        else:
                            M[index, index] += 1 / dY ** 2
                    if i < Nx - 1:
                        if domain[i+1, j] != 0:
                            M[index, (i + 1) * Nx + (j)] = 1 / dX ** 2
                        else:
                            M[index, index] += 1 / dX ** 2
                    if j < Ny - 1:
                        if domain[i, j+1] != 0:
                            M[index, (i) * Ny + (j + 1)] = 1 / dY ** 2
                        else:
                            M[index, index] += 1 / dY ** 2

        [eigval, eigvect, M] = self.find_choose_eig(M)
        
        return [eigval, eigvect] 
    
    def draw_3D(self, sln):
        L = self.grid_length
        x = np.linspace(0, L, sln.shape[1])
        y = np.linspace(0, L, sln.shape[0])
        x, y = np.meshgrid(x, y)
        z = sln 
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        # Plot the surface
        if self.shape_type == "contour":
            ax.plot(self.interp_x * L, self.interp_y * L, 0, c="k", zorder=3)
        surface = ax.plot_surface(x, y, z, cmap="coolwarm",linewidth=0, antialiased=False, zorder=0)
        
        ax.set_xlabel("X", fontsize=self.fontsize)
        ax.set_ylabel("Y", fontsize=self.fontsize)

        cbar = fig.colorbar(surface, shrink=0.5, aspect=10)
        plt.show()
        plt.close()
    
    def draw_cmap(self, sln):
        L = self.grid_length
        x = np.linspace(0, L, sln.shape[1])
        y = np.linspace(0, L, sln.shape[0])

        fig, ax = plt.subplots()
        im = ax.pcolormesh(x, y, sln, cmap="coolwarm", zorder=1)
        if self.shape_type == "contour":
            ax.plot(self.interp_x * L, self.interp_y * L, c="k", zorder=3)
        
        ax.set_xlabel("X", fontsize=self.fontsize)
        ax.set_ylabel("Y", fontsize=self.fontsize)
        
        cbar = fig.colorbar(im, aspect=10)
        plt.show()
        plt.close()

    def play(self, bc="dirichlet", eigmode=0, draw_format="3D"):
        
        grid_length, v = self.grid_length, self.v

        if self.shape_type == "contour":
            grid = self.make_contour_grid()
        else:
            grid = self.make_rect_grid()

        # self.draw_cmap(grid)

        if bc == "neumann":
            [eigval, eigvect] = self.solve_wave_eqn_neumann(grid, grid_length, grid_length)
        else:
            [eigval, eigvect] = self.solve_wave_eqn_dirichlet(grid, grid_length, grid_length)
        
        freq = v * np.sqrt(np.abs(eigval[eigmode])) / (2 * np.pi)
        # print("λ" + str(eigmode) + " =", round(eigval[eigmode],2))
        print("f" + str(eigmode) + " =", round(freq, 2))

        wave = eigvect[:, eigmode].reshape(grid.shape)

        if draw_format == "3D":
            self.draw_3D(wave)
        else:
            self.draw_cmap(wave)
        
        return freq, wave

def santiana():
    X = np.array([
        0,0.271,1.097,2.931,5.732,9.465,13.614,\
        19.032,24.318,29.191,33.593,39.147,43.463,\
        48.183,52.381,56.586,61.094,66.302,70.448,\
        75.38,80.575,84.738,89.326,93.521,97.244,\
        101.928,106.346,109.991,113.339,115.502,\
        117.642,119.402,120.704,121.25
        ])
    
    Y = np.array([
        0,8.139,17.695,26.087,31.925,36.373,39.856,\
        42.875,44.515,45.286,45.5,45.12,44.287,\
        42.822,41.041,38.838,36.10,32.787,30.621,\
        29.75,30.816,32.076,33.348,34.199,34.50,\
        33.951,32.353,30.223,27.424,24.834,21.092,\
        16.569,10.364,0
        ])
    # Because the guitar is symmetric, we can double the x values and double and flip the y values.
    # This creates a plot that draws out a guitar from (0,0) and moving counterclockwise
    X = np.concatenate((X, np.flipud(X[:-1])))
    Y = np.concatenate((-Y, np.flipud(Y[:-1])))

    # Convert millimetres to metres
    X *= 1e-3
    Y *= 1e-3
    return X, Y

def spinner():
    X = np.array([0,1,2,3,4,5])
    Y = np.array([0,1,0.8,1,4,0])
    X = np.concatenate((X, np.flipud(X[:-1])))
    Y = np.concatenate((-Y, Y[1:]))
    return X, Y

def fork():
    X = np.array([0, 1, 2, 3, 4, 5, 6, 4, 3.5, 3, 2.5, 2, 0])
    Y = np.array([0, 4, 1, 4, 1, 4, 0, -1, -3.8, -4, -3.8, -1, 0])
    return X, Y

def wacky():
    X = np.array([-3, -2.5, -2, -1, 0, 1, 2, 1.5, 1, 0, -2, -3])
    Y = np.array([0, 1, 4, 3, 4, 1, 0.5, -4, -2, -5, -0.3, 0])
    return X, Y

def circle():
    Theta = np.linspace(0, 2 * np.pi, 100)
    X = np.sin(Theta)
    Y = np.cos(Theta)
    return X, Y

def rect():
    return 0.2122, 0.1485

def error_test_1():
    return [0, 1, 3], "I'm a string!"

def error_test_2():
    return 12, np.array([1,2,3])

x, y = santiana()

v = lambda T: np.sqrt(1.4 * constants.Boltzmann * (T + 273.15) / (28.96 * constants.u))
temp = 22.4

WaveEquationResonsanceHunter(
    x, y, v(temp), 30
    ).play(
            bc="neumann",
            eigmode=7,
            draw_format="3D"
            )
# %%