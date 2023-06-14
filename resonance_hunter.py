#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class WaveEquationResonsanceHunter():
    
    def __init__(self, input_X, input_Y, phase_velocity=343, N=30, spline_degree=3, fontsize=16):

        def error_msg(): 
            print("Sorry, your input coordinates are not valid.\nPlease input two arrays for a contour or two numbers for a rectangle.")
            quit()

        self.input_X = input_X
        self.input_Y = input_Y
        self.spline_degree = spline_degree
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

        tck, u = interpolate.splprep([X, Y], s=0, k=self.spline_degree, per=True)
        xx, yy = interpolate.splev(np.linspace(0, 1, 5 * self.N), tck)

        if length(xx) >= length(yy):
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

        self.draw_cmap(grid)

        if bc == "neumann":
            [eigval, eigvect] = self.solve_wave_eqn_neumann(grid, grid_length, grid_length)
        else:
            [eigval, eigvect] = self.solve_wave_eqn_dirichlet(grid, grid_length, grid_length)
        
        freq = v * np.sqrt(np.abs(eigval[eigmode])) / (2 * np.pi)
        border = 100 * "#"
        print(border)
        print("The eigenvalue is:\nλ" + str(eigmode) + " =", round(eigval[eigmode],2))
        print(border)
        print("The frequency is:\nf" + str(eigmode) + " =", round(freq, 2))

        wave = eigvect[:, eigmode].reshape(grid.shape)

        if draw_format == "3D":
            self.draw_3D(wave)
        else:
            self.draw_cmap(wave)
        
        return freq, wave

# %%