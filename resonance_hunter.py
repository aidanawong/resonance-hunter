#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on March 30, 2023 at 3:25 PM

Author: Aidan Wong

Description: Creates a 2D domain that is interpolated from coordinates that describe its boundary. 
Solves the wave equation on that 2D domain.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

class WaveEquationResonsanceHunter():
    
    def __init__(self, input_X, input_Y, phase_velocity=343, N=30, spline_degree=3, fontsize=14):
        """
        Initializing constants and settings.
        
        input_X: 1D np.ndarray - x coordinates of boundary for the domain.
        input_Y: 1D np.ndarray - Y coordinates of boundary for the domain.
        phase_velocity: float or int - speed of the waves. Only used for finding the frequency.
        N: int - Resolution of 2D grid.
        spline_degree: int - Between and including 1 to 5. For smooth domains, 3 is recommended. Choose 1 for linear domains.
        fontsize: int - Size of letters for axes.
        """

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
            # Simple closed contours
            self.shape_type = "contour"
            self.interp_x, self.interp_y = self.process_boundary()  
        elif type(input_X) == int or type(input_X) == float:
            # Rectangles
            if type(input_Y) == int or type(input_Y) == float:
                self.input_X, self.input_Y = np.abs(self.input_X), np.abs(self.input_Y)
                self.shape_type = "rect"
                self.grid_length = np.max([self.input_X, self.input_Y])
            else:
                error_msg()
        else:
            error_msg()
    
    def play(self, bc="dirichlet", eigmode=0, draw_format="3D"):
        """
        Runs the entire class to solve and present the wave.

        bc: str - Boundary conditions, dirichlet or neumann
        eigmode: int - Eigenmode. The higher the eigenmode the greater the frequency.
        draw_format: str - "3D" or "heatmap", describes the figure format
        """

        grid_length, v = self.grid_length, self.v

        if self.shape_type == "contour":
            grid = self.make_contour_grid()
        else:
            grid = self.make_rect_grid()

        # Draw the domain
        self.draw_cmap(grid)

        if bc == "neumann":
            [eigval, eigvect] = self.solve_wave_eqn_neumann(grid, grid_length)
        else:
            [eigval, eigvect] = self.solve_wave_eqn_dirichlet(grid, grid_length)
        
        # Find the frequency and eigenvalues given an eigmode
        freq = v * np.sqrt(np.abs(eigval[eigmode])) / (2 * np.pi)
        border = 30 * "#"
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
    
    def process_boundary(self):
        """
        Prepares the boundary to be transformed into a grid by interpolation, 
        scaling to a unit square, and transforming back to the original size.  
        """

        def shift(u, ref): return 0.5 * (np.ptp(ref) - np.ptp(u)) - np.min(u)
        
        tck, u = splprep([self.input_X, self.input_Y], s=0, k=self.spline_degree, per=True)
        xx, yy = splev(np.linspace(0, 1, 5 * self.N), tck)

        # Determine which is the longer side
        if np.ptp(xx) >= np.ptp(yy):
            scale = 1/np.ptp(xx)
            x_shift = np.abs(np.min(xx))
            y_shift = shift(yy, xx)
            self.grid_length = np.ptp(self.input_X)
        else:
            scale = 1/np.ptp(yy)
            x_shift = shift(xx, yy)
            y_shift = np.abs(np.min(yy))
            self.grid_length = np.ptp(self.input_Y)

        xx = scale * (xx + x_shift)
        yy = scale * (yy + y_shift)
        return (xx, yy)
    
    def make_rect_grid(self):
        """
        Creates a grid where points in a domain denoted by a 1.
        The domain will have the length and width of input_X and input_Y
        """

        xx, yy, L = self.input_X, self.input_Y, self.grid_length
        grid = np.zeros([self.N,self.N])
        X_mg, Y_mg = np.meshgrid(np.linspace(-L,L,self.N),np.linspace(-L,L,self.N))
        grid[(-xx<X_mg)&(X_mg<xx)&(-yy<Y_mg)&(Y_mg<yy)] = 1
        return grid
    
    def make_contour_grid(self):
        """
        Creates a grid where points in a domain denoted by a 1.
        The domain will be determined by the processed coordinates from a contour.
        """
        
        # Makes sure the edges of the grid are 0 so the domain does not spill over
        N_cut = self.N - 2
        grid = np.zeros([N_cut,N_cut])

        # Scales the coordinates to the grid.
        x, y = self.interp_x, self.interp_y
        x = np.ceil(x * N_cut - 1) 
        y = np.ceil(y * N_cut - 1) 
        
        # Draws the boundary on the grid
        coords = np.c_[y,x]
        for coord in coords:
            xi, yi = int(coord[0]), int(coord[1])
            grid[xi, yi] = 1
        
        # After the grid's boundary is plotted with 1's, the inside is filled with 1's.
        # This completes the domain.
        A = np.maximum.accumulate(grid,1)
        B = np.fliplr(np.maximum.accumulate(np.fliplr(grid),1))
        C = np.flipud(np.maximum.accumulate(np.flipud(grid), 0))
        D = np.maximum.accumulate(grid, 0)
        E = np.logical_and(np.logical_and(A,B),
                            np.logical_and(C,D)
                            ).astype(int)
        
        # Adds the edges of the grid again
        grid = np.pad(E, pad_width=1, mode='constant', constant_values=0)
        return grid

    def find_choose_eig(self, M):
        """
        Finds the eigenvalues and eigenvectors, discarding ones that equal zero.

        M: 2D np.ndarray matrix - The grid consisting of the finite differences.
        """

        # Find eigenvalues and eigenvectors
        eigval, eigvect = np.linalg.eig(M)
        eigval = np.real(eigval)
        eigvect = np.real(eigvect)

        # Discard eigenvalues that equal zero
        not_zero = (eigval!=0)
        eigval = eigval[not_zero]
        eigvect = eigvect[: , not_zero]

        # Sort by increasing eigenvalues
        idx = np.argsort(np.abs(eigval))
        eigval = eigval[idx]
        eigvect = eigvect[:, idx]

        return [eigval, eigvect, M]

    def solve_wave_eqn_dirichlet(self, domain, L):
        """
        Solves the wave equation with the finite difference method given Dirichlet conditions

        domain: 2D np.ndarray - The grid composed of 1's and 0's. The equation is solved wherever there are 1's.
        L: float or int - Physical length of the grid's X axis. Equivalent to self.grid_length
        """

        Nx, Ny = domain.shape
        M = np.zeros([Nx * Ny, Nx * Ny])

        dX = L / (Nx - 1)
        dY = L / (Ny - 1)

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
    
    def solve_wave_eqn_neumann(self, domain, L):
        """
        Solves the wave equation with the finite difference method given Neumann conditions

        domain: 2D np.ndarray - The grid composed of 1's and 0's. The equation is solved wherever there are 1's.
        L: float or int - Physical length of the grid's X axis. Equivalent to self.grid_length
        """

        Nx, Ny = domain.shape
        M = np.zeros([Nx * Ny, Nx * Ny])

        dX = L / (Nx - 1)
        dY = L / (Ny - 1)

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
        """
        Draws the solution on a 3D plot.

        sln: 2D np.ndarray - The wave's values
        """

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
        """
        Draws the solution on a heatmap.

        sln: 2D np.ndarray - The wave's values
        """

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