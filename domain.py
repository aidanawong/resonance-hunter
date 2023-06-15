#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on April 28, 2023 at 5:24 PM

Author: Aidan Wong

Description: A list of sample coordinates for simple closed contours.
Two numbers are interpreted as length and width for a rectangular domain.
"""

import numpy as np

def santiana():
    """
    Coordinates describing a classical Santos Hernandez guitar
    Source: N. Bellido, Guitare Classique 'dans le style' SANTOS HERNANDEZ, 
    url:https://laguitarra-blog.com/wp-content/uploads/2011/12/santoshernandez.pdf
    """

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
    # A two bladed object
    X = np.array([0,1,2,3,4,5])
    Y = np.array([0,1,0.8,1,4,0])
    X = np.concatenate((X, np.flipud(X[:-1])))
    Y = np.concatenate((-Y, Y[1:]))
    return X, Y

def fork():
    # A fork with three prongs
    X = np.array([0, 1, 2, 3, 4, 5, 6, 4, 3.5, 3, 2.5, 2, 0])
    Y = np.array([0, 4, 1, 4, 1, 4, 0, -1, -3.8, -4, -3.8, -1, 0])
    return X, Y

def wacky():
    # Randomly, strangely fox shaped object
    X = np.array([-3, -2.5, -2, -1, 0, 1, 2, 1.5, 1, 0, -2, -3])
    Y = np.array([0, 1, 4, 3, 4, 1, 0.5, -4, -2, -5, -0.3, 0])
    return X, Y

def circle():
    # A unit circle
    Theta = np.linspace(0, 2 * np.pi, 100)
    X = np.sin(Theta)
    Y = np.cos(Theta)
    return X, Y

def rect():
    # Rectangular cavity, units in metres
    return 0.2122, 0.1485

def error_test_1():
    # Causes an error
    return [0, 1, 3], "I'm a string!"

def error_test_2():
    return 12, np.array([1,2,3])