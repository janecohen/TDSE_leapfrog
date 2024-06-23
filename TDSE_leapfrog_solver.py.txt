#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:21:49 2024

@author: janecohen
"""

#%% USER: run cell ONCE to define all imports and subroutines 

import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.optimize import fsolve
import matplotlib.animation as animation
from scipy.sparse import diags
import matplotlib.colors as colors
import matplotlib.cm as cmx

"Clean up old variables and plots"
def clean_all():
    plt.close('all') # close all plots
    for var in ['a', 'A', 'b', 'dt', 'h', 'k0', 'p', 'probabilityDensity', 'q', 'sigma', 't', 
                'tlen', 'tmax', 'V', 'waveFunction', 'x', 'x0', 'xlen', 'x_expt']:
        if var in globals():
            del globals()[var] # delete all global variables
    
"Velocity-Verlet ODE solver"
def leapfrog(diffeqn, x0, v0, dt):
    hh = dt/2
    x1 = x0 + diffeqn(x0, v0)[0] * hh
    v1 = v0 + diffeqn(x1, v0)[1] * dt
    x1 = x1 + diffeqn(x1, v1)[0] * hh
    return x1, v1 

"Slicing method"
def tdse_slicing(R, I):
    
    tmpI = b*I # 1st/3rd term
    dRdt = a*I # 2nd term
    dRdt[1:-1] += tmpI[:-2] + tmpI[2:]
    
    tmpR = -b*R # 1st/3rd term
    dIdt = -a*R # 2nd term
    dIdt[1:-1] += tmpR[:-2] + tmpR[2:]
    
    # periodic boundary conditions: wrap around for the first and last points
    dRdt[0] += tmpI[-1] + tmpI[1]
    dRdt[-1] += tmpI[0] + tmpI[-2]
    dIdt[0] += tmpR[-1] + tmpR[1]
    dIdt[-1] += tmpR[0] + tmpR[-2]
    return [dRdt, dIdt]  

"Matrix method"
def tdse_matrix(R,I):
    dRdt = A@I
    dIdt = -A@R
    return [dRdt, dIdt]

"Matrix types"
def sparse_matrix(a,b, xlen):
    offsets = [0, 1, -1, xlen-1, -(xlen-1)]  
    diagonals = [a, b*np.ones((xlen-1)), b*np.ones((xlen-1)), b, b]  # add 'b' to corners for periodicity
    M = diags(diagonals, offsets, shape=(xlen, xlen), format='csr')
    return M

def simple_matrix(a,b, xlen):
    M = np.zeros((xlen, xlen))
    np.fill_diagonal(M, a)
    np.fill_diagonal(M[1:], b) 
    np.fill_diagonal(M[:, 1:], b)  
    M[0, -1] = b 
    M[-1, 0] = b 
    return M

"Initial waveform"
def initial_wavefuntion(x_0, sigma_0, k_0, xList):
    return (sigma_0*np.sqrt(np.pi))**(-1/2) * np.exp(-(xList-x_0)**2 / (2*sigma_0**2) + 1j*k_0*xList)
      
"Plot wave packets"
def plot_wave_packets(xList, probDens, potential, timeStamp, timeLabel, arrows, ylim, potential_scaling):
    # set up plot
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['figure.dpi']= 120
    fig, ax = plt.subplots(figsize = (6,4))
    ax.set_ylim(-0.1,ylim)
    
    # scale and plot potential
    potential_scaled = potential/potential_scaling
    ax.plot(xList, potential_scaled, color='k', linestyle = 'dotted')
    
    # plot waveforms at set time stamps
    colors = ['coral', 'hotpink', 'mediumorchid', 'royalblue']
    for stamp, lab, col in zip(timeStamp, timeLabel, colors):
        probDensityAtTime = probDens[:, stamp]
        stampText = "$t/T_0$ = " + str(lab)
        ax.plot(xList, probDensityAtTime, color = col, label=stampText)
        
        # plot direction arrows for waveforms
        if (arrows and stamp < tlen):
            # find the peak of the curve
            peakIndex = np.argmax(probDensityAtTime)
            peakX = xList[peakIndex]
            peakY = probDensityAtTime[peakIndex]
            
            # find the peak of the curve
            probDensityAtTimeNext = probDens[:, stamp+20]
            peakIndexNext = np.argmax(probDensityAtTimeNext)
            peakXNext = xList[peakIndexNext]

            # find direction peak is moving
            if (peakX > peakXNext):
                direction = -1 # left
            else:
                direction = 1  # right
       
            # plot the arrow at the peak
            ax.arrow(peakX, peakY, direction, 0, head_width=0.03, head_length=0.2, fc=col, ec=col)

    ax.set_title('Wave Packets')
    ax.set_xlabel('$x$ (a.u.)')
    ax.set_ylabel('$|\psi(x,t)|^2$ (a.u.)')
    ax.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
     
"Contour plot of probability density"
def plot_contour(tList, xList, probDens):
    X, Y = np.meshgrid(tList, xList)
    fig, ax = plt.subplots(figsize = (6,4))
    contour = ax.contourf(X, Y, probDens, cmap='gnuplot2')
    cbar = fig.colorbar(contour)
    
    # plot expectation value of x    
    if ('x_expt' in globals()):
        ax.plot(tList, x_expt, color='white', linewidth=2, linestyle='--')
     
    ax.set_title('Probability Density')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$ (a.u.)')
    plt.tight_layout()
    plt.show()
    
"Run animation of wave packets"
def run_animation(x, psiAmp, potential, ds, ylim, xlim, potential_scaling):
    # set up plot window
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))

    ax1.set_title('Wavefunction over time')
    ax1.set_xlabel('$x$ (a.u.)')
    ax1.set_ylabel('$|\psi(x,t)|^2$ (a.u.)')
    ax1.set_xlim(-xlim, xlim)
    ax1.set_ylim(-0.1,ylim)

    # create empty line
    line1, = ax1.plot([], [], lw=4, color='hotpink')
    
    # downsample
    psiAmp_downsampled = psiAmp[:,::ds] 
    x_downsampled = x[::ds]
    
    # scale and plot potential
    potential_scaled = potential/potential_scaling
    ax1.plot(x, potential_scaled, linestyle='dotted', color='k')
    
    # animation function
    def update(i):
        line1.set_data(x, psiAmp_downsampled[:,i])
        return line1,

    # calling the animation function
    global anim
    anim = animation.FuncAnimation(fig, update, frames=tlen, interval=1, blit=True)
    
    plt.show()
    
"Probability density calculation using Leapfrog integrator"
def probability_density(x0, sigma, k0, V, xList, tList, run_method, use_sparse_matrix):
    
    xlen = len(xList)
    tlen = len(tList)
    
    # initial waveform
    psi_initial = initial_wavefuntion(x0, sigma, k0, xList)
    
    R0 = np.real(psi_initial)
    I0 = np.imag(psi_initial)
    
    R = np.zeros((xlen,tlen))
    I = np.zeros((xlen,tlen))
    
    R[:,0] = R0
    I[:,0] = I0

    global A
    if (use_sparse_matrix == True):
        A = sparse_matrix(a, b, xlen)
    else:
        A = simple_matrix(a, b, xlen)
    
    for i in range(1,tlen):
        R0, I0 = leapfrog(run_method, R0, I0, dt)
        R[:,i]= R0
        I[:,i] = I0    

    psi = R + 1j*I
    return psi # returns complex waveform

"Keyword function mapper"
keyword_function_mapper = {
    "matrix": tdse_matrix,
    "slicing": tdse_slicing
}


"Virial Theorem"
def virial_theorem(waveFunction, probabilityDensity):
    # 2 <T> = n <V> 
    
    sumT = 0
    sumV = 0
    for psiComplex, probDens in zip(waveFunction.transpose(), probabilityDensity.transpose()):
        
        # kinetic energy <T> 
        gradPsi = np.gradient(psiComplex, h) 
        gradProb = np.abs(gradPsi)**2 
        sumT += np.sum(gradProb*0.5)*h
        
        # potential energy <V> 
        integrandV = V*probDens
        sumV += np.sum(integrandV)*h
        
    T_expt = sumT/tlen
    V_expt = sumV/tlen
    
    return T_expt, V_expt

"Position expectation value"
def position_expectation_value(waveFUntion, probabilityDensity, scale):
    # expectation value of the position <x>
    
    x_expt = []
    for psiComplex, probDens in zip(waveFunction.transpose(), probabilityDensity.transpose()):
        val = np.sum(probDens*x)*h
        val_scaled = val/scale
        x_expt.append(val_scaled)
        
    return x_expt


#%%
"QUESTION 1 - Free Space Propagation"

#%% USER: edit parameters and run this cell to set up the free space propagation problem (Q1)

clean_all() # run cleaner to remove previous variables and plots

"Initial conditions"
x0 = -5 # centre of Gaussian
sigma = 1 # width of Gaussian
k0 = 5 # initial avg. momentum

"Spatial and time parameters"
xlen = 1001 # number of points on space grid
tlen = 15000
x = np.linspace(-10,10,xlen) # -10 to +10 in a.u.
h = (np.max(x) - np.min(x)) / (xlen - 1)
dt = 0.5*h**2 # time step
tmax = dt*tlen
t = np.linspace(0, tmax, tlen) # time values

"Potential"
V = np.zeros((xlen)) 
a = 1/h**2 + V
b = -1/(2*h**2)

"USER CONTROLS"
# choice of derivative method
choice = "slicing" #  "matrix" or "slicing" 
run_method = keyword_function_mapper.get(choice)

# choice of matrix type
use_sparse_matrix = True # True: use sparse matrix; False: use full matrix

"Solve"
# run solver and find probability density
waveFunction = probability_density(x0, sigma, k0, V, x, t, run_method, use_sparse_matrix)
probabilityDensity = np.abs(waveFunction)**2


#%% USER: edit parameters and run this cell to show graphics for the free space propagation problem (Q1)

timeStamp = [20,3000,5000,13000]
timeLabel = [round(timeStamp[0]/(tlen/2), 2), round(timeStamp[1]/(tlen/2),2), 
             round(timeStamp[2]/(tlen/2),2), round(timeStamp[3]/(tlen/2),2)]


wavepackets = True # turn on or off to display wavepacket plot
save_wavepackets = False # turn on or off to save wavepacket plot
contour = True # turn on or off to display contour plot
save_contour = False # turn on or off to save contour plot
animate = True # turn on or off to display animation

if wavepackets:
    # x, y, potential, snapshots, display arrows, ylimit, potential scale
    plot_wave_packets(x, probabilityDensity, V, timeStamp, timeLabel, True, 0.6, 1) 
    if (save_wavepackets == True):
        #plt.savefig('./Desktop/ENPH479/Assignment3/Q1_wavepackets.pdf', format='pdf', dpi=1200, bbox_inches = 'tight')
        plt.savefig('/Users/janecohen/Desktop/ENPH479/Assignment3/Q1_wavepackets.pdf', format='pdf', dpi=1200, bbox_inches = 'tight')

if contour:
    # t, x, y
    plot_contour(t, x, probabilityDensity)
    if (save_contour == True):
        plt.savefig('/Users/janecohen/Desktop/ENPH479/Assignment3/Q1_contour.pdf', format='pdf', dpi=1200, bbox_inches = 'tight')


if animate:
    # x, y, potential, downsample, ylimit, xlimit, potential scale
    run_animation(x, probabilityDensity, V, 40, 0.6, 10, 1) 


#%%
"QUESTION 2 - Harmonic Oscillator"

#%% USER: edit parameters and run this cell to set up the harmonic oscillator (Q2)

clean_all() # run cleaner to remove previous variables and plots

"Initial conditions"
x0 = -5 # centre of Gaussian
sigma = 0.5 # width of Gaussian
k0 = 0 # initial avg. momentum

"Spatial and time parameters"
xlen = 1001 # number of points on space grid
x = np.linspace(-10, 10, xlen) # -10 to +10 in a.u.
h = (np.max(x) - np.min(x)) / (xlen - 1)
dt = 0.5*h**2 # time step
T = 2*np.pi
tmax = 2*T
tlen = int(tmax/dt)
t = np.linspace(0, tmax, tlen) # time values

"Potential"
V = (1/2)*(x**2)

a = 1/h**2 + V
b = -1/(2*h**2)

"USER CONTROLS"
# choice of derivative method
choice = "slicing" #  "matrix" or "slicing" 
run_method = keyword_function_mapper.get(choice)

# choice of matrix type
use_sparse_matrix = True # True: use sparse matrix; False: use full matrix

"Solve"
# run solver and find probability density
waveFunction = probability_density(x0, sigma, k0, V, x, t, run_method, use_sparse_matrix)
probabilityDensity = np.abs(waveFunction)**2

"Virial Theorem"
T_expt, V_expt = virial_theorem(waveFunction, probabilityDensity)


#%% USER: edit parameters and run this cell to show graphics for the harmonic oscillator (Q2)
timeStamp = [20,3000,19000,25000]
timeLabel = [round(timeStamp[0]/(tlen/2), 2), round(timeStamp[1]/(tlen/2),2), 
             round(timeStamp[2]/(tlen/2),2), round(timeStamp[3]/(tlen/2),2)]

wavepackets = True # turn on or off to display wavepacket plot
save_wavepackets = False # turn on or off to save wavepacket plot
contour = True # turn on or off to display contour plot
save_contour = False # turn on or off to save contour plot
animate = True # turn on or off to display animation


if wavepackets:
    # x, y, potential, snapshots, display arrows, ylimit, potential scale 
    plot_wave_packets(x, probabilityDensity, V, timeStamp, timeLabel, True, 1.2, 50) 
    if (save_wavepackets == True):
        plt.savefig('/Users/janecohen/Desktop/ENPH479/Assignment3/Q2_wavepackets.pdf', format='pdf', dpi=1200, bbox_inches = 'tight')

if contour:
    # t, x, y
    plot_contour(t, x, probabilityDensity)
    if (save_contour == True):
        plt.savefig('/Users/janecohen/Desktop/ENPH479/Assignment3/Q2_contour.pdf', format='pdf', dpi=1200, bbox_inches = 'tight')

if animate:
    # x, y, potential, downsample, ylimit, xlimit, potential scale
    run_animation(x, probabilityDensity, V, 50, 1.2, 10, 50) 
    
print("2<T>:", 2.*T_expt)
print("2<V>:", 2.*V_expt)


#%%
"QUESTION 3 - Double Well Potential"

#%% USER: edit parameters and run this cell to set up the double well potential (Q3)

clean_all() # run cleaner to remove previous variables and plots

"Initial conditions"
x0 = -np.sqrt(2)
sigma = 0.5
k0 = 0

"Spatial and time parameters"
xlen = 500 # number of points on space grid
x = np.linspace(-5, 5, xlen) # -10 to +10 in a.u.
h = (np.max(x) - np.min(x)) / (xlen - 1)
dt = 0.5*h**2 # time step
T = 4*np.pi
tmax = 2*T
tlen = int(tmax/dt)
t = np.linspace(0, tmax, tlen) # time values

"Potential"
q = 1
p = 4
V = q*x**4 - p*x**2

a = 1/h**2 + V
b = -1/(2*h**2)

"USER CONTROLS"
# choice of derivative method
choice = "slicing" #  "matrix" or "slicing" 
run_method = keyword_function_mapper.get(choice)

# choice of matrix type
use_sparse_matrix = True # True: use sparse matrix; False: use full matrix

"Solve"
# run solver and find probability density
waveFunction = probability_density(x0, sigma, k0, V, x, t, run_method, use_sparse_matrix)
probabilityDensity = np.abs(waveFunction)**2

"Position expectation value"
x_expt = position_expectation_value(waveFunction, probabilityDensity, 0.5)


#%% USER: edit parameters and run this cell to show graphics for the double well potential (Q3)
timeStamp = [20,18500,81500,120000]
timeLabel = [round(timeStamp[0]/(tlen/2), 2), round(timeStamp[1]/(tlen/2),2), 
             round(timeStamp[2]/(tlen/2),2), round(timeStamp[3]/(tlen/2),2)]

wavepackets = True # turn on or off to display wavepacket plot
save_wavepackets = False # turn on or off to save wavepacket plot
contour = True # turn on or off to display contour plot
save_contour = False # turn on or off to save contour plot
animate = True # turn on or off to display animation

if wavepackets:
    # x, y, potential, snapshots, display arrows, ylimit, potential scale 
    plot_wave_packets(x, probabilityDensity, V, timeStamp, timeLabel, False, 1.2, 50) 
    if (save_wavepackets == True):
        plt.savefig('/Users/janecohen/Desktop/ENPH479/Assignment3/Q3_wavepackets.pdf', format='pdf', dpi=1200, bbox_inches = 'tight')

if contour:
    plot_contour(t, x, probabilityDensity)
    if (save_contour == True):
        plt.savefig('/Users/janecohen/Desktop/ENPH479/Assignment3/Q3_contour.pdf', format='pdf', dpi=1200, bbox_inches = 'tight')

if animate:
    # x, y, potential, downsample, ylimit, xlimit, potential scale
    run_animation(x, probabilityDensity, V, 50, 1.2, 5, 100) 



#%% The end :)

