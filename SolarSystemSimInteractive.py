#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:41:26 2021

@author: Santiago R.

This is an interactive, Mayavi based version of the Jupyter Notebook on Solar 
System Simulations using NASA Horizons data. Using the Sun, the main 8 planets and 
their current locations, get initial data on and simulate the rough trajectory 
of any object such as a comet, asteroid, dwarf planet etc.

"""

import numpy as np
import time
import scipy.constants as cs
from astropy.time import Time
from astroquery.jplhorizons import Horizons


#Conversion Units
AU = 149597870700
D = 24*60*60

#define functions for the program

#Vectorial acceleration function
def a_t( r, m, epsilon):
    """
    Function of matrices returning the gravitational acceleration
    between N-bodies within a system with positions r and masses m
    -------------------------------------------------
    r  is a N x 3 matrix of object positions
    m is a N x 1 vector of object masses
    epsilon is the softening factor to prevent numerical errors
    a is a N x 3 matrix of accelerations
    -------------------------------------------------
    """  
    G = cs.gravitational_constant/(AU**3)
    # positions r = [x,y,z] for all bodies in the N-Body System
    x = r[:,0:1]
    y = r[:,1:2]
    z = r[:,2:3]
    # matrices that store each pairwise body separation for each [x,y,z] direction: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    #matrix 1/r^3 for the absolute value of all pairwise body separations together and 
    #resulting acceleration components in each [x,y,z] direction  
    inv_r3 = (dx**2 + dy**2 + dz**2 + epsilon**2)**(-1.5)
    ax = G * (dx * inv_r3) @ m
    ay = G * (dy * inv_r3) @ m
    az = G * (dz * inv_r3) @ m
    # pack together the three acceleration components
    a = np.hstack((ax,ay,az))
    return a

def omega(N, t_max, dt):
    """
    Dummy acceleration function to give an estimate of the total memory
    consumption for a simulation with N bodies, total simulation time 
    t_max and timestep dt
    -------------------------------------------------
    N is the amount of bodies in the simulation
    t_max is the total simulation time in years
    dt is the timestep for each integration in days
    -------------------------------------------------
    """
    epsilon = 1
    G = 1
    r = np.ones((N,3))
    v = np.ones((N,3)) 
    """although the velocities aren't actually taken into account for computing the 
    acceleration, they will be stored as Nx3 matrices for exactly as many iterations 
    in the integration loop itself and take up memory accordingly"""
    m = np.ones((N,1))
    # positions r = [x,y,z] for all bodies in the N-Body System
    x = r[:,0:1]
    y = r[:,1:2]
    z = r[:,2:3]
    # matrices that store each pairwise body separation for each [x,y,z] direction: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    #matrix 1/r^3 for the absolute value of all pairwise body separations together and 
    #resulting acceleration components in each [x,y,z] direction 
    inv_r3 = (dx**2 + dy**2 + dz**2 + epsilon**2)**(-1.5)
    ax = G * (dx * inv_r3) @ m
    ay = G * (dy * inv_r3) @ m
    az = G * (dz * inv_r3) @ m
    # pack together the three acceleration components
    a = np.hstack((ax,ay,az))
    # sum the memory usage of each matrix storing the positions, distances and accelerations
    memory_usage_per_iteration = r.nbytes + v.nbytes + x.nbytes + y.nbytes + z.nbytes + dx.nbytes + dy.nbytes + dz.nbytes + ax.nbytes + ay.nbytes + az.nbytes + inv_r3.nbytes + a.nbytes 
    total_memory_usage = memory_usage_per_iteration * (t_max)/(dt*10e6) 
    return total_memory_usage #in megabytes




def simulate_solar_system(N,dN,starting_values): #
    t0_sim_start = time.time()
    t = 0
    t_max = 24*60*60*N #N day simulation time
    dt = 60*60*24*dN #dN day time step
    epsilon_s = 0.01 #softening default value
    r_i = starting_values[0]/AU
    v_i = starting_values[1]/AU
    m_i = starting_values[2]
    a_i = a_t(r_i, m_i, epsilon_s)
    ram_usage_estimate = omega(len(r_i), t_max, dt) #returns the estimated ram usage for the simulation
    # Simulation Main Loop using a Leapfrog Kick-Drift-Kick Algorithm
    k = int(t_max/dt)
    r_save = np.zeros((r_i.shape[0],3,k+1))
    r_save[:,:,0] = r_i
    for i in range(k):
        # (1/2) kick
        v_i += a_i * dt/2.0
        # drift
        r_i += v_i * dt
        # update accelerations
        a_i = a_t(r_i, m_i, epsilon_s)
        # (2/2) kick
        v_i += a_i * dt/2.0
        # update time
        t += dt
        #update list
        r_save[:,:,i+1] = r_i
    sim_time = time.time()-t0_sim_start
    print('The required computation time for the N-Body Simulation was', round(sim_time,3), 'seconds. The estimated memory usage was', round(ram_usage_estimate,3), 'megabytes of RAM.')
    return r_save

#Main Program Loop

from traits.api import *
from traitsui.api import *
from traitsui.api import DateEditor
from mayavi import mlab
import random
import sys

class SimulationTool( HasTraits ):
   """ Simulation object """
   
   

   
   PlanetaryObjects = Enum('Sun','Inner Planets', 'Outer Planets', 'Inner and outer Planets',
      desc="Choose which celestial bodies to account for in the simulation",
      label="Simulated Objects", )

   dt = CInt(1,
      desc="The timestep for the simulation, in days",
      label="Timestep (days)", )



   info_string = Str(
        'Solar System Simulation Tool'
    )
   
   multi_date = List(Date)
   multi_select_editor = DateEditor(
        allow_future=True,
        multi_select=True,
        shift_to_select=True,
        on_mixed_select='max_change',
        # Qt ignores these setting and always shows only 1 month:
        months=1,
        padding=30, )

   obj_id = List(Str,
                 desc="The Horizons IDs of the additional objects to simulate",
                 label="Horizons IDs")
   



   def _obj_id_default(self):
       return ['Add some IDs']

   def _add_fired(self):
       new_item = "Item%d" % random.randint(3, 999)
       self.obj_id.append(new_item)

   def _clear_fired(self):
       self.obj_id = []
       
   RunSim = Bool(label="Run Simulation")
   
   SaveFig = Bool(label="Save Figure")
   
   close_result = False

   traits_view = View(
        Item('info_string', show_label=False, style='readonly'),
        '_',
        Group(
            Item('PlanetaryObjects', label='Simulated Objects'),
            Item('dt', label='Timestep (days)'),
            Item('obj_id', label='Horizons IDs'),
            '_',
        ),
        
        Group(
            Item(
                'multi_date',
                editor=multi_select_editor,
                style='custom',
                label='Pick 2:',
            ),
            '_',
            label='Select Range of Dates for the Simulation',
            
        ),
        
        Group(
            Item('RunSim', label='Check to Run Simulation'),
            Item('SaveFig', label='Check to Save 3D Plot'),),
        buttons=[OKButton],
        resizable=False,
        icon = "SSS_ToolLogoMain.png",
        width = 350
    )
   
   
   
   
   
   

   def _multi_date_changed(self):
        """Print each time the date value is changed in the editor."""
        td = self.multi_date[-1]-self.multi_date[0]
        t_max = td.days
        t_0 = str(self.multi_date[0])
        print("The starting date for the simulation is ", str(self.multi_date[0]),"and lasts for", t_max, "days")
 
    
   def simulate(self):
       "Simulates the solar system in the specified region with the added object in Horizons"
       td = self.multi_date[-1]-self.multi_date[0]
       t_max = td.days
       t_0 = str(self.multi_date[0])
       r_list = []
       v_list = []
       m_list = [[1.989e30],[3.285e23],[4.867e24],[5.972e24],[6.39e23],[1.8989e27],[5.683e26],[8.681e25],[1.024e26],[1.309e22]] #Object masses for Sun-Pluto
       plot_colors = [(1.0, 0.0, 0.),(0.25, 0.25, 0.25),(1.0,0.8,0.8),(0.0, 0.4470, 0.7410),(0.9500, 0.3250, 0.0980),(0.6350, 0.0780, 0.1840),(0.9290, 0.6940, 0.1250),(0, 0.75, 0.75),(0.0, 0.0, 1.0),(0.25, 0.25, 0.25)]
       plot_labels = ['Barycenter','Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune','Pluto']
       if self.PlanetaryObjects == str("Sun"):
           m_list = [m_list[0]]
           plot_colors = [plot_colors[0]]
           obj = Horizons(id=0, location="@sun", epochs=Time(t_0).jd, id_type='id').vectors()
           r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
           v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]
           r_list.append(r_obj)
           v_list.append(v_obj)
               
       if self.PlanetaryObjects == str("Inner Planets"):
           m_list = m_list[0:5]
           plot_colors = plot_colors[0:5]
           for i in range(0,5):
               obj = Horizons(id=i, location="@sun", epochs=Time(t_0).jd, id_type='id').vectors()
               r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
               v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]
               r_list.append(r_obj)
               v_list.append(v_obj)
           
       elif self.PlanetaryObjects == str("Outer Planets"):
           m_outer = m_list[5:10]
           m_list = [m_list[0]]+m_outer
           plot_colors_outer = plot_colors[5:10]
           plot_colors = [plot_colors[0]]+plot_colors_outer
           obj = Horizons(id=0, location="@sun", epochs=Time(t_0).jd, id_type='id').vectors()
           r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
           v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]
           r_list.append(r_obj)
           v_list.append(v_obj)
           for i in range(5,10):
               obj = Horizons(id=i, location="@sun", epochs=Time(t_0).jd, id_type='id').vectors()
               r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
               v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]
               r_list.append(r_obj)
               v_list.append(v_obj)
       elif self.PlanetaryObjects == str("Inner and outer Planets"):
           for i in range(0,10):
               obj = Horizons(id=i, location="@sun", epochs=Time(t_0).jd, id_type='id').vectors()
               r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
               v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]
               r_list.append(r_obj)
               v_list.append(v_obj)
       def add_simulation_object(Id_obj,t_0,m_obj, plot_color, plot_label):
           obj = Horizons(id=Id_obj, location="@sun", epochs=Time(t_0).jd, id_type='id').vectors()
           r_obj = [obj['x'][0], obj['y'][0], obj['z'][0]]
           v_obj = [obj['vx'][0], obj['vy'][0], obj['vz'][0]]
           r_list.append(r_obj)
           v_list.append(v_obj)
           m_list.append([m_obj])
           plot_colors.append(plot_color)
           plot_labels.append(plot_label)
       for i in range(0,len(self.obj_id),1):
           try:
               add_simulation_object(self.obj_id[i],t_0,1e12, (0.0,0.0,0.0),self.obj_id[i])
           except ValueError:
               print("The ID", self.obj_id[i] ," is not valid.  Try again...")
       #Convert object staring value lists to numpy
       r_i = np.array(r_list)*AU
       v_i = np.array(v_list)*AU/D
       m_i = np.array(m_list)
       #pack together as list for the simulation function
       horizons_data = [r_i,v_i,m_i]
       #Run simulation for t_max years at a dt day time-step
       r_save = simulate_solar_system(t_max,self.dt,horizons_data) 

#      Plotting
       mlab.figure(bgcolor=(1.0,1.0,1.0),fgcolor=(0.0,0.0,0.0))
       
       for i in range(0,len(r_i),1): #Plots any objects in the simulation
           mlab.plot3d(r_save[i,0,:],r_save[i,1,:],r_save[i,2,:], line_width=1.0, color=plot_colors[i])
#       for i in range(0,len(self.obj_id),1):
#           mlab.plot3d(r_save[len(r_i)-i,0,:],r_save[len(r_i)-i,1,:],r_save[len(r_i)-i,2,:], line_width=1.0, color=(0.0, 0.0, 0.0))
       mlab.axes(xlabel='AU', ylabel='AU', zlabel='AU',nb_labels=5)
       if self.SaveFig == True:
           mlab.savefig("SSS_Exports/Sim"+str(np.datetime64('now'))+".jpg",size=(1024,1024))
       mlab.show()


if  __name__ == "__main__":
   init = SimulationTool()
   while True:
       init.configure_traits()
       if init.RunSim == False:
           break
       init.simulate()