# -*- coding: utf-8 -*-

"""

Created on Mon May 15 19:14:53 2023
@author: James Bader & Laurent Montesi

"""

import pandas as pd
import time
import numpy as np
import math
import ast
import matplotlib
from matplotlib import pyplot as plt
import scipy
from scipy.spatial import SphericalVoronoi, geometric_slerp
from matplotlib import patches
from matplotlib.path import Path
import random

def MVOauto(numphi,numtheta, pergeosmvodatafilepath):
    #numphi = 4 #number of bounding levels per latitude
    #numtheta = 6 #number of bounding levels per latitude
    xVertex,yVertex,zVertex,thetaCellEdge,phiCellEdge = defineGrid(numphi,numtheta)
    axDemo = plotGrid(xVertex,yVertex,zVertex,thetaCellEdge,phiCellEdge,-100,35)
    dataPatch = initializePatch(numphi,numtheta,thetaCellEdge,phiCellEdge, xVertex, yVertex, zVertex)
    pergeosdataframe=load_dataframe(pergeosmvodatafilepath)
    meltdataframe=pergeos_mvodata_to_xyz_points(pergeosdataframe)
    #dataPatch= find_melt_in_patch(dataPatch, meltdataframe)
    
    return (dataPatch)

def defineGrid(numphi,numtheta):
# define the edges and bounds of the cells given a number of longitude (theta) and latitude (phi)   
# xVertex,yVertex,zVertex,aCell,thetaCellEdge,phiCellEdge = defineGrid(numphi,numtheta)

    thetaCellEdge = np.linspace(0, 2*np.pi, numtheta) # longitude in radians
    phiCellEdge = np.deg2rad(np.linspace(0, 90, numphi)) # latitude in radians
 #   areaCell = np.diff(np.sin(phiCellEdge))*(2*np.pi/(numtheta-1)) # Area for each band
    thetaVertex, phiVertex = np.meshgrid(thetaCellEdge, phiCellEdge) # Coordinates of each cell corner
    xVertex = np.cos(thetaVertex)*np.cos(phiVertex) # x-coordinates
    yVertex = np.sin(thetaVertex)*np.cos(phiVertex) # y-coordinates
    zVertex = np.sin(phiVertex) # z-coordinates
 #   aCell = np.tile(areaCell,(numtheta,1))
    return (xVertex,yVertex,zVertex,thetaCellEdge,phiCellEdge)

def x_coordinate_from_spherical(theta, phi):
    return(np.cos(theta)*np.cos(phi)) # x-coordinates
def y_coordinate_from_spherical(theta, phi):
    return(np.sin(theta)*np.cos(phi)) # y-coordinates
def z_coordinate_from_spherical(theta, phi):
    return(np.sin(phi))

def initializePatch(numphi,numtheta,thetaCellEdge,phiCellEdge, xVertex, yVertex, zVertex): 
    # define the patches on the unit sphere where the information will be gathered
    # dataPatch = initializePatch(numphi,numtheta,thetaCellEdge,phiCellEdge)
    # bounds of each patch
    thetaMin, phiMin = np.meshgrid(thetaCellEdge[:-1], phiCellEdge[:-1]) 
    thetaMax, phiMax = np.meshgrid(thetaCellEdge[1:], phiCellEdge[1:])
    # area of each patch
    patchArea=(np.sin(phiMax)-np.sin(phiMin))*(thetaMax-thetaMin) 
    # TO DO LATER: do not use the final latitude. Append information for the cap
    numPatch = (numphi-1)*(numtheta-1) # number f patches
    # initialize columns TO DO: ADD THE X< Y< Z COORDINATES OF PATCH EDGES? 
    dataPatch = pd.DataFrame(index=np.arange(numPatch),columns=['Theta Minimum','Phi Minimum','Theta Maximum','Phi Maximum','Patch Area','Melt Volume','Sum of Elongation','Melt Volume per Patch Area'])
  
    dataPatch['Theta Minimum']= np.ravel(thetaMin)
    dataPatch['Theta Maximum']= np.ravel(thetaMax)
    dataPatch['Phi Minimum']= np.ravel(phiMin)
    dataPatch['Phi Maximum']= np.ravel(phiMax)
    dataPatch['Patch Area']= np.ravel(patchArea)


    return (dataPatch)


def pergeos_mvodata_to_xyz_points(pergeosdataframe):
    tot_xdat=[]
    tot_zdat=[]
    tot_ydat=[]
    theta_zerothreesixty=[]
    tot_vol=[]
    tot_shape=[]
    tot_phidat=[]
    meltpocket_data=pd.DataFrame()
    
    for subvolume in pergeosdataframe.index:
        phidat =list(ast.literal_eval(pergeosdataframe['Column_1'][subvolume]))
        thetadat = list(ast.literal_eval(pergeosdataframe['Column_0'][subvolume]))
        shapedat=list(ast.literal_eval(pergeosdataframe['PoreShape'][subvolume]))
        voldat=list(ast.literal_eval(pergeosdataframe['PoreVolume'][subvolume]))

        for i in range(len(thetadat)):        
            if thetadat[i]<0:
                theta_zerothreesixty.append(360+thetadat[i])##Do some conversions since python wants data from 0-360degrees and PerGeos gives it in -180 to 180.
            else:
                theta_zerothreesixty.append(thetadat[i])
            tot_xdat.append(np.sin(np.deg2rad(phidat[i]))*(np.cos(np.deg2rad(theta_zerothreesixty[i]))))
            tot_ydat.append(np.sin(np.deg2rad(phidat[i]))*(np.sin(np.deg2rad(theta_zerothreesixty[i]))))
            tot_zdat.append(np.cos(np.deg2rad(phidat[i])))
            tot_vol.append(voldat[i])
            tot_shape.append(shapedat[i])
            tot_phidat.append(phidat[i])
    
    meltpocket_data['X Coordinates']=tot_xdat
    meltpocket_data['Y Coordinates']=tot_ydat
    meltpocket_data['Z Coordinates']=tot_zdat
    meltpocket_data['Melt Pocket Volume']=tot_vol
    meltpocket_data['Melt Pocket Shape']=tot_shape
    meltpocket_data['Theta']=theta_zerothreesixty
    meltpocket_data['Phi']=tot_phidat
    return(meltpocket_data)

    
"""
def find_melt_in_patch(dataPatch, meltdataframe):
    for patch in dataPatch.index:
        single_patch_df=meltdataframe[(meltdataframe['Theta'] <= dataPatch.index[patch]['Theta Maximum'])\
                                      & (meltdataframe['Theta'] > dataPatch.index[patch]['Theta Minimum'])\
                                          & (meltdataframe['Phi'] <= dataPatch.index[patch]['Phi Maximum']) \
                                              & (meltdataframe['Phi'] > dataPatch.index[patch]['Phi Minimum'])]
        dataPatch.index[patch]['Melt Volume']=sum(single_patch_df['Melt Pocket Volume'])
    return(dataPatch)
 """
    
def load_dataframe(dataframepath):
    dataFrame = pd.read_csv(dataframepath)
    return(dataFrame)

def compute_3Dpatch__edges(dataPatch):
    ##compute the x y z coordinates of each corner for each patch
    points=100
    for patch in dataPatch.index:
        ##compute the path for each edge 
        
        
        
        
        
        dataPatch.index[Patch]['3D Patch Edge']
    return(dataPatch)
    
    
    

#    cellDF = pd.DataFrame(columns = ['xVertex','yVertex','zVertex','longitudeBound','latitudeBound','Area'])

    



def plotGrid(xVertex,yVertex,zVertex,thetaCellEdge,phiCellEdge,thetaDemo,phiDemo):

# visualize the grid and add an example vector

# axDemo = plotGrid(xVertex,yVertex,zVertex,thetaCellEdge,phiCellEdge,-100,35):

    figDemo = plt.figure()

    axDemo=figDemo.add_subplot(1,1,1,projection='3d')
    axDemo.scatter(xVertex,yVertex,zVertex,color='black')


    ntheta=len(thetaCellEdge)
    nphi=len(phiCellEdge)

    nlatPlot=20
    nlonPlot=nlatPlot*4

    latPlotVec = np.linspace(0,np.pi/2,nlatPlot)
    lonPlotVec = np.linspace(0,2*np.pi,nlonPlot)

    for i in range(len(phiCellEdge)):
        axDemo.plot(np.cos(lonPlotVec)*np.cos(phiCellEdge[i]),np.sin(lonPlotVec)*np.cos(phiCellEdge[i]),np.sin(phiCellEdge[i])*np.linspace(1,1,nlonPlot),color='black')

    for i in range(len(thetaCellEdge)):
        axDemo.plot(np.cos(thetaCellEdge[i])*np.cos(latPlotVec),np.sin(thetaCellEdge[i])*np.cos(latPlotVec),np.sin(latPlotVec),color='black')

    # Illustative vector;

    teta=np.deg2rad(thetaDemo)
    ph=np.deg2rad(phiDemo)
    
    xDemo =(np.cos(teta)*np.cos(ph))
    yDemo =(np.sin(teta)*np.cos(ph))
    zDemo =(np.sin(ph))
    
    lengthDemo=np.sqrt(2)
    axDemo.quiver(0,0,0,xDemo*lengthDemo,yDemo*lengthDemo,zDemo*lengthDemo,color='blue')
    axDemo.scatter([0,xDemo],[0,yDemo],[0,zDemo],color='blue',marker='o')
    axDemo.set_box_aspect([1,1,0.5])

    

    return (axDemo)

    

   

"""

Weird start for sorting



for i in range(lonEdge)-1

    [v,ID]=find(lon>=lonEdge&lon<lonEdge+1)

    for latEdge

        [v,IDlatinlon]=find(lon>=lonEdge&lon<lonEdge+1)

        volInCell(lonEdge,latEdge)=sum(vol(ID(IDlatinlon)))
"""

