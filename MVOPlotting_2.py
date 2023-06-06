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
    meltvolumedataframe=pergeos_mvodata_to_xyz_points(pergeosdataframe)
    dataPatch= find_melt_in_patch(dataPatch, meltvolumedataframe)
    dataPatch=compute_thetaphi_patch_edge(dataPatch)
    dataPatch=compute_3Dpatch_edges(dataPatch)
    
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
def z_coordinate_from_spherical(phi):
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

    
def find_melt_in_patch(dataPatch, meltdataframe):
    for i in range(len(dataPatch.index)):##For every patch...
        ##crate a dataframe with only melt pockets with the phi, theta range of that patch (including max but not including
        ##lower bound, and convert the bounds to degrees to be compatible with the melt pocket data from PerGeos)
        single_patch_df=meltdataframe[(meltdataframe['Theta'] <= np.rad2deg(dataPatch.at[i,'Theta Maximum']))\
                                          & (meltdataframe['Theta'] > np.rad2deg(dataPatch.at[i,'Theta Minimum']))\
                                              & (meltdataframe['Phi'] <= np.rad2deg(dataPatch.at[i,'Phi Maximum'])) \
                                                  & (meltdataframe['Phi'] > np.rad2deg(dataPatch.at[i,'Phi Minimum']))]        
        dataPatch.at[i,'Melt Volume']=sum(single_patch_df['Melt Pocket Volume'])##store the sums in the dataPatch dataframe
    print(sum(meltdataframe['Melt Pocket Volume']))
    print(sum(dataPatch['Melt Volume']))##Off by 100th of a percent but shouldn't be?
    return(dataPatch)
    
def load_dataframe(dataframepath):
    dataFrame = pd.read_csv(dataframepath)
    return(dataFrame)

# def empty_dataframe_column_asobject(dataframe, column_name):
#     dataframe.at[0, column_name]=[0]
#     dataframe[column_name]=dataframe[column_name].astype(object)
#     return(dataPatch)

def compute_thetaphi_patch_edge(dataPatch):
    points=50##number of points along each edge
    dataPatch.at[0, 'Theta Edges']=[0]
    dataPatch.at[0, 'Phi Edges']=[0]
    dataPatch['Theta Edges']=dataPatch['Theta Edges'].astype(object)
    dataPatch['Phi Edges']=dataPatch['Phi Edges'].astype(object)
    ##initialize empty dataframe columns which can be filled with a list
    
    for i in range(len(dataPatch.index)):
        theta_edges=[]
        phi_edges=[]##make all theta edges clockwise from the bottom left corner of the patch
        theta_edges.append(np.linspace(dataPatch.at[i, 'Theta Maximum'],\
                dataPatch.at[i, 'Theta Maximum'], points))
        theta_edges.append(np.linspace(dataPatch.at[i, 'Theta Maximum'],\
                dataPatch.at[i, 'Theta Minimum'], points))
        theta_edges.append(np.linspace(dataPatch.at[i, 'Theta Minimum'],\
                dataPatch.at[i, 'Theta Minimum'], points))
        theta_edges.append(np.linspace(dataPatch.at[i, 'Theta Minimum'],\
                dataPatch.at[i, 'Theta Maximum'], points))
            
        ##make all phi edges clockwise from the bottom left corner of the patch
        phi_edges.append(np.linspace(dataPatch.at[i, 'Phi Minimum'],\
                dataPatch.at[i, 'Phi Maximum'], points))
        phi_edges.append(np.linspace(dataPatch.at[i, 'Phi Maximum'],\
                dataPatch.at[i, 'Phi Maximum'], points))    
        phi_edges.append(np.linspace(dataPatch.at[i, 'Phi Maximum'],\
                dataPatch.at[i, 'Phi Minimum'], points))    
        phi_edges.append(np.linspace(dataPatch.at[i, 'Phi Minimum'],\
                dataPatch.at[i, 'Phi Minimum'], points))    
            
        dataPatch.at[i, 'Theta Edges']=np.ravel(theta_edges)
        dataPatch.at[i, 'Phi Edges']=np.ravel(phi_edges)
        
    return(dataPatch)


def compute_3Dpatch_edges(dataPatch):
    
    ##initialize empty column where each cell can be filled by a list
    ##Should this be a separate function? it's tediuous but we want
    ##dataPatch to have this list for each patch
    dataPatch.at[0,'3D X-Coordinate of Patch Edge']=[0]
    dataPatch['3D X-Coordinate of Patch Edge']=dataPatch['3D X-Coordinate of Patch Edge'].astype(object)
    dataPatch.at[0,'3D Y-Coordinate of Patch Edge']=[0]
    dataPatch['3D Y-Coordinate of Patch Edge']=dataPatch['3D Y-Coordinate of Patch Edge'].astype(object)
    dataPatch.at[0,'3D Z-Coordinate of Patch Edge']=[0]
    dataPatch['3D Z-Coordinate of Patch Edge']=dataPatch['3D Z-Coordinate of Patch Edge'].astype(object)
    
    for i in range(len(dataPatch.index)):
        ##compute the path for each edge 
        x_edges=[]
        y_edges=[]##list for all theta, phi values of path around the patch
        z_edges=[]
        ##compute the x coordinates for the left-most edge, then the top edge, then the 
        ##right-most edge, then the bottom edge. 
        for j in range(len(dataPatch.at[i, 'Theta Edges'])):
            
            x_edges.append(x_coordinate_from_spherical(dataPatch.at[i, 'Theta Edges'], dataPatch.at[i, 'Phi Edges']))
            y_edges.append(y_coordinate_from_spherical(dataPatch.at[i, 'Theta Edges'], dataPatch.at[i, 'Phi Edges']))
            z_edges.append(z_coordinate_from_spherical(dataPatch.at[i, 'Phi Edges']))

                                                                         
        ##assign the coordinates for points along the patch in the cell of dataPatch
        dataPatch.at[i,'3D X-Coordinate of Patch Edge']=np.ravel(x_edges)
        dataPatch.at[i,'3D Y-Coordinate of Patch Edge']=np.ravel(y_edges)
        dataPatch.at[i,'3D Z-Coordinate of Patch Edge']=np.ravel(z_edges)
        
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

    
