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
    ##dataPatch, meltpocketDF, stat_dataframe, polaraxis=MVOauto(5,20, 'H:/DataBackup_03242022/DataExtraction/0705/200cubes_POREINERTIA.csv')
    #to run, enter line above (edit file name and path)
    
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
    dataPatch=melt_volume_per_area(dataPatch)
    
    dataPatch=load_n_set_colormap(dataPatch, 'Melt Volume per Patch Area')
    polarfigure, polaraxis=initialize_polar_plot()
    plot_patch_polarplot(dataPatch, polaraxis)
    
    stat_dataframe=initialize_stat_dataframe()
    stat_dataframe=construct_the_orientation_matrix(stat_dataframe, meltvolumedataframe)
    stat_dataframe=find_orientation_matrix_eigenvalues(stat_dataframe, meltvolumedataframe)
    stat_dataframe=find_mean_vec_length(stat_dataframe, meltvolumedataframe)
    
    polaraxis=plot_eigenvectors(stat_dataframe, polaraxis)
    
    
    return (dataPatch, meltvolumedataframe, stat_dataframe, polaraxis)

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


##equations assume phi is latitude (Z=0 phi=0) and theta is angle from X towards Y axis
def x_coordinate_from_spherical(theta, phi):
    return(np.cos(theta)*np.cos(phi)) # x-coordinates
def y_coordinate_from_spherical(theta, phi):
    return(np.sin(theta)*np.cos(phi)) # y-coordinates 
def z_coordinate_from_spherical(phi):
    return(np.sin(phi))

def azimuth_from_xy(x, y):
    azimuth=np.arccos(x/(np.sqrt(x*x+y*y)))
    return(azimuth)
def inclination_from_z(z):
    inclination=np.arcsin(z)
    return(inclination)

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
    tot_xdat=[]##initialize lists for each variable associated with each melt pocket
    tot_zdat=[]
    tot_ydat=[]
    theta_zerothreesixty=[]
    tot_vol=[]
    tot_shape=[]
    tot_phidat=[]
    meltpocket_data=pd.DataFrame()##initialize an empty dataframe to store melt pocket measurements
    
    for subvolume in pergeosdataframe.index:
        phidat =list(ast.literal_eval(pergeosdataframe['Column_1'][subvolume]))##for each subvolume, load all melt pocket measurements
        thetadat = list(ast.literal_eval(pergeosdataframe['Column_0'][subvolume]))
        shapedat=list(ast.literal_eval(pergeosdataframe['PoreShape'][subvolume]))
        voldat=list(ast.literal_eval(pergeosdataframe['PoreVolume'][subvolume]))

        for i in range(len(thetadat)):                
            phidat[i]=90-phidat[i]    ##Get latitude, not colat, as phi, for subsequent calculations
            if thetadat[i]<0:
                theta_zerothreesixty.append(360+thetadat[i])##Do some conversions since python wants data from 0-360degrees and PerGeos gives it in -180 to 180.
            else:
                theta_zerothreesixty.append(thetadat[i])
            tot_vol.append(voldat[i])
            tot_shape.append(shapedat[i])
            tot_phidat.append(phidat[i])
            ##use the most recently accessed theta, phi for calculations
            tot_xdat.append(x_coordinate_from_spherical(np.deg2rad(theta_zerothreesixty[-1]), np.deg2rad(tot_phidat[-1])))
            tot_ydat.append(y_coordinate_from_spherical(np.deg2rad(theta_zerothreesixty[-1]), np.deg2rad(tot_phidat[-1])))
            tot_zdat.append(z_coordinate_from_spherical(np.deg2rad(tot_phidat[-1])))##get x, y, z values for each unit vector associated with each melt pocket's theta and phi (equations assumme theta from X towards Y, and phi is latitude)
    
    meltpocket_data['X Coordinates']=tot_xdat##store all measurements in the melt pocket dataframe
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
        single_patch_df=meltdataframe[(meltdataframe['Theta'] < np.rad2deg(dataPatch.at[i,'Theta Maximum']))\
                                          & (meltdataframe['Theta'] >= np.rad2deg(dataPatch.at[i,'Theta Minimum']))\
                                              & (meltdataframe['Phi'] < np.rad2deg(dataPatch.at[i,'Phi Maximum'])) \
                                                  & (meltdataframe['Phi'] >= np.rad2deg(dataPatch.at[i,'Phi Minimum']))]        
        dataPatch.at[i,'Melt Volume']=sum(single_patch_df['Melt Pocket Volume'])##store the sums in the dataPatch dataframe
    print(sum(meltdataframe['Melt Pocket Volume']))
    print(sum(dataPatch['Melt Volume']))##Off by 100th of a percent but shouldn't be? perhaps missing MP at 90 phi, theta 360?
    return(dataPatch)
    
def load_dataframe(dataframepath):
    dataFrame = pd.read_csv(dataframepath)
    return(dataFrame)

# def empty_dataframe_column_asobject(dataframe, column_name):
#     dataframe.at[0, column_name]=[0]
#     dataframe[column_name]=dataframe[column_name].astype(object)
#     return(dataPatch)


def compute_thetaphi_patch_edge(dataPatch):
    ##compute the theta and phis along the patch edges so we can plot a patch from the path
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


def initialize_polar_plot():
    ##initialize and nicely set a polar plot for the patches to be graphed
    polar_figure_object=plt.figure()
    polar_axis_object=polar_figure_object.add_subplot(projection='polar', frameon=False)
    # polar_axis_object.set_rmax(.5*np.pi)
    polar_axis_object.set_theta_direction(-1)
    polar_axis_object.set_theta_zero_location('N')
    polar_axis_object.set_rlim(bottom=.5*np.pi, top=0)## set so it plots latitude (default would be colat), with phi=90 degrees in center of polar plot.
    ##nicely label the phi ticks
    polar_axis_object.set_rgrids(radii=[0, np.deg2rad(22.5), np.deg2rad(45), np.deg2rad(67.5), np.deg2rad(90)], labels=['0', '', '45', '', '90'])
    return(polar_figure_object, polar_axis_object)


def melt_volume_per_area(dataPatch):
    dataPatch['Melt Volume per Patch Area']=dataPatch['Melt Volume']/dataPatch['Patch Area']
    return(dataPatch)

def load_n_set_colormap(dataPatch, desired_column_for_color):
    ##function to load colormap and assign value to column of dataPatch
    colormap_name='Blues'
    
    ##You can input manual min, max values for color (useful for plotting several datasets against each other)
    # vmin=40000
    # vmax=850000
    ##Or let it find the min and max value
    vmin=min(dataPatch[desired_column_for_color])
    vmax=max(dataPatch[desired_column_for_color])
    
    colormap=matplotlib.cm.get_cmap(colormap_name)
    colormap_scaled=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)## set the color map limits to the min and max of volume
    
    dataPatch.at[0, 'Patch Color']=[0]
    dataPatch['Patch Color']=dataPatch['Patch Color'].astype(object)##make the empty column for Patch Color (must be type=object since color is a list of RGBA values)
    for i in range(len(dataPatch.index)):
        dataPatch.at[i,'Patch Color']=colormap(colormap_scaled(dataPatch.at[i,desired_column_for_color]))#find each patch color
    return(dataPatch)


def construct_polarpath(thetas, phis):
    ##construct the path object from the theta and phi values along the patch edge
    verts=[]    
    codes=[]
    for i in range(len(thetas)):
        verts.append([thetas[i], phis[i]])
    for i in range(len(thetas)):
        if i == 0:
            codes.append(Path.MOVETO)##start the path from the first vert
        else:
            codes.append(Path.LINETO)#and build the line, clockwise around the patch
    path=Path(verts, codes)
    return(path)


def plot_patch_polarplot(dataPatch ,polaraxisobject):
    ##plot each patch from dataPatch
    for i in range(len(dataPatch.index)):
        path=construct_polarpath(dataPatch.at[i, 'Theta Edges'], dataPatch.at[i, 'Phi Edges'])
        patchvol=patches.PathPatch(path, facecolor=dataPatch.at[i, 'Patch Color'])##make the patch from the path
        polaraxisobject.add_patch(patchvol)#plot this on the polar plot
    return


def initialize_stat_dataframe():
    stat_dataframe=pd.DataFrame(columns=['Orientation Matrix', 'Eigenvalues', 'Eigenvectors', 'R-bar', 'beta', 'alpha'])
    return(stat_dataframe)


def construct_the_orientation_matrix(stat_dataframe, melt_pocket_dataframe):
    THE_orientation_matrix=np.zeros((3,3))##initialize the orientation matrix
    
    ##solve, element-wise, for each orientation matrix component and sum them to get each overall unique orientation matrix components
    THE_orientation_matrix[0,0]=sum(melt_pocket_dataframe['X Coordinates']*melt_pocket_dataframe['X Coordinates']*melt_pocket_dataframe['Melt Pocket Volume'])
    THE_orientation_matrix[0,1]=sum(melt_pocket_dataframe['X Coordinates']*melt_pocket_dataframe['Y Coordinates']*melt_pocket_dataframe['Melt Pocket Volume'])
    THE_orientation_matrix[0,2]=sum(melt_pocket_dataframe['X Coordinates']*melt_pocket_dataframe['Z Coordinates']*melt_pocket_dataframe['Melt Pocket Volume'])
    THE_orientation_matrix[1,1]=sum(melt_pocket_dataframe['Y Coordinates']*melt_pocket_dataframe['Y Coordinates']*melt_pocket_dataframe['Melt Pocket Volume'])
    THE_orientation_matrix[2,2]=sum(melt_pocket_dataframe['Z Coordinates']*melt_pocket_dataframe['Z Coordinates']*melt_pocket_dataframe['Melt Pocket Volume'])
    THE_orientation_matrix[1,2]=sum(melt_pocket_dataframe['Y Coordinates']*melt_pocket_dataframe['Z Coordinates']*melt_pocket_dataframe['Melt Pocket Volume'])
    
    ##complete the matrix with the repeated components
    THE_orientation_matrix[2,1]=THE_orientation_matrix[1,2]
    THE_orientation_matrix[2,0]=THE_orientation_matrix[0,2]
    THE_orientation_matrix[1,0]=THE_orientation_matrix[0,1]
    
    ##store the orientation matrix in the statistics dataframe
    stat_dataframe.at[0,'Orientation Matrix']=THE_orientation_matrix
    return(stat_dataframe)


def find_mean_vec_length(stat_dataframe, melt_pocket_dataframe):
    ##find cumulative x, y, z coordinate, weighted by melt pocket volume
    sumx=sum(melt_pocket_dataframe['X Coordinates']*melt_pocket_dataframe['Melt Pocket Volume'])
    sumy=sum(melt_pocket_dataframe['Y Coordinates']*melt_pocket_dataframe['Melt Pocket Volume'])
    sumz=sum(melt_pocket_dataframe['Z Coordinates']*melt_pocket_dataframe['Melt Pocket Volume'])
    summelt=sum(melt_pocket_dataframe['Melt Pocket Volume'])##sum melt pocket volume to solve for mean vector length
    mean_resultant_vector_length=np.sqrt(sumx*sumx+sumy*sumy+sumz*sumz)/summelt##square x's, y's, z's, take square root, and divide by volume to find mean resultant vector length
    ##store in the statistics dataframe
    stat_dataframe.at[0,'R-bar']=mean_resultant_vector_length
    return(stat_dataframe)


def find_orientation_matrix_eigenvalues(stat_dataframe, melt_pocket_dataframe):
    normeig=[]
    [eigval, normeigvec] = np.linalg.eig(stat_dataframe.at[0,'Orientation Matrix'])     
    for i in range(len(eigval)):
        normeig.append(eigval[i]/sum(melt_pocket_dataframe['Melt Pocket Volume']))##find the unit eigenvalues (remove scaling by melt pocket volume)
    
    ##sort eigenvalues from greatest to smallest and keep the eigenvectors associated with each in the right order
    normeig=np.array(normeig)
    idx=normeig.argsort()[::-1]
    normeig=normeig[idx]
    normeigvec=normeigvec[:,idx]
    
    ##store the eigenvalues and eigenvectors associated with the orientation matrix in the stat dataframe
    stat_dataframe.at[0,'Eigenvalues']=normeig
    stat_dataframe.at[0,'Eigenvectors']=normeigvec
    return(stat_dataframe)


def plot_eigenvectors(stat_dataframe, polaraxis):
    eigenvectors=stat_dataframe.at[0, 'Eigenvectors']
    eigenvalues=stat_dataframe.at[0, 'Eigenvalues']
    for i in range(3):##for each eigenvector
        if eigenvectors[2, i]<0:
            eigenvectors[:, i]=eigenvectors[:, i]*-1
        theta=azimuth_from_xy(eigenvectors[0, i], eigenvectors[1, i])
        phi=inclination_from_z(eigenvectors[2, i])
        polaraxis.scatter(theta, phi, s=500*eigenvalues[i], zorder=5)##plot a marker for the theta, phi, for each eigenvector scaled to each eigenvalue's magntitude
    return(polaraxis)





