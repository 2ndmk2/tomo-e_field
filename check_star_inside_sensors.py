#!/usr/bin/env python
# -*- coding: utf-8 *-
## Author: Masataka Aizawa

import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
def take_sensor_id_and_pos(df):
    """ Get sensor id & position:
    Parameters:
        df: (DataFrame)
            dataframe containing sensor information
    """
    foc_x_arr = []
    foc_y_arr = []

    det_id = df['det_id'].unique()
    for d in det_id:
        df_now = df.loc[df.det_id==d,['foc_x','foc_y']] 
        foc_x_now = df_now["foc_x"].values
        foc_y_now = df_now["foc_y"].values
        foc_x_arr.append([foc_x_now[0], foc_x_now[1], foc_x_now[3], foc_x_now[2]])
        foc_y_arr.append([foc_y_now[0], foc_y_now[1], foc_y_now[3], foc_y_now[2]])
    foc_x_arr = np.array(foc_x_arr)
    foc_y_arr = np.array(foc_y_arr)
    return det_id, foc_x_arr, foc_y_arr

def distance_min(rect, obj_pos):
    """ Compute minimum distance edges of rectangular
    Parameters:
        rect: (ndarray)
            [4*2] array describe (x,y) for 4 points
        obj_pos: (ndarray)
            1d array contains object position
    """

    p3=obj_pos

    p1=rect[0]
    p2=rect[1]
    d1=np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
    
    p1=rect[1]
    p2=rect[2]
    d2=np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
    p1=rect[2]
    p2=rect[3]
    d3=np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1) )
    
    p1=rect[3]
    p2=rect[0]
    d4=np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
    
    return np.min([d1, d2, d3, d4])

def check_points_inside_sensors(points, det_id, foc_x_arr, foc_y_arr):
    """ Check whether points are in sensors. Compute minimum distance to edges of sensors if they are inside.
    Parameters:
        points: (ndarray)
            [N*2] array describe (x,y) for N positions of targets
        det_id: (ndarray)
            array of M detector IDs
        foc_x_arr: (ndarray)
            [M, 4] array for x-coordiantes for detector vortices
        foc_y_arr: (ndarray)
            [M, 4] array for y-coordiantes for detector vortices    
    """        

    inside_or_not_arr = []
    dist_arr = []

    for point in points:
        ## choose closest sensors
        dist = np.sum((foc_x_arr - point[0])**2 +  (foc_y_arr - point[1])**2, axis=1)
        i_min = np.argmin(dist) 
        rect = np.array([foc_x_arr[i_min],foc_y_arr[i_min]]).T
        point_obj = Point(point)
        polygon = Polygon(rect)

        inside_or_not = polygon.contains(point_obj )

        if inside_or_not:
            inside_or_not_arr.append(inside_or_not)
            dist_arr.append(distance_min(rect, point))
        else:
            inside_or_not_arr.append(inside_or_not)
            dist_arr.append(-1)
    return np.array(inside_or_not_arr), np.array(dist_arr)

