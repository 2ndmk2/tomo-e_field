import fieldchecker
import io
from astropy.wcs import WCS
from astropy.coordinates import Longitude, Latitude, Angle, SkyCoord
import pandas as pd
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import check_star_inside_sensors
import importlib
import argparse
import astropy.units as u



parser = argparse.ArgumentParser()
parser.add_argument('--file')   
parser.add_argument('--output', default="selected")   
parser.add_argument('--ra_cen', default = 50)   
parser.add_argument('--dec_cen', default = 50)   
args = parser.parse_args()
catalogfile = args.file
ra_cen = float(args.ra_cen)
dec_cen = float(args.dec_cen)
output = str(args.output)
tel_wd = 10
dist_thr = 30

## make telescope FOV
csv = io.StringIO(fieldchecker.__pix2foc_str__)
ra_tel, dec_tel =  fieldchecker.LongitudeLike(ra_cen), fieldchecker.LongitudeLike(dec_cen)
pixscale = fieldchecker.Arcsecond(1.189)
tomoe_info = pd.read_csv("./data/tomoe_instrument.csv")
gc = tomoe_info[['foc_x','foc_y']].mean()
delta,phi = fieldchecker.calc_deltaphi(gc.foc_x, gc.foc_y, pixscale)
ra_opt,dec_opt = fieldchecker.calc_pointing(ra_tel, dec_tel, delta, phi)
proj = fieldchecker.get_projection(ra_opt, dec_opt, pixscale)
det_id, foc_x_arr, foc_y_arr =check_star_inside_sensors.take_sensor_id_and_pos(tomoe_info)


## load catalog
catalog = pd.read_csv(catalogfile )
ra = catalog["ra"]
dec =  catalog["dec"]
mag = catalog["catalog_mag"]

## select targets in FOV
mask = (ra.values>ra_cen - tel_wd ) *  (ra.values<ra_cen + tel_wd ) * \
(dec.values>dec_cen - tel_wd) *  (dec.values<dec_cen + tel_wd)
mask_target_can = catalog[mask]
c = SkyCoord(ra=mask_target_can["ra"]*u.degree, dec=mask_target_can["dec"]*u.degree, frame='icrs')
points_can = proj.wcs_world2pix(np.array([c.ra, c.dec]).T,0)
inside_or_not_arr,dist_arr = check_star_inside_sensors.check_points_inside_sensors(points_can  , det_id, foc_x_arr, foc_y_arr)
mask_can = inside_or_not_arr * (dist_arr>dist_thr)

## plot
detectors = fieldchecker.detector_footprint(tomoe_info)
focpix = np.array(tomoe_info[['foc_x', 'foc_y']]).reshape([-1,2])
fig,ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': proj})
ax.add_collection(detectors)
ax.scatter(0.0, 0.0, marker='x', label='optical center')
ax.scatter(gc['foc_x'], gc['foc_y'], marker='x', label='gravity center')
ax.scatter(focpix[:,0],focpix[:,1],10,color='k',marker='.')
ax.scatter(points_can [:,0],points_can [:,1], 1,color='k',marker='.')
ax.scatter(points_can[mask_can,0],points_can[mask_can,1], 5,color='r',marker='.')
plt.savefig("%s.png" % output)
#plt.show()


## visibility function
visibility = np.zeros_like(ra.values)
count =0
for i in range(len(visibility)):
    if mask[i]:
        if mask_can[count]:
            visibility[i] = 1
        count+=1


## output to csv file
catalog["visibility"] = visibility.astype("int")
catalog.to_csv("%s.csv" % output)
