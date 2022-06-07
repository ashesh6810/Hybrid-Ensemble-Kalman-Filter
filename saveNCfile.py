import numpy as np
import netCDF4 as nc4

def savenc(x,lon,lat,filename):
    f = nc4.Dataset(filename,'w', format='NETCDF4')
    tempgrp = f.createGroup('Temp_data')
    tempgrp.createDimension('lon', len(lon))
    tempgrp.createDimension('lat', len(lat))
    tempgrp.createDimension('level', 2)
    tempgrp.createDimension('time', None)
    
    longitude = tempgrp.createVariable('Longitude', 'f4', 'lon')
    latitude = tempgrp.createVariable('Latitude', 'f4', 'lat')  
#    levels = tempgrp.createVariable('Levels', 'i4', 'level')
    psi = tempgrp.createVariable('PSI', 'f4', ('time','lon','lat','level'))
    time = tempgrp.createVariable('Time', 'i4', 'time')

    longitude[:] = lon
    latitude[:] = lat
    psi[:,:,:,:] = x
  
    f.close()

