import pyproj
import numpy as np
import xarray as xr
import rioxarray as xrio
from mpi4py import MPI
from time import process_time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class projectUTM(object):

    def __init__(self, filename=None, dsin=None):

        # Read the input info
        if filename is not None:
            self.ds = xr.open_dataset(filename)
            self.ds = self.ds[['erodep', 'te']]
        else:
            self.ds = dsin[['erodep', 'te']]
        self.res = self.ds.longitude.values[1] - self.ds.longitude.values[0]
        self.dlon = 20
        self.dlat = 15

        return

    def projectData(self):

        tstart = process_time()

        # Split the global grids into tiles and project in Cartesian.
        self._shiftValues()
        self._makeBnds()

        if rank == 0 and self.verbose:
            print(
                "--- Build cartesian tiles (%0.02f seconds)"
                % (process_time() - tstart),
                flush=True
            )
        return

    def _shiftValues(self):

        dval = self.ds.erodep.values.copy()
        tval = self.ds.te.values.copy()

        lx = int((dval.shape[1]-1)/2)
        shifted = np.zeros(dval.shape)
        shifted[:, :lx+1] = dval[:, lx:]
        shifted[:, lx+1:] = dval[:, :lx]
        shiftte = np.zeros(tval.shape)
        shiftte[:, :lx+1] = tval[:, lx:]
        shiftte[:, lx+1:] = tval[:, :lx]

        self.ds['shifted'] = (('latitude', 'longitude'), shifted)
        self.ds['shifted'].attrs['units'] = 'metres'

        self.ds['shiftte'] = (('latitude', 'longitude'), shiftte)
        self.ds['shiftte'].attrs['units'] = 'metres'

        return

    def _makeBnds(self):

        lonrange = np.arange(-140, 140 + self.dlon, self.dlon)
        latrange = np.arange(-75, 75 + self.dlat, self.dlat)

        listbnds = []
        for lon in range(len(lonrange)-1):
            min_lon = lonrange[lon]
            max_lon = lonrange[lon+1]
            for lat in range(len(latrange)-1):
                min_lat = latrange[lat]
                max_lat = latrange[lat+1]
                listbnds.append([min_lon, max_lon, min_lat, max_lat])

        self.listbnds = []
        if size == 1:
            self.listbnds = listbnds.copy()
        elif size == 2:
            proclist = np.array_split(
                np.arange(len(listbnds), dtype=int), size
                )
            self.listbnds = listbnds[proclist[rank][0]:
                                     proclist[rank][-1] + 1]
        else:
            if size <= 142:
                proclist = np.array_split(
                                np.arange(len(listbnds), dtype=int), size - 2
                                )
            else:
                tmplist = np.array_split(
                                np.arange(len(listbnds), dtype=int), 140
                                )
                tmplist[len(tmplist):] = tmplist * (int(size/140)+1)
                proclist = tmplist[:size]
            if rank > 1:
                self.listbnds = listbnds[proclist[rank-2][0]:
                                         proclist[rank-2][-1] + 1]

        self.list_clip_utm = []
        for k in range(len(self.listbnds)):
            dll = 10
            if self.listbnds[k][2] <= -30 or self.listbnds[k][2] >= 30:
                dll = 20
            if self.listbnds[k][2] <= -45 or self.listbnds[k][2] >= 45:
                dll = 30
            if self.listbnds[k][2] <= -60 or self.listbnds[k][2] >= 60:
                dll = 40
            tclip_utm = self._makeUTM(k, dll)
            self.list_clip_utm.append(tclip_utm)

        # Get polar grids
        if rank == 0:
            self.north_utm = self._getNorthPole()
        if size > 1:
            if rank == 1:
                self.south_utm = self._getSouthPole()
        else:
            self.south_utm = self._getSouthPole()

        return

    def _makeUTM(self, k, dll=10):

        min_lon = self.listbnds[k][0] - 5
        max_lon = self.listbnds[k][1] + 5
        min_lat = self.listbnds[k][2] - 5
        max_lat = self.listbnds[k][3] + 5
        self.ds.rio.write_crs(4326, inplace=True)

        lon_bnds = [min_lon, max_lon]
        lat_bnds = [min_lat, max_lat]
        ds_clip = self.ds.sel(latitude=slice(*lat_bnds),
                              longitude=slice(*lon_bnds))

        ds_srtm = ds_clip.rename({'longitude': 'x', 'latitude': 'y'})
        epsg = ds_srtm.rio.estimate_utm_crs()
        clip_utm = ds_srtm.rio.reproject(epsg)
        clip_utm = clip_utm.sortby(clip_utm.x)
        clip_utm = clip_utm.sortby(clip_utm.y)
        x_bnds = [clip_utm.x.min(), clip_utm.x.max()]
        y_bnds = [clip_utm.y.min(), clip_utm.y.max()]

        lon_bnds2 = [min_lon-dll, max_lon+dll]
        lat_bnds2 = [min_lat-dll, max_lat+dll]
        ds_clip2 = self.ds.sel(latitude=slice(*lat_bnds2),
                               longitude=slice(*lon_bnds2))
        clip_utm2 = ds_clip2.rio.reproject(epsg)
        clip_utm2 = clip_utm2.sortby(clip_utm2.x)
        clip_utm2 = clip_utm2.sortby(clip_utm2.y)

        ds2_clip = clip_utm2.sel(x=slice(*x_bnds), y=slice(*y_bnds))
        ds2_clip = ds2_clip.where(ds2_clip.erodep < 20000)

        return self._interpUTM(ds2_clip)

    def _interpUTM(self, ds):

        xmax, xmin = ds.x.values.max(), ds.x.values.min()
        ymax, ymin = ds.y.values.max(), ds.y.values.min()
        dx = (xmax-xmin)/len(ds.x.values)
        dy = (ymax-ymin)/len(ds.y.values)
        rdx = np.round(dx, -3)
        rdy = np.round(dy, -3)

        rxmin = np.round(xmin, -3)
        if rxmin < xmin:
            rxmin += 1000
        rxmax = np.round(xmax, -3)
        if rxmax > xmax:
            rxmax -= 1000

        rymin = np.round(ymin, -3)
        if rymin < ymin:
            rymin += 1000
        rymax = np.round(ymax, -3)
        if rymax > ymax:
            rymax -= 1000
        newx = np.arange(rxmin, xmax, rdx)
        newy = np.arange(rymin, ymax, rdy)

        return ds.interp(x=newx, y=newy, method="cubic")

    def _getSouthPole(self):

        # Polar Stereographic South (71S,0E) epsg:3031
        source_crs = 'epsg:3031'
        target_crs = 'epsg:4326'
        latlon_to_spolar = pyproj.Transformer.from_crs(target_crs, source_crs)
        sblon, sblat = self._lon_lat_box([-180, 180], [-90, -60],
                                         refinement=100)
        sbx, sby = latlon_to_spolar.transform(sblat, sblon)

        sxmin, sxmax = sbx.min(), sbx.max()
        symin, symax = sby.min(), sby.max()

        southpole = self.ds.sel(latitude=slice(*[-90, -35]))
        southpole.rio.write_crs(target_crs, inplace=True)
        sPolar = southpole.rio.reproject(source_crs)
        sPolar = sPolar.sortby(sPolar.x)
        sPolar = sPolar.sortby(sPolar.y)
        southClip = sPolar.sel(x=slice(*[sxmin, sxmax]), 
                               y=slice(*[symin, symax]))

        return self._interpUTM(southClip)

    def _getNorthPole(self):
        # Polar Stereographic North (60N,0E) epsg:3995
        source_crs = 'epsg:3995'
        target_crs = 'epsg:4326'
        latlon_to_npolar = pyproj.Transformer.from_crs(target_crs, source_crs)
        nblon, nblat = self._lon_lat_box([-180, 180], [60, 90],
                                         refinement=100)
        nbx, nby = latlon_to_npolar.transform(nblat, nblon)

        nxmin, nxmax = nbx.min(), nbx.max()
        nymin, nymax = nby.min(), nby.max()

        northpole = self.ds.sel(latitude=slice(*[35, 90]))
        northpole.rio.write_crs(target_crs, inplace=True)
        nPolar = northpole.rio.reproject(source_crs)
        nPolar = nPolar.sortby(nPolar.x)
        nPolar = nPolar.sortby(nPolar.y)
        northClip = nPolar.sel(x=slice(*[nxmin, nxmax]),
                               y=slice(*[nymin, nymax]))

        return self._interpUTM(northClip)

    def _lon_lat_box(self, lon_bounds, lat_bounds, refinement=2):

        lons = []
        lats = []
        lons.append(np.linspace(lon_bounds[0], lon_bounds[-1],
                                num=refinement))
        lats.append(np.linspace(lat_bounds[0], lat_bounds[0],
                                num=refinement))
        lons.append(np.linspace(lon_bounds[-1], lon_bounds[-1],
                                num=refinement))
        lats.append(np.linspace(lat_bounds[0], lat_bounds[-1],
                                num=refinement))
        lons.append(np.linspace(lon_bounds[-1], lon_bounds[0],
                                num=refinement))
        lats.append(np.linspace(lat_bounds[-1], lat_bounds[-1],
                                num=refinement))
        lons.append(np.linspace(lon_bounds[0], lon_bounds[0],
                                num=refinement))
        lats.append(np.linspace(lat_bounds[-1], lat_bounds[0],
                                num=refinement))

        return np.concatenate(lons), np.concatenate(lats)


