import pickle
import numpy as np
from mpi4py import MPI
from gflex.f2d import F2D

import xarray as xr
from scipy import spatial
from rasterio.warp import transform

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class computeFlex(object):

    def __init__(self, young, nu, rho_m, rho_s, flexgrd):

        self.flexNGrid = None
        self.flexSGrid = None
        self.flexgrd = flexgrd
        self.young = young
        self.nu = nu
        self.rho_m = rho_m
        self.rho_s = rho_s

        return

    def getFlex(self):

        grd_flex = []
        bnd_flex = []
        grd_flex_edges = []
        bnd_flex_edges = []
        for k in range(len(self.listbnds)):
            if self.listbnds[k][0] >= -140 and self.listbnds[k][0] < 140:
                bnd_flex.append(self.listbnds[k])
                grd_flex.append(self._cmptFlex(self.list_clip_utm[k],
                                               shifted=False))
            if self.listbnds[k][0] >= -40 and self.listbnds[k][0] < 40:
                bnd_flex_edges.append(self.listbnds[k])
                grd_flex_edges.append(self._cmptFlex(self.list_clip_utm[k],
                                                     shifted=True))
        flexGrids = []
        for k in range(len(grd_flex)):
            tmp = grd_flex[k]
            # tmp = self._reproject_ll(bnd_flex[k], tmp, 1, tmp.rio.crs)
            tmp = self._back2LonLat(tmp['flex'].to_dataset(), bnd_flex[k])
            flexGrids.append(tmp)
        self.flexGrids = flexGrids.copy()

        flexEdges = []
        for k in range(len(grd_flex_edges)):
            tmp = grd_flex_edges[k]
            # tmp = self._reproject_ll(bnd_flex[k], tmp, 1, tmp.rio.crs)
            tmp = self._back2LonLat(tmp['flex'].to_dataset(),
                                    bnd_flex_edges[k])
            flexEdges.append(tmp)
        self.flexEdges = flexEdges.copy()

        # Save local files on disk
        if rank > 0:
            if len(flexGrids) > 0:
                file_grids = 'flexGrids_'+str(rank)+'.pickle'
                with open(file_grids, 'wb') as file:
                    pickle.dump(flexGrids, file)
            if len(flexEdges) > 0:
                file_edges = 'flexEdges_'+str(rank)+'.pickle'
                with open(file_edges, 'wb') as file:
                    pickle.dump(flexEdges, file)

        # Run gflex on polar grids
        if rank == 0:
            north_flex = self._cmptFlex(self.north_utm, shifted=False)
            flexNGrid = self._reproject_ll([-180, 180, 75, 90], north_flex,
                                           2, north_flex.rio.crs)
            # tmp = self._makeLonLat(north_flex, [-180, 180, 75, 90])[['flex']]
            # flexNGrid = self._interpLonLat2(tmp, [-180, 180, 75, 90])
            flexNGrid.flex.rio.write_nodata(np.nan, inplace=True)
            self.flexNGrid = flexNGrid.rio.interpolate_na()

        if size > 1:
            if rank == 1:
                south_flex = self._cmptFlex(self.south_utm, shifted=False)
                flexSGrid = self._reproject_ll([-180, 180, -90, -75],
                                               south_flex, 2,
                                               south_flex.rio.crs)
                # tmp = self._makeLonLat(south_flex,
                #                        [-180, 180, -90, -75])[['flex']]
                # flexSGrid = self._interpLonLat2(tmp, [-180, 180, -90, -75])
                flexSGrid.flex.rio.write_nodata(np.nan, inplace=True)
                flexSGrid = flexSGrid.rio.interpolate_na()
                south_grid = 'sthGrid.pickle'
                with open(south_grid, 'wb') as file:
                    pickle.dump(flexSGrid, file)
        else:
            south_flex = self._cmptFlex(self.south_utm, shifted=False)
            flexSGrid = self._reproject_ll([-180, 180, -90, -75], south_flex,
                                           2, south_flex.rio.crs)

            flexSGrid.flex.rio.write_nodata(np.nan, inplace=True)
            self.flexSGrid = flexSGrid.rio.interpolate_na()

        return

    def _reproject_ll(self, bnds, utmds, step, src_crs,
                      var='flex', dst_crs='EPSG:4326'):

        xv, yv = np.meshgrid(utmds.x, utmds.y)
        datav = utmds[var].values.flatten().copy()

        lon, lat = transform(src_crs, dst_crs, xv.flatten(), yv.flatten())
        lonlat = np.dstack([np.asarray(lon), np.asarray(lat)])[0]
        tree_ll = spatial.cKDTree(lonlat, leafsize=10)

        if step == 1:
            lon = np.arange(bnds[0], bnds[1] + self.res, self.res)
        else:
            lon = np.arange(bnds[0] + self.res, bnds[1], self.res)
        lat = np.arange(bnds[2], bnds[3] + self.res, self.res)
        nlon, nlat = np.meshgrid(lon, lat)
        nshape = nlon.shape
        nlonlat = np.dstack([nlon.flatten(), nlat.flatten()])[0]

        dists, ids = tree_ll.query(nlonlat, k=3)
        oIDs = np.where(dists[:, 0] == 0)[0]
        dists[oIDs, :] = 0.001
        wghts = 1.0 / dists ** 2
        denum = 1.0 / np.sum(wghts, axis=1)
        denum[oIDs] = 0.0

        ndata = np.sum(wghts * datav[ids], axis=1) * denum
        if len(oIDs) > 0:
            ndata[oIDs] = datav[ids[oIDs, 0]]
        ndata = np.reshape(ndata, nshape)

        # Build an xarray
        ds = xr.Dataset({
            'tmp':
            xr.DataArray(ndata, coords=dict(
                latitude=lat,
                longitude=lon),
                dims=("latitude", "longitude"))
            })
        ds = ds.rename({'tmp': var})
        ds.rio.write_crs(dst_crs, inplace=True)

        return ds

    def _makeLonLat(self, utmds, bnds):

        test = utmds.copy()
        test.rio.write_crs(test.rio.crs, inplace=True)
        testll = test.rio.reproject('EPSG:4326')
        testll = testll.sortby(testll.x)
        testll = testll.sortby(testll.y)
        testll = testll.rename({'x': 'longitude', 'y': 'latitude'})
        lon_bnds = [bnds[0]-self.res*2, bnds[1]+self.res*2]
        lat_bnds = [bnds[2]-self.res*2, bnds[3]+self.res*2]
        map = testll.sel(latitude=slice(*lat_bnds), longitude=slice(*lon_bnds))
        map = map.where(map.flex < 200000)

        return map

    def _interpLonLat(self, ds, bnds):

        rxmin, rxmax = bnds[0], bnds[1]
        rymin, rymax = bnds[2], bnds[3]
        newx = np.arange(rxmin, rxmax + self.res, self.res)
        newy = np.arange(rymin, rymax + self.res, self.res)

        return ds.interp(longitude=newx, latitude=newy, method="nearest")

    def _interpLonLat2(self, ds, bnds):

        rxmin, rxmax = bnds[0], bnds[1]
        rymin, rymax = bnds[2], bnds[3]
        newx = np.arange(rxmin + self.res, rxmax, self.res)
        newy = np.arange(rymin, rymax + self.res, self.res)

        return ds.interp(longitude=newx, latitude=newy, method="nearest")

    def _back2LonLat(self, utm_ds, bnds):

        tmp = self._makeLonLat(utm_ds, bnds)

        return self._interpLonLat(tmp, bnds)

    def _cmptFlex(self, xrutm, shifted=False):

        if self.flexgrd > 1:
            xcoord = xrutm.x.values.copy()
            ycoord = xrutm.y.values.copy()
            xrutm = xrutm.coarsen(y=self.flexgrd, x=self.flexgrd,
                                  boundary='pad').mean()

        if shifted:
            cload = xrutm.shifted.values.copy()
            cte = xrutm.shiftte.values.copy()
        else:
            cload = xrutm.erodep.values.copy()
            cte = xrutm.te.values.copy()

        # z = xrutm.elevation.values.copy()
        dx = xrutm.x.values[1]-xrutm.x.values[0]
        dy = xrutm.y.values[1]-xrutm.y.values[0]
        flex = F2D()
        flex.Quiet = True

        flex.Method = "FD"
        flex.PlateSolutionType = "vWC1994"
        flex.Solver = "direct"

        flex.g = 9.8
        flex.rho_fill = 0.0
        flex.E = self.young
        flex.nu = self.nu
        flex.rho_m = self.rho_m

        flex.Te = cte.copy()

        flex.qs = cload * self.rho_s * flex.g
        # r, c = np.where(z < 0)
        # flex.qs[r, c] = cload[r, c] * (flex.rho_fill-rho_water) * flex.g
        flex.dx = dx
        flex.dy = dy

        # Boundary conditions:
        flex.BC_E = "Periodic"  # west boundary condition
        flex.BC_W = "Periodic"  # east boundary condition
        flex.BC_S = "Periodic"  # south boundary condition
        flex.BC_N = "Periodic"  # north boundary condition

        flex.initialize()
        flex.run()
        flex.finalize()

        xrutm['flex'] = (['y', 'x'], flex.w)
        if self.flexgrd > 1:
            xrutm = xrutm.interp(x=xcoord, y=ycoord, method="nearest")
            xrutm.flex.rio.write_nodata(np.nan, inplace=True)
            # xrutm.elevation.rio.write_nodata(np.nan, inplace=True)
            xrutm.erodep.rio.write_nodata(np.nan, inplace=True)
            xrutm.te.rio.write_nodata(np.nan, inplace=True)
            # xrutm.shiftz.rio.write_nodata(np.nan, inplace=True)
            xrutm.shifted.rio.write_nodata(np.nan, inplace=True)
            xrutm.shiftte.rio.write_nodata(np.nan, inplace=True)
            return xrutm.rio.interpolate_na()

        return xrutm
