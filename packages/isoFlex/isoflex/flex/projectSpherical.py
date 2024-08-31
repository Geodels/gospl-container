import os
import glob
import pickle
import numpy as np
import xarray as xr
from pathlib import Path
from scipy import spatial


class projectSpherical(object):

    def __init__(self, fileout, size):

        self.size = size
        self.fileout = fileout
        self.flexds = None
        self.simflex = None

        return

    def _lonlat2xyz(self, lonlat, radius):

        rlon = np.radians(lonlat[:, 0])
        rlat = np.radians(lonlat[:, 1])

        coords = np.zeros((len(lonlat), 3))
        coords[:, 0] = np.cos(rlat) * np.cos(rlon) * radius
        coords[:, 1] = np.cos(rlat) * np.sin(rlon) * radius
        coords[:, 2] = np.sin(rlat) * radius

        return coords

    def buildLonLatMesh(self, data, res=0.25, nghb=4):

        radius = np.sqrt(
            data[0, 0] ** 2
            + data[0, 1] ** 2
            + data[0, 2] ** 2
        )
        tree1 = spatial.cKDTree(data[:, :3], leafsize=10)

        nx = int(360.0 / res) + 1
        ny = int(180.0 / res) + 1
        lon = np.linspace(-180.0, 180.0, nx)
        lat = np.linspace(-90.0, 90.0, ny)
        self.lon = lon.copy()
        self.lat = lat.copy()

        mlon, mlat = np.meshgrid(lon, lat)
        lonlati = np.dstack([mlon.flatten(), mlat.flatten()])[0]
        coords = self._lonlat2xyz(lonlati, radius)
        tree2 = spatial.cKDTree(coords, leafsize=10)

        dists, self.ids = tree1.query(coords, k=nghb)
        self.oIDs = np.where(dists[:, 0] == 0)[0]
        dists[self.oIDs, :] = 0.001

        self.nx = nx
        self.ny = ny
        self.wghts = 1.0 / dists ** 2
        self.denum = 1.0 / np.sum(self.wghts, axis=1)
        self.denum[self.oIDs] = 0.0

        edi = np.sum(self.wghts * data[self.ids, 3], axis=1) * self.denum
        tei = np.sum(self.wghts * data[self.ids, 4], axis=1) * self.denum

        if len(self.oIDs) > 0:
            edi[self.oIDs] = data[self.ids[self.oIDs, 0], 3]
            tei[self.oIDs] = data[self.ids[self.oIDs, 0], 4]

        edi = np.reshape(edi, (self.ny, self.nx))
        tei = np.reshape(tei, (self.ny, self.nx))

        dists, self.ids2 = tree2.query(data[:, :3], k=nghb)
        self.oIDs2 = np.where(dists[:, 0] == 0)[0]
        dists[self.oIDs2, :] = 0.001
        self.wghts2 = 1.0 / dists ** 2
        self.denum2 = 1.0 / np.sum(self.wghts2, axis=1)
        self.denum2[self.oIDs2] = 0.0

        # Build an xarray
        ds = xr.Dataset({
            'erodep':
            xr.DataArray(edi, coords=dict(
                latitude=self.lat,
                longitude=self.lon),
                dims=("latitude", "longitude"))
            })
        ds['te'] = (('latitude', 'longitude'), tei)

        return ds

    def updateData(self, data):

        edi = np.sum(self.wghts * data[self.ids, 0], axis=1) * self.denum
        tei = np.sum(self.wghts * data[self.ids, 1], axis=1) * self.denum

        if len(self.oIDs) > 0:
            edi[self.oIDs] = data[self.ids[self.oIDs, 0], 0]
            tei[self.oIDs] = data[self.ids[self.oIDs, 0], 1]

        edi = np.reshape(edi, (self.ny, self.nx))
        tei = np.reshape(tei, (self.ny, self.nx))

        # Build an xarray
        ds = xr.Dataset({
            'erodep':
            xr.DataArray(edi, coords=dict(
                latitude=self.lat,
                longitude=self.lon),
                dims=("latitude", "longitude"))
            })
        ds['te'] = (('latitude', 'longitude'), tei)

        return ds

    def _getFlexData(self, fds, fdse):

        nfds = xr.Dataset({
            'empty':
            xr.DataArray(np.nan, coords=dict(
                latitude=self.ds.latitude.values,
                longitude=self.ds.longitude.values),
                dims=("latitude", "longitude"))
            })

        shape = nfds.empty.values.shape
        sflexe = fdse.flex.values.copy()

        newflex = np.zeros(shape)
        newflex[:, 1:-1] = fds.flex.values.copy()

        datalat_min = fdse.latitude.values.min()
        datalat_max = fdse.latitude.values.max()

        ilat_min = list(nfds.latitude.values).index(
            nfds.sel(latitude=datalat_min, method='nearest').latitude)
        ilat_max = list(nfds.latitude.values).index(
            nfds.sel(latitude=datalat_max, method='nearest').latitude)
        ilon140 = list(nfds.longitude.values).index(
            nfds.sel(longitude=140.0, method='nearest').longitude)
        nlon140 = list(nfds.longitude.values).index(
            nfds.sel(longitude=-140.0, method='nearest').longitude)

        ilon0 = list(fdse.longitude.values).index(
            fdse.sel(longitude=0.0, method='nearest').longitude)
        ilon40 = list(fdse.longitude.values).index(
            fdse.sel(longitude=40.0, method='nearest').longitude)
        nlon40 = list(fdse.longitude.values).index(
            fdse.sel(longitude=-40.0, method='nearest').longitude)

        newflex[ilat_min:ilat_max+1, 0:nlon140 + 1] = sflexe[:, ilon0:ilon40+1]
        newflex[ilat_min:ilat_max+1, ilon140:] = sflexe[:, nlon40:ilon0+1]
        newflex[0:ilat_min, 0] = newflex[0: ilat_min, 1]
        newflex[0:ilat_min, -1] = newflex[0: ilat_min, -2]
        newflex[ilat_max+1:, 0] = newflex[ilat_max+1:, 1]
        newflex[ilat_max+1:, -1] = newflex[ilat_max+1:, -2]

        nfds['flex'] = (('latitude', 'longitude'), newflex)
        nfds['flex'].attrs['units'] = 'metres'

        del newflex

        return nfds[['flex']]

    def projectLonLat(self):

        # Combine the local grids into a single one
        # flexds = None
        flexdse = None
        flexds = self.flexNGrid.copy()

        if len(self.flexGrids) == 1:
            flexds = flexds.combine_first(self.flexGrids[0])

        for k in range(1, len(self.flexGrids)):
            if k == 1:
                flexds = self.flexGrids[0].combine_first(self.flexGrids[1])
            else:
                flexds = flexds.combine_first(self.flexGrids[k])
        for k in range(1, len(self.flexEdges)):
            if k == 1:
                flexdse = self.flexEdges[0].combine_first(self.flexEdges[1])
            else:
                flexdse = flexdse.combine_first(self.flexEdges[k])

        if self.size > 1:
            with open('sthGrid.pickle', 'rb') as file:
                flexSGrid = pickle.load(file)
        else:
            flexSGrid = self.flexSGrid

        # Add the ones built on other processors
        for r in range(1, self.size):
            file_grids = 'flexGrids_'+str(r)+'.pickle'
            if Path(file_grids).is_file():
                rank_flexg = None
                # Open the file in binary mode
                with open(file_grids, 'rb') as file:
                    # Deserialize and retrieve the variable from the file
                    rank_flexg = pickle.load(file)
                    if flexds is None:
                        for k in range(1, len(rank_flexg)):
                            if k == 1:
                                flexds = rank_flexg[0].combine_first(
                                    rank_flexg[1]
                                    )
                            else:
                                flexds = flexds.combine_first(
                                    rank_flexg[k]
                                    )
                    else:
                        for k in range(0, len(rank_flexg)):
                            flexds = flexds.combine_first(
                                rank_flexg[k]
                                )

            file_gridse = 'flexEdges_'+str(r)+'.pickle'
            if Path(file_gridse).is_file():
                rank_flexe = None
                # Open the file in binary mode
                with open(file_gridse, 'rb') as file:
                    # Deserialize and retrieve the variable from the file
                    rank_flexe = pickle.load(file)
                    if flexdse is None:
                        if len(rank_flexe) == 1:
                            flexdse = rank_flexe[0].copy()
                        for k in range(1, len(rank_flexe)):
                            if k == 1:
                                flexdse = rank_flexe[0].combine_first(
                                    rank_flexe[1])
                            else:
                                flexdse = flexdse.combine_first(rank_flexe[k])
                    else:
                        for k in range(0, len(rank_flexe)):
                            flexdse = flexdse.combine_first(rank_flexe[k])

        flexds = flexds.combine_first(self.flexNGrid)
        flexds = flexds.combine_first(flexSGrid)
        self.flexds = self._getFlexData(flexds, flexdse)

        if self.fileout is not None:
            self.flexds.to_netcdf(self.fileout)
        for filename in glob.glob("./*.pickle"):
            os.remove(filename)

        return

    def interp2umesh(self):

        regflex = self.flexds.flex.values.flatten()
        flex = np.sum(self.wghts2 * regflex[self.ids2], axis=1) * self.denum2

        if len(self.oIDs2) > 0:
            flex[self.oIDs2] = regflex[self.ids2[self.oIDs2, 0]]

        self.simflex = flex.copy()

        return