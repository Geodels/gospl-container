import os
import sys
import petsc4py
import numpy as np
import pandas as pd

from operator import itemgetter

if "READTHEDOCS" not in os.environ:
    from ruamel.yaml import YAML
    from scipy.interpolate import interp1d

petsc4py.init(sys.argv)
MPIrank = petsc4py.PETSc.COMM_WORLD.Get_rank()


class ReadYaml(object):
    """
    Class for reading simulation input file and initialising model parameters.

    Definition of input parameters is provided in the `User Documentation <https://gospl.readthedocs.io/en/latest/inputfile.html>`_
    """

    def __init__(self, filename):
        """
        Parsing YAML file.

        :arg filename: input filename (.yml YAML file)
        """

        if self.showlog:
            self.log = petsc4py.PETSc.Log()
            self.log.begin()

        # Check input file exists
        self.finput = filename
        try:
            with open(filename) as finput:
                pass
        except IOError:
            print("Unable to open file: ", filename, flush=True)
            raise IOError("The input file is not found...")

        # Open YAML file
        with open(filename, "r") as finput:
            yaml = YAML(typ='rt')
            self.input = yaml.load(finput)

        if MPIrank == 0 and "name" in self.input.keys() and self.verbose:
            print(
                "The following model will be run:     {}".format(self.input["name"]),
                flush=True,
            )

        # Read simulation parameters
        self._readDomain()
        self._readTime()
        self._readSPL()
        self._readHillslope()
        self._readSealevel()
        self._readTectonic()
        self._readErofactor()
        self._readPlate()
        self._readRain()
        self._readCompaction()
        self._readIce()
        self._readFlex()
        self._readGFlex()
        self._readOrography()
        self._readOut()

        self.radius = 6378137.0
        self.gravity = 9.81
        self.tNow = self.tStart
        self.saveTime = self.tNow
        if self.strat > 0:
            self.saveStrat = self.tNow + self.strat
        else:
            self.saveStrat = self.tEnd + self.tout

        # In case of restarting simulation
        if self.rStep > 0:
            rNow = self.tStart + self.rStep * self.tout

            if self.raindata is not None:
                for k in range(len(self.raindata)):
                    if self.raindata["start"][k] < rNow and k < len(self.raindata) - 1:
                        self.raindata.loc[k, ["start"]] = rNow - self.tout
                    elif (
                        self.raindata["start"][k] < rNow and k == len(self.raindata) - 1
                    ):
                        self.raindata.loc[k, ["start"]] = rNow
                self.raindata = self.raindata[self.raindata["start"] >= rNow]
                self.raindata.reset_index(drop=True, inplace=True)
                self.rainNb = len(self.raindata)

            if self.sedfacdata is not None:
                for k in range(len(self.sedfacdata)):
                    if self.sedfacdata["start"][k] < rNow and k < len(self.sedfacdata) - 1:
                        self.sedfacdata.loc[k, ["start"]] = rNow - self.tout
                    elif (
                        self.sedfacdata["start"][k] < rNow and k == len(self.sedfacdata) - 1
                    ):
                        self.sedfacdata.loc[k, ["start"]] = rNow
                self.sedfacdata = self.sedfacdata[self.sedfacdata["start"] >= rNow]
                self.sedfacdata.reset_index(drop=True, inplace=True)
                self.sedfactNb = len(self.sedfacdata)

        return

    def _readDomain(self):
        """
        Read domain definition, boundary conditions and flow direction parameters.
        """

        try:
            domainDict = self.input["domain"]
        except KeyError:
            print(
                "Key 'domain' is required and is missing in the input file!", flush=True
            )
            raise KeyError("Key domain is required in the input file!")

        try:
            self.flowDir = domainDict["flowdir"]
        except KeyError:
            self.flowDir = 6

        try:
            self.flowExp = domainDict["flowexp"]
        except KeyError:
            self.flowExp = 0.5

        try:
            self.boundCond = domainDict["bc"]
        except KeyError:
            self.boundCond = '1111'

        try:
            meshFile = domainDict["npdata"]
        except KeyError:
            print(
                "Key 'npdata' is required and is missing in the 'domain' declaration!",
                flush=True,
            )
            raise KeyError("Compressed numpy dataset definition is not defined!")

        self.meshFile = meshFile + ".npz"

        try:
            with open(self.meshFile) as meshfile:
                meshfile.close()

        except IOError:
            print("Unable to open numpy dataset: {}".format(self.meshFile), flush=True)
            raise IOError("The numpy dataset is not found...")

        try:
            self.fast = domainDict["fast"]
        except KeyError:
            self.fast = False

        try:
            self.interp = domainDict["interp"]
        except KeyError:
            self.interp = 3

        try:
            self.seaDepo = domainDict["seadepo"]
        except KeyError:
            self.seaDepo = True

        self._extraDomain()

        return

    def _extraDomain(self):
        """
        Read domain additional information.
        """

        domainDict = self.input["domain"]

        try:
            self.excessIn = domainDict["excess"]
        except KeyError:
            self.excessIn = False

        try:
            self.overlap = domainDict["overlap"]
        except KeyError:
            self.overlap = 1

        try:
            dataFile = domainDict["nperodep"]
            self.dataFile = dataFile + ".npz"
            with open(self.dataFile) as dataFile:
                dataFile.close()
        except KeyError:
            self.dataFile = None

        try:
            self.nodep = domainDict["nodep"]
        except KeyError:
            self.nodep = False

        try:
            strataFile = domainDict["npstrata"]
            self.strataFile = strataFile + ".npz"
            with open(self.strataFile) as strataFile:
                strataFile.close()
        except KeyError:
            self.strataFile = None

        try:
            self.fitMarine = domainDict["fitmarine"]
        except KeyError:
            self.fitMarine = False

        return

    def _readTime(self):
        """
        Read simulation time declaration.
        """

        try:
            timeDict = self.input["time"]
        except KeyError:
            print(
                "Key 'time' is required and is missing in the input file!", flush=True
            )
            raise KeyError("Key time is required in the input file!")

        try:
            self.tStart = timeDict["start"]
        except KeyError:
            print(
                "Key 'start' is required and is missing in the 'time' declaration!",
                flush=True,
            )
            raise KeyError("Simulation start time needs to be declared.")

        try:
            self.tEnd = timeDict["end"]
        except KeyError:
            print(
                "Key 'end' is required and is missing in the 'time' declaration!",
                flush=True,
            )
            raise KeyError("Simulation end time needs to be declared.")

        if self.tEnd <= self.tStart:
            raise ValueError("Simulation end/start times do not make any sense!")

        try:
            self.dt = timeDict["dt"]
        except KeyError:
            print(
                "Key 'dt' is required and is missing in the 'time' declaration!",
                flush=True,
            )
            raise KeyError("Simulation discretisation time step needs to be declared.")

        try:
            self.tout = timeDict["tout"]
        except KeyError:
            self.tout = self.tEnd - self.tStart
            print(
                "Output time interval 'tout' has been set to {} years".format(
                    self.tout
                ),
                flush=True,
            )

        self._addTime(timeDict)

        return

    def _addTime(self, timeDict):
        """
        Read additional time parameters.
        """

        try:
            self.rStep = timeDict["rstep"]
        except KeyError:
            self.rStep = 0

        if self.tStart + self.tout > self.tEnd:
            self.tout = self.tEnd - self.tStart
            print(
                "Output time interval was changed to {} years to match the end time".format(
                    self.tout
                ),
                flush=True,
            )

        if self.tout < self.dt:
            self.tout = self.dt
            print(
                "Output time interval was changed to {} years to match the time step dt".format(
                    self.dt
                ),
                flush=True,
            )

        try:
            self.tecStep = timeDict["tec"]
        except KeyError:
            self.tecStep = self.tout

        try:
            self.strat = timeDict["strat"]
        except KeyError:
            self.strat = 0

        if self.tout < self.tecStep:
            self.tecStep = self.tout
            print(
                "Output time interval and tectonic forcing time step have been adjusted to match each others.",
                flush=True,
            )

        if self.tout < self.strat:
            self.strat = self.tout
            print(
                "Output time interval and stratal time step \
                 have been adjusted to match each others.",
                flush=True,
            )

        if self.tecStep > 0:
            if self.tout % self.tecStep != 0:
                print(
                    "When declaring tectonic time interval, the value should be divisible by the output time interval.",
                    flush=True,
                )
                raise ValueError("Tectonic time interval definition is wrong!")

        if self.strat > 0:
            if self.tout % self.strat != 0:
                print(
                    "When declaring stratal time interval, the value should be divisible by the output time interval.",
                    flush=True,
                )
                raise ValueError("Stratal time interval definition is wrong!")
            self.stratNb = int((self.tEnd - self.tStart) / self.strat) + 1
        else:
            self.stratNb = 0

        return

    def _readSPL(self):
        """
        Read surface processes bedrock parameters.
        """

        try:
            splDict = self.input["spl"]
            try:
                self.K = splDict["K"]
            except KeyError:
                print(
                    "When using the Surface Process Model definition of coefficient K is required.",
                    flush=True,
                )
                raise ValueError("Surface Process Model: K coefficient not found.")
            try:
                self.coeffd = splDict["d"]
            except KeyError:
                self.coeffd = 0.0
            try:
                self.fDepa = splDict["fDa"]
            except KeyError:
                self.fDepa = 0.0
            try:
                self.fDepm = splDict["fDm"]
            except KeyError:
                self.fDepm = 0.0
            try:
                self.spl_m = splDict["m"]
            except KeyError:
                self.spl_m = 0.5
            try:
                self.dmthd = splDict["mthd"]
            except KeyError:
                self.dmthd = 1
        except KeyError:
            self.K = 1.0e-12
            self.coeffd = 0.0
            self.fDepa = 0.0
            self.fDepm = 30.0
            self.spl_m = 0.5
            self.dmthd = 1

        return

    def _readHillslope(self):
        """
        Read hillslope parameters.
        """

        try:
            hillDict = self.input["diffusion"]

            try:
                self.Cda = hillDict["hillslopeKa"]
            except KeyError:
                print(
                    "When declaring diffusion processes, the coefficient hillslopeKa is required.",
                    flush=True,
                )
                raise ValueError(
                    "Hillslope: Cd coefficient for aerial environment not found."
                )
            try:
                self.Cdm = hillDict["hillslopeKm"]
            except KeyError:
                print(
                    "When declaring diffusion processes, the coefficient hillslopeKm is required.",
                    flush=True,
                )
                raise ValueError(
                    "Hillslope: Cd coefficient for marine environment not found."
                )
            try:
                self.oFill = hillDict["oFill"]
            except KeyError:
                self.oFill = -6000.0

        except KeyError:
            self.Cda = 0.0
            self.Cdm = 0.0
            self.sedimentK = 1.0e3

        self._extraHillslope()

        return

    def _extraHillslope(self):
        """
        Read extra hillslope parameters.
        """

        try:
            hillDict = self.input["diffusion"]

            try:
                self.smthD = hillDict["smthDep"]
            except KeyError:
                self.smthD = 1.0e3
            try:
                self.clinSlp = hillDict["clinSlp"]
            except KeyError:
                self.clinSlp = 1.0e-6
            try:
                self.diffNb = hillDict["diffNb"]
            except KeyError:
                self.diffNb = 1

        except KeyError:
            self.smthD = 1.0e3
            self.clinSlp = 1.0e-7
            self.diffNb = 1

        self.clinSlp = max(1.0e-7, self.clinSlp)

        return

    def _readSealevel(self):
        """
        Define sealevel evolution.
        """

        seafile = None
        sealevel = 0.0
        self.seafunction = None
        try:
            seaDict = self.input["sea"]
            try:
                sealevel = seaDict["position"]
                try:
                    seafile = seaDict["curve"]
                except KeyError:
                    seafile = None
            except KeyError:
                try:
                    seafile = seaDict["curve"]
                except KeyError:
                    seafile = None
        except KeyError:
            sealevel = 0.0

        if seafile is not None:
            try:
                with open(seafile) as fsea:
                    fsea.close()
                    try:
                        seadata = pd.read_csv(
                            seafile,
                            sep=r",",
                            engine="c",
                            header=None,
                            na_filter=False,
                            dtype=np.float64,
                            low_memory=False,
                        )

                    except ValueError:
                        try:
                            seadata = pd.read_csv(
                                seafile,
                                sep=r"\s+",
                                engine="c",
                                header=None,
                                na_filter=False,
                                dtype=np.float64,
                                low_memory=False,
                            )

                        except ValueError:
                            print(
                                "The sea-level file is not well formed: it should be comma or tab separated",
                                flush=True,
                            )
                            raise ValueError("Wrong formating of sea-level file.")
            except IOError:
                print("Unable to open file: ", seafile)
                raise IOError("The sealevel file is not found...")

            seadata[1] += sealevel
            if seadata[0].min() > self.tStart:
                tmpS = []
                tmpS.insert(0, {0: self.tStart, 1: seadata[1].iloc[0]})
                seadata = pd.concat([pd.DataFrame(tmpS), seadata], ignore_index=True)
            if seadata[0].max() < self.tEnd:
                tmpE = []
                tmpE.insert(0, {0: self.tEnd, 1: seadata[1].iloc[-1]})
                seadata = pd.concat([seadata, pd.DataFrame(tmpE)], ignore_index=True)
            self.seafunction = interp1d(seadata[0], seadata[1], kind="linear")
        else:
            year = np.linspace(self.tStart, self.tEnd + self.dt, num=11, endpoint=True)
            seaval = np.full(len(year), sealevel)
            self.seafunction = interp1d(year, seaval, kind="linear")

        return

    def _storeTectonic(self, k, tecStart, zMap, tMap, tStep, tEnd, tecdata):
        """
        Record tectonic conditions.

        :arg k: tectonic event number
        :arg tecStart: tectonic event start time
        :arg zMap: horizontal tectonic displacement file
        :arg tMap: vertical tectonic displacement file
        :arg tStep: tectonic time step
        :arg tEnd: tectonic event end time
        :arg tecdata: pandas dataframe storing each tectonic event

        :return: appended tecdata
        """

        if tMap is not None:
            if self.meshFile != tMap:
                try:
                    with open(tMap) as tecfile:
                        tecfile.close()

                except IOError:
                    print("Unable to open tectonic file: {}".format(tMap), flush=True)
                    raise IOError(
                        "The tectonic file {} is not found for climatic event {}.".format(
                            tMap, k
                        )
                    )
        else:
            tMap = "empty"

        if zMap is not None:
            if self.meshFile != zMap:
                try:
                    with open(zMap) as tecfile:
                        tecfile.close()

                except IOError:
                    print("Unable to open tectonic file: {}".format(zMap), flush=True)
                    raise IOError(
                        "The tectonic file {} is not found for tectonic event {}.".format(
                            zMap, k
                        )
                    )
        else:
            zMap = "empty"

        if tMap == "empty" and zMap == "empty":
            print(
                "For each tectonic event a tectonic grid (mapH or mapV) is required.",
                flush=True,
            )
            raise ValueError("Tectonic event {} has no tectonic map (map).".format(k))

        tmpTec = []
        tmpTec.insert(0, {"start": tecStart, "tMap": tMap, "zMap": zMap})

        if k == 0:
            tecdata = pd.DataFrame(tmpTec, columns=["start", "tMap", "zMap"])
        else:
            tecdata = pd.concat(
                [tecdata, pd.DataFrame(tmpTec, columns=["start", "tMap", "zMap"])],
                ignore_index=True,
            )

        if tStep is not None:
            if tEnd is not None:
                tectime = tecStart + tStep
                while tectime < tEnd:
                    tmpTec = []
                    tmpTec.insert(0, {"start": tectime, "tMap": tMap, "zMap": zMap})
                    tecdata = pd.concat(
                        [
                            tecdata,
                            pd.DataFrame(tmpTec, columns=["start", "tMap", "zMap"]),
                        ],
                        ignore_index=True,
                    )
                    tectime = tectime + tStep

        return tecdata

    def _defineTectonic(self, k, tecSort, tecdata):
        """
        Define tectonic conditions.

        :arg k: tectonic event number
        :arg tecSort: sorted tectonic event
        :arg tecdata: pandas dataframe storing each tectonic event
        :return: appended tecdata
        """

        tecStart = None
        tEnd = None
        tStep = None
        tMap = None
        zMap = None

        try:
            tecStart = tecSort[k]["start"]
        except Exception:
            print("For each tectonic event a start time is required.", flush=True)
            raise ValueError("Tectonic event {} has no parameter start".format(k))
        try:
            tMap = tecSort[k]["mapH"] + ".npz"
        except Exception:
            pass
        try:
            zMap = tecSort[k]["mapV"] + ".npz"
        except Exception:
            pass
        try:
            tStep = self.tecStep
        except Exception:
            pass
        try:
            tEnd = tecSort[k]["end"]
        except Exception:
            pass

        tecdata = self._storeTectonic(k, tecStart, zMap, tMap, tStep, tEnd, tecdata)

        return tecdata

    def _readTectonic(self):
        """
        Parse tectonic forcing conditions.
        """

        tecdata = None
        try:
            tecDict = self.input["tectonic"]
            tecSort = sorted(tecDict, key=itemgetter("start"))
            for k in range(len(tecSort)):
                tecdata = self._defineTectonic(k, tecSort, tecdata)

            if tecdata["start"][0] > self.tStart:
                tmpTec = []
                tmpTec.insert(
                    0, {"start": self.tStart, "tMap": "empty", "zMap": "empty"}
                )
                tecdata = pd.concat(
                    [pd.DataFrame(tmpTec, columns=["start", "tMap", "zMap"]), tecdata],
                    ignore_index=True,
                )
            self.tecdata = tecdata[tecdata["start"] >= self.tStart]

            if self.rStep > 0:
                rNow = self.tStart + self.rStep * self.tout
                for k in range(len(self.tecdata)):
                    if self.tecdata["start"][k] < rNow and k < len(self.tecdata) - 1:
                        self.tecdata.loc[k, ["start"]] = rNow - self.tout
                    elif self.tecdata["start"][k] < rNow and k == len(self.tecdata) - 1:
                        self.tecdata.loc[k, ["start"]] = rNow
                self.tecdata = self.tecdata[self.tecdata["start"] >= rNow]
                self.tecdata.reset_index(drop=True, inplace=True)

        except KeyError:
            self.tecdata = None

        return

    def _defineErofactor(self, k, sStart, sMap, sUniform, sedfacdata):
        """
        Define sediment surface erodibility factor conditions.

        :arg k: erodibility factor map number
        :arg sStart: erodibility factor map start time
        :arg sMap: erodibility factor map file event
        :arg sUniform: erodibility factor uniform value event
        :arg sedfacdata: pandas dataframe storing each erodibility factor map
        :return: appended sedfacdata
        """

        if sMap is None and sUniform is None:
            print(
                "For each erodibility factor map a factor value (uniform) or a factor \
                grid (map) is required.",
                flush=True,
            )
            raise ValueError(
                "Sediment erodibility factor {} has no value (uniform) or a \
                map (map).".format(
                    k
                )
            )

        tmpErof = []
        if sMap is None:
            tmpErof.insert(
                0,
                {"start": sStart, "rUni": sUniform, "sMap": None, "sKey": None},
            )
        else:
            tmpErof.insert(
                0,
                {
                    "start": sStart,
                    "sUni": None,
                    "sMap": sMap[0] + ".npz",
                    "sKey": sMap[1],
                },
            )

        if k == 0:
            sedfacdata = pd.DataFrame(tmpErof, columns=["start", "sUni", "sMap", "sKey"])
        else:
            sedfacdata = pd.concat(
                [
                    sedfacdata,
                    pd.DataFrame(tmpErof, columns=["start", "sUni", "sMap", "sKey"]),
                ],
                ignore_index=True,
            )

        return sedfacdata
    
    def _readErofactor(self):
        """
        Parse erodibility factor based on surface geology.
        """

        sedfacdata = None
        try:
            sedDict = self.input["sedfactor"]
            sedSort = sorted(sedDict, key=itemgetter("start"))
            for k in range(len(sedSort)):
                sStart = None
                sUniform = None
                sMap = None
                try:
                    sStart = sedSort[k]["start"]
                except Exception:
                    print(
                        "For each sediment factor a start time is required.", flush=True
                    )
                    raise ValueError(
                        "Sediment factor map {} has no parameter start".format(k)
                    )
                try:
                    sUniform = sedSort[k]["uniform"]
                except Exception:
                    pass
                try:
                    sMap = sedSort[k]["map"]
                except Exception:
                    pass

                if sMap is not None:
                    try:
                        with open(sMap[0] + ".npz") as sedfacfile:
                            sedfacfile.close()

                    except IOError:
                        print(
                            "Unable to open sediment factor file: {}.npz".format(sMap[0]),
                            flush=True,
                        )
                        raise IOError(
                            "The sediment factor file {} is not found for event {}.".format(
                                sMap[0] + ".npz", k
                            )
                        )
                    mdata = np.load(sMap[0] + ".npz")
                    sedfacSet = mdata.files

                    try:
                        sedKey = mdata[sMap[1]]
                        if sedKey is not None:
                            pass
                    except KeyError:
                        print(
                            "Field name {} is missing from sediment factor file {}.npz".format(
                                sMap[1], sMap[0]
                            ),
                            flush=True,
                        )
                        print(
                            "The following fields are available: {}".format(sedfacSet),
                            flush=True,
                        )
                        print("Check your sediment factor file fields definition...", flush=True)
                        raise KeyError(
                            "Field name for sediment factor is not defined correctly or does not exist!"
                        )

                sedfacdata = self._defineErofactor(k, sStart, sMap, sUniform, sedfacdata)

            if sedfacdata["start"][0] > self.tStart:
                tmpSedF = []
                tmpSedF.insert(
                    0, {"start": self.tStart, "sUni": 1.0, "sMap": None, "sKey": None}
                )
                sedfacdata = pd.concat(
                    [
                        pd.DataFrame(
                            tmpSedF, columns=["start", "sUni", "sMap", "sKey"]
                        ),
                        sedfacdata,
                    ],
                    ignore_index=True,
                )
            self.sedfacdata = sedfacdata.copy()  
            self.sedfacdata.reset_index(drop=True, inplace=True)
            self.sedfactNb = len(self.sedfacdata)

        except KeyError:
            self.sedfactNb = 0
            self.sedfacdata = None

        return
    
    def _storePlate(self, k, pStart, pMap, pTec, pEnd, platedata):
        """
        Record plate movement conditions.

        :arg k: plate event number
        :arg pStart: plate event start time
        :arg pMap: plate advection information file
        :arg pTec: plate uplift/subsidence file
        :arg pEnd: tectonic event end time
        :arg platedata: pandas dataframe storing each plate movement event

        :return: appended platedata
        """

        if pMap is not None:
            try:
                with open(pMap) as platefile:
                    platefile.close()

            except IOError:
                print("Unable to open plate file: {}".format(pMap), flush=True)
                raise IOError(
                    "The plate file {} is not found for event {}.".format(pMap, k)
                )
        else:
            pMap = "empty"

        if pTec is not None:
            try:
                with open(pTec) as platetec:
                    platetec.close()

            except IOError:
                print(
                    "Unable to open uplift/subsidence file in plates: {}".format(pTec),
                    flush=True,
                )
                raise IOError(
                    "The plate file {} is not found for event {}.".format(pTec, k)
                )
        else:
            pTec = "empty"

        tmpPlate = []
        tmpPlate.insert(0, {"start": pStart, "pMap": pMap, "pTec": pTec})

        if k == 0:
            platedata = pd.DataFrame(tmpPlate, columns=["start", "pMap", "pTec"])
        else:
            platedata = pd.concat(
                [platedata, pd.DataFrame(tmpPlate, columns=["start", "pMap", "pTec"])],
                ignore_index=True,
            )

        return platedata

    def _definePlate(self, k, plateSort, platedata):
        """
        Define plate rotation IDs.

        :arg k: plate movement event number
        :arg plateSort: sorted plate ids event through time
        :arg platedata: pandas dataframe storing each plate IDs
        :return: appended platedata
        """

        pStart = None
        pEnd = None
        pMap = None
        pTec = None

        try:
            pStart = plateSort[k]["start"]
        except Exception:
            print("For each tectonic event a start time is required.", flush=True)
            raise ValueError("Tectonic event {} has no parameter start".format(k))
        try:
            pMap = plateSort[k]["plate"] + ".npz"
        except Exception:
            pass
        try:
            pTec = plateSort[k]["upsub"] + ".npz"
        except Exception:
            pass
        try:
            pEnd = plateSort[k]["end"]
        except Exception:
            pass

        platedata = self._storePlate(k, pStart, pMap, pTec, pEnd, platedata)

        return platedata

    def _readPlate(self):
        """
        Parse plates movement forcing conditions.
        """

        platedata = None
        try:
            plateDict = self.input["plates"]

            plateSort = sorted(plateDict, key=itemgetter("start"))
            for k in range(len(plateSort)):
                platedata = self._definePlate(k, plateSort, platedata)

            if platedata["start"][0] > self.tStart:
                tmpPlate = []
                tmpPlate.insert(
                    0, {"start": self.tStart, "pMap": "empty", "pTec": "empty"}
                )
                platedata = pd.concat(
                    [
                        pd.DataFrame(tmpPlate, columns=["start", "pMap", "pTec"]),
                        platedata,
                    ],
                    ignore_index=True,
                )
            self.platedata = platedata[platedata["start"] >= self.tStart]
            if self.rStep > 0:
                self.platedata = platedata[
                    platedata["start"] >= self.tStart + self.rStep * self.tout
                ]
            self.platedata.reset_index(drop=True, inplace=True)

            if self.rStep > 0:
                rNow = self.tStart + self.rStep * self.tout
                for k in range(len(self.platedata)):
                    if (
                        self.platedata["start"][k] < rNow
                        and k < len(self.platedata) - 1
                    ):
                        self.platedata.loc[k, ["start"]] = rNow - self.tout
                    elif (
                        self.platedata["start"][k] < rNow
                        and k == len(self.platedata) - 1
                    ):
                        self.platedata.loc[k, ["start"]] = rNow
                self.platedata = self.platedata[self.platedata["start"] >= rNow]
                self.platedata.reset_index(drop=True, inplace=True)

        except KeyError:
            self.platedata = None

        return

    def _defineRain(self, k, rStart, rMap, rUniform, raindata):
        """
        Define precipitation conditions.

        :arg k: precipitation event number
        :arg rStart: precipitation event start time
        :arg rMap: precipitation map file event
        :arg rUniform: precipitation uniform value event
        :arg raindata: pandas dataframe storing each precipitation event
        :return: appended raindata
        """

        if rMap is None and rUniform is None:
            print(
                "For each climate event a rainfall value (uniform) or a rainfall \
                grid (map) is required.",
                flush=True,
            )
            raise ValueError(
                "Climate event {} has no rainfall value (uniform) or a rainfall \
                map (map).".format(
                    k
                )
            )

        tmpRain = []
        if rMap is None:
            tmpRain.insert(
                0,
                {"start": rStart, "rUni": rUniform, "rMap": None, "rKey": None},
            )
        else:
            tmpRain.insert(
                0,
                {
                    "start": rStart,
                    "rUni": None,
                    "rMap": rMap[0] + ".npz",
                    "rKey": rMap[1],
                },
            )

        if k == 0:
            raindata = pd.DataFrame(tmpRain, columns=["start", "rUni", "rMap", "rKey"])
        else:
            raindata = pd.concat(
                [
                    raindata,
                    pd.DataFrame(tmpRain, columns=["start", "rUni", "rMap", "rKey"]),
                ],
                ignore_index=True,
            )

        return raindata

    def _readRain(self):
        """
        Parse rain forcing conditions.
        """

        raindata = None
        try:
            rainDict = self.input["climate"]
            rainSort = sorted(rainDict, key=itemgetter("start"))
            for k in range(len(rainSort)):
                rStart = None
                rUniform = None
                rMap = None
                try:
                    rStart = rainSort[k]["start"]
                except Exception:
                    print(
                        "For each climate event a start time is required.", flush=True
                    )
                    raise ValueError(
                        "Climate event {} has no parameter start".format(k)
                    )
                try:
                    rUniform = rainSort[k]["uniform"]
                except Exception:
                    pass
                try:
                    rMap = rainSort[k]["map"]
                except Exception:
                    pass

                if rMap is not None:
                    if self.meshFile != rMap[0] + ".npz":
                        try:
                            with open(rMap[0] + ".npz") as rainfile:
                                rainfile.close()

                        except IOError:
                            print(
                                "Unable to open rain file: {}.npz".format(rMap[0]),
                                flush=True,
                            )
                            raise IOError(
                                "The rain file {} is not found for climatic event {}.".format(
                                    rMap[0] + ".npz", k
                                )
                            )
                        mdata = np.load(rMap[0] + ".npz")
                        rainSet = mdata.files
                    else:
                        mdata = np.load(self.meshFile)
                        rainSet = mdata.files
                    try:
                        rainKey = mdata[rMap[1]]
                        if rainKey is not None:
                            pass
                    except KeyError:
                        print(
                            "Field name {} is missing from rain file {}.npz".format(
                                rMap[1], rMap[0]
                            ),
                            flush=True,
                        )
                        print(
                            "The following fields are available: {}".format(rainSet),
                            flush=True,
                        )
                        print("Check your rain file fields definition...", flush=True)
                        raise KeyError(
                            "Field name for rainfall is not defined correctly or does not exist!"
                        )

                raindata = self._defineRain(k, rStart, rMap, rUniform, raindata)

            if raindata["start"][0] > self.tStart:
                tmpRain = []
                tmpRain.insert(
                    0, {"start": self.tStart, "rUni": 0.0, "rMap": None, "rKey": None}
                )
                raindata = pd.concat(
                    [
                        pd.DataFrame(
                            tmpRain, columns=["start", "rUni", "rMap", "rKey"]
                        ),
                        raindata,
                    ],
                    ignore_index=True,
                )
            self.raindata = raindata.copy()  # [raindata["start"] >= self.tStart]
            self.raindata.reset_index(drop=True, inplace=True)
            self.rainNb = len(self.raindata)

        except KeyError:
            self.raindata = None

        return

    def _readCompaction(self):
        """
        Read compaction parameters.
        """

        try:
            compDict = self.input["compaction"]
            try:
                self.phi0s = compDict["phis"]
            except KeyError:
                self.phi0s = 0.49
            try:
                self.z0s = compDict["z0s"]
            except KeyError:
                self.z0s = 3700.0

        except KeyError:
            self.phi0s = 0.49
            self.z0s = 3700.0

        return

    def _readFlex(self):
        """
        Parse flexural isostasy variables.
        """

        try:
            flexDict = self.input["flexure"]
            self.flexOn = True
            try:
                self.reg_dx = flexDict["regdx"]
            except KeyError:
                raise ValueError("Flexure definition: regular grid spacing is required.")
            try:
                self.flex_rhoa = flexDict["rhoa"]
            except KeyError:
                self.flex_rhoa = 3250.0
            try:
                self.flex_eet = flexDict["thick"]
            except KeyError:
                self.flex_eet = 10000.0
            try:
                self.flex_rhos = flexDict["rhoc"]
            except KeyError:
                self.flex_rhos = 2400.0
        except KeyError:
            self.flexOn = False

        return
    
    def _readGFlex(self):
        """
        Parse global flexural isostasy variables.
        """

        try:
            flexGDict = self.input["gflex"]
            self.gflexOn = True
            try:
                interpf = flexGDict["interS"]
            except KeyError:
                print(
                    "Key 'interS' is required and is missing in the 'gflex' declaration!",
                    flush=True,
                )
                raise KeyError("Compressed numpy dataset definition is not defined in gflex declaration!")
            self.Interp = interpf + ".npz"
            try:
                with open(self.Interp) as meshfile:
                    meshfile.close()
            except IOError:
                print("Unable to open numpy dataset: {}".format(self.Interp), flush=True)
                raise IOError("The numpy dataset is not found...")
            
            try:
                interpf2 = flexGDict["interR"]
            except KeyError:
                print(
                    "Key 'interR' is required and is missing in the 'gflex' declaration!",
                    flush=True,
                )
                raise KeyError("Compressed numpy dataset definition is not defined in gflex declaration!")
            self.rInterp = interpf2 + ".npz"
            try:
                with open(self.rInterp) as meshfile:
                    meshfile.close()
            except IOError:
                print("Unable to open numpy dataset: {}".format(self.interR), flush=True)
                raise IOError("The numpy dataset is not found...")
            
            try:
                self.gflexproc = flexGDict["procs"]
            except KeyError:
                self.gflexproc = 10

            try:
                self.gflexStep = flexGDict["step"]
            except KeyError:
                print(
                    "Key 'step' is required and is missing in the 'gflex' declaration!",
                    flush=True,
                )
                raise KeyError("Simulation global flexural timestep needs to be declared.")
        except KeyError:
            self.gflexStep = None
            self.gflexOn = False

        return
    
    def _readOrography(self):
        """
        Parse orographic precipitation variables.
        
        latitude (float) : Coriolis effect decreases as latitude decreases
        precip_base (float) : non-orographic, uniform precipitation rate [mm/h], usually [0, 10]
        precip_min (float) : minimum precipitation [mm/h] when precipitation rate <= 0
        wind_speed (float) : [m/s]
        wind_dir (float) : wind direction [0: north, 270: west]
        conv_time (float) : cloud water to hydrometeor conversion time [s]
        fall_time (float) : hydrometeor fallout time [s]
        nm (float) : moist stability frequency [1/s]
        hw (float) : water vapor scale height [m]
        cw (float) : uplift sensitivity [kg/m^3], product of saturation water vapor sensitivity ref_density [kg/m^3]
                    and environmental lapse rate (lapse_rate_m / lapse_rate)

        """

        try:
            oroDict = self.input["orography"]
            self.oroOn = True
            
            try:
                self.reg_dx = oroDict["regdx"]
            except KeyError:
                raise ValueError("Orographic definition: regular grid spacing is required.")
            try:
                self.wind_latitude = oroDict["latitude"]
                if self.wind_latitude > 90 or self.wind_latitude < -90:
                    print(
                        "Latitude for orographic rain needs to be between -90 and 90.",
                        flush=True,
                    )
                    raise ValueError("Latitude value not appropriately set.")
            except KeyError:
                self.wind_latitude = 0.0
            try:
                self.wind_speed = oroDict["wind_speed"]
            except KeyError:
                self.wind_speed = 10.
            try:
                self.wind_dir = oroDict["wind_dir"]
            except KeyError:
                self.wind_dir = 0.0
            try:
                self.oro_nm = oroDict["nm"]
            except KeyError:
                self.oro_nm = 0.01
            try:
                self.oro_hw = oroDict["hw"]
            except KeyError:
                self.oro_hw = 3400.0
            try:
                lapse_rate = oroDict["env_lapse_rate"]
            except KeyError:
                lapse_rate = -4.0
            try:
                lapse_rate_m = oroDict["moist_lapse_rate"]
            except KeyError:
                lapse_rate_m = -7.0
            try:
                ref_density = oroDict["ref_density"]
            except KeyError:
                ref_density = 7.4e-3
            self.oro_cw = ref_density * lapse_rate_m / lapse_rate
            try:
                self.oro_conv_time = oroDict["conv_time"]
            except KeyError:
                self.oro_conv_time = 1000.0
            try:
                self.oro_fall_time = oroDict["fall_time"]
            except KeyError:
                self.oro_fall_time = 1000.0
            try:
                self.oro_precip_base = oroDict["precip_base"]
            except KeyError:
                self.oro_precip_base = 7.0
            try:
                self.oro_precip_min = oroDict["precip_min"]
            except KeyError:
                self.oro_precip_min = 0.01
            try:
                self.rainfall_frequency = oroDict["rainfall_frequency"]
            except KeyError:
                self.rainfall_frequency = 1

        except KeyError:
            self.oroOn = False

        return

    def _readIce(self):
        """
        Parse ice flow variables.
        """

        try:
            iceDict = self.input["ice"]
            self.iceOn = True
            try:
                self.gaussIce = iceDict["gauss"]
            except KeyError:
                print("Check your ice fields definition...", flush=True)
                raise KeyError(
                    "Field name gauss is not defined correctly or does not exist!"
                )
            try:
                self.elaH = iceDict["hela"]
            except KeyError:
                self.elaH = 1800.0
            try:
                self.iceH = iceDict["hice"]
            except KeyError:
                self.iceH = 2100.0
            try:
                self.scaleIce = iceDict["fice"]
            except KeyError:
                self.scaleIce = 1.0
            try:
                self.Kice = iceDict["Ki"]
            except KeyError:
                self.Kice = 0.0
        except KeyError:
            self.iceOn = False

        return
    
    def _readOut(self):
        """
        Parse output directory.
        """

        try:
            outDict = self.input["output"]
            try:
                self.outputDir = outDict["dir"]
            except KeyError:
                self.outputDir = "output"
            try:
                self.makedir = outDict["makedir"]
            except KeyError:
                self.makedir = True
        except KeyError:
            self.outputDir = "output"
            self.makedir = True

        if self.rStep > 0:
            self.makedir = False

        return
