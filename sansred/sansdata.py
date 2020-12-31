"""
SANS data format
================

Internal data representation for SANS data.
"""

import sys
import datetime
from copy import copy, deepcopy
import json
from io import BytesIO
from collections import OrderedDict

import numpy as np

from dataflow.lib.uncertainty import Uncertainty
from dataflow.lib.exporters import exports_HDF5, exports_text
from vsansred.vsansdata import RawVSANSData, _toDictItem

IS_PY3 = sys.version_info[0] >= 3

IGNORE_CORNER_PIXELS = True

def _b(s):
    if IS_PY3:
        return s.encode('utf-8')
    else:
        return s

def _s(b):
    if IS_PY3:
        return b.decode('utf-8') if hasattr(b, 'decode') else b
    else:
        return b

class RawSANSData(RawVSANSData):
    suffix = ".sans"

class SansData(object):
    """SansData object used for storing values from a sample file (not div/mask).
       Stores the array of data as a Uncertainty object (detailed in uncertainty.py)
       Stores all metadata
       q, qx, qy, theta all get updated with values throughout the reduction process
       Tsam and Temp are just used for storage across modules (in wireit)
    """
    def __init__(self, data=None, metadata=None, q=None, qx=None, qy=None, aspect_ratio=1.0,
                 xlabel="X", ylabel="Y",
                 theta=None, Tsam=None, Temp=None, 
                 qx_min=None, qx_max=None, qy_min=None, qy_max=None,
                 qx_lo=None, qx_hi=None, qy_lo=None, qy_hi=None,
                 attenuation_corrected=False,
                 sensitivity_corrected=False, ):
        if isinstance(data, np.ndarray):
            # Data is counts, so variance is counts.  Set variance on zero counts to 1
            self.data = Uncertainty(data, data + (data==0))
        else:
            self.data = data
        self.metadata = metadata if metadata is not None else {}
        # There are many places where q was not set, i think i fixed most,
        # but there might be more; be wary
        self.q = q
        self.qx = qx
        self.qy = qy
        #bin edges
        self.qx_hi = qx_hi
        self.qy_hi = qy_hi
        self.qx_lo = qx_lo
        self.qy_lo = qy_lo
        self.qx_max = qx_max
        self.qy_max = qy_max
        self.qx_min = qx_min
        self.qy_min = qy_min
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.aspect_ratio = aspect_ratio
        self.theta = theta
        self.attenuation_corrected = attenuation_corrected
        self.sensitivity_corrected = sensitivity_corrected

        self.Tsam = None #Tsam and Temp are used to store the transmissions for later use
        self.Temp = None
    # Note that I have not defined an inplace subtraction
    def __sub__(self, other):
        result = self.copy()
        if isinstance(other, SansData):
            result.data = self.data - other.data
        else:
            result.data = self.data - other
        return result
    # Actual subtraction
    def __sub1__(self, other):
        result = self.copy()
        if isinstance(other, SansData):
            result.data = self.data - other.data
        else:
            result.data = self.data - other
        return result
    def __add__(self, other):
        result = self.copy()
        if isinstance(other, SansData):
            result.data = self.data + other.data
        else:
            result.data = self.data + other
        return result
    def __rsub__(self, other):
        result = self.copy()
        result.data = other - self.data
        return result
    def __truediv__(self, other):
        result = self.copy()
        if isinstance(other, SansData):
            result.data = self.data/other.data
        else:
            result.data = self.data/other
        return result
    def __mul__(self, other):
        result = self.copy()
        if isinstance(other, SansData):
            result.data = self.data * other.data
        else:
            result.data = self.data * other
        return result

    def copy(self):
        return SansData(copy(self.data), deepcopy(self.metadata),
                        q=copy(self.q), qx=copy(self.qx), qy=copy(self.qy),
                        theta=copy(self.theta), aspect_ratio=self.aspect_ratio,
                        qx_min=self.qx_min, qx_max=self.qx_max, qy_min=self.qy_min, qy_max=self.qy_max,
                        qx_lo=self.qx_lo, qx_hi=self.qx_hi, qy_lo=self.qy_lo,qy_hi=self.qy_hi,
                        xlabel=self.xlabel, ylabel=self.ylabel,
                        attenuation_corrected=self.attenuation_corrected,
                        sensitivity_corrected=self.sensitivity_corrected,
                       )

    def __copy__(self):
        return self.copy()

    #def __str__(self):
        #return self.data.x.__str__()
    #def __repr__(self):
        #return self.__str__()

    def get_plottable(self):
        data = self.data.x.astype("float")
        xdim = data.shape[0]
        ydim = data.shape[1]
        if not (np.abs(data) > 1e-10).any():
            zmin = 0.0
            zmax = 1.0
        else:
            zmin = data[data > 1e-10].min()
            if IGNORE_CORNER_PIXELS:
                mask = np.ones(data.shape, dtype='bool')
                mask[0, 0] = mask[-1, 0] = mask[-1, -1] = mask[0, -1] = 0.0
                zmax = data[mask].max()
            else:
                zmax = data.max()
        if self.qx is None or self.qy is None:
            xmin = 0.0
            xmax = 128
            ymin = 0.0
            ymax = 128
        else:
            xmin = self.qx_min if self.qx_min is not None else self.qx.min()
            xmax = self.qx_max if self.qx_max is not None else self.qx.max()
            ymin = self.qy_min if self.qy_min is not None else self.qy.min()
            ymax = self.qy_max if self.qy_max is not None else self.qy.max()
        plottable_data = {
            'entry': self.metadata['entry'],
            'type': '2d',
            'z':  [data.flatten().tolist()],
            'title': _s(self.metadata['run.filename'])+': ' + _s(self.metadata['sample.labl']),
            #'metadata': self.metadata,
            'options': {
                'fixedAspect': {
                    #'fixAspect': True,
                    #'aspectRatio': 1.0
                }
            },
            'dims': {
                'xmax': xmax,
                'xmin': xmin,
                'ymin': ymin,
                'ymax': ymax,
                'xdim': xdim,
                'ydim': ydim,
                'zmin': zmin,
                'zmax': zmax,
            },
            'xlabel': self.xlabel,
            'ylabel': self.ylabel,
            'zlabel': 'Intensity (I)',
        }
        if self.aspect_ratio is not None:
            plottable_data['options']['fixedAspect'] = {
                'fixAspect': True,
                'aspectRatio': self.aspect_ratio
            }
        return plottable_data

    def get_metadata(self):
        metadata = {}
        metadata.update(_toDictItem(self.metadata, convert_bytes=True))
        return metadata

class Sans1dData(object):
    properties = ['x', 'v', 'dx', 'dv', 'xlabel', 'vlabel', 'xunits', 'vunits', 'xscale', 'vscale', 'metadata', 'fit_function']

    def __init__(self, x, v, dx=0, dv=0, xlabel="", vlabel="", xunits="", vunits="", xscale="linear", vscale="linear", metadata=None, fit_function=None):
        self.x = x
        self.v = v
        self.dx = dx
        self.dv = dv
        self.xlabel = xlabel
        self.vlabel = vlabel
        self.xunits = xunits
        self.vunits = vunits
        self.xscale = xscale
        self.vscale = vscale
        self.metadata = metadata if metadata is not None else {}
        self.fit_function = fit_function


    def to_dict(self):
        props = dict([(p, getattr(self, p, None)) for p in self.properties])
        return _toDictItem(props, convert_bytes=True)

    def get_plottable(self):
        label = "%s: %s" % (self.metadata['run.experimentScanID'], self.metadata['sample.labl'])
        xdata = self.x.tolist()
        ydata = self.v.tolist()
        if not hasattr(self.dx, 'tolist'):
            xerr = np.zeros_like(xdata).tolist()
        else:
            xerr = np.sqrt(self.dx).tolist()
        yerr = np.sqrt(self.dv).tolist()
        data = [[x, y, {"yupper": y+dy, "ylower": y-dy, "xupper": x+dx, "xlower": x-dx}] for x,y,dx,dy in zip(xdata, ydata, xerr, yerr)]
        plottable = {
            "type": "1d",
            "options": {
                "axes": {
                    "xaxis": {"label": self.xlabel},
                    "yaxis": {"label": self.vlabel}
                },
                "series": [{"label": label}]
            },
            "data": [data]
        }
        return plottable

    def get_metadata(self):
        return self.to_dict()

    @exports_text(name="column")
    def to_column_text(self):
        with BytesIO() as fid:
            fid.write(_b("# %s\n" % json.dumps(_toDictItem(self.metadata, convert_bytes=True)).strip("{}")))
            columns = {"columns": [self.xlabel, self.vlabel, "uncertainty", "resolution"]}
            units = {"units": [self.xunits, self.vunits, self.vunits, self.xunits]}
            fid.write(_b("# %s\n" % json.dumps(columns).strip("{}")))
            fid.write(_b("# %s\n" % json.dumps(units).strip("{}")))
            np.savetxt(fid, np.vstack([self.x, self.v, self.dv, self.dx]).T, fmt="%.10e")
            fid.seek(0)
            value = fid.read()

        return {
            "name": getattr(self, "name", "default_name"),
            "entry": self.metadata.get("entry", "default_entry"),
            "file_suffix": ".sans1d.dat",
            "value": value.decode('utf-8'),
        }

class SansIQData(object):
    def __init__(self, I=None, dI=None, Q=None, dQ=None, meanQ=None, ShadowFactor=None, label='', metadata=None):
        self._I = I
        self._dI = dI
        self._Q = Q
        self._dQ = dQ
        self._meanQ = meanQ
        self._ShadowFactor = ShadowFactor
        self.label = label
        self.metadata = metadata if metadata is not None else {}
        self.mask_indices = []
        self.mask_reference = np.arange(len(I))

    def append(self,data,sort=True):
        self._I            = np.append(self.I,data.I)
        self._dI           = np.append(self.dI,data.dI)
        self._Q            = np.append(self.Q,data.Q)
        self._dQ           = np.append(self.dQ,data.dQ)
        self._meanQ        = np.append(self.meanQ,data.meanQ)
        self._ShadowFactor = np.append(self.ShadowFactor,data.ShadowFactor)
        self.mask_indices = []
        self.mask_reference = np.arange(len(self._I))

        if sort:
            sort_mask = np.argsort(self._Q)
            self._I = self._I[sort_mask]
            self._dI = self._dI[sort_mask]
            self._Q = self._Q[sort_mask]
            self._dQ = self._dQ[sort_mask]
            self._meanQ = self._meanQ[sort_mask]
            self._ShadowFactor = self._ShadowFactor[sort_mask]
    @property
    def mask(self):
        return ~np.in1d(self.mask_reference,self.mask_indices)
    @property
    def I(self):
        return self._I[self.mask]
    @property
    def dI(self):
        return self._dI[self.mask]
    @property
    def Q(self):
        return self._Q[self.mask]
    @property
    def dQ(self):
        return self._dQ[self.mask]
    @property
    def meanQ(self):
        return self._meanQ[self.mask]
    @property
    def ShadowFactor(self):
        return self._ShadowFactor[self.mask]

    
    def get_plottable(self):
        columns = OrderedDict([
            ('Q', {'label': "Q", 'units': "1/Ang", 'errorbars': 'dQ'}),
            ('I', {'label': "I(Q)", 'units': "1/cm", 'errorbars': 'dI'}),
            ('meanQ', {'label': 'Mean Q', 'units': "1/Ang"}),
            ('Q^4', {'label': 'Q^4', 'units': '1/Ang**4'}),
            ('I*Q^4', {'label': 'I * Q^4', 'units': '1/cm * 1/Ang**4'}),
            ('I*Q^2', {'label': 'I * Q^2', 'units': '1/cm * 1/Ang**2'}),
        ])
        datas = OrderedDict([
            ("Q", {"values": self.Q.tolist(), "errorbars": self.dQ.tolist()}),
            ("I", {"values": self.I.tolist(), "errorbars": self.dI.tolist()}),
            ("meanQ", {"values": self.meanQ.tolist(), "errorbars": self.dQ.tolist()}),
            ("Q^4", {"values": (self.meanQ**4).tolist()}),
            ("I*Q^4", {"values": (self.I * self.meanQ**4).tolist()}),
            ("I*Q^2", {"values": (self.I * self.meanQ**2).tolist()}),
        ])
        
        name = self.metadata.get("run.experimentScanID", "default_name")
        entry_name = self.metadata.get("sample.description", "default_entry")
        series = [{"label": "%s:%s" % (name, entry_name)}]
        xcol = "meanQ"
        ycol = "I"
        plottable = {
            "type": "nd",
            "title": "%s:%s" % (name, entry_name),
            "entry": entry_name,
            "columns": columns,
            "options": {
                "series": series,
                "xtransform": "log",
                "ytransform": "log",
                "axes": {
                    "xaxis": {"label": "%s(%s)" % (columns[xcol]["label"], columns[xcol]["units"])},
                    "yaxis": {"label": "%s(%s)" % (columns[ycol]["label"], columns[ycol]["units"])}
                },
                "xcol": xcol,
                "ycol": ycol,
                "errorbar_width": 0
            },
            "datas": datas
        }
        return plottable

    @exports_text(name="column")
    def to_column_text(self):
        # export to 6-column format compatible with SASVIEW
        # The 6 columns are | Q (1/A) | I(Q) (1/cm) | std. dev. I(Q) (1/cm) | sigmaQ | meanQ | ShadowFactor|
        column_names = ["Q", "I", "dI", "dQ", "meanQ", "ShadowFactor"]
        column_values = [getattr(self, cn) for cn in column_names]
        labels = ["Q (1/A)", "I(Q) (1/cm)", "std. dev. I(Q) (1/cm)", "sigmaQ", "meanQ", "ShadowFactor"]

        with BytesIO() as fid:
            fid.write(_b("# %s\n" % json.dumps(_toDictItem(self.metadata, convert_bytes=True)).strip("{}")))
            fid.write(_b("# %s\n" % json.dumps({"columns": labels}).strip("{}")))
            np.savetxt(fid, np.vstack(column_values).T, fmt="%.10e")
            fid.seek(0)
            value = fid.read()

        return {
            "name": _s(self.metadata.get("name", "default_name")),
            "entry": _s(self.metadata.get("entry", "default_entry")),
            "file_suffix": ".sansIQ.dat",
            "value": value.decode('utf-8'),
        }

    @exports_HDF5(name="NXcanSAS")
    def to_NXcanSAS(self):
        import h5py

        fid = BytesIO()
        h5_item = h5py.File(fid, 'w')

        entry_name = self.metadata.get("entry", "entry")
        nxentry = h5_item.create_group(entry_name)
        nxentry.attrs.update({
            "NX_class": "NXentry",
            "canSAS_class": "SASentry",
            "version": "1.0"
        })
        nxentry["definition"] = "NXcanSAS"
        nxentry["run"] = "<see the documentation>"
        nxentry["title"] = self.metadata["sample.description"]
        datagroup = nxentry.create_group("data")
        datagroup.attrs.update({
            "NX_class": "NXdata",
            "canSAS_class": "SASdata",
            "signal": "I",
            "I_axes": "<see the documentation>",
            "Q_indices": 1
        })
        datagroup["I"] = self.I
        datagroup["I"].attrs["units"] = "1/cm"
        datagroup["dI"] = self.dI
        datagroup["dI"].attrs["units"] = "1/cm"
        datagroup["Q"] = self.Q
        datagroup["Q"].attrs["units"] = "1/Ang"
        datagroup["dQ"] = self.dQ
        datagroup["dQ"].attrs["units"] = "1/Ang"
        datagroup["meanQ"] = self.meanQ
        datagroup["meanQ"].attrs["units"] = "1/Ang"
        datagroup["ShadowFactor"] = self.ShadowFactor
        datagroup["ShadowFactor"].attrs["units"] = "1/Ang"

        return {
            "name": _s(self.metadata.get("name", "default_name")),
            "entry": _s(self.metadata.get("entry", "default_entry")),
            "file_suffix": ".sansIQ.nx.h5",
            "value": h5_item,
        }

class Parameters(object):
    def __init__(self, params=None):
        self.params = params

    def get_metadata(self):
        return self.params

    def get_plottable(self):
        #return {"entry": "entry", "type": "params", "params": _toDictItem(self.metadata)}
        return {"entry": "entry", "type": "params", "params": _toDictItem(self.params)}
