from dataflow import core as df
from dataflow.automod import make_modules

from . import steps
from . import templates
from .sansdata import RawSANSData, SANSData2D, SANSData1D, SANSDataIQ, Parameters

INSTRUMENT = "ncnr.sans"

def define_instrument():
    # Define modules
    modules = make_modules(steps.ALL_ACTIONS, prefix=INSTRUMENT+'.')

    # Define data types
    sansraw = df.DataType(INSTRUMENT+".raw", RawSANSData)
    sans2d = df.DataType(INSTRUMENT+".sans2d", SANSData2D)
    sans1d = df.DataType(INSTRUMENT+".sans1d", SANSData1D)
    sansIQ = df.DataType(INSTRUMENT+".sansIQ", SANSDataIQ)
    params = df.DataType(INSTRUMENT+".params", Parameters)
    #offset_data = df.DataType(INSTRUMENT+".offset_data", dict)

    # Define instrument
    sans = df.Instrument(
        id=INSTRUMENT,
        name='NCNR SANS',
        menu=[('steps', modules)],
        datatypes=[sansraw, sans2d, sans1d, sansIQ, params],
        template_defs=df.load_templates(templates),
        )

    # Register instrument
    df.register_instrument(sans)
    return sans
