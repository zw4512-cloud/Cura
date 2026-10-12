# Converted from a Tool plugin to an Extension plugin (see
# TopologyOptimizeExtension.py for why): Cura's Extension entry point needs
# no "tool"/"extension" metadata of its own -- this mirrors Ultimaker/Cura's
# own plugins/PostProcessingPlugin/__init__.py exactly (getMetaData()
# returns {} and register() returns {"extension": <instance>}).

from . import TopologyOptimizeExtension


def getMetaData():
    return {}


def register(app):
    return {"extension": TopologyOptimizeExtension.TopologyOptimizeExtension()}
