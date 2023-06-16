# Copyright 2023 by U. Naepels and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

import sys
import os
import inspect
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolutionStatus, TerminationCondition
import logging
import numpy as np
import math
import importlib
import csv
import inspect
import typing
import copy
import time

import mpisppy.log
from mpisppy import global_toc
from mpisppy import MPI
import mpisppy.utils.sputils as sputils
import mpisppy.spopt
from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.utils.wxbarwriter import WXBarWriter
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.confidence_intervals.ciutils as ciutils
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import mpisppy.utils.wxbarutils as wxbarutils
import mpisppy.utils.rho_utils as rho_utils
import mpisppy.utils.find_rho as find_rho
import mpisppy.phbase as phbase


############################################################################
def W_Convergence(mname, cfg):#, ph_object):
    """Interface to compute and write gradient cost
    
    Args:
       ph_object (PHBase): ph object
       cfg (Config): config object

    Attributes:
       c (dict): gradient cost

    """
    new_ph_object = _create_ph_object(mname, cfg)#, ph_object)
    new_ph_object.disable_W_and_prox()
    new_ph_object.solve_loop()
    #self.ph_object.reenable_W_and_prox()
    wxbarutils.write_W_to_file(new_ph_object, "wconv_w.csv", sep_files=False)
    write_xs_to_file(new_ph_object, "wconv_xs.csv")
    phbase._Compute_Xbar(new_ph_object, verbose=False)
    wxbarutils.write_xbar_to_file(new_ph_object, "wconv_xbar.csv")

def write_xs_to_file(PHB, fname, sep_files=False):
    if (sep_files):
        for (sname, scenario) in PHB.local_scenarios.items():
            (v._value for v in node.nonant_vardata_list)
            scenario_xs = {v.name: v._value 
                           for node in scenario._mpisppy_node_list
                           for v in node.nonant_vardata_list}
            scenario_fname = os.path.join(fname, sname + '_weights.csv')
            with open(scenario_fname, 'w') as f:
                for (vname, val) in scenario_xs.items():
                    row = ','.join([vname, str(val)]) + '\n'
                    f.write(row)
    else:
        local_xs = {(sname, v.name): v._value
                    for (sname, scenario) in PHB.local_scenarios.items()
                    for node in scenario._mpisppy_node_list
                    for v in node.nonant_vardata_list}
        comm = PHB.comms['ROOT']
        xs = comm.gather(local_xs, root=0)
        if (PHB.cylinder_rank == 0):
            with open(fname, 'a') as f:
                for x in xs:
                    for (key, val) in x.items():
                        sname, vname = key[0], key[1]
                        row = ','.join([sname, vname, str(val)]) + '\n'
                        f.write(row)


def _create_ph_object(mname, original_cfg):#, ph_object):
    #write W
    w_fname="../../examples/farmer/temp_w.csv"
    #wxbarutils.write_W_to_file(ph_object, w_fname, sep_files=False)

    #create new PH object
    try:
        model_module = importlib.import_module(mname)
    except:
        raise RuntimeError(f"Could not import module: {mname}")
    cfg = copy.deepcopy(original_cfg)
    cfg.max_iterations = 0 #we only need x0 here       

    scenario_creator = model_module.scenario_creator
    scenario_denouement = model_module.scenario_denouement
    scen_names_creator_args = inspect.getfullargspec(model_module.scenario_names_creator).args #partition requires to do that                    
    if scen_names_creator_args[0] == 'cfg':
        all_scenario_names = model_module.scenario_names_creator(cfg)
    else :
        all_scenario_names = model_module.scenario_names_creator(cfg.num_scens)
    scenario_creator_kwargs = model_module.kw_creator(cfg)
    variable_probability = None
    if hasattr(model_module, '_variable_probability'):
        variable_probability = model_module._variable_probability
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=None,
                              variable_probability=variable_probability)
    list_of_spoke_dict = list()
    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()
    if wheel.strata_rank == 0:  # don't do this for bound ranks 
        new_ph_object = wheel.spcomm.opt

    # set old Ws in new ph object
    wxbarutils.set_W_from_file(w_fname, new_ph_object, rank=0, sep_files=False, disable_check=True)
    #os.remove(w_fname)

    return new_ph_object
