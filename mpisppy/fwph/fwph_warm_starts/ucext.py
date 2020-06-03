from mpisppy.fixer import Fixer
from mpisppy.utils.wxbarreader import WXBarReader
from mpisppy.utils.wxbarwriter import WXBarWriter
import pyomo.environ as pyo
import numpy as np
import hashlib # For cycle detection
import mpisppy.extension

from mpisppy.utils.sputils import create_EF

class UCExtension(mpisppy.extension.Extension):

    def __init__(self, ph, rank, n_proc):
        self.ph = ph
        self.rank = rank

        self.fixer_object  = None
        if ('fixeroptions' in ph.PHoptions):
            self.fixer_object = Fixer(ph, rank, n_proc)

        self.use_lagrangian   = False
        if ('use_lagrangian' in ph.PHoptions):
            self.use_lagrangian = ph.PHoptions['use_lagrangian']
        self.reader_object = WXBarReader(ph, rank, n_proc)
        self.writer_object = WXBarWriter(ph, rank, n_proc)

    def print_model(self, name='Scenario0', print_obj=False):
        ph = self.ph
        scenario = ph.local_scenarios[name]
        scenario.pprint()
        if (print_obj):
            obj = scenario.Total_Cost_Objective
            print('Objective value:', pyo.value(obj))
                    
    def pre_iter0(self):
        if (self.fixer_object):
            self.fixer_object.pre_iter0()
        self.reader_object.pre_iter0(self.reader_object)
        self.writer_object.pre_iter0(self.writer_object)
                                        
    def post_iter0(self):
        if (self.fixer_object):
            self.fixer_object.post_iter0()
        self.reader_object.post_iter0(self.reader_object)
        self.writer_object.post_iter0(self.writer_object)

        if (self.use_lagrangian):
            print('WARNING: No longer support serial Lagrangian computation')
            bound = np.inf
            if (self.rank == 0):
                print('  0 xxxxxxxxxx {bound:12.4f}'.format(bound=bound))

        # primal = _compute_farmer_conv(self.ph)
        # dual   = _compute_farmer_dist(self.ph)
        # print(f'{0:03d} {primal:12.4e} {dual:12.4e}')

    def miditer(self, PHIter, conv):
        if (self.fixer_object):
            self.fixer_object.miditer(PHIter, conv)
        self.reader_object.miditer(PHIter, conv)
        self.writer_object.miditer(PHIter, conv)
        if (self.rank==0 and PHIter == 1):
            print(f'Trival bound: {self.ph.trivial_bound:.4f} '
                   '(Only valid if no prox term is used in iter0)') 
        if (self.rank == 0):
            print('solve_loop() itr', PHIter-1, 'conv =', self.ph.conv)
        # primal = _compute_farmer_conv(self.ph)
        # dual   = _compute_farmer_dist(self.ph)
        # print(f'{PHIter:03d} {primal:12.4e} {dual:12.4e}')

    def _compute_farmer_conv(ph): 
        ''' Compute the relative L2 distance between the current xbar
            and the optimal xstar for the continuous (convex) farmer problem.
        '''
        true = {
            ('ROOT', 0):  80,
            ('ROOT', 1): 250,
            ('ROOT', 2): 170,
        }
        random_scenario_name = list(ph.local_scenarios.keys())[0]
        scenario = ph.local_scenarios[random_scenario_name]
        num = 0.
        den = 0.
        for (node_name, ix) in scenario._nonant_indexes:
            v  = scenario._xbars[node_name, ix].value
            vs = true[node_name, ix]
            num += (v - vs) * (v - vs)
            den += vs * vs
        return np.sqrt(num / den)

    def _compute_farmer_dist(ph):
        ''' Compute the expected relative L2 distance between the 
            current dual weights and the optimal dual weights for the
            continuous (convex) farmer problem.

            Notes:
                Not parallelized---just a temporary debugger
        '''
        true = { # Dual weights
            ('Scenario0', 'ROOT', 0):  14.0,
            ('Scenario0', 'ROOT', 1):  56.0,
            ('Scenario0', 'ROOT', 2): -70.0,
            ('Scenario1', 'ROOT', 0): -84.333333333333333,
            ('Scenario1', 'ROOT', 1): 134.666666666666666,
            ('Scenario1', 'ROOT', 2): -50.333333333333333,
            ('Scenario2', 'ROOT', 0):  70.333333333333333,
            ('Scenario2', 'ROOT', 1):-190.666666666666666,
            ('Scenario2', 'ROOT', 2): 120.333333333333333,
        }
        errs = {'Scenario' + str(i): 0. for i in range(3)}
        for (name, scenario) in ph.local_scenarios.items():
            num = 0.
            den = 0.
            for node in scenario._PySPnode_list:
                for (ix, var) in enumerate(node.nonant_vardata_list):
                    w  = scenario._Ws[node.name, ix].value
                    ws = true[name, node.name, ix]
                    num += (w - ws) * (w - ws)
                    den += ws * ws
            errs[name] += (1/3) * np.sqrt(num / den)
        err = sum(errs[name] for name in errs.keys())
        return err

    def enditer(self, PHIter):
        if (self.fixer_object):
            self.fixer_object.enditer(PHIter)
        self.reader_object.enditer(PHIter)
        self.writer_object.enditer(PHIter)
        print_soln_codes(self, PHIter)

    def post_everything(self, PHIter, conv):
        if (self.fixer_object):
            self.fixer_object.post_everything(PHIter, conv)
        self.reader_object.post_everything(PHIter, conv)
        self.writer_object.post_everything(PHIter, conv)

        # Custom fixer post-processing
        if (self.rank != 0):
            return
        if (not self.fixer_object):
            return
        print('Fixed', self.fixer_object.fixed_so_far, 'unique variables')
        # Solve the resulting EF with all variables fixed
        scenario_names = self.ph.all_scenario_names
        scenario_creator = self.ph.scenario_creator
        ef = create_EF(scenario_names, scenario_creator, creator_options=None)

        # Pick out any random scenario
        sname = list(ph.local_scenarios.keys())[0]
        scenario = ph.local_scenarios[sname]
        nonant_vars = {var.name: var
                        for node in scenario._PySPnode_list
                        for var  in node.nonant_vardata_list}

        # Fix all the nonant vars in the EF
        for (pair, var) in ef.ref_vars.items():
            vname = extract_varname(var.name)
            if nonant_vars[vname].is_fixed():
                var.fix(nonant_vars[vname].value) 

        solver = pyo.SolverFactory('gurobi')
        solver.solve(ef, tee=True)

    def extract_varname(ef_varname):
        # HARDCODE FOR UC (is it?)
        vname = ef_varname[19:].split('_')
        vname = '_'.join(vname[:-1])
        return vname

    def print_soln_codes(self, PHIter):
        codes = []
        for (name, scenario) in self.ph.local_scenarios.items():
            code = get_soln_code(scenario)
            codes.append(f'({name}) {code:8d}')
        print(f'{PHIter:3d}', ' '.join(codes))

    def get_soln_code(scenario):
        root = scenario._PySPnode_list[0] # Only one node b/c two-stage
        var  = np.array([pyo.value(v) for v in root.nonant_vardata_list])
        strr = ''.join(['1' if np.abs(1-val) < 1e-7 else '0' for val in var])
        # code = int(strr, 2)
        strr = strr.encode('utf-8')
        code = int(hashlib.sha256(strr).hexdigest(), 16) % 10**8
        return code

    if __name__=='__main__':
        print('No main for ucext.py')
