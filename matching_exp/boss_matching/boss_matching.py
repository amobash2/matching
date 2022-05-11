# BOSS (balance optimization subset selection) matching script is used
# to generate two sub-groups with minimum imbalance between covariates of 
# cases from control and treatment groups:
# 1) control cases
# 2) treatment cases
#
# Inputs:
# 1) control cases & treatment cases: 
#       features: a list of features used for matching
#       control cases:
#       pd.DataFrame(columns = ["customer_id", *features])
#       treatment cases:
#       pd.DataFrame(columns = ["customer_id", *features])
#
#       
# Outputs:
# two sub-groups from control and treatment cases (ensuring every treatment case is assigned)

import os
import pandas as pd
import numpy as np
import argparse
import time
import math
from typing import List, Dict, Optional
from datetime import datetime
try:
    import ortools
except:
    raise Exception("Please install Google OR Tools first!")
from ortools.sat.python import cp_model
from matching_exp.utils import read_input_data

# We use CP_SAT solver from ortools.
# The requirement of CP_SAT solver to solve an integer program 
# is to have all integer coefficients. The proposed trick is to
# multiply coefficients with a large number, such that coefficients
# are integer. 
BIG_LINEAR = 100000

class BossData:
    """
    This class is used to:
    + Preprocess control and treatment data for the BOSS model. 
      - Control and treatment cases will be converted to indices
      - objective function coefficients will be calculated
         . absolute differences of feature values
         . first order raw sample moment
         . second order raw sample moment
    + Calculate upper bounds for integer variables
      - Estimate reasonable upper bounds for integer variables
        used for linearization of the model. A tighter upper bound
        can speed up convergance of ortools solver.
    """
    def __init__(self,
                 subset_control_customers: Dict,
                 subset_treatment_customers: Dict,
                 features: List[str]):
        self.control = subset_control_customers
        self.treatment = subset_treatment_customers
        self.features = features
        # Getting indices to set number of variables
        self.control_idx = list(self.control.keys())
        self.treatment_idx = list(self.treatment.keys())
        self.number_features = len(self.features)
        self.coefficients = self._preprocess_data()
        self.treatment_moments = self._cal_treatment_moments()
        self.control_moments_coef = self._cal_control_moments()
        self.ub_z, self.ub_y = self._find_ub_z_y()
        print("UB Z = ", self.ub_z)
        print("UB Y = ", self.ub_y)

    def _get_feature_name(self, f_idx: int):
        return self.features[f_idx]

    def _get_feature_val(self, data_idx: int, f_idx: int, control_treatment: str = "c"):
        feature_name = self._get_feature_name(f_idx)
        if control_treatment == "c":
            return self.control[data_idx][feature_name]
        else:
            return self.treatment[data_idx][feature_name]

    def _get_diff_feature(self, c_idx: int, t_idx: int, f_idx: int):
        control_val = self._get_feature_val(c_idx, f_idx, "c")
        treatment_val = self._get_feature_val(t_idx, f_idx, "t")
        return abs(control_val - treatment_val)

    def _get_treatment_feature_col(self, f_idx: int):
        feature_name = self._get_feature_name(f_idx)
        return [self.treatment[k][feature_name] for k in self.treatment.keys()]

    def _get_control_feature_col(self, f_idx: int):
        feature_name = self._get_feature_name(f_idx)
        return [self.control[k][feature_name] for k in self.control.keys()]

    def _preprocess_data(self):
        """
        There are three types of coefficients that the model uses:
        1) absolute differences of feature values between a control case and a treatment case
        2) first order raw sample moment of a feature for a control case : to approximate average difference
        3) second order raw sample moment of a feature for a treatment case : to approximate variance difference
        """
        coefficients = {"distance": {}, "first_moment": {}, "second_moment": {}}
        # absolute difference coefficients
        for i in self.control_idx:
            for k in self.treatment_idx:
                for f in range(self.number_features):
                    coefficients["distance"][(i, k , f)] = self._get_diff_feature(i, k, f)

        for i in self.control_idx:
            for f in range(self.number_features):
                # first order moment coefficients
                coefficients["first_moment"][(i, f)] = 1.0/float(len(self.treatment_idx)) * \
                                                       self._get_feature_val(i, f, "c")
                # second order moment coefficients
                coefficients["second_moment"][(i, f)] = 1.0/float(len(self.treatment_idx)) * \
                                                        pow(self._get_feature_val(i, f, "c"),2)
        return coefficients

    def _cal_treatment_moments(self):
        # calculating first order and second raw sample moments for all treatment cases for each feature
        treatment_moments = {"first_moment": {}, "second_moment": {}}
        for f in range(self.number_features):
            treatment_moments["first_moment"][f] = \
                math.ceil(BIG_LINEAR * 1.0/float(len(self.treatment_idx)) * \
                    np.sum(self._get_treatment_feature_col(f)))

            treatment_moments["second_moment"][f] = \
                math.ceil(BIG_LINEAR * 1.0/float(len(self.treatment_idx)) * \
                    np.sum(list(map(lambda x: pow(x, 2), self._get_treatment_feature_col(f)))))
        return treatment_moments

    def _cal_control_moments(self):
        # calculating first order and second raw sample moments for all control cases for each feature
        control_moments_coef = {"first_moment": {}, "second_moment": {}}
        for f in range(self.number_features):
            for i in self.control_idx:
                control_moments_coef["first_moment"][(i, f)] = \
                    math.ceil(BIG_LINEAR * 1.0/float(len(self.treatment_idx)) * \
                    self._get_feature_val(i, f, "c"))
                
                control_moments_coef["second_moment"][(i, f)] = \
                    math.ceil(BIG_LINEAR * 1.0/float(len(self.treatment_idx)) * \
                    pow(self._get_feature_val(i, f, "c"), 2))
        return control_moments_coef

    def _find_ub_z_y(self):
        # finding upper bounds for linearization integer decision variables
        # tighter realistic upper bounds will speed up convergance of the solver
        ub_z, ub_y = {}, {}
        for f in range(self.number_features):
            ub_z[f] = math.ceil(abs(BIG_LINEAR * max(self._get_control_feature_col(f)) - \
                             self.treatment_moments["first_moment"][f]))
            ub_y[f] = math.ceil(abs(max(BIG_LINEAR * list(map(lambda x: pow(x, 2), self._get_control_feature_col(f)))) - \
                             self.treatment_moments["second_moment"][f]))
        return ub_z, ub_y

class BossModel:
    """
    This class is used to:
    + Initiate ortools solver with provided parameters
    + Build the mathematical model
      - Defining decision variables
        . binary assignment variables
        . integer linearization variables
      - Objective contains three components: absolute difference of values, first order raw sample moment, second order raw sample moment
      - There are multiple constraints described in the documentation
        . ensuring each treatment case will be assigned to at least one case
        . limitation on a case assignment to multiple treatment cases
        . ensuring all treatment cases are assigned
        . linearization constraints (due to existance of non-smooth absolute values, we need to introduce auxiliary decision variables and constraints)
    + Run the model, post-process populated results
      - Please note that only if the model terminates with FEASIBLE or OPTIMAL status we can populate a matching solution
        . A model can result in UNSOLVED, UNKNOWN, FEASIBLE or OPTIMAL status. We only like to observe OPTIMAL or FEASIBLE status.
          * UNSOLVED or UNKNOWN status means either the model was too big and computational resources and provided solution time were not enough to allow
            the solver to find a solution. It can also mean that the input data is resulted in an infeasible model.
          * FEASIBLE status means that the model was able to find at least one solution that satisfies all model constraints
          * OPTIMAL status means that the model was able to find the best (or one of the best) solutions that minimized the objective and satisifes all constraints
    """
    def __init__(self,
                 data: BossData,
                 max_control_case_to_treatment: int = 5,
                 first_moment_weight: float = 1,
                 second_moment_weight: float = 1,
                 solution_time_limit_seconds: int = 600,
                 use_second_moment: bool = False):
        self.data = data
        self.max_control_case_to_treatment = max_control_case_to_treatment
        self.first_moment_weight = first_moment_weight
        self.second_moment_weight = second_moment_weight
        self.solution_time_limit_seconds = solution_time_limit_seconds
        self.use_second_moment = use_second_moment
        # Creating model and solver objects
        self.mdl = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        # Defining decision variables
        self.x_var, \
            self.z_var, \
            self.y_var = self._define_variables()
        # Adding constraints and objective to the model object
        # Run the solver to minimize model objective
        self.status = self._run_model()
        print(self._status_name(self.status))
        # Post-process output of the model only if the model status is FEASIBLE or OPTIMAL
        self.output, \
            self.z_variables, \
            self.y_variables, \
            self.rh_z, \
            self.rh_y = self._process_output(self.status)

    def _define_variables(self):
        x_var, z_var, y_var = {}, {}, {}
        for i in self.data.control_idx:
            for k in self.data.treatment_idx:
                x_var[(i, k)] = self.mdl.NewBoolVar(f"x_{i}_{k}")

        for f in range(self.data.number_features):
            z_var[f] = self.mdl.NewIntVar(0, self.data.ub_z[f], f"z_{f}")
            y_var[f] = self.mdl.NewIntVar(0, self.data.ub_y[f], f"y_{f}")
        return x_var, z_var, y_var

    def _run_model(self):
        self._total_assign_constraints()
        self._control_assign_constraints()
        self._treatment_assign_constraints()
        self._z_linearization_constraints()
        if self.use_second_moment:
            self._y_linearization_constraints()
        self._objective()
        self.solver.parameters.max_time_in_seconds = \
            self.solution_time_limit_seconds
        self.solver.parameters.num_search_workers = 8
        solution_printer = cp_model.ObjectiveSolutionPrinter()
        status = self.solver.SolveWithSolutionCallback(self.mdl, solution_printer)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print("Objective value =", self.solver.ObjectiveValue())
        return status

    def _objective(self):
        distance_component = sum(self.x_var[(i, k)] * math.ceil(BIG_LINEAR *self.data.coefficients["distance"][(i, k, f)]) for i in self.data.control_idx for k in self.data.treatment_idx for f in range(self.data.number_features))
        first_moment_component = sum(self.z_var[f] for f in range(self.data.number_features))
        second_moment_component = 0
        if self.use_second_moment:
            second_moment_component = sum(self.y_var[f] for f in range(self.data.number_features))
        # Defining objective
        self.mdl.Minimize(distance_component +
                          self.first_moment_weight * first_moment_component +
                          self.second_moment_weight * second_moment_component)

    def _total_assign_constraints(self):
        # Ensuring all treatment cases have an assigned control case
        self.mdl.Add(sum(self.x_var[(i, k)] for i in self.data.control_idx for k in self.data.treatment_idx) == len(self.data.treatment_idx))

    def _control_assign_constraints(self):
        # Ensuring a control case is not assigned to more than max_control_case_to_treatment number of treatment cases
        for i in self.data.control_idx:
            self.mdl.Add(sum(self.x_var[(i, k)] for k in self.data.treatment_idx) <= self.max_control_case_to_treatment)

    def _treatment_assign_constraints(self):
        # Ensuring each treatment case will have a case assigned
        for k in self.data.treatment_idx:
            self.mdl.Add(sum(self.x_var[(i, k)] for i in self.data.control_idx) == 1)

    def _z_linearization_constraints(self):
        # Linearization constraints for first order raw sample moment
        for f in range(self.data.number_features):
            rh = sum(self.x_var[(i, k)]* (self.data.control_moments_coef["first_moment"][(i, f)]) for i in self.data.control_idx for k in self.data.treatment_idx) - \
                self.data.treatment_moments["first_moment"][f]
            self.mdl.Add(self.z_var[f] >= rh)
            self.mdl.Add(self.z_var[f] <= rh)

    def _y_linearization_constraints(self):
        # Linearization constraints for second order raw sample moment
        for f in range(self.data.number_features):
            rh = sum(self.x_var[(i, k)]* (self.data.control_moments_coef["second_moment"][(i, f)]) for i in self.data.control_idx for k in self.data.treatment_idx) - \
                self.data.treatment_moments["second_moment"][f]
            self.mdl.Add(self.y_var[f] >= rh)
            self.mdl.Add(self.y_var[f] <= rh)

    def _status_name(self, status):
        if status == cp_model.OPTIMAL:
            return "OPTIMAL"
        elif status == cp_model.FEASIBLE:
            return "FEASIBLE"
        else:
            return "NOT SOLVED"

    def _process_output(self, status):
        # Extract values of decision variables from solver
        x_variables, \
            z_variables, \
            y_variables, \
            rh_z, \
            rh_y = {}, {}, {}, {}, {}
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for i in self.data.control_idx:
                for k in self.data.treatment_idx:
                    x_variables[(i, k)] = self.solver.Value(self.x_var[(i, k)])
            for f in range(self.data.number_features):
                z_variables[f] = self.solver.Value(self.z_var[f])
                y_variables[f] = self.solver.Value(self.y_var[f])

                rh_z[f] = sum(self.solver.Value(self.x_var[(i, k)])* (self.data.control_moments_coef["first_moment"][(i, f)]) for i in self.data.control_idx for k in self.data.treatment_idx) - \
                          self.data.treatment_moments["first_moment"][f]
                rh_y[f] = sum(self.solver.Value(self.x_var[(i, k)])* (self.data.control_moments_coef["second_moment"][(i, f)]) for i in self.data.control_idx for k in self.data.treatment_idx) - \
                          self.data.treatment_moments["second_moment"][f]
        return x_variables, z_variables, y_variables, rh_z, rh_y

class BossMatching:
    """
    The main class in boss matching which initializes BossData and BossModel objects
    boss_assignment function is the main public matching function.
    _sub_boss_assignment function is the main private matching function which will be called as needed.
    """
    def __init__(self, 
                 control_customers: pd.DataFrame(),
                 treatment_customers: pd.DataFrame(),
                 features: List[str],
                 max_control_case_to_treatment: int = 5):
        self.control_customers = control_customers
        self.treatment_customers = treatment_customers
        self.features = features
        self.features.extend(["sum"])
        self.normalized_features = [f"normalized_{f}" for f in self.features]
        self.max_control_case_to_treatment = max_control_case_to_treatment

    def _find_customer_id(self,
                          idx: int,
                          control_treatment: str = "c"):
        # Using this function to find customer IDs for populating output of matching
        if control_treatment == "c":
            return self.control_customers.iloc[idx]["customer_id"]
        else:
            return self.treatment_customers.iloc[idx]["customer_id"]

    def _sub_boss_assignment(self,
                             control_data: Dict,
                             treatment_data: Dict,
                             output_path: str,
                             save_on_storage: bool = False,
                             event_name: Optional[str] = None,
                             solution_time_limit_seconds: int = 600,
                             use_second_moment: bool = False):
        # Preparing Boss data
        boss_data = BossData(control_data, treatment_data, self.normalized_features)
        # Runing Boss model
        model_output = BossModel(boss_data,
                                 self.max_control_case_to_treatment,
                                 solution_time_limit_seconds = solution_time_limit_seconds,
                                 use_second_moment = use_second_moment).output
        assignments = {k:v for (k,v) in model_output.items() if v == 1} 

        # Populating output of matching
        output = pd.DataFrame(columns = ["customer_id", "neighbor_0"])
        
        for k, _ in assignments.items():
            output = output.append({"customer_id": self._find_customer_id(k[1], "t"),
                                    "neighbor_0": self._find_customer_id(k[0], "c")}, ignore_index=True)

        if save_on_storage:
            output.to_csv(f"{output_path}/cust_pairs.csv", index = False)
        return output

    def boss_assignment(self,
                        output_path: str,
                        save_on_storage: bool = False,
                        solution_time_limit_seconds: int = 600,
                        use_second_moment: bool = False):
        # The main public function for BossMatching class
        # calling private function _sub_boss_assignment as needed

        control_data = self.control_customers.to_dict(orient="index")
        treatment_data = self.treatment_customers.to_dict(orient="index")
        output = self._sub_boss_assignment(control_data,
                                           treatment_data,
                                           output_path,
                                           save_on_storage,
                                           solution_time_limit_seconds,
                                           use_second_moment)

        return output

def boss_main(args):
    # This is the entry function to boss matching
    features = args.features.split(",")
    control_customers, \
        treatment_customers = read_input_data(args.input_path, features = features)


    all_output = pd.DataFrame()
    # Run boss matching daily
    # The current example data is set to single cal_month date and hence this loop
    # will only be called once
    all_output = BossMatching(control_customers,
                                treatment_customers,
                                features,
                                args.max_control_case_to_treatment).boss_assignment(args.output_path,
                                                                                    args.save_on_storage, 
                                                                                    args.solution_time_limit_seconds,
                                                                                    args.use_second_moment)

    if args.save_on_storage:
        print(f"Recording file at : {args.output_path}")
        all_output.to_csv(f"{args.output_path}/cust_pairs.csv")
    return all_output


