import argparse
from ctypes import util

from matching_exp import utils
from matching_exp.utils import str2bool
from matching_exp.boss_matching import boss_main
import time

def main(args):
    #Processing input data
    control_customers, \
        treatment_customers = utils.read_input_data(input_path=args.input_path, features=args.features.split(","))

    start_time = time.time()
    output = boss_main(args)
    print(output.head())

    print(f"Evaluation finished at {round((time.time() - start_time)/60.0, 5)} minutes!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running matching")
    parser.add_argument("--input_path", required=True, help="Path to control.csv and treatment.csv files")
    parser.add_argument("--features", default="feature_1,feature_2,feature_3", help="Comma separated list of features")
    parser.add_argument("--output_path", required=True, help="Path to record output data")
    # If storing data in the output folder
    parser.add_argument("--save_on_storage", default=False, type=str2bool)
    # Maximum number of treatment cases that can have same control case assigned to
    parser.add_argument("--max_control_case_to_treatment", default=5, type=int)
    # Maximum solution time allowed for the CP_SAT solver
    parser.add_argument("--solution_time_limit_seconds", default=600, type=int)
    # Minimize for second moment in the objective function
    parser.add_argument("--use_second_moment", default=False, type=str2bool)
    args = parser.parse_args()
    main(args)


    