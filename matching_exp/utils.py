import os
import pandas as pd
import argparse
from typing import List
from sklearn.preprocessing import StandardScaler

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def process_dataframe(data: pd.DataFrame(), features: List[str]):
    data["sum"] = data[features].sum(axis=1)
    cur_features = features.copy()
    cur_features.extend(["sum"])
    normalized_features = [f"normalized_{f}" for f in cur_features]
    data[normalized_features] = pd.DataFrame(StandardScaler().fit_transform(data[cur_features]))
    return data

def read_input_data(input_path:str, features: List[str], is_test: bool = False, test_on_main_data: bool = False):
    control_name, treatment_name = "control.csv", "treatment.csv"
    if is_test and not test_on_main_data:
        control_name, treatment_name = "test_control.csv", "test_treatment.csv"

    cur_control_customers = pd.read_csv(os.path.join(input_path, control_name)).drop_duplicates().reset_index()
    cur_control_customers = process_dataframe(cur_control_customers, features)

    cur_treatment_customers = pd.read_csv(os.path.join(input_path, treatment_name)).drop_duplicates().reset_index()
    cur_treatment_customers = process_dataframe(cur_treatment_customers, features)

    return cur_control_customers, cur_treatment_customers

def features_names(features, is_control: bool = False):
    new_features = {}
    feature_str = "_treatment"
    if is_control:
        feature_str = "_control"
    for f in features:
        new_features[f] = f+feature_str
    return new_features

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')