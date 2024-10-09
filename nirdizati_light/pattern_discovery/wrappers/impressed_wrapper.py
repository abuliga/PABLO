import argparse
import pickle
from nirdizati_light.pattern_discovery.utils.Alignment_Check import Alignment_Checker
from joblib import Parallel, delayed
import random
import os
import networkx as nx
import numpy as np
import pandas as pd
import pm4py
from paretoset import paretoset
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
from nirdizati_light.pattern_discovery.utils.Auto_IMPID import AutoPatternDetection
from nirdizati_light.pattern_discovery.utils.IMIPD import VariantSelection, create_pattern_attributes, Trace_graph_generator, Pattern_extension,\
    plot_only_pattern, Single_Pattern_Extender
from sklearn.model_selection import train_test_split
import itertools


def impressed_wrapper(df,output_path,discovery_type,case_id,activity,timestamp,outcome,outcome_type,delta_time,
                      max_gap,max_extension_step,factual_outcome,likelihood,encoding,testing_percentage,extension_style,data_dependency,
    model,pattern_extension_strategy,aggregation_style,frequency_type, distance_style,trace_encoding, only_event_attributes):
    # Load the log
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    Log_graph_address = output_path + '/EventLogGraph.pickle'
    if os.path.exists(Log_graph_address):
        EventLog_graphs = pickle.load(open(Log_graph_address, "rb"))
        log_graph_exist = True
        print("Event log graph loaded successfully.")
    else:
        log_graph_exist = False
        EventLog_graphs = dict()
    pareto_features = ['Outcome_Interest', 'Frequency_Interest', 'likelihood', 'Case_Distance_Interest']
    pareto_sense = ['max', 'max', 'max', 'min']
    #pareto_features = ['Outcome_Interest', 'Frequency_Interest', 'likelihood', 'Case_Distance_Interest']
    #pareto_sense = ['max', 'max', 'max', 'min']
    df[activity] = df[activity].astype('string')
    df[activity] = df[activity].str.replace("_", "")
    df[activity] = df[activity].str.replace("-", "")
    df[activity] = df[activity].str.replace(".", "")
    # Remove all _ , - and space form the column names
    df.columns = df.columns.str.replace("_", "")
    df.columns = df.columns.str.replace("-", "")
    df.columns = df.columns.str.replace(" ", "")
    df.columns = df.columns.str.replace(".", "")
    timestamp = timestamp.replace("_", "")
    timestamp = timestamp.replace("-", "")
    timestamp = timestamp.replace(" ", "")
    case_id = case_id.replace("_", "")
    case_id = case_id.replace("-", "")
    case_id = case_id.replace(" ", "")
    activity = activity.replace("_", "")
    activity = activity.replace("-", "")
    activity = activity.replace(" ", "")
    outcome = outcome.replace("_", "")
    outcome = outcome.replace("-", "")
    outcome = outcome.replace(" ", "")
    try:
        df[timestamp] = pd.to_datetime(df[timestamp])
    except:
        print('The timestamp column is not in the correct format. Please convert it to datetime format.')
    df[case_id] = df[case_id].astype('string')
    outcomes = df[outcome].unique()
    if outcome_type == 'binary':
        for i, out in enumerate(outcomes):
            df.loc[df[outcome] == str(out), outcome] = i
        df[outcome] = df[outcome].astype('uint8')
    elif outcome_type == 'numerical':
        df[outcome] = df[outcome].astype('float32')

    color_dict_address = output_path + '/color_dict.pickle'
    if os.path.exists(color_dict_address):
        color_act_dict = pickle.load(open(color_dict_address, "rb"))
    else:
        color_codes = ["#" + ''.join([random.choice('000123456789ABCDEF') for i in range(6)])
                       for j in range(len(df[activity].unique()))]

        color_act_dict = dict()
        counter = 0
        for act in df[activity].unique():
            color_act_dict[act] = color_codes[counter]
            counter += 1
        color_act_dict['start'] = 'k'
        color_act_dict['end'] = 'k'
        pickle.dump(color_act_dict, open(color_dict_address, "wb"))

    patient_data = df[[case_id, likelihood, outcome]]
    patient_data.drop_duplicates(subset=[case_id], inplace=True)
    patient_data = patient_data.reset_index(drop=True)
    patient_data.loc[:, list(df[activity].unique())] = 0
    selected_variants = VariantSelection(df, case_id, activity, timestamp)
    for case in selected_variants["case:concept:name"].unique():
        if not log_graph_exist:
            EventLog_graphs[case] = Trace_graph_generator(df, delta_time, case, color_act_dict, case_id,
                                                          activity, timestamp)
        Other_cases = \
            selected_variants.loc[selected_variants["case:concept:name"] == case, 'case:CaseIDs'].tolist()[0]
        trace = df.loc[df[case_id] == case, activity].tolist()
        for act in np.unique(trace):
            Number_of_act = trace.count(act)
            for Ocase in Other_cases:
                patient_data.loc[patient_data[case_id] == Ocase, act] = Number_of_act
                if not log_graph_exist:
                    EventLog_graphs[Ocase] = EventLog_graphs[case].copy()

    # save the event log graph
    if not log_graph_exist:
        pickle.dump(EventLog_graphs, open(Log_graph_address, "wb"))
        print("Event log graph created successfully.")

    if discovery_type == 'auto':
        patient_data[case_id] = patient_data[case_id].astype('string')
        df[case_id] = df[case_id].astype('string')
        AutoDetection = AutoPatternDetection(EventLog_graphs, selected_variants,
                                             max_extension_step, max_gap,
                                             testing_percentage, df, patient_data, case_id,
                                             activity, outcome, outcome_type, timestamp,
                                             pareto_features, pareto_sense, delta_time,
                                             color_act_dict, output_path,
                                             factual_outcome, extension_style, data_dependency, aggregation_style,
                                             pattern_extension_strategy, model, frequency_type, distance_style, only_event_attributes)

        train_X, test_X = AutoDetection.AutoStepWise_PPD()

    train_X.to_csv(output_path + "/training_encoded_log.csv", index=False)
    test_X.to_csv(output_path + "/testing_encoded_log.csv", index=False)
    #TODO: Add decision tree training here with rule extraction tomorrow
    #TODO: Plot the frequency curves for all generated patterns
    return train_X,test_X