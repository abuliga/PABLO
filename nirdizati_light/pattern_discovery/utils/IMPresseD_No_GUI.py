import argparse
import pickle
from Alignment_Check import Alignment_Checker
from joblib import Parallel, delayed
import random
import os
import networkx as nx
import numpy as np
import pandas as pd
from paretoset import paretoset
from Auto_IMPID import AutoPatternDetection
from IMIPD import VariantSelection, create_pattern_attributes, Trace_graph_generator, Pattern_extension, \
    plot_only_pattern, Single_Pattern_Extender

parser = argparse.ArgumentParser(description="IMPresseD")
parser.add_argument('--log_path', default='../datasets/sepsis_2_trunc30.csv', type=str)
parser.add_argument('--output_path', default='../output/sepsis_2_test', type=str)
parser.add_argument('--discovery_type', default='auto', type=str, choices=['interactive', 'auto'])
parser.add_argument('--encoding', default=True, type=bool, help='whether to encode the pattern in interactive mode')
parser.add_argument('--case_id', default='case:concept:name', type=str)
parser.add_argument('--activity', default='concept:name', type=str)
parser.add_argument('--timestamp', default='time:timestamp', type=str)
parser.add_argument('--outcome', default='case:label', type=str)
parser.add_argument('--outcome_type', default='binary', type=str, choices=['binary', 'numerical'])
parser.add_argument('--delta_time', default=-1, type=float, help='delta time in seconds')
parser.add_argument('--max_gap', default=10, type=float, help='maximum gap between events for eventual relations')
parser.add_argument('--max_extension_step', default=2, type=int,
                    help='maximum number of steps for extension in auto mode')
parser.add_argument('--testing_percentage', default=0.2, type=float, help='percentage of the testing data in auto mode')
parser.add_argument('--factual_outcome', default=0, type=float, help='the outcome of the inquiry case')
parser.add_argument('--likelihood', default='likelihood', type=str)
parser.add_argument('--extension_style', default='Pareto', type=str, help='Pareto or All')
parser.add_argument('--data_dependency', default="dependent", type=str, help='[dependent or independent], '
                                                                             '"dependent": data_aware pattern discovery'
                                                                             '"independent": data-agnostic pattern'
                                                                             ' discovery')
parser.add_argument('--aggregation_style', default="pareto", type=str,
                    help='whether to aggregate "all", "none", non-dominated from "pareto" front or "mix" both'
                         ' aggregated and non-aggregated features [all, none, pareto, mix]')

parser.add_argument('--pattern_extension_strategy', default="activities", type=str, choices=['activities', 'attributes'])

parser.add_argument('--frequency_type', default="absolute", type=str, choices=['absolute', 'relative'])


#log_path = 'datasets/Production_variants/Production_eventually_following_same_resource_fixed.csv'
log_path = 'datasets/Production_variants/eventually_Production_activity_durations_means.csv'
discover_type = 'auto'
Encoding = True
case_id = 'case:concept:name'
activity = 'concept:name'
timestamp = 'time:timestamp'
outcome = 'case:label'
outcome_type = 'binary'
delta_time = 1
max_gap = 10
max_extension_step = 2
testing_percentage = 0.2
factual_outcome = 0
likelihood = 'likelihood'
extension_style = 'Pareto'
data_dependency = 'dependent'
# Parameters for the pareto optimization
pareto_features = ['Outcome_Interest', 'Frequency_Interest', 'Case_Distance_Interest']
pareto_sense = ['max', 'max', 'min']
model = 'DT'
pattern_extension_strategy = 'activities'
aggregation_style = 'mix'
frequency_type = 'absolute'
distance_style = 'all'  # 'case' or 'all'
#output_path = 'output/Production_variants/Production_eventually_following_same_resource_fixed/%sAgg_%sActivites_%spool_revisedsimilarity_%s_freq%s' % (aggregation_style, pattern_extension_strategy,
#                                                                                                                                                  distance_style,str(len(pareto_features)),frequency_type)
output_path = 'output/Production_variants/eventually_Production_activity_durations_means/%sAgg_%sActivites_%spool_revisedsimilarity_%s_freq%s' % (aggregation_style, pattern_extension_strategy,
                                                                                                                                                 distance_style,str(len(pareto_features)),frequency_type)

'''
args = parser.parse_args()
case_id = args.case_id
activity = args.activity
timestamp = args.timestamp
outcome = args.outcome
outcome_type = args.outcome_type
delta_time = args.delta_time
max_gap = args.max_gap
factual_outcome = args.factual_outcome
likelihood = args.likelihood
output_path = args.output_path
discover_type = args.discovery_type
max_extension_step = args.max_extension_step
testing_percentage = args.testing_percentage
Encoding = args.encoding
extension_style = args.extension_style
data_dependency = args.data_dependency
aggregation_style = args.aggregation_style
aggregation_style = aggregation_style.lower()
frequency_type = args.frequency_type
'''


if not os.path.exists(output_path):
    os.makedirs(output_path)

Log_graph_address = "/".join(output_path.split("/")[:-1]) + '/EventLogGraph.pickle'
if os.path.exists(Log_graph_address):
    EventLog_graphs = pickle.load(open(Log_graph_address, "rb"))
    log_graph_exist = True
    print("Event log graph loaded successfully.")
else:
    log_graph_exist = False
    EventLog_graphs = dict()


# Load the log
df = pd.read_csv(log_path, sep=",")
# df.drop(['timesincelastevent'], axis=1, inplace=True)

df[activity] = df[activity].astype('string')
df[activity] = df[activity].str.replace("_", "")
df[activity] = df[activity].str.replace("-", "")
# Remove all _ , - and space form the column names
df.columns = df.columns.str.replace("_", "")
df.columns = df.columns.str.replace("-", "")
df.columns = df.columns.str.replace(" ", "")
df[timestamp] = pd.to_datetime(df[timestamp])
df[case_id] = df[case_id].astype('string')

color_dict_address = "/".join(output_path.split("/")[:-1]) + '/color_dict.pickle'
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

patient_data_address = "/".join(output_path.split("/")[:-1]) + '/patient_data.csv'
if os.path.exists(patient_data_address):
    patient_data = pd.read_csv(patient_data_address)
    patient_data_exist = True
else:
    patient_data_exist = False

if not patient_data_exist:
    patient_data = df[[case_id, outcome]]
    patient_data.drop_duplicates(subset=[case_id], inplace=True)
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

if not patient_data_exist:
    patient_data.to_csv(patient_data_address, index=False)
    print("Patient data created successfully.")


if discover_type == 'interactive':
    activity_attributes = create_pattern_attributes(patient_data, outcome, factual_outcome,
                                                    list(df[activity].unique()), outcome_type)

    Objectives_attributes = activity_attributes[pareto_features]
    mask = paretoset(Objectives_attributes, sense=pareto_sense)
    paretoset_activities = activity_attributes[mask]
    paretoset_activities.to_csv(output_path + '/paretoset_1.csv', index=False)
    All_pareto_patterns = paretoset_activities['patterns'].tolist()

    for pattern in list(paretoset_activities['patterns']):
        G = nx.DiGraph()
        G.add_node(1, value=pattern, parallel=False, color=color_act_dict[pattern])
        pickle.dump(G, open(output_path + '/%s_interactive.pickle' % pattern, "wb"))

    # ask the user to select the pattern of interest
    print("Please select the pattern of interest from the following list:")
    print(paretoset_activities['patterns'].tolist())
    Core_activity = input("Enter the name of the pattern of interest: ")

    all_pattern_dictionary = dict()
    all_extended_patterns = dict()
    EventLog_graphs = dict()
    Patterns_Dictionary = dict()
    filtered_cases = df.loc[df[activity] == Core_activity, case_id]
    filtered_main_data = df[df[case_id].isin(filtered_cases)]
    for case in filtered_main_data[case_id].unique():
        case_data = filtered_main_data[filtered_main_data[case_id] == case]
        if case not in EventLog_graphs.keys():
            Trace_graph = Trace_graph_generator(filtered_main_data, delta_time,
                                                case, color_act_dict,
                                                case_id, activity, timestamp)

            EventLog_graphs[case] = Trace_graph.copy()
        else:
            Trace_graph = EventLog_graphs[case].copy()

        Patterns_Dictionary = Pattern_extension(case_data, Trace_graph, Core_activity,
                                                case_id, Patterns_Dictionary, max_gap)

    # patient_data.loc[:, list(Patterns_Dictionary.keys())] = 0
    #
    # for PID in Patterns_Dictionary:
    #     for CaseID in np.unique(Patterns_Dictionary[PID]['Instances']['case']):
    #         variant_frequency_case = Patterns_Dictionary[PID]['Instances']['case'].count(CaseID)
    #         patient_data.loc[patient_data[case_id] == CaseID, PID] = variant_frequency_case

    Alignment_Check = Alignment_Checker(case_id, outcome)
    for pattern_name in list(Patterns_Dictionary.keys()):
        Pattern = Patterns_Dictionary[pattern_name]['pattern']
        patient_data[pattern_name] = 0
        patient_data = Alignment_Check.check_pattern_alignment(EventLog_graphs, patient_data, Pattern, pattern_name)

    pattern_attributes = create_pattern_attributes(patient_data, outcome,
                                                   factual_outcome, list(Patterns_Dictionary.keys()), outcome_type)

    Objectives_attributes = pattern_attributes[pareto_features]
    mask = paretoset(Objectives_attributes, sense=pareto_sense)
    paretoset_patterns = pattern_attributes[mask]
    all_pattern_dictionary.update(Patterns_Dictionary)
    All_pareto_patterns.extend(paretoset_patterns['patterns'].tolist())

    paretoset_patterns.to_csv(output_path + '/paretoset_2.csv', index=False)

    # save all patterns in paretofront in json format
    for pattern in paretoset_patterns['patterns']:
        P_graph = Patterns_Dictionary[pattern]['pattern']
        pickle.dump(P_graph, open(output_path + '/%s_interactive.pickle' % pattern, "wb"))

    # parallelize the plotting of the patterns
    Parallel(n_jobs=6)(delayed(plot_only_pattern)(Patterns_Dictionary, row['patterns'], color_act_dict, output_path)
                       for ticker, row in paretoset_patterns.iterrows())

    # extend the patterns
    continue_extending = input("Enter 1 if you want to continue extending patterns or 0 to stop: ")
    continue_extending = int(continue_extending)
    counter = 3
    while continue_extending == 1:
        # ask the user to select the pattern of interest
        print("Please select the pattern of interest from the following list:")
        print(paretoset_patterns['patterns'].tolist())
        Core_pattern = input("Enter to the name of the pattern of interest: ")
        while any(nx.get_edge_attributes(Patterns_Dictionary[Core_pattern]['pattern'], 'eventually').values()):
            print("Patterns including eventually relations are not supported yet for extension")
            Core_pattern = input("Enter to the name of the pattern of interest or -1 to stop extension: ")
            if Core_pattern == '-1':
                break

        all_extended_patterns.update(Patterns_Dictionary)
        all_extended_patterns, Patterns_Dictionary = Single_Pattern_Extender(
            all_extended_patterns,
            Core_pattern,
            EventLog_graphs,
            df, max_gap, activity, case_id)

        for pattern_name in Patterns_Dictionary.keys():
            Pattern = Patterns_Dictionary[pattern_name]['pattern']
            patient_data[pattern_name] = 0
            patient_data = Alignment_Check.check_pattern_alignment(EventLog_graphs, patient_data, Pattern, pattern_name)

        pattern_attributes = create_pattern_attributes(patient_data, outcome,
                                                       factual_outcome, list(Patterns_Dictionary.keys()),
                                                       outcome_type)
        Objectives_attributes = pattern_attributes[pareto_features]
        mask = paretoset(Objectives_attributes, sense=pareto_sense)
        paretoset_patterns = pattern_attributes[mask]
        paretoset_patterns.to_csv(output_path + '/paretoset_%s.csv' % counter, index=False)

        for pattern in paretoset_patterns['patterns']:
            P_graph = Patterns_Dictionary[pattern]['pattern']
            pickle.dump(P_graph, open(output_path + '/%s_interactive.pickle' % pattern, "wb"))

        All_pareto_patterns.extend(paretoset_patterns['patterns'].tolist())
        counter += 1
        # parallelize the plotting of the patterns
        Parallel(n_jobs=6)(delayed(plot_only_pattern)(Patterns_Dictionary, row['patterns'], color_act_dict, output_path)
                           for ticker, row in paretoset_patterns.iterrows())

        continue_extending = input("Enter 1 if you want to continue extending patterns or 0 to stop: ")
        continue_extending = int(continue_extending)

    if Encoding:
        Encoded_patterns = patient_data[All_pareto_patterns]
        Encoded_patterns.loc[:, case_id] = patient_data[case_id]
        Encoded_patterns.loc[:, outcome] = patient_data[outcome]
        Encoded_patterns.to_csv(output_path + '/EncodedPatterns_InteractiveMode.csv', index=False)

if discover_type == 'auto':
    patient_data[case_id] = patient_data[case_id].astype('string')
    df[case_id] = df[case_id].astype('string')
    AutoDetection = AutoPatternDetection(EventLog_graphs, max_extension_step, max_gap,
                                         testing_percentage, df, patient_data, case_id,
                                         activity, outcome, outcome_type, timestamp,
                                         pareto_features, pareto_sense, delta_time,
                                         color_act_dict, output_path,
                                         factual_outcome, extension_style, data_dependency, aggregation_style,
                                         pattern_extension_strategy, model, frequency_type, distance_style)

    train_X, test_X = AutoDetection.AutoStepWise_PPD()

    train_X.to_csv(output_path + "/training_encoded_log.csv", index=False)
    test_X.to_csv(output_path + "/testing_encoded_log.csv", index=False)
