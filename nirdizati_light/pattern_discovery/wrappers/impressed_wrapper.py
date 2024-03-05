import argparse
import pickle
from nirdizati_light.pattern_discovery.utils.Alignment_Check import alignment_check,Alignment_Checker
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
from nirdizati_light.pattern_discovery.utils.Auto_IMPID import AutoStepWise_PPD
from nirdizati_light.pattern_discovery.utils.IMIPD import VariantSelection, create_pattern_attributes, Trace_graph_generator, Pattern_extension,\
    plot_only_pattern, Single_Pattern_Extender
from sklearn.model_selection import train_test_split
import itertools


def impressed_wrapper(df,output_path,discovery_type,case_id,activity,timestamp,outcome,outcome_type,delta_time,
                      max_gap,max_extension_step,factual_outcome,likelihood,encoding,testing_percentage,pareto_only):
    # Load the log
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pareto_features = ['Outcome_Interest', 'Frequency_Interest', 'likelihood']
    pareto_sense = ['max', 'max', 'max']
    df = df[[case_id, activity, timestamp, outcome, likelihood]]
    df[activity] = df[activity].astype('string')
    df[activity] = df[activity].str.replace("_", "-")
    df[timestamp] = pd.to_datetime(df[timestamp],format='mixed')
    df[case_id] = df[case_id].astype('string')
    outcomes = df[outcome].unique()
    if outcome_type == 'binary':
        for i, out in enumerate(outcomes):
            df.loc[df[outcome] == str(out), outcome] = i
        df[outcome] = df[outcome].astype('uint8')
    elif outcome_type == 'numerical':
        df[outcome] = df[outcome].astype('float32')

    color_codes = ["#" + ''.join([random.choice('000123456789ABCDEF') for i in range(6)])
                   for j in range(len(df[activity].unique()))]

    color_act_dict = dict()
    counter = 0
    for act in df[activity].unique():
        color_act_dict[act] = color_codes[counter]
        counter += 1
    color_act_dict['start'] = 'k'
    color_act_dict['end'] = 'k'

    patient_data = df[[case_id, likelihood, outcome]]
    patient_data.drop_duplicates(subset=[case_id], inplace=True)
    patient_data.loc[:, list(df[activity].unique())] = 0
    selected_variants = VariantSelection(df, case_id, activity, timestamp)
    for case in selected_variants["case:concept:name"].unique():
        Other_cases = \
            selected_variants.loc[selected_variants["case:concept:name"] == case, 'case:CaseIDs'].tolist()[0]
        trace = df.loc[df[case_id] == case, activity].tolist()
        for act in np.unique(trace):
            Number_of_act = trace.count(act)
            for Ocase in Other_cases:
                patient_data.loc[patient_data[case_id] == Ocase, act] = Number_of_act


    if discovery_type == 'interactive':
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

        Alignment_Check = Alignment_Checker(case_id)
        for pattern_name in list(Patterns_Dictionary.keys()):
            Pattern = Patterns_Dictionary[pattern_name]['pattern']
            patient_data[pattern_name] = 0
            patient_data = Alignment_Check.check_pattern_alignment(EventLog_graphs, patient_data, Pattern, pattern_name)
    
        pattern_attributes = create_pattern_attributes(patient_data, outcome,
                                                   factual_outcome, list(Patterns_Dictionary.keys()), outcome_type)

        Objectives_attributes = pattern_attributes[pareto_features]
        mask = paretoset(Objectives_attributes, sense=pareto_sense)
        if pareto_only:
            paretoset_patterns = pattern_attributes[mask]
        else:
            paretoset_patterns = pattern_attributes
        all_pattern_dictionary.update(Patterns_Dictionary)
        All_pareto_patterns.extend(paretoset_patterns['patterns'].tolist())
        paretoset_patterns_to_save = paretoset_patterns.copy()
        pattern_activities = [list(nx.get_node_attributes(
            Patterns_Dictionary[m]['pattern'], 'value').values()) for m in
                              list(paretoset_patterns['patterns'])]
        pattern_relations = [list(nx.get_edge_attributes(
            Patterns_Dictionary[m]['pattern'], 'eventually').values()) for m in
                             list(paretoset_patterns['patterns'])]
        pattern_activities = [list(itertools.chain(*e)) for e in zip(pattern_activities, pattern_relations)]
        paretoset_patterns_to_save['activities'] = pattern_activities
        paretoset_patterns_to_save.to_csv(output_path + '/paretoset_2.csv', index=False)
        # parallelize the plotting of the patterns
        for pattern in paretoset_patterns['patterns']:
            P_graph = Patterns_Dictionary[pattern]['pattern']
            pickle.dump(P_graph, open(output_path + '/%s_interactive.pickle' % pattern, "wb"))

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
            paretoset_patterns = pattern_attributes
            if pareto_only:
                paretoset_patterns = pattern_attributes[mask]
            else:
                paretoset_patterns = pattern_attributes
            for pattern in paretoset_patterns['patterns']:
                P_graph = Patterns_Dictionary[pattern]['pattern']
                pickle.dump(P_graph, open(output_path + '/%s_interactive.pickle' % pattern, "wb"))

            paretoset_patterns_to_save = paretoset_patterns.copy()

            pattern_activities = [list(nx.get_node_attributes(
                Patterns_Dictionary[m]['pattern'], 'value').values()) for m in
                                  list(paretoset_patterns['patterns'])]
            pattern_relations = [list(nx.get_edge_attributes(
                Patterns_Dictionary[m]['pattern'], 'eventually').values()) for m in
                                 list(paretoset_patterns['patterns'])]
            pattern_activities = [list(itertools.chain(*e)) for e in zip(pattern_activities, pattern_relations)]
            paretoset_patterns_to_save['activities'] = pattern_activities
            paretoset_patterns_to_save.to_csv(output_path + '/paretoset_%s.csv' % counter, index=False)
            All_pareto_patterns.extend(paretoset_patterns['patterns'].tolist())
            counter += 1
            # parallelize the plotting of the patterns
            Parallel(n_jobs=6)(delayed(plot_only_pattern)(Patterns_Dictionary, row['patterns'], color_act_dict, output_path)
                               for ticker, row in paretoset_patterns.iterrows())

            continue_extending = input("Enter 1 if you want to continue extending patterns or 0 to stop: ")
            continue_extending = int(continue_extending)
        if encoding:
            # TODO: FILTERED PATTERNS TO NOT INCLUDE THE SINGULAR ACTIVITES
            All_pareto_patterns = [x for x in All_pareto_patterns if x not in set(paretoset_activities['patterns'])]
            Encoded_patterns = patient_data[All_pareto_patterns]
            Encoded_patterns.loc[:, case_id] = patient_data[case_id]
            Encoded_patterns.loc[:, outcome] = patient_data[outcome]
            Encoded_patterns.to_csv(output_path + '/EncodedPatterns_InteractiveMode.csv', index=False)
            if outcome_type == 'binary':
                train, test = train_test_split(Encoded_patterns, test_size=testing_percentage, random_state=42,
                                           stratify=patient_data[outcome])
            else:
                train, test = train_test_split(Encoded_patterns, test_size=testing_percentage, random_state=42)
            train_X = train
            test_X = test
            test_ids = test_X.loc[:,case_id]
    if discovery_type == 'auto':
        train_X, test_X,test_ids = AutoStepWise_PPD(max_extension_step, max_gap,
                                           testing_percentage, df, patient_data, case_id,
                                           activity, outcome, outcome_type, timestamp,
                                           pareto_features, pareto_sense, delta_time,
                                           color_act_dict, output_path,
                                           factual_outcome,pareto_only)
        
    train_X.to_csv(output_path + "/training_encoded_log.csv", index=False)
    test_X.to_csv(output_path + "/testing_encoded_log.csv", index=False)
    #TODO: Add decision tree training here with rule extraction tomorrow
    #TODO: Plot the frequency curves for all generated patterns
    return train_X,test_X,test_ids