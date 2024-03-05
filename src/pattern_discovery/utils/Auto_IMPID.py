from paretoset import paretoset
import networkx as nx
from networkx.readwrite import json_graph
import json
import pm4py
import pickle
from joblib import Parallel, delayed
import networkx.algorithms.isomorphism as iso
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
from nirdizati_light.pattern_discovery.utils.Alignment_Check import alignment_check,Alignment_Checker
from sklearn.model_selection import train_test_split
from nirdizati_light.pattern_discovery.utils.IMIPD import create_pattern_attributes, Pattern_extension,Single_Pattern_Extender,plot_only_pattern, Trace_graph_generator
from nirdizati_light.pattern_discovery.utils.tools import create_embedded_pattern_in_trace, update_pattern_dict, Pattern_Extender
import itertools

def AutoStepWise_PPD(Max_extension_step, Max_gap_between_events, test_data_percentage, data, patient_data,
                     case_id, activity, outcome, outcome_type, timestamp,
                     pareto_features, pareto_sense, d_time, color_act_dict, save_path, factual_outcome,pareto_only):

    # split test and train data
    if outcome_type == 'binary':
        train, test = train_test_split(patient_data, test_size=test_data_percentage, random_state=42,
                                       stratify=patient_data[outcome])
    else:
        train, test = train_test_split(patient_data, test_size=test_data_percentage, random_state=42)

    # Filter the data based on case_id
    train_ids, test_ids = train[case_id], test[case_id]
    train_data = patient_data[patient_data[case_id].isin(train_ids)]
    test_data = patient_data[patient_data[case_id].isin(test_ids)]

    All_pareto_patterns = []
    activity_attributes = create_pattern_attributes(train_data, outcome, factual_outcome,
                                                    list(data[activity].unique()), outcome_type)

    Objectives_attributes = activity_attributes[pareto_features]
    mask = paretoset(Objectives_attributes, sense=pareto_sense)
    paretoset_activities = activity_attributes[mask]
    from pandas import concat
    paretoset_activities = concat(
        [paretoset_activities, activity_attributes[activity_attributes['patterns'] == 'activityA']])
    All_pareto_patterns.extend(list(paretoset_activities['patterns']))

    for pattern in list(paretoset_activities['patterns']):
        G = nx.DiGraph()
        G.add_node(1, value=pattern, parallel=False, color=color_act_dict[pattern])
        pickle.dump(G, open(save_path + '/%s.pickle' % pattern, "wb"))


    Extended_patterns_at_stage = dict()
    All_extended_patterns_1_list = []
    EventLog_graphs = dict()
    Alignment_Check = Alignment_Checker(case_id)

    if pareto_only:
        #Patterns_for_extension = list(paretoset_activities['patterns'])
        Patterns_for_extension = list(activity_attributes['patterns'])
    else:
        Patterns_for_extension = list(activity_attributes['patterns'])

    for Core_activity in Patterns_for_extension:
        filtered_cases = data.loc[data[activity] == Core_activity, case_id]
        filtered_main_data = data[data[case_id].isin(filtered_cases)]
        new_patterns_for_core = []
        for case in filtered_main_data[case_id].unique():
            case_data = filtered_main_data[filtered_main_data[case_id] == case]

            if case not in EventLog_graphs.keys():
                Trace_graph = Trace_graph_generator(filtered_main_data, d_time,
                                                    case, color_act_dict, case_id, activity, timestamp)

                EventLog_graphs[case] = Trace_graph.copy()
            else:
                Trace_graph = EventLog_graphs[case].copy()
            Extended_patterns_at_stage, new_patterns_for_core = Pattern_extension(case_data, Trace_graph, Core_activity,
                                                                                  case_id, Extended_patterns_at_stage,
                                                                                  Max_gap_between_events,
                                                                                  new_patterns_for_core)


        All_extended_patterns_1_list.extend(new_patterns_for_core)
        for pattern_name in new_patterns_for_core:
            Pattern = Extended_patterns_at_stage[pattern_name]['pattern']
            patient_data[pattern_name] = 0
            patient_data = Alignment_Check.check_pattern_alignment(EventLog_graphs, patient_data, Pattern, pattern_name)

    new_train_data = patient_data[patient_data[case_id].isin(train_data[case_id])]
    new_test_data = patient_data[patient_data[case_id].isin(test_data[case_id])]

    pattern_attributes = create_pattern_attributes(new_train_data, outcome, factual_outcome,
                                                   All_extended_patterns_1_list, outcome_type)
    Objectives_attributes = pattern_attributes[pareto_features]
    mask = paretoset(Objectives_attributes, sense=pareto_sense)
    paretoset_patterns = pattern_attributes[mask]
    #paretoset_patterns = pd.concat([paretoset_patterns,pattern_attributes[[pattern_attributes['activity']=='activityA']]])
    paretoset_patterns_to_save = paretoset_patterns.copy()

    pattern_activities = [list(nx.get_node_attributes(
        Extended_patterns_at_stage[m]['pattern'], 'value').values()) for m in
                          list(paretoset_patterns['patterns'])]
    pattern_relations = [list(nx.get_edge_attributes(
        Extended_patterns_at_stage[m]['pattern'], 'eventually').values()) for m in
                         list(paretoset_patterns['patterns'])]
    pattern_activities = [list(itertools.chain(*e)) for e in zip(pattern_activities, pattern_relations)]
    paretoset_patterns_to_save['activities'] = pattern_activities
    paretoset_patterns_to_save.to_csv(save_path + '/paretoset.csv',mode='a', index=False)
    All_pareto_patterns.extend(list(paretoset_patterns['patterns']))

    # save all patterns in paretofront in json format
    for pattern in paretoset_patterns['patterns']:
        P_graph = Extended_patterns_at_stage[pattern]['pattern']
        pickle.dump(P_graph, open(save_path + '/%s.pickle' % pattern, "wb"))


    # parallelize the plotting of the patterns
    Parallel(n_jobs=6)(delayed(plot_only_pattern)(Extended_patterns_at_stage, row['patterns'], color_act_dict, save_path)
                        for ticker, row in paretoset_patterns.iterrows())

    train_X = new_train_data[All_pareto_patterns]
    test_X = new_test_data[All_pareto_patterns]
    train_X['Case_ID'] = new_train_data[case_id]
    test_X['Case_ID'] = new_test_data[case_id]
    train_X['Outcome'] = new_train_data[outcome]
    test_X['Outcome'] = new_test_data[outcome]

    if pareto_only:
        Patterns_for_extension = list(paretoset_patterns['patterns'])
        #Patterns_for_extension = list(pattern_attributes['patterns'])
    else:
        Patterns_for_extension = list(pattern_attributes['patterns'])

    All_extended_patterns_dict = Extended_patterns_at_stage.copy()
    for ext in range(1, Max_extension_step):
        print("extension number %s " % (ext + 1))
        new_patterns_per_extension = []
        eventual_counter = 0

        for chosen_pattern_ID in Patterns_for_extension:
            if any(nx.get_edge_attributes(All_extended_patterns_dict[chosen_pattern_ID]['pattern'],
                                          'eventually').values()):
                eventual_counter += 1
                continue

            All_extended_patterns_dict, Extended_patterns_at_stage = Single_Pattern_Extender(
                All_extended_patterns_dict,
                chosen_pattern_ID,
                EventLog_graphs, data, Max_gap_between_events, activity, case_id)

            new_patterns_per_extension.extend(Extended_patterns_at_stage.keys())
        # Extension_2_patterns_list, Extended_patterns_at_stage = \
        #     Pattern_Extender(Extended_patterns_at_stage,
        #                      EventLog_graphs, data, case_id, activity, Max_gap_between_events)
        if eventual_counter == len(Patterns_for_extension):
            break
        for pattern_name in new_patterns_per_extension:
            Pattern = All_extended_patterns_dict[pattern_name]['pattern']
            patient_data[pattern_name] = 0
            patient_data = Alignment_Check.check_pattern_alignment(EventLog_graphs, patient_data, Pattern, pattern_name)

        train_patient_data = patient_data[patient_data[case_id].isin(train_data[case_id])]
        test_patient_data = patient_data[patient_data[case_id].isin(test_data[case_id])]

        try:
            pattern_attributes = create_pattern_attributes(train_patient_data, outcome, factual_outcome,
                                                           new_patterns_per_extension, outcome_type)
    
            Objectives_attributes = pattern_attributes[pareto_features]
            mask = paretoset(Objectives_attributes, sense=pareto_sense)
            paretoset_patterns = pattern_attributes[mask]
            All_pareto_patterns.extend(list(paretoset_patterns['patterns']))
    
            if pareto_only:
                Patterns_for_extension = list(paretoset_patterns['patterns'])
            else:
                Patterns_for_extension = list(pattern_attributes['patterns'])
            paretoset_patterns_to_save = paretoset_patterns.copy()

            pattern_activities = [list(nx.get_node_attributes(
                Extended_patterns_at_stage[m]['pattern'], 'value').values()) for m in
                                  list(paretoset_patterns['patterns'])]
            pattern_relations = [list(nx.get_edge_attributes(
                Extended_patterns_at_stage[m]['pattern'], 'eventually').values()) for m in
                                 list(paretoset_patterns['patterns'])]
            pattern_activities = [list(itertools.chain(*e)) for e in zip(pattern_activities, pattern_relations)]
            paretoset_patterns_to_save['activities'] = pattern_activities
            paretoset_patterns_to_save.to_csv(save_path + '/paretoset.csv' % ext,mode='a', index=False)
            # save all patterns in paretofront in json format
            for pattern in paretoset_patterns['patterns']:
                P_graph = All_extended_patterns_dict[pattern]['pattern']
                pickle.dump(P_graph, open(save_path + '/%s.pickle' % pattern, "wb"))
            Parallel(n_jobs=6)(
                delayed(plot_only_pattern)(Patterns_Dictionary, row['patterns'], color_act_dict, output_path)
                for ticker, row in paretoset_patterns.iterrows())

            train_X = train_patient_data[All_pareto_patterns]
            test_X = test_patient_data[All_pareto_patterns]
            train_X['Case_ID'] = train_patient_data[case_id]
            test_X['Case_ID'] = test_patient_data[case_id]
            train_X['Outcome'] = train_patient_data[outcome]
            test_X['Outcome'] = test_patient_data[outcome]
            #train_X['AMOUNT_REQ'] = train_patient_data['AMOUNT_REQ']
            #test_X['AMOUNT-REQ'] = test_patient_data['AMOUNT_REQ']
        except:
            continue

    return train_X, test_X,test_ids
