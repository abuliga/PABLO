import os
from sklearn.preprocessing import LabelEncoder
from paretoset import paretoset
import networkx as nx
import pickle
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from sklearn import preprocessing
from nirdizati_light.pattern_discovery.utils.IMIPD import create_pattern_attributes, Pattern_extension, Single_Pattern_Extender, \
    plot_only_pattern, Single_Pattern_Extender_with_attributes, filtering_cases
import numpy as np
from nirdizati_light.pattern_discovery.utils.Alignment_Check import Alignment_Checker
import pandas as pd
import re


class AutoPatternDetection:
    def __init__(self, EventLog_graphs, Max_extension_step, Max_gap_between_events, test_data_percentage, data,
                 patient_data,
                 case_id, activity, outcome, outcome_type, timestamp,
                 pareto_features, pareto_sense, d_time, color_act_dict,
                 save_path, factual_outcome, extension_style='Pareto', data_dependency=None,
                 aggregation_style="all", pattern_extension_strategy='activities', model='DT',
                 frequency_type='absolute', distance_style='case'):

        self.EventLog_graphs = EventLog_graphs
        self.frequency_type = frequency_type
        self.Max_extension_step = Max_extension_step
        self.Max_gap_between_events = Max_gap_between_events
        self.case_id = case_id
        self.activity = activity
        self.outcome = outcome
        self.outcome_type = outcome_type
        self.timestamp = timestamp
        self.pareto_features = pareto_features
        self.interest_dimension = len(pareto_features)
        self.pareto_sense = pareto_sense
        self.d_time = d_time
        self.color_act_dict = color_act_dict
        self.save_path = save_path
        self.factual_outcome = factual_outcome
        self.extension_style = extension_style
        self.data_dependency = data_dependency
        self.aggregation_style = aggregation_style
        self.data = data
        self.patient_data = patient_data
        self.pattern_extension_strategy = pattern_extension_strategy
        self.model = model

        if outcome_type == 'binary':
            self.train, self.test = train_test_split(patient_data, test_size=test_data_percentage, random_state=42,
                                                     stratify=patient_data[outcome])
        else:
            self.train, self.test = train_test_split(patient_data, test_size=test_data_percentage, random_state=42)

        # determine which features are case attributes and which are event attributes
        self.Case_attributes = []
        self.Case_attributes_numerical = []
        self.Event_categorical_attributes = []
        self.Event_numerical_attributes = []
        for feature in self.data.columns:
            if feature not in [self.case_id, self.activity, self.timestamp]:
                if len(self.data.groupby([self.case_id, feature]).size().reset_index()) > len(
                        self.data[self.case_id].unique()):
                    if pd.api.types.is_numeric_dtype(self.data[feature]):
                        self.Event_numerical_attributes.append(feature)
                    else:
                        self.Event_categorical_attributes.append(feature)
                else:
                    if pd.api.types.is_numeric_dtype(self.data[feature]):
                        self.Case_attributes_numerical.append(feature)
                    self.Case_attributes.append(feature)

        self.Case_attributes_categorical = list((set(self.Case_attributes) -
                                                set(self.Case_attributes_numerical)) - {self.outcome})
        self.agg_dict = {}
        self.agg_dict_only_case = {}
        for col in self.Event_numerical_attributes:
            self.agg_dict[col] = 'sum'
        for col in self.Case_attributes:
            self.agg_dict[col] = lambda x: x.mode()[0] if not x.mode().empty else None
            self.agg_dict_only_case[col] = lambda x: x.mode()[0] if not x.mode().empty else None
        for col in self.Event_categorical_attributes:
            self.agg_dict[col] = lambda x: str(list(x))

        # aggregation for delta time between events of a pattern
        self.agg_dict['dtime'] = 'sum'

        # self.pairwise_distances_array, self.pair_cases, self.start_search_points = None, None, None
        self.pairwise_distances_array, self.pair_cases, self.start_search_points = self.creat_pairwise_distance(style=distance_style)


    def AutoStepWise_PPD(self):
        Alignment_Check = Alignment_Checker(self.case_id, self.outcome, self.outcome_type)
        train_ids, test_ids = self.train[self.case_id], self.test[self.case_id]
        All_pareto_patterns = []
        discovered_single_patterns = []
        data_dependent_rules = dict()
        if self.data_dependency == 'dependent':
            self.data[self.case_id].astype(str)
            df_train = self.data[self.data[self.case_id].isin(train_ids)]
            for pattern in df_train[self.activity].unique():
                Temp_data_pattern = self.data[self.data[self.activity] == pattern]
                Temp_data_pattern.drop([self.activity, self.timestamp], axis=1, inplace=True)
                self.patient_data, data_dependent_patterns, pattern_name, rules_per_node = \
                    Alignment_Check.pattern_annotation(self.patient_data, Temp_data_pattern, pattern,
                                                       self.data_dependency, self.model)

                if data_dependent_patterns is not None:
                    discovered_single_patterns.append(pattern)
                    discovered_single_patterns.extend(data_dependent_patterns)
                    # self.patient_data.drop(pattern_name, axis=1, inplace=True) # not removing the core pattern from the pool
                    for P in data_dependent_patterns:
                        data_dependent_rules[P] = rules_per_node[int(P.split('+')[1].split('_')[1])]
                else:
                    discovered_single_patterns.append(pattern)
        else:
            discovered_single_patterns = list(self.data[self.activity].unique())

        # Filter the data based on case_id
        train_data = self.patient_data[self.patient_data[self.case_id].isin(train_ids)]
        test_data = self.patient_data[self.patient_data[self.case_id].isin(test_ids)]
        activity_attributes = create_pattern_attributes(train_data, self.outcome, self.factual_outcome,
                                                        discovered_single_patterns, self.outcome_type,
                                                        self.frequency_type, self.pairwise_distances_array,
                                                        self.pair_cases, self.start_search_points,
                                                        self.interest_dimension)

        activity_attributes.to_csv(self.save_path + '/activity_attributes_step0.csv', index=False)

        All_pareto_patterns, paretoset_activities = self.create_pareto_front(activity_attributes, All_pareto_patterns)

        # remove keys in data_dependent_rules that are not in the pareto front
        if self.extension_style == 'Pareto':
            for key in list(data_dependent_rules.keys()):
                if key not in All_pareto_patterns:
                    data_dependent_rules.pop(key)

        for pattern in list(paretoset_activities['patterns']):
            G = nx.DiGraph()
            if '+' in pattern:
                pattern_parent = pattern.split('+')[0]
                G.add_node(1, value=pattern, parallel=False, color=self.color_act_dict[pattern_parent])
                G.graph['rule'] = data_dependent_rules[pattern]
                # write rules in a text file
                with open(self.save_path + '/%s_rules.txt' % pattern, 'w') as f:
                    f.write(data_dependent_rules[pattern])

            else:
                G.add_node(1, value=pattern, parallel=False, color=self.color_act_dict[pattern])
                G.graph['rule'] = None

            pickle.dump(G, open(self.save_path + '/%s.pickle' % pattern, "wb"))

        if self.extension_style == 'Pareto':
            Patterns_for_extension = list(paretoset_activities['patterns'])
        elif self.extension_style == 'All':
            Patterns_for_extension = list(activity_attributes['patterns'])
        else:
            raise ValueError('extension_style should be either "Pareto" or "All"')

        if self.Max_extension_step == 0:
            train_X, test_X = self.prepare_data_for_saving(train_data, test_data, All_pareto_patterns)
            return train_X, test_X

        columns_new_DTCase = [self.case_id, 'act', 'instance', 'dtime']
        # columns_new_DTCase = [self.case_id, 'act', 'instance']
        columns_new_DTCase.extend(
            self.data.drop([self.case_id, self.activity, self.timestamp], axis=1).columns.tolist())

        extended_parents = []
        Extended_patterns_at_stage = dict()
        All_extended_patterns_1_list = []

        for pattern_to_extend in Patterns_for_extension:
            if (self.pattern_extension_strategy == 'activities') or ('+' not in pattern_to_extend):
                Core_activity = pattern_to_extend.split('+')[0]
                if Core_activity in extended_parents:
                    continue
                extended_parents.append(Core_activity)
                filtered_cases = self.data.loc[self.data[self.activity] == Core_activity, self.case_id]
                filtered_main_data = self.data[self.data[self.case_id].isin(filtered_cases)]
                new_patterns_for_core = []
                for case in filtered_main_data[self.case_id].unique():
                    if case not in train_ids.to_list():
                        continue
                    case_data = filtered_main_data[filtered_main_data[self.case_id] == case]
                    Trace_graph = self.EventLog_graphs[case].copy()
                    Extended_patterns_at_stage, new_patterns_for_core = Pattern_extension(case_data, Trace_graph,
                                                                                          Core_activity,
                                                                                          self.case_id,
                                                                                          Extended_patterns_at_stage,
                                                                                          self.Max_gap_between_events,
                                                                                          None,
                                                                                          new_patterns_for_core)

                All_extended_patterns_1_list.extend(new_patterns_for_core)
                for pattern_name in new_patterns_for_core:
                    Pattern = Extended_patterns_at_stage[pattern_name]['pattern']
                    self.patient_data.loc[:, pattern_name] = 0
                    self.patient_data, DT_Case_Pattern = \
                        Alignment_Check.check_pattern_alignment(self.EventLog_graphs, self.patient_data, Pattern,
                                                                pattern_name)
                    if len(DT_Case_Pattern) == 0:  # in case the pattern is not found in the data
                        continue

                    if self.data_dependency == 'dependent':
                        DT_Case_Pattern.columns = columns_new_DTCase
                        Aggregated_DT_case_patterns = self.feature_aggregation(DT_Case_Pattern, Core_activity,
                                                                               data_dependent_rules)
                        self.patient_data, data_dependent_patterns, removed_pattern, rules_per_node = \
                            Alignment_Check.pattern_annotation(self.patient_data, Aggregated_DT_case_patterns,
                                                               pattern_name,
                                                               self.data_dependency, self.model)

                        if data_dependent_patterns is not None:
                            All_extended_patterns_1_list.extend(data_dependent_patterns)
                            # All_extended_patterns_1_list.remove(removed_pattern)
                            for p in data_dependent_patterns:
                                Extended_patterns_at_stage[p] = Extended_patterns_at_stage[removed_pattern].copy()
                                data_dependent_rules[p] = rules_per_node[int(p.split('+')[1].split('_')[1])]

            elif (self.pattern_extension_strategy == 'attributes') and ('+' in pattern_to_extend):
                if pattern_to_extend in extended_parents:
                    continue
                filtered_cases = filtering_cases(self.data, self.activity, self.case_id, pattern_to_extend,
                                                 data_dependent_rules)
                Core_activity = pattern_to_extend.split('+')[0]
                extended_parents.append(pattern_to_extend)
                filtered_main_data = self.data[self.data[self.case_id].isin(filtered_cases)]
                new_patterns_for_core = []
                for case in filtered_main_data[self.case_id].unique():
                    if case not in train_ids.to_list():
                        continue
                    case_data = filtered_main_data[filtered_main_data[self.case_id] == case]
                    Trace_graph = self.EventLog_graphs[case].copy()
                    Extended_patterns_at_stage, new_patterns_for_core = Pattern_extension(case_data, Trace_graph,
                                                                                          Core_activity,
                                                                                          self.case_id,
                                                                                          Extended_patterns_at_stage,
                                                                                          self.Max_gap_between_events,
                                                                                          pattern_to_extend,
                                                                                          new_patterns_for_core)
                All_extended_patterns_1_list.extend(new_patterns_for_core)
                for pattern_name in new_patterns_for_core:
                    Pattern = Extended_patterns_at_stage[pattern_name]['pattern']
                    self.patient_data.loc[:, pattern_name] = 0
                    self.patient_data, DT_Case_Pattern = \
                        Alignment_Check.check_pattern_alignment(self.EventLog_graphs, self.patient_data, Pattern,
                                                                pattern_name)
                    if len(DT_Case_Pattern) == 0:  # in case the pattern is not found in the data
                        continue
                    DT_Case_Pattern.columns = columns_new_DTCase
                    Aggregated_DT_case_patterns = self.feature_aggregation(DT_Case_Pattern, Core_activity,
                                                                           data_dependent_rules)
                    self.patient_data, data_dependent_patterns, removed_pattern, rules_per_node = \
                        Alignment_Check.pattern_annotation(self.patient_data, Aggregated_DT_case_patterns, pattern_name,
                                                           self.data_dependency, self.model)

                    if data_dependent_patterns is not None:
                        All_extended_patterns_1_list.extend(data_dependent_patterns)
                        # All_extended_patterns_1_list.remove(removed_pattern)
                        for p in data_dependent_patterns:
                            Extended_patterns_at_stage[p] = Extended_patterns_at_stage[removed_pattern].copy()
                            data_dependent_rules[p] = rules_per_node[int(p.split('+')[-1].split('_')[1])]

        new_train_data = self.patient_data[self.patient_data[self.case_id].isin(train_ids)]
        new_test_data = self.patient_data[self.patient_data[self.case_id].isin(test_ids)]

        pattern_attributes = create_pattern_attributes(new_train_data, self.outcome, self.factual_outcome,
                                                       All_extended_patterns_1_list, self.outcome_type,
                                                       self.frequency_type,
                                                       self.pairwise_distances_array,
                                                       self.pair_cases, self.start_search_points,
                                                       self.interest_dimension)

        pattern_attributes.to_csv(self.save_path + '/activity_attributes_step1.csv', index=False)

        All_pareto_patterns, paretoset_patterns = self.create_pareto_front(pattern_attributes, All_pareto_patterns)
        if self.extension_style == 'Pareto':
            Patterns_for_extension = list(paretoset_patterns['patterns'])
        elif self.extension_style == 'All':
            Patterns_for_extension = list(pattern_attributes['patterns'])
        else:
            raise ValueError('extension_style should be either "Pareto" or "All"')

        # Add if Pareto else:
        if self.extension_style == 'Pareto':
            for key in list(data_dependent_rules.keys()):
                if key not in All_pareto_patterns:
                    data_dependent_rules.pop(key)

        # save all patterns in paretofront in json format
        for pattern in paretoset_patterns['patterns']:
            P_graph = Extended_patterns_at_stage[pattern]['pattern']
            rules_number = pattern.count("+")
            P_graph.graph['rule'] = []
            search_start = 0
            for r in range(rules_number):
                rule_index = pattern.find("rule_", search_start)
                search_start = rule_index + 1
                P_graph.graph['rule'].append(data_dependent_rules[pattern[:rule_index + 6]])

            pickle.dump(P_graph, open(self.save_path + '/%s.pickle' % pattern, "wb"))
            plot_only_pattern(Extended_patterns_at_stage, pattern, self.color_act_dict, self.save_path)

        train_X, test_X = self.prepare_data_for_saving(new_train_data, new_test_data, All_pareto_patterns)

        All_extended_patterns_dict = Extended_patterns_at_stage.copy()
        for ext in range(1, self.Max_extension_step):
            print("extension number %s " % (ext + 1))
            new_patterns_per_extension = []
            eventual_counter = 0
            for chosen_pattern_ID in Patterns_for_extension:
                if any(nx.get_edge_attributes(All_extended_patterns_dict[chosen_pattern_ID]['pattern'],
                                              'eventually').values()):
                    eventual_counter += 1
                    continue

                if (self.pattern_extension_strategy == 'activities') or ('+' not in chosen_pattern_ID):

                    chosen_pattern_ID = chosen_pattern_ID.split('+')[0]
                    if chosen_pattern_ID in extended_parents:
                        continue
                    extended_parents.append(chosen_pattern_ID)
                    All_extended_patterns_dict, Extended_patterns_at_stage = Single_Pattern_Extender(
                        All_extended_patterns_dict,
                        chosen_pattern_ID,
                        self.EventLog_graphs, self.data, self.Max_gap_between_events, self.activity, self.case_id)
                elif (self.pattern_extension_strategy == 'attributes') and ('+' in chosen_pattern_ID):
                    # chosen_pattern_ID = chosen_pattern_ID.split('+')[0]
                    if chosen_pattern_ID in extended_parents:
                        continue
                    extended_parents.append(chosen_pattern_ID)
                    All_extended_patterns_dict, Extended_patterns_at_stage = Single_Pattern_Extender_with_attributes(
                        All_extended_patterns_dict,
                        chosen_pattern_ID,
                        self.EventLog_graphs, self.data, self.Max_gap_between_events, self.activity,
                        self.case_id, data_dependent_rules, chosen_pattern_ID)

                new_patterns_per_extension.extend(Extended_patterns_at_stage.keys())

            if eventual_counter >= len(Patterns_for_extension):
                break

            data_annotated_pattern = new_patterns_per_extension.copy()
            for pattern_name in new_patterns_per_extension:
                Pattern = All_extended_patterns_dict[pattern_name]['pattern']
                self.patient_data.loc[:, pattern_name] = 0
                self.patient_data, DT_Case_Pattern = \
                    Alignment_Check.check_pattern_alignment(self.EventLog_graphs, self.patient_data, Pattern,
                                                            pattern_name)
                if len(DT_Case_Pattern) == 0:  # in case the pattern is not found in the data
                    continue

                if self.data_dependency == 'dependent':
                    DT_Case_Pattern.columns = columns_new_DTCase
                    # foundational_pattern = "_".join(pattern_name.split("_")[:-1])
                    # if the pattern is not extended given the data attrbiutes, here the founfational pattern can just be the single core activity
                    foundational_pattern = pattern_name.split('_')[0]
                    Aggregated_DT_case_patterns = self.feature_aggregation(DT_Case_Pattern, foundational_pattern,
                                                                           data_dependent_rules)

                    self.patient_data, data_dependent_patterns, removed_pattern, rules_per_node = \
                        Alignment_Check.pattern_annotation(self.patient_data, Aggregated_DT_case_patterns,
                                                           pattern_name, self.data_dependency, self.model)

                    if data_dependent_patterns is not None:
                        data_annotated_pattern.extend(data_dependent_patterns)
                        # data_annotated_pattern.remove(removed_pattern)
                        for p in data_dependent_patterns:
                            All_extended_patterns_dict[p] = All_extended_patterns_dict[removed_pattern].copy()
                            if self.pattern_extension_strategy == 'activities':
                                data_dependent_rules[p] = rules_per_node[int(p.split('+')[1].split('_')[1])]
                            elif self.pattern_extension_strategy == 'attributes':
                                data_dependent_rules[p] = rules_per_node[int(p.split('+')[-1].split('_')[1])]

            print('Final patterns', data_annotated_pattern)
            train_patient_data = self.patient_data[self.patient_data[self.case_id].isin(train_data[self.case_id])]
            test_patient_data = self.patient_data[self.patient_data[self.case_id].isin(test_data[self.case_id])]

            pattern_attributes = create_pattern_attributes(train_patient_data, self.outcome, self.factual_outcome,
                                                           data_annotated_pattern, self.outcome_type,
                                                           self.frequency_type,
                                                           self.pairwise_distances_array,
                                                           self.pair_cases, self.start_search_points,
                                                           self.interest_dimension)

            pattern_attributes.to_csv(self.save_path + '/activity_attributes_step%s.csv' % (ext + 1), index=False)

            All_pareto_patterns, paretoset_patterns = self.create_pareto_front(pattern_attributes, All_pareto_patterns)

            # remove keys in data_dependent_rules that are not in the pareto front
            if self.extension_style == 'Pareto':
                for key in list(data_dependent_rules.keys()):
                    if key not in All_pareto_patterns:
                        data_dependent_rules.pop(key)

            if self.extension_style == 'Pareto':
                Patterns_for_extension = list(paretoset_patterns['patterns'])
            elif self.extension_style == 'All':
                Patterns_for_extension = list(pattern_attributes['patterns'])

            # save all patterns in paretofront in json format
            for pattern in paretoset_patterns['patterns']:
                P_graph = All_extended_patterns_dict[pattern]['pattern']
                rules_number = pattern.count("+")
                P_graph.graph['rule'] = []
                search_start = 0
                for r in range(rules_number):
                    rule_index = pattern.find("rule_", search_start)
                    search_start = rule_index + 1
                    P_graph.graph['rule'].append(data_dependent_rules[pattern[:rule_index + 6]])

                pickle.dump(P_graph, open(self.save_path + '/%s.pickle' % pattern, "wb"))
                plot_only_pattern(All_extended_patterns_dict, pattern, self.color_act_dict, self.save_path)

            train_X, test_X = self.prepare_data_for_saving(train_patient_data, test_patient_data, All_pareto_patterns)

        return train_X, test_X

    def create_pareto_front(self, activity_attributes, All_pareto_patterns):
        Objectives_attributes = activity_attributes[self.pareto_features]
        mask = paretoset(Objectives_attributes, sense=self.pareto_sense)
        paretoset_activities = activity_attributes[mask]
        All_pareto_patterns.extend(list(paretoset_activities['patterns']))
        return All_pareto_patterns, paretoset_activities

    def expanding_event_attributes(self, DT_Case_Pattern, Aggregated_DT_case_patterns, feature_to_aggregate):

        DT_Case_Pattern['unique_act'] = DT_Case_Pattern.groupby([self.case_id, 'act']).cumcount().astype(str) + \
                                        '-' + DT_Case_Pattern['act']

        for att in feature_to_aggregate:
            if att in self.Event_numerical_attributes:
                pivot_df = DT_Case_Pattern.pivot_table(index=self.case_id, columns='unique_act', values=att,
                                                       fill_value=0)
                pivot_df.columns = [f"{att}-{act}" for act in pivot_df.columns]
                Aggregated_DT_case_patterns = Aggregated_DT_case_patterns.join(pivot_df, on=self.case_id,
                                                                               how='left').fillna(0)

            elif att in self.Event_categorical_attributes:
                string_df = DT_Case_Pattern.pivot_table(index=self.case_id, columns='unique_act', values=att,
                                                        aggfunc=lambda x: ' '.join(x.dropna()), fill_value='')
                string_df.columns = [f"{att}-{act}" for act in string_df.columns]
                Aggregated_DT_case_patterns = Aggregated_DT_case_patterns.join(string_df, on=self.case_id,
                                                                               how='left').fillna('')

        return Aggregated_DT_case_patterns

    def feature_aggregation(self, DT_Case_Pattern, foundational_pattern, data_dependent_rules):

        if self.aggregation_style == "all":
            Aggregated_DT_case_patterns = DT_Case_Pattern.groupby([self.case_id, 'instance']).agg(
                self.agg_dict).reset_index()

        elif self.aggregation_style == "none":
            Aggregated_DT_case_patterns = DT_Case_Pattern.groupby([self.case_id, 'instance']).agg(
                self.agg_dict_only_case).reset_index()

            feature_to_aggregate = self.Event_numerical_attributes + self.Event_categorical_attributes
            Aggregated_DT_case_patterns = self.expanding_event_attributes(DT_Case_Pattern, Aggregated_DT_case_patterns,
                                                                          feature_to_aggregate)
        elif self.aggregation_style == "pareto":
            important_features = []
            agg_dict = self.agg_dict.copy()
            foundational_rules = [data_dependent_rules[key] for key in data_dependent_rules.keys() if
                                  key.startswith(foundational_pattern)]
            # Don't aggregate the important features from previous steps, if there is any
            for rule in foundational_rules:
                # text_pattern = r"\('(\w+)'(?:<=|<|>=|>) ([\d\.]+)\)"
                text_pattern = r"\('([^)]+)'\s*(<=|<|>=|>)\s*([\d.]+)\)"
                matches = re.findall(text_pattern, rule)
                important_features = [match[0].split("_")[0].split("-")[0] for match in matches]
                important_features = list(np.unique(important_features))
                for feat in important_features:
                    if feat in self.Case_attributes:
                        important_features.remove(feat)
                for fe in important_features:
                    try:
                        agg_dict.pop(fe)
                    except:
                        continue

            Aggregated_DT_case_patterns = DT_Case_Pattern.groupby([self.case_id, 'instance']).agg(
                agg_dict).reset_index()

            Aggregated_DT_case_patterns = self.expanding_event_attributes(DT_Case_Pattern, Aggregated_DT_case_patterns,
                                                                          important_features)

        elif self.aggregation_style == "mix":
            # mix aggregated and non-aggregated features
            Aggregated_DT_case_patterns = DT_Case_Pattern.groupby([self.case_id, 'instance']).agg(
                self.agg_dict).reset_index()
            feature_to_aggregate = self.Event_numerical_attributes + self.Event_categorical_attributes
            Aggregated_DT_case_patterns = self.expanding_event_attributes(DT_Case_Pattern, Aggregated_DT_case_patterns,
                                                                          feature_to_aggregate)

        else:
            raise ValueError('aggregation_style should be either "all" or "none" or "pareto" or "mix"')

        Aggregated_DT_case_patterns.drop('instance', axis=1, inplace=True)
        return Aggregated_DT_case_patterns

    def prepare_data_for_saving(self, train_data, test_data, All_pareto_patterns):
        train_X = train_data[All_pareto_patterns]
        test_X = test_data[All_pareto_patterns]
        train_X['Case_ID'] = train_data[self.case_id]
        test_X['Case_ID'] = test_data[self.case_id]
        train_X['Outcome'] = train_data[self.outcome]
        test_X['Outcome'] = test_data[self.outcome]

        return train_X, test_X

    def creat_pairwise_distance(self, style='case'):
        # calculate the pairwise case distance or load it from the file if it is already calculated
        distance_address = self.save_path + '/dist_%s.pickle' % style
        if not os.path.exists(distance_address):
            pairwise_distances_array = self.calculate_pairwise_case_distance(style)
            with open(distance_address, 'wb') as f:
                pickle.dump(pairwise_distances_array, f)

        else:
            with open(distance_address, 'rb') as f:
                pairwise_distances_array = pickle.load(f)

        pair_cases = [(a, b) for idx, a in enumerate(self.patient_data.index) for b in
                      self.patient_data.index[idx + 1:]]
        case_size = len(self.patient_data)
        i = 0
        start_search_points = []
        for k in range(case_size):
            start_search_points.append(k * case_size - (i + k))
            i += k

        return pairwise_distances_array, pair_cases, start_search_points

    def calculate_pairwise_case_distance(self, style='case'):
        agg_dict = dict()
        for col in self.Event_numerical_attributes:
            if 'time' in col or 'duration' in col:
                continue # we don't want to aggregate time features
            agg_dict[col] = 'mean'
        for col in self.Case_attributes:
            agg_dict[col] = lambda x: x.mode()[0] if not x.mode().empty else None


        Aggregated_data = self.data.groupby([self.case_id]).agg(agg_dict).reset_index()

        # X_features = self.data.drop([self.case_id, self.outcome, self.activity], axis=1)
        if style == 'case':
            num_col = self.Case_attributes_numerical
            cat_col = self.Case_attributes_categorical
            X_features = Aggregated_data.drop([self.case_id, self.outcome], axis=1)
        else:
            # frequency encode the categorical event attributes
            for col in self.Event_categorical_attributes:
                frequency = self.data.groupby([self.case_id, col])[col].agg('count').reset_index(name='frequency')
                pivoted = frequency.pivot_table(index=self.case_id, columns=col, values='frequency',
                                                fill_value=0).reset_index()
                Aggregated_data = pd.concat([Aggregated_data, pivoted], axis=1)

            X_features = Aggregated_data.drop([self.case_id, self.outcome], axis=1)
            cat_col = self.Case_attributes_categorical
            num_col = list(set(X_features.columns.tolist()) - set(cat_col))

        if self.outcome in num_col:
            num_col.remove(self.outcome)

        Cat_exists = False
        cat_dist = None
        Num_exists = False
        numeric_dist = None

        le = LabelEncoder()
        for col in cat_col:
            X_features[col] = le.fit_transform(X_features[col])

        if len(cat_col) > 0:
            Cat_exists = True
            cat_dist = pdist(X_features[cat_col].values, 'jaccard')

        if len(num_col) > 0:
            Num_exists = True
            numeric_dist = pdist(X_features[num_col].astype('float64').values, 'euclid')
            normalizer = preprocessing.MinMaxScaler()
            x = numeric_dist.reshape((-1, 1))
            numeric_dist = normalizer.fit_transform(x)
            del x
            numeric_dist = numeric_dist.reshape(len(numeric_dist))

        if Cat_exists and Num_exists:
            Combined_dist = ((len(cat_col) * cat_dist) + numeric_dist) / (1 + len(cat_col))
            return Combined_dist

        elif Cat_exists and not Num_exists:
            return cat_dist

        elif Num_exists and not Cat_exists:
            return numeric_dist
