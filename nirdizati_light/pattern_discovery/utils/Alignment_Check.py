import argparse
import random
import networkx as nx
import pandas as pd
from nirdizati_light.pattern_discovery.utils.IMIPD import Trace_graph_generator
from scipy.stats import pointbiserialr, pearsonr
from sklearn.metrics import roc_auc_score
import glob
import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import export_text
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

class Alignment_Checker:
    def __init__(self, case_id, outcome, outcome_type='binary'):
        self.children_left = None
        self.children_right = None
        self.feature = None
        self.threshold = None
        self.case_id = case_id
        self.outcome = outcome
        self.outcome_type = outcome_type
        self.likelihood = 'likelihood'

    def concatenate_lists(self, lists):
        concatenated_list = []
        for sublist in lists:
            concatenated_list.extend(sublist)
        return concatenated_list

    def generate_combinations(self, all_instances_list):
        def generate(current_combination, index):
            if index == len(all_instances_list):
                combinations.append(list(current_combination))
                return
            for instance in all_instances_list[index]:
                current_combination.append(instance)
                generate(current_combination, index + 1)
                current_combination.pop()

        combinations = []
        generate([], 0)
        concatenated_combinations = [self.concatenate_lists(combination) for combination in combinations]
        return concatenated_combinations

    def check_consecutive_edges(self, graph):
        nodes = list(graph.nodes())
        nodes.sort()
        # Check if there are no edges at all
        if len(graph.edges()) == 0:
            return False
        if any([v for v in nx.get_edge_attributes(graph, 'eventually').values()]):
            return False
        combined = False
        for i in range(len(nodes) - 1):
            if not graph.has_edge(nodes[i], nodes[i + 1]):
                combined = True
                break
        return combined

    def following_patterns_alignment(self, trace_node_value_exists, sub_nodes_value, pattern_length):
        start_search = 0
        instances = 0
        collected_indexes = {instances: []}
        core = sub_nodes_value[0]
        while start_search <= (len(trace_node_value_exists) - pattern_length) and core in trace_node_value_exists[
                                                                                          start_search:]:
            core_index = trace_node_value_exists.index(core, start_search)
            collected_indexes[instances].append(core_index)
            start_search = core_index + 1
            founded_idx = core_index + 1
            for node in sub_nodes_value[1:]:
                if node in trace_node_value_exists[founded_idx:]:
                    following_node_index = trace_node_value_exists.index(node, founded_idx)
                    if self.Trace_graph.has_edge(core_index, following_node_index):
                        collected_indexes[instances].append(following_node_index)
                        core_index = following_node_index
                        founded_idx = following_node_index + 1
                    else:
                        break

            if len(collected_indexes[instances]) == pattern_length:
                instances += 1
                collected_indexes[instances] = []
            else:
                collected_indexes[instances] = []

        self.remove_empty_lists(collected_indexes)
        return collected_indexes

    def remove_empty_lists(self, dictionary):
        keys_to_remove = [key for key, value in dictionary.items() if not value]
        for key in keys_to_remove:
            del dictionary[key]

    def parallel_patterns_alignment(self, trace_node_value_exists, sub_nodes_value, pattern_length):
        start_search = 0
        instances = 0
        collected_indexes = {instances: []}
        core = sub_nodes_value[0]
        while start_search <= len(trace_node_value_exists) - pattern_length and core in trace_node_value_exists[
                                                                                        start_search:]:
            core_index = trace_node_value_exists.index(core, start_search)
            if self.Trace_graph.nodes[core_index]['parallel']:
                start_search = core_index + 1
                pattern_lower_bound = max(core_index - pattern_length + 1, 0)
                pattern_upper_bound = min(pattern_lower_bound + pattern_length, len(trace_node_value_exists))
                while pattern_upper_bound <= core_index + pattern_length:
                    collected_indexes[instances].append(core_index)
                    for node in sub_nodes_value[1:]:
                        if node in trace_node_value_exists[pattern_lower_bound: pattern_upper_bound]:
                            idx = trace_node_value_exists.index(node, pattern_lower_bound, pattern_upper_bound)
                            if self.Trace_graph.nodes[idx]['parallel']:
                                collected_indexes[instances].append(idx)

                    if len(collected_indexes[instances]) == pattern_length:
                        collected_indexes[instances].sort()
                        instances += 1
                        collected_indexes[instances] = []
                    else:
                        collected_indexes[instances] = []

                    pattern_lower_bound += 1
                    pattern_upper_bound += 1

            else:
                start_search = core_index + 1

        self.remove_empty_lists(collected_indexes)
        return collected_indexes

    def two_eventual_alignment(self, trace_node_value_exists, sub_nodes_value, pattern_length):
        instances = 0
        start_search = 0
        collected_indexes = {instances: []}
        core = sub_nodes_value[0]
        eventually_node = sub_nodes_value[1]
        while start_search < len(trace_node_value_exists) - pattern_length and \
                core in trace_node_value_exists[start_search:]:
            core_index = trace_node_value_exists.index(core, start_search)
            search_for_eventually = core_index + 2
            while search_for_eventually < len(trace_node_value_exists) and \
                    eventually_node in trace_node_value_exists[search_for_eventually:]:
                second_index = trace_node_value_exists.index(eventually_node, search_for_eventually)
                collected_indexes[instances].append(core_index)
                collected_indexes[instances].append(second_index)
                instances += 1
                collected_indexes[instances] = []
                search_for_eventually = second_index + 1

            start_search = core_index + 1

        self.remove_empty_lists(collected_indexes)
        return collected_indexes

    def Combined_pattern_alignment(self, trace_node_value_exists, Pattern):
        parallel_nodes = set()
        following_nodes = set()
        Parallel_Pattern = Pattern.copy()
        Following_Pattern = Pattern.copy()
        pattern_nodes = list(Pattern.nodes())
        pattern_nodes.sort()
        for i in range(len(pattern_nodes) - 1):
            if not Pattern.has_edge(pattern_nodes[i], pattern_nodes[i + 1]):
                parallel_nodes.add(pattern_nodes[i])
                parallel_nodes.add(pattern_nodes[i + 1])
            else:
                following_nodes.add(pattern_nodes[i])
                following_nodes.add(pattern_nodes[i + 1])

        following_nodes = following_nodes.difference(parallel_nodes)
        Parallel_Pattern.remove_nodes_from(list(following_nodes))
        Following_Pattern.remove_nodes_from(list(parallel_nodes))

        No_Direct_Pattern = False
        if len(following_nodes) > 0:
            following_nodes = list(following_nodes)
            following_nodes.sort()
            Pattern_number = following_nodes[0]
            Direct_Patterns = {Pattern_number: [following_nodes[0]]}
            for i in range(1, len(following_nodes)):
                if following_nodes[i] == Direct_Patterns[Pattern_number][-1] + 1:
                    Direct_Patterns[Pattern_number].append(following_nodes[i])
                else:
                    Pattern_number = following_nodes[i]
                    Direct_Patterns[Pattern_number] = [following_nodes[i]]
        else:
            No_Direct_Pattern = True
            Direct_Patterns = dict()

        parallel_nodes = list(parallel_nodes)
        parallel_nodes.sort()
        Pattern_number = parallel_nodes[0]
        Parallel_Patterns = {Pattern_number: [parallel_nodes[0]]}
        for i in range(1, len(parallel_nodes)):
            if not Parallel_Pattern.has_edge(Pattern_number, parallel_nodes[i]):
                Parallel_Patterns[Pattern_number].append(parallel_nodes[i])
            else:
                Pattern_number = parallel_nodes[i]
                Parallel_Patterns[Pattern_number] = [parallel_nodes[i]]

        if not No_Direct_Pattern:
            for pattern in Direct_Patterns:
                temp_pattern = Following_Pattern.copy()
                for P in Direct_Patterns:
                    if P != pattern:
                        temp_pattern.remove_nodes_from(Direct_Patterns[P])
                Direct_Patterns[pattern] = temp_pattern

        for pattern in Parallel_Patterns:
            temp_pattern = Parallel_Pattern.copy()
            for P in Parallel_Patterns:
                if P != pattern:
                    temp_pattern.remove_nodes_from(Parallel_Patterns[P])
            Parallel_Patterns[pattern] = temp_pattern

        if not No_Direct_Pattern:
            Direct_Patterns_Temp = Direct_Patterns.copy()
            for pattern in Direct_Patterns_Temp:
                F_nodes_value = [v for v in nx.get_node_attributes(Direct_Patterns[pattern], 'value').values()]
                if len(F_nodes_value) == 0:
                    Direct_Patterns.pop(pattern)
                    continue
                Direct_Patterns[pattern] = self.following_patterns_alignment(trace_node_value_exists, F_nodes_value,
                                                                             len(F_nodes_value))

        for pattern in Parallel_Patterns:
            F_nodes_value = [v for v in nx.get_node_attributes(Parallel_Patterns[pattern], 'value').values()]
            Parallel_Patterns[pattern] = self.parallel_patterns_alignment(trace_node_value_exists, F_nodes_value,
                                                                          len(F_nodes_value))
        collected_indexes = dict()

        if not No_Direct_Pattern and (len(Parallel_Patterns) == 0 or len(Direct_Patterns) == 0):
            return collected_indexes

        if No_Direct_Pattern:
            Min_instances = min([len(Parallel_Patterns[pattern]) for pattern in Parallel_Patterns])
        else:
            Min_instances = min([len(Parallel_Patterns[pattern]) for pattern in Parallel_Patterns],
                                [len(Direct_Patterns[pattern]) for pattern in Direct_Patterns])
            Min_instances.sort()

        if not No_Direct_Pattern and Min_instances[0] == 0:
            # incomplete alignment of the combined pattern
            return collected_indexes

        elif No_Direct_Pattern and Min_instances == 0:
            # incomplete alignment of the combined pattern
            return collected_indexes

        else:
            Patterns_instance_number = [(pattern, len(Parallel_Patterns[pattern])) for pattern in Parallel_Patterns]
            if not No_Direct_Pattern:
                Patterns_instance_number.extend(
                    [(pattern, len(Direct_Patterns[pattern])) for pattern in Direct_Patterns])
            Patterns_instance_number.sort(key=lambda x: x[1])
            if not No_Direct_Pattern:
                All_instances = {**Parallel_Patterns, **Direct_Patterns}
            else:
                All_instances = Parallel_Patterns.copy()
            All_instances = dict(sorted(All_instances.items()))
            All_instances_list = []
            for key in All_instances:
                instance_per_key = []
                for i in All_instances[key]:
                    instance_per_key.append(All_instances[key][i])
                All_instances_list.append(instance_per_key)

            combined_instances = self.generate_combinations(All_instances_list)
            ID = 0
            for combination in combined_instances:
                consecutive = True
                for i in range(len(combination) - 1):
                    if combination[i + 1] - combination[i] != 1:
                        consecutive = False
                        break
                if consecutive:
                    collected_indexes[ID] = combination
                    ID += 1

        return collected_indexes

    def check_pattern_alignment(self, EventLog_graphs, patient_data, Pattern, pattern_name):
        # create a dataframe for recording the time-dependent features
        DT_Case_Pattern = pd.DataFrame(columns=[self.case_id, 'act', 'instance', 'dtime'])
        # check the type of pattern
        Parallel_pattern = len(Pattern.edges()) == 0 and len(Pattern.nodes()) > 1
        Combined_pattern = self.check_consecutive_edges(Pattern)
        Eventual_sub_pattern = any([v for v in nx.get_edge_attributes(Pattern, 'eventually').values()])
        # sub_edges_values = [v for v in nx.get_edge_attributes(Pattern, 'eventually').values()]
        sub_nodes_value = [v for v in nx.get_node_attributes(Pattern, 'value').values()]
        pattern_length = len(sub_nodes_value)
        for case in EventLog_graphs:
            self.Trace_graph = EventLog_graphs[case].copy()
            trace_nodes_value = [v for v in nx.get_node_attributes(self.Trace_graph, 'value').values()]
            trace_node_value_exists = trace_nodes_value.copy()
            trace_edges_dtime = [v for v in nx.get_edge_attributes(self.Trace_graph, 'dtime').values()]
            existence = 0
            for v in sub_nodes_value:
                if v in trace_nodes_value:
                    existence += 1
                    trace_nodes_value.remove(v)

            collected_indexes = dict()
            if existence >= len(sub_nodes_value):
                if not Eventual_sub_pattern and not Parallel_pattern and not Combined_pattern:
                    collected_indexes = self.following_patterns_alignment(trace_node_value_exists, sub_nodes_value,
                                                                          pattern_length)

                elif not Eventual_sub_pattern and not Combined_pattern and Parallel_pattern:
                    collected_indexes = self.parallel_patterns_alignment(trace_node_value_exists, sub_nodes_value,
                                                                         pattern_length)

                elif not Eventual_sub_pattern and Combined_pattern:
                    # break pattern into parallel and following patterns
                    collected_indexes = self.Combined_pattern_alignment(trace_node_value_exists, Pattern)

                elif Eventual_sub_pattern:
                    if pattern_length == 2:
                        collected_indexes = self.two_eventual_alignment(trace_node_value_exists, sub_nodes_value,
                                                                        pattern_length)
                    else:
                        eventually_node = sub_nodes_value[-1]
                        # remove the last node from the pattern
                        Temp_Pattern = Pattern.copy()
                        Temp_sub_nodes_value = [v for v in nx.get_node_attributes(Temp_Pattern, 'value').values()]
                        Temp_sub_nodes_value.remove(eventually_node)
                        Temp_pattern_length = len(Temp_sub_nodes_value)
                        Temp_Pattern.remove_node(list(Pattern.nodes)[-1])
                        # check the type of pattern
                        Temp_Parallel_pattern = len(Temp_Pattern.edges()) == 0 and len(Temp_Pattern.nodes()) > 1
                        Temp_Combined_pattern = self.check_consecutive_edges(Temp_Pattern)
                        Temp_sub_nodes_value = [v for v in nx.get_node_attributes(Temp_Pattern, 'value').values()]
                        search_for_eventually = 0
                        if Temp_Parallel_pattern:
                            collected_indexes = self.parallel_patterns_alignment(trace_node_value_exists,
                                                                                 Temp_sub_nodes_value,
                                                                                 Temp_pattern_length)
                        elif Temp_Combined_pattern:
                            collected_indexes = self.Combined_pattern_alignment(trace_node_value_exists, Temp_Pattern)
                        else:
                            collected_indexes = self.following_patterns_alignment(trace_node_value_exists,
                                                                                  Temp_sub_nodes_value,
                                                                                  Temp_pattern_length)

                        instance = 0
                        final_collected_indexes = dict()
                        for ins in collected_indexes:
                            if len(collected_indexes[ins]) > 0:
                                search_for_eventually = max(collected_indexes[ins]) + 2
                                while search_for_eventually < len(trace_node_value_exists) - 1 and \
                                        eventually_node in trace_node_value_exists[search_for_eventually:]:
                                    eventually_index = trace_node_value_exists.index(eventually_node,
                                                                                     search_for_eventually)
                                    final_collected_indexes[instance] = collected_indexes[ins].copy()
                                    final_collected_indexes[instance].append(eventually_index)
                                    instance += 1
                                    search_for_eventually = eventually_index + 1

                        collected_indexes = final_collected_indexes.copy()
                        self.remove_empty_lists(collected_indexes)

            patient_data.loc[patient_data[self.case_id] == case, pattern_name] = len(collected_indexes)
            # add a new row to DT_Case_Pattern for each pattern instance
            for instance in collected_indexes:
                for ind in range(len(collected_indexes[instance])):
                    pattern_instance = {self.case_id: case,
                                        'act': nx.get_node_attributes(self.Trace_graph, 'value').values().mapping[
                                            collected_indexes[instance][ind]], 'instance': instance}
                    # pattern_instance = {self.case_id: case,
                    #                     'act': nx.get_node_attributes(self.Trace_graph, 'value')[
                    #                         collected_indexes[instance][ind]], 'instance': instance}
                    event_features = nx.get_node_attributes(
                        self.Trace_graph, 'event_data').values().mapping[collected_indexes[instance][ind]]

                    # event_features = nx.get_node_attributes(
                    #     self.Trace_graph, 'event_data')[collected_indexes[instance][ind]]
                    for id, feature in enumerate(event_features):
                        pattern_instance["col%s" % id] = feature

                    dtime_total = 0
                    for idx in range(collected_indexes[instance][0], collected_indexes[instance][ind]):
                        try:
                            dtime_total += nx.get_edge_attributes(self.Trace_graph, 'dtime').values().mapping[
                                (idx, idx + 1)]
                        except:
                            continue

                    pattern_instance['dtime'] = dtime_total / 60

                    new_row = pd.DataFrame(pattern_instance, index=[0])
                    DT_Case_Pattern = pd.concat([DT_Case_Pattern, new_row], ignore_index=True)

        # DT_Case_Pattern.drop(['dtime'], axis=1, inplace=True)
        return patient_data, DT_Case_Pattern


    def pattern_annotation(self, patient_data, DT_Case_Pattern, pattern_name, feature='dependent',model='decision_tree'):
        if feature == 'dependent':
            X = DT_Case_Pattern.drop(columns=[self.case_id, self.outcome,self.likelihood])
            y = DT_Case_Pattern[self.outcome]
            # remove irrelevant features to pattern is all values are the same
            X = X.loc[:, (X != X.iloc[0]).any()]
            if len(y.unique()) < 2:
                return patient_data, None, None, None
            if len(X.columns) == 1 and len(X) > 1:
                if self.outcome_type == 'binary':
                    y_binary = y.apply(lambda x: 1 if x == 'deviant' else 0)
                    # y_binary = y
                    corr, p_value = pointbiserialr(X[X.columns[0]], y_binary)
                elif self.outcome_type == 'numerical':
                    corr, p_value = pearsonr(X[X.columns[0]], y)
                else:
                    raise ValueError('Unknown outcome type (binary or numerical)')

                if p_value < 0.05:
                    tree, rules, features_in_rule = self.train_decision_tree(X, y)
                    if self.outcome_type == 'binary':
                        classes_in_rules = [line.split('class: ')[1] for line in rules.split('\n') if 'class:' in line]
                    elif self.outcome_type == 'numerical':
                        classes_in_rules = [line.split('value: ')[1] for line in rules.split('\n') if 'value:' in line]
                    else:
                        raise ValueError('Unknown outcome type (binary or numerical)')
                    if len(np.unique(classes_in_rules)) < 2:
                        return patient_data, None, None, None

                    rule_indices = tree.tree_.apply(X[features_in_rule].values.astype('float32'))
                    result_df = pd.DataFrame({'rule_index': rule_indices})
                    result_df = pd.concat([DT_Case_Pattern, result_df], axis=1)

                    data_dependent_patterns = []
                    for ANP in np.sort(result_df['rule_index'].unique()):
                        aligned_cases = result_df[result_df['rule_index'] == ANP][self.case_id].tolist()
                        annotated_pattern = pattern_name + '+rule_' + str(ANP)
                        patient_data.loc[:, annotated_pattern] = 0
                        data_dependent_patterns.append(annotated_pattern)
                        for case in aligned_cases:
                            patient_data.loc[patient_data[self.case_id] == case, annotated_pattern] = \
                                aligned_cases.count(case)

                    # Leaves
                    self.children_left = tree.tree_.children_left
                    self.children_right = tree.tree_.children_right
                    self.feature = tree.tree_.feature
                    self.threshold = tree.tree_.threshold

                    leave_id = tree.apply(X[features_in_rule].values.astype('float32'))

                    paths = {}
                    for leaf in np.unique(leave_id):
                        path_leaf = []
                        self.find_path(0, path_leaf, leaf)
                        paths[leaf] = np.unique(np.sort(path_leaf))

                    rules_per_node = {}
                    for key in paths:
                        rules_per_node[key] = self.get_rule(paths[key], X[features_in_rule].columns)

                    return patient_data, data_dependent_patterns, pattern_name, rules_per_node
                else:
                    return patient_data, None, None, None

            if len(X.columns) > 1:
                minimum_case_per_feature = 0
                if len(X) >= minimum_case_per_feature * len(X.columns):
                    # Identify numerical and categorical columns
                    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
                    categorical_features = X.select_dtypes(include=['object']).columns


                    # One-hot encode the categorical features
                    if model == 'DT':
                        if len(categorical_features) > 0:
                            X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
                        # normalize the numerical features
                        # X[numerical_features] = (X[numerical_features] - X[numerical_features].mean()) / X[
                        #     numerical_features].std()
                        tree, rules, features_in_rule = self.train_decision_tree(X, y)
                        print(rules)
                    elif model == 'XGB':
                        for column in categorical_features:
                            X[column] = X[column].astype('category')
                        tree, rules, features_in_rule = self.train_decision_tree_xgboost(X, y)
                    if self.outcome_type == 'binary':
                        classes_in_rules = [line.split('class: ')[1] for line in rules.split('\n') if 'class:' in line]
                    elif self.outcome_type == 'numerical':
                        classes_in_rules = [line.split('value: ')[1] for line in rules.split('\n') if 'value:' in line]
                    else:
                        raise ValueError('Unknown outcome type (binary or numerical)')

                    if len(np.unique(classes_in_rules)) < 2:
                        return patient_data, None, None, None
                    data_dependent_patterns = []
                    rule_indices = tree.tree_.apply(X[features_in_rule].values.astype('float32'))
                    result_df = pd.DataFrame({'rule_index': rule_indices}, index=DT_Case_Pattern.index)
                    result_df = pd.concat([DT_Case_Pattern, result_df], axis=1)
                    for ANP in np.sort(result_df['rule_index'].unique()):
                        aligned_cases = result_df[result_df['rule_index'] == ANP][self.case_id].tolist()
                        annotated_pattern = pattern_name + '+rule_' + str(ANP)
                        patient_data.loc[:, annotated_pattern] = 0
                        data_dependent_patterns.append(annotated_pattern)
                        for case in aligned_cases:
                            patient_data.loc[patient_data[self.case_id] == case, annotated_pattern] = \
                                aligned_cases.count(case)

                    # Leaves
                    self.children_left = tree.tree_.children_left
                    self.children_right = tree.tree_.children_right
                    self.feature = tree.tree_.feature
                    self.threshold = tree.tree_.threshold

                    leave_id = tree.apply(X[features_in_rule].values.astype('float32'))

                    paths = {}
                    for leaf in np.unique(leave_id):
                        path_leaf = []
                        self.find_path(0, path_leaf, leaf)
                        paths[leaf] = np.unique(np.sort(path_leaf))

                    rules_per_node = {}
                    for key in paths:
                        rules_per_node[key] = self.get_rule(paths[key], X[features_in_rule].columns)

                    return patient_data, data_dependent_patterns, pattern_name, rules_per_node

                else:
                    return patient_data, None, None, None
            else:
                return patient_data, None, None, None

        elif feature == 'independent':
            return patient_data, None, None, None

        else:
            raise ValueError('Unknown feature dependency type (dependent or independent)')

    def find_path(self, node_numb, path, x):
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False
        if (self.children_left[node_numb] != -1):
            left = self.find_path(self.children_left[node_numb], path, x)
        if (self.children_right[node_numb] != -1):
            right = self.find_path(self.children_right[node_numb], path, x)
        if left or right:
            return True
        path.remove(node_numb)
        return False

    def get_rule(self, path, column_names):
        mask = ''
        for index, node in enumerate(path):
            # We check if we are not in the leaf
            if index != len(path) - 1:
                # Do we go under or over the threshold ?
                if (self.children_left[node] == path[index + 1]):
                    mask += "('{}'<= {}) \t ".format(column_names[self.feature[node]], self.threshold[node])
                else:
                    mask += "('{}'> {}) \t ".format(column_names[self.feature[node]], self.threshold[node])
        # We insert the & at the right places
        mask = mask.replace("\t", "&", mask.count("\t") - 1)
        mask = mask.replace("\t", "")
        return mask

    def train_decision_tree(self, X, y):
        # Split the data into training and test sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, y_train = X, y

        # Initial Decision Tree to identify important features
        if self.outcome_type == 'binary':
            initial_tree = DecisionTreeClassifier(random_state=42)
            initial_tree.fit(X_train, y_train)
        elif self.outcome_type == 'numerical':
            initial_tree = DecisionTreeRegressor(random_state=42)
            initial_tree.fit(X_train, y_train)
        else:
            raise ValueError('Unknown outcome type (binary or numerical)')

        # Feature selection
        selector = SelectFromModel(initial_tree, prefit=True, threshold="mean")
        X_train_selected = selector.transform(X_train)
        # X_test_selected = selector.transform(X_test)

        # Train the decision tree on selected features
        feature_names = [X.columns[i] for i in selector.get_support(indices=True)]
        if self.outcome_type == 'binary':
            final_tree = DecisionTreeClassifier(max_depth=len(feature_names)+1,
                                                min_samples_split=0.2, random_state=42, ccp_alpha=0.05)
        elif self.outcome_type == 'numerical':
            final_tree = DecisionTreeRegressor(max_depth=len(feature_names)+1,
                                               min_samples_split=0.2, random_state=42, ccp_alpha=0.05)
        # final_tree = DecisionTreeClassifier(max_depth=len(feature_names), min_samples_split=0.2, random_state=42)
        final_tree.fit(X_train_selected, y_train)

        # Export the tree to text format
        tree_rules = export_text(final_tree, feature_names=feature_names)

        return final_tree, tree_rules, feature_names
    '''
    def train_decision_tree_xgboost(self, X, y):
        # Split the data into training and test sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, y_train = X, y

        # Initial Decision Tree to identify important features
        if self.outcome_type == 'binary':
            initial_tree = XGBClassifier(random_state=42,enable_categorical=True)
            initial_tree.fit(X_train, y_train)
        elif self.outcome_type == 'numerical':
            initial_tree = XGBRegressor(random_state=42)
            initial_tree.fit(X_train, y_train)

        # Feature selection
        selector = SelectFromModel(initial_tree, prefit=True, threshold="mean")

        X_train_selected = selector.transform(X_train)
        # X_test_selected = selector.transform(X_test)

        # Train the decision tree on selected features
        feature_names = [X.columns[i] for i in selector.get_support(indices=True)]
        if self.outcome_type == 'binary':
            final_tree = XGBClassifier(n_estimators=1,max_depth=len(feature_names),
                                                #min_child_weight=0.2,
                                       random_state=42,enable_categorical=True
                                       #, ccp_alpha=0.05
                                       )

        elif self.outcome_type == 'numerical':
            final_tree = XGBRegressor(n_estimators=1, max_depth=len(feature_names),
                                               #min_samples_split=0.2,
                                      random_state=42,enable_categorical=True
                                      #ccp_alpha=0.05
                                      )
        # final_tree = DecisionTreeClassifier(max_depth=len(feature_names), min_samples_split=0.2, random_state=42)
        final_tree.fit(X_train[feature_names], y_train)

        # Export the tree to text format
        tree_rules = export_text(final_tree, feature_names=feature_names)

        return final_tree, tree_rules, feature_names
    '''
    def extract_leaves(self, tree_rules):
        leaves_list = []
        for line in tree_rules.split('\n'):
            if 'class:' in line:
                leaves_list.append(line.split('class:')[1].split(' ')[0])

        return leaves_list

    def calculate_gini_index_numerical(self, outcome, feature, threshold):

        left_mask = feature <= threshold
        right_mask = feature > threshold

        left_outcome = outcome[left_mask]
        right_outcome = outcome[right_mask]

        total_samples = len(outcome)

        # Gini index for the left node
        gini_left = 1 - sum((np.bincount(left_outcome) / len(left_outcome)) ** 2)

        # Gini index for the right node
        gini_right = 1 - sum((np.bincount(right_outcome) / len(right_outcome)) ** 2)

        # Weighted Gini index
        weighted_gini = (len(left_outcome) / total_samples) * gini_left + (
                len(right_outcome) / total_samples) * gini_right

        return weighted_gini


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Alignment Checking")
    parser.add_argument('--log_path', default='../full_filtered.csv', type=str)
    parser.add_argument('--pattern_folder', default='./output/full_filtered', type=str)
    parser.add_argument('--case_id', default='case:concept:name', type=str)
    parser.add_argument('--activity', default='concept:name', type=str)
    parser.add_argument('--timestamp', default='Complete Timestamp', type=str)
    parser.add_argument('--outcome', default='case:label', type=str)
    parser.add_argument('--delta_time', default=1, type=float, help='delta time in seconds')
    args = parser.parse_args()

    case_id = args.case_id
    activity = args.activity
    timestamp = args.timestamp
    outcome = args.outcome
    pattern_folder = args.pattern_folder
    delta_time = args.delta_time

    # Load the log
    df = pd.read_csv(args.log_path)
    df = df[[case_id, activity, timestamp, outcome]]
    df[activity] = df[activity].astype('string')
    df[activity] = df[activity].str.replace("_", "-")
    df[timestamp] = pd.to_datetime(df[timestamp])
    df[case_id] = df[case_id].astype('string')
    patient_data = df.drop_duplicates(subset=case_id, keep='first')

    color_codes = ["#" + ''.join([random.choice('000123456789ABCDEF') for i in range(6)])
                   for j in range(len(df[activity].unique()))]

    color_act_dict = dict()
    counter = 0
    for act in df[activity].unique():
        color_act_dict[act] = color_codes[counter]
        counter += 1
    color_act_dict['start'] = 'k'
    color_act_dict['end'] = 'k'

    EventLog_graphs = dict()
    for case in df[case_id].unique():
        case_data = df[df[case_id] == case]
        if case not in EventLog_graphs.keys():
            Trace_graph = Trace_graph_generator(df, delta_time,
                                                case, color_act_dict,
                                                case_id, activity, timestamp)

            EventLog_graphs[case] = Trace_graph.copy()
        else:
            Trace_graph = EventLog_graphs[case].copy()

    # Load the pattern
    Pattern_files = glob.glob(pattern_folder + '/*.pickle')
    Alignment_Check = Alignment_Checker(case_id, 'label')
    for pattern in Pattern_files:
        pattern_name = os.path.basename(pattern).split('.')[0]
        patient_data.loc[:, pattern_name] = 0
        Pattern = pickle.load(open(pattern, 'rb'))
        patient_data = Alignment_Check.check_pattern_alignment(EventLog_graphs, patient_data, Pattern, pattern_name)

    patient_data.to_csv(pattern_folder + '/Alignment_Checked_event_log.csv', index=False)


def alignment_check(log_df,case_id, activity, timestamp, outcome, pattern_folder, delta_time):

    # Load the log
    #df = pd.read_csv(args.log_path)
    df = log_df
    df = df[[case_id, activity, timestamp, outcome]]
    df[activity] = df[activity].astype('string')
    df[activity] = df[activity].str.replace("_", "-")
    df[timestamp] = pd.to_datetime(df[timestamp])
    df[case_id] = df[case_id].astype('string')
    patient_data = df.drop_duplicates(subset=case_id, keep='first')

    color_codes = ["#" + ''.join([random.choice('000123456789ABCDEF') for i in range(6)])
                   for j in range(len(df[activity].unique()))]

    color_act_dict = dict()
    counter = 0
    for act in df[activity].unique():
        color_act_dict[act] = color_codes[counter]
        counter += 1
    color_act_dict['start'] = 'k'
    color_act_dict['end'] = 'k'

    EventLog_graphs = dict()
    for case in df[case_id].unique():
        case_data = df[df[case_id] == case]
        if case not in EventLog_graphs.keys():
            Trace_graph = Trace_graph_generator(df, delta_time,
                                                case, color_act_dict,
                                                case_id, activity, timestamp)

            EventLog_graphs[case] = Trace_graph.copy()
        else:
            Trace_graph = EventLog_graphs[case].copy()

    # Load the pattern
    Pattern_files = glob.glob(pattern_folder + '/*.pickle')
    #remove dist_all.pickle, EventLogGraph.pickle, and color_dict.pickle
    Pattern_files = [x for x in Pattern_files if 'dist_all' not in x]
    Pattern_files = [x for x in Pattern_files if 'EventLogGraph' not in x]
    Pattern_files = [x for x in Pattern_files if 'color_dict' not in x]
    Alignment_Check = Alignment_Checker(case_id, outcome)
    for pattern in Pattern_files:
        pattern_name = os.path.basename(pattern).split('.')[0]
        patient_data[pattern_name] = 0
        Pattern = pickle.load(open(pattern, 'rb'))
        patient_data, _ = Alignment_Check.check_pattern_alignment(EventLog_graphs, patient_data, Pattern, pattern_name)

    return patient_data