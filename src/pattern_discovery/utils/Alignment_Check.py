import argparse
import random
import networkx as nx
import pandas as pd
from nirdizati_light.pattern_discovery.utils.IMIPD import Trace_graph_generator
import glob
import os
import pickle


class Alignment_Checker:
    def __init__(self, case_id):
        self.case_id = case_id

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
                Patterns_instance_number.extend([(pattern, len(Direct_Patterns[pattern])) for pattern in Direct_Patterns])
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

            existence = 0
            for v in trace_nodes_value:
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

        return patient_data



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
    Alignment_Check = Alignment_Checker(case_id)
    for pattern in Pattern_files:
        pattern_name = os.path.basename(pattern).split('.')[0]
        patient_data[pattern_name] = 0
        Pattern = pickle.load(open(pattern, 'rb'))
        patient_data = Alignment_Check.check_pattern_alignment(EventLog_graphs, patient_data, Pattern, pattern_name)

    return patient_data