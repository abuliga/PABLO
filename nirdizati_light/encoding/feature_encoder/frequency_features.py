from collections import Counter
from datetime import timedelta

from pandas import DataFrame
from pm4py.objects.log.obj import EventLog, Trace, Event

from nirdizati_light.encoding.constants import get_max_prefix_length, get_prefix_length, TaskGenerationType
from nirdizati_light.labeling.common import add_label_column

PREFIX_ = 'prefix_'


def frequency_features(log: EventLog, prefix_length, padding, prefix_length_strategy: str, labeling_type, generation_type, feature_list: list = None, target_event: str = None) -> DataFrame:
    if feature_list is None:
        max_prefix_length = get_max_prefix_length(log, prefix_length, prefix_length_strategy, target_event)
        feature_list = _compute_columns(log, max_prefix_length, padding)
    encoded_data = []
    for trace in log:
        trace_prefix_length = get_prefix_length(trace, prefix_length, prefix_length_strategy, target_event)
        if len(trace) <= prefix_length - 1 and not padding:
            # trace too short and no zero padding
            continue
        if generation_type == TaskGenerationType.ALL_IN_ONE.value:
            for event_index in range(1, min(trace_prefix_length + 1, len(trace) + 1)):
                encoded_data.append(_trace_to_row(trace, event_index, feature_list, padding, labeling_type))
        else:
            encoded_data.append(_trace_to_row(trace, trace_prefix_length, feature_list, padding, labeling_type))

    return DataFrame(columns=feature_list, data=encoded_data)


def _compute_columns(log: EventLog, prefix_length: int, padding: bool) -> list:
    """trace_id, prefixes, any other columns, label

    """
    ret_val = ["trace_id"]
    ret_val += sorted(list({
       event['concept:name']
       for trace in log
       for event in trace[:prefix_length]
    }))
    ret_val += ['0'] if padding else []
    ret_val += ['label']

    return ret_val


def _trace_to_row(trace: Trace, prefix_length: int, columns: list, padding: bool = True, labeling_type: str = None) -> list:
    """Row in data frame"""
    trace_row = [ trace.attributes['concept:name'] ]

    if len(trace) <= prefix_length - 1 and not padding:
        pass
        trace += [
            Event({
                'concept:name': '0',
                'time:timestamp': trace[len(trace)] + timedelta(hours=i)
            })
            for i in range(len(trace), prefix_length + 1)
        ]

    occurences = Counter([
        event['concept:name']
        for event in trace[:prefix_length]
    ])
    cleaned_comumns = columns[1:-1]
    trace_row += [ occurences[col] for col in cleaned_comumns ]
    trace_row += [ add_label_column(trace, labeling_type, prefix_length) ]
    return trace_row

