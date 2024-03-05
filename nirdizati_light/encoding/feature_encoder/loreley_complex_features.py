from functools import reduce

from pandas import DataFrame
from pm4py.objects.log.obj import Trace, EventLog

from nirdizati_light.encoding.constants import get_max_prefix_length, get_prefix_length, TaskGenerationType, PrefixLengthStrategy
from nirdizati_light.labeling.common import add_label_column

ATTRIBUTE_CLASSIFIER = None

PREFIX_ = 'prefix_'
PREFIX = 'prefix'
single_prefix = True


def loreley_complex_features(log: EventLog, prefix_length, padding, prefix_length_strategy: str, labeling_type, generation_type, feature_list: list = None, target_event: str = None) -> DataFrame:
    max_prefix_length = get_max_prefix_length(log, prefix_length, prefix_length_strategy, target_event)
    columns, additional_columns = _columns_complex(log, max_prefix_length, feature_list)
    columns_number = len(columns)
    encoded_data = []
    for trace in log:
        trace_prefix_length = get_prefix_length(trace, prefix_length, prefix_length_strategy, target_event)
        if len(trace) <= prefix_length - 1 and not padding:
            # trace too short and no zero padding
            continue
        if generation_type == TaskGenerationType.ALL_IN_ONE.value:
            for event_index in range(1, min(trace_prefix_length + 1, len(trace) + 1)):
                encoded_data.append(_trace_to_row(trace, event_index, additional_columns, prefix_length_strategy, padding, columns, labeling_type))
        else:
            encoded_data.append(_trace_to_row(trace, trace_prefix_length, additional_columns, prefix_length_strategy, padding, columns, labeling_type))
    #change prefiz_i to prefix and update feature lsit
    return DataFrame(columns=columns, data=encoded_data)


def _get_global_trace_attributes(log: EventLog):
    # retrieves all traces in the log and returns their intersection
    attributes = list(reduce(set.intersection, [set(trace._get_attributes().keys()) for trace in log]))
    trace_attributes = [attr for attr in attributes if attr not in ["concept:name", "time:timestamp", "label"]]
    return sorted(trace_attributes)


def _get_global_event_attributes(log):
    """Get log event attributes that are not name or time
    """
    # retrieves all events in the log and returns their intersection
    attributes = list(reduce(set.intersection, [set(event._dict.keys()) for trace in log for event in trace]))
    event_attributes = [attr for attr in attributes if attr not in ["concept:name"]]
    return sorted(event_attributes)


def _compute_additional_columns(log) -> dict:
    return {'trace_attributes': _get_global_trace_attributes(log),
            'event_attributes': _get_global_event_attributes(log)}


def _columns_complex(log, prefix_length: int, feature_list: list = None) -> tuple:
    additional_columns = _compute_additional_columns(log)
    columns = ['trace_id']
    columns += additional_columns['trace_attributes']
    for i in range(1, prefix_length + 1):
        for additional_column in additional_columns['event_attributes']:
            columns.append(additional_column + "_" + str(i))
    columns.insert(len(columns), PREFIX)

    columns += ['label']
    if feature_list is not None:
        assert(list(feature_list) == columns)
    return columns, additional_columns


def _data_complex(trace: Trace, prefix_length: int, additional_columns: dict) -> list:
    """Creates list in form [1, value1, value2, 2, ...]

    Appends values in additional_columns
    """
    event_list = []
    data = [trace.attributes.get(att, 0) for att in additional_columns['trace_attributes']]
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = event["concept:name"]
        event_list.append(event_name)

        for att in additional_columns['event_attributes']:
            data.append(event.get(att, '0'))
    data.append(event_list)
    return data


def _trace_to_row(trace: Trace, prefix_length: int, additional_columns, prefix_length_strategy: str, padding, columns: list, labeling_type) -> list:
    trace_row = [trace.attributes["concept:name"]]
    trace_row += _data_complex(trace, prefix_length, additional_columns)
    if padding or prefix_length_strategy == PrefixLengthStrategy.PERCENTAGE.value:
        trace_row += [0 for _ in range(len(trace_row), len(columns) - 1)]
    trace_row += [add_label_column(trace, labeling_type, prefix_length)]
    return trace_row
