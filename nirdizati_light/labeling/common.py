from enum import Enum
from nirdizati_light.utils.log_metrics import events_by_date, resources_by_date, new_trace_start
from nirdizati_light.utils.time_metrics import elapsed_time_id, remaining_time_id, count_on_event_day, duration
from pm4py.objects.log.obj import EventLog, Trace

def get_intercase_attributes(log, encoding):
    """Dict of kwargs
    These intercase attributes are expensive operations!!!
    """
    # Expensive operations
    executed_events = events_by_date(log) if encoding.add_executed_events else None
    resources_used = resources_by_date(log) if encoding.add_resources_used else None
    new_traces = new_trace_start(log) if encoding.add_new_traces else None
    kwargs = {'executed_events': executed_events, 'resources_used': resources_used, 'new_traces': new_traces}
    # 'label': label}  TODO: is it really necessary to add this field in the dict?
    return kwargs

class LabelTypes(Enum):
    NEXT_ACTIVITY = 'next_activity'
    ATTRIBUTE_STRING = 'label_attribute_string'
    REMAINING_TIME = 'remaining_time'
    DURATION = 'duration'
    NO_LABEL = 'no_label'

class ThresholdTypes(Enum):
    THRESHOLD_MEAN = 'threshold_mean'
    THRESHOLD_CUSTOM = 'threshold_custom'
    NONE = 'none'


def add_label_column(trace, labeling_type, prefix_length: int):
    """TODO COMMENT ME
    """
    if labeling_type == LabelTypes.NEXT_ACTIVITY.value:
        return next_event_name(trace, prefix_length)
    elif labeling_type == LabelTypes.ATTRIBUTE_STRING.value:
        return trace.attributes['label']
    elif labeling_type == LabelTypes.REMAINING_TIME.value:
        return remaining_time_id(trace, prefix_length)
    elif labeling_type == LabelTypes.DURATION.value:
        return duration(trace)
    elif labeling_type == LabelTypes.NO_LABEL.value:
        return 0
    else:
        raise Exception('Label not set please select one of LabelTypes(Enum) values!')




def next_event_name(trace: list, prefix_length: int):
    """Return the event event_name at prefix length or 0 if out of range.

    """
    if prefix_length < len(trace):
        next_event = trace[prefix_length]
        name = next_event['concept:name']
        return name
    else:
        return 0


