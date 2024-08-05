import logging

import pm4py
from pandas import read_csv,DataFrame
from pm4py import read_xes,format_dataframe,convert_to_event_log
logger = logging.getLogger(__name__)


def import_log_csv(path):
    dataframe = read_csv(path,sep=',')
    dataframe = format_dataframe(dataframe, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    event_log = convert_to_event_log(dataframe)
    return event_log




def get_log(filepath, dataset_confs):
    """Read in event log from disk
    Uses xes_importer to parse log.
    Outputs event log in pm4py format.
    """
    logger.info("\t\tReading in log from {}".format(filepath))

    # uses the xes, or csv importer depending on file type
    # event_log = read_xes(filepath)
    event_log = read_csv(filepath, sep=',')
    event_log = pm4py.format_dataframe(event_log, case_id=[*dataset_confs.case_id_col.values()][0],
                                       activity_key=[*dataset_confs.activity_col.values()][0],
                                       timestamp_key=[*dataset_confs.timestamp_col.values()][0])
    if isinstance(event_log,DataFrame):
        event_log = convert_to_event_log(event_log)
    return event_log
