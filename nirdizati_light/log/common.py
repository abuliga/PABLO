import logging

import pm4py
from pandas import read_csv,DataFrame
from pm4py import read_xes,format_dataframe,convert_to_event_log
logger = logging.getLogger(__name__)


def import_log_csv(path, dataset_confs):
    dataframe = read_csv(path,sep=';')
    dataframe = format_dataframe(dataframe, case_id=[*dataset_confs.case_id_col.values()][0],
                                       activity_key=[*dataset_confs.activity_col.values()][0],
                                       timestamp_key=[*dataset_confs.timestamp_col.values()][0])
    event_log = convert_to_event_log(dataframe)
    return event_log




def get_log(filepath, dataset_confs):
    """Read in event log from disk
    Uses xes_importer to parse log.
    """
    logger.info("\t\tReading in log from {}".format(filepath))
    # uses the xes, or csv importer depending on file type
    if filepath.endswith('.csv'):
        event_log = import_log_csv(filepath, dataset_confs)
    elif filepath.endswith('.xes'):
        event_log = read_xes(filepath)
    if isinstance(event_log,DataFrame):
        event_log = convert_to_event_log(event_log)
    return event_log
