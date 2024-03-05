import logging
from enum import Enum

from pandas import DataFrame
from pm4py.objects.log.obj import EventLog, Trace, Event

from nirdizati_light.encoding.data_encoder import Encoder
from nirdizati_light.encoding.feature_encoder.complex_features import complex_features
from nirdizati_light.encoding.feature_encoder.frequency_features import frequency_features
from nirdizati_light.encoding.feature_encoder.loreley_complex_features import loreley_complex_features
from nirdizati_light.encoding.feature_encoder.loreley_features import loreley_features
from nirdizati_light.encoding.feature_encoder.simple_features import simple_features
from nirdizati_light.encoding.feature_encoder.simple_trace_features import simple_trace_features
from nirdizati_light.encoding.feature_encoder.binary_features import binary_features
from nirdizati_light.encoding.time_encoding import time_encoding

logger = logging.getLogger(__name__)


class EncodingType(Enum):
    SIMPLE = 'simple'
    FREQUENCY = 'frequency'
    COMPLEX = 'complex'
    DECLARE = 'declare'
    LORELEY = 'loreley'
    LORELEY_COMPLEX = 'loreley_complex'
    SIMPLE_TRACE = 'simple_trace'
    BINARY = 'binary'

class EncodingTypeAttribute(Enum):
    LABEL = 'label'
    ONEHOT = 'onehot'


TRACE_TO_DF = {
    EncodingType.SIMPLE.value : simple_features,
    EncodingType.FREQUENCY.value : frequency_features,
    # EncodingType.FREQUENCY.value : frequency_features,
    EncodingType.COMPLEX.value : complex_features,
    # EncodingType.DECLARE.value : declare_features,
    EncodingType.LORELEY.value: loreley_features,
    EncodingType.LORELEY_COMPLEX.value: loreley_complex_features,
    EncodingType.SIMPLE_TRACE.value: simple_trace_features,
    EncodingType.BINARY.value: binary_features,
}


def get_encoded_df(log: EventLog, CONF: dict=None, encoder: Encoder=None, train_cols: DataFrame=None, train_df=None) -> (Encoder, DataFrame):
    logger.debug('SELECT FEATURES')
    df = TRACE_TO_DF[CONF['feature_selection']](
        log,
        prefix_length=CONF['prefix_length'],
        padding=CONF['padding'],
        prefix_length_strategy=CONF['prefix_length_strategy'],
        labeling_type=CONF['labeling_type'],
        generation_type=CONF['task_generation_type'],
        feature_list=train_cols,
        target_event=CONF['target_event'],
    )

    logger.debug('EXPLODE DATES')
    df = time_encoding(df, CONF['time_encoding'])

    logger.debug('ALIGN DATAFRAMES')
    if train_df is not None:
        _, df = train_df.align(df, join='left', axis=1)

    if not encoder:
        logger.debug('INITIALISE ENCODER')
        encoder = Encoder(df=df, attribute_encoding=CONF['attribute_encoding'],feature_selection=CONF['feature_selection'],
                          prefix_length=CONF['prefix_length'])
    logger.debug('ENCODE')
    encoder.encode(df=df)

    return encoder, df
