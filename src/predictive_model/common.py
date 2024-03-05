from enum import Enum

import numpy as np
from funcy import flatten
from pandas import DataFrame
from nirdizati_light.encoding.common import EncodingType

class ClassificationMethods(Enum):
    RANDOM_FOREST = 'randomForestClassifier'
    KNN = 'knn'
    XGBOOST = 'xgboost'
    SGDCLASSIFIER = 'SGDClassifier'
    PERCEPTRON = 'perceptron'
    LSTM = 'lstm'
    MLP = 'mlp'
    SVM = 'svc'
    DT = 'DecisionTree'


class RegressionMethods(Enum):
    RANDOM_FOREST = 'randomForestRegressor'


def get_tensor(CONF, df: DataFrame):
    #trace_attributes = [att for att in df.columns if '_' not in att]
    if CONF['feature_selection'] == EncodingType.SIMPLE_TRACE.value:
        trace_attributes = [att for att in df.columns if 'prefix' not in att]
    else:
        trace_attributes = [att for att in df.columns if 'prefix' not in att and '_' not in att]
    event_attributes = [att[:-2] for att in df.columns if att[-2:] == '_1']

    reshaped_data = {
            trace_index: {
                prefix_index:
                    list(flatten(
                        feat_values if isinstance(feat_values, tuple) else [feat_values]
                        for feat_name, feat_values in trace.items()
                        if feat_name in trace_attributes + [event_attribute + '_' + str(prefix_index) for event_attribute in event_attributes]
                    ))
                for prefix_index in range(1, CONF['prefix_length'] + 1)
            }
            for trace_index, trace in df.iterrows()
    }
    #        for prefix in reshaped_data[trace]:
            #   for trace in reshaped_data:
                #   len(reshaped_data[trace][prefix])
    flattened_features = max(
        len(reshaped_data[trace][prefix])
        for trace in reshaped_data
        for prefix in reshaped_data[trace]
    )

    tensor = np.zeros((
        len(df),                # sample
        CONF['prefix_length'],  # time steps
        flattened_features      # features x single time step (trace and event attributes)
    ))
    keys = list(reshaped_data.keys())
    for trace_index in reshaped_data:  # prefix
        for prefix_index in reshaped_data[trace_index]:  # steps of the prefix
            for single_flattened_value in range(len(reshaped_data[trace_index][prefix_index])):
                try:
                    tensor[keys.index(trace_index), prefix_index - 1, single_flattened_value] = reshaped_data[trace_index][prefix_index][single_flattened_value]
                except IndexError:
                    print(f'IndexError: trace_index={trace_index}, prefix_index={prefix_index}, single_flattened_value={single_flattened_value}')
                    print(f'len(reshaped_data[trace_index][prefix_index])={len(reshaped_data[trace_index][prefix_index])}')
                    print(f'tensor.shape={tensor.shape}')
                    raise

    return tensor

def shape_label_df(df: DataFrame):

    labels_list = df['label'].tolist()
    labels = np.zeros((len(labels_list), int(max(df['label'].nunique(), int(max(df['label'].values))) +1)))
    for label_idx, label_val in enumerate(labels_list):
        labels[int(label_idx), int(label_val)] = 1

    return labels