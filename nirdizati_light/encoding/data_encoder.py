import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder

PADDING_VALUE = 0
#onehot and minmaxscaler not fully done

class Encoder:
    def __init__(self, df: DataFrame = None, attribute_encoding=None,feature_selection=None,prefix_length=None):
        self.attribute_encoding = attribute_encoding
        self.feature_selection = feature_selection
        self.prefix_length = prefix_length
        self._label_encoder = {}
        self._numeric_encoder = {}
        self._label_dict = {}
        self._label_dict_decoder = {}
        self._scaled_values = {}
        self._unscaled_values = {}
        for column in df:
            if column != 'trace_id':
                if not is_numeric_dtype(df[column].dtype):#or (is_numeric_dtype(df[column].dtype) and np.any(df[column] < 0)):
                    #if column == 'prefix':
                    #    print('column:', column)
                    #else:
                    #    print('column:', column, 'considered NOT number, set values are:', set(tuple(row) for row in df[column]))
                    if attribute_encoding == 'label':
                        if column == 'label':
                            self._label_encoder[column] = LabelEncoder().fit(
                                sorted(df[column].apply(lambda x: str(x))))
                            classes = self._label_encoder[column].classes_
                            transforms = self._label_encoder[column].transform(classes)
                            self._label_dict[column] = dict(zip(classes, transforms))
                            self._label_dict_decoder[column] = dict(zip(transforms, classes))
                        else:
                            self._label_encoder[column] = LabelEncoder().fit(
                                sorted(pd.concat([pd.Series([str(PADDING_VALUE)]), df[column].apply(lambda x: str(x))])))
                            classes = self._label_encoder[column].classes_
                            transforms = self._label_encoder[column].transform(classes)
                            self._label_dict[column] = dict(zip(classes, transforms))
                            self._label_dict_decoder[column] = dict(zip(transforms, classes))
                    elif attribute_encoding == "onehot":
                        if column == 'label':
                            self._label_encoder[column] = LabelEncoder().fit(
                                sorted(df[column].apply(lambda x: str(x))))
                            classes = self._label_encoder[column].classes_
                            transforms = self._label_encoder[column].transform(classes)
                            self._label_dict[column] = dict(zip(classes, transforms))
                            self._label_dict_decoder[column] = dict(zip(transforms, classes))
                        else:
                            #padded_values = pd.concat([pd.Series([str(PADDING_VALUE)]), df[column].apply(lambda x: str(x))])
                            #label_enc = pd.DataFrame(LabelEncoder().fit_transform(sorted(padded_values)))
                            self._label_encoder[column] = OneHotEncoder(drop='if_binary', sparse_output=False,
                                       handle_unknown='ignore').fit(df[column].astype(str).values.reshape(-1,1))
                            categories = self._label_encoder[column].categories_[0].reshape(-1, 1)
                            transforms = [tuple(enc) for enc in self._label_encoder[column].transform(categories)]
                            classes = list(categories.flatten())
                            self._label_dict[column] = dict(zip(classes, transforms))
                            self._label_dict_decoder[column] = dict(zip(transforms, classes))

                else:
                    self._numeric_encoder[column] = MinMaxScaler().fit(
                        df[column].values.reshape(-1,1)
                    )
                    unscaled = df[column].values
                    scaled = self._numeric_encoder[column].transform(df[column].values.reshape(-1,1)).flatten()
                    self._scaled_values[column] = scaled
                    self._unscaled_values[column] = unscaled
                    print('column:', column, 'considered number, top 5 values are:', list(df[column][:5]))

    def encode(self, df: DataFrame) -> None:
        for column in df:
            if column != 'trace_id':
                if column in self._label_encoder:
                    try:
                        df[column] = df[column].apply(lambda x: self._label_dict[column].get(str(x), PADDING_VALUE))
                    except:
                        print('Error')
                else:
                    try:
                        df[column] = self._numeric_encoder[column].transform(df[column].values.reshape(-1,1)).flatten()
                    except:
                        print('Error')


    def decode(self, df: DataFrame) -> None:
        for column in df:
                if column != 'trace_id':
                        if column in self._label_encoder:
                            df[column] = df[column].apply(lambda x: self._label_dict_decoder[column].get(x, PADDING_VALUE))
                        else:
                            df[column] = self._numeric_encoder[column].inverse_transform(df[column].values.reshape(-1,1)).flatten()

    def decode_row(self, row) -> np.array:
        decoded_row = []
        for column, value in row.iteritems():
            if column != 'trace_id':
                if column in self._label_encoder:
                     decoded_row += [self._label_dict_decoder[column].get(value, PADDING_VALUE)]
                elif column in self._numeric_encoder:
                    decoded_row += [self._numeric_encoder[column].inverse_transform(np.array(value).reshape(-1,1))[0][0]]
            else:
                decoded_row += [value]
        return np.array(decoded_row)

    def decode_column(self, column, column_name) -> np.array:
        decoded_column = []
        if column != 'trace_id':
            if column_name in self._encoder:
                if not is_numeric_dtype(df[column].dtype):
                    decoded_column += [self._label_dict_decoder[column_name].get(x, PADDING_VALUE) for x in column]
                else:
                    decoded_column += [self._unscaled_values[column_name].get(x) for x in column]
        else:
            decoded_column += list(column)
        return np.array(decoded_column)

    def get_values(self, column_name):
        if not is_numeric_dtype(df[column].dtype):
            return (self._label_dict[column_name].keys(), self._label_dict_decoder[column_name].keys())
