import random

from pandas import Series, DataFrame


def randomise_features(feedback, train_df) -> DataFrame:
    randomised_df = train_df.copy()

    for classes in feedback:
        for single_feedback in feedback[classes]:
            target_columns = [column_name for column_name, _ in single_feedback]

            possible_values = {column: list(set(train_df[column].values)) for column in target_columns}

            randomised_df = randomised_df.apply(
                lambda row:
                    row if row['label'] != classes and any([row[column] != value for column, value in single_feedback])
                    else Series({
                        column:
                            random.choice(possible_values[column]) if column in target_columns
                            else value
                        for column, value in row.iteritems()
                    }),
                axis=1
            )

    randomised_df = randomised_df[train_df.columns]

    return randomised_df

