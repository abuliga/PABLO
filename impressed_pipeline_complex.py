import logging
import warnings
import os
import numpy as np
import pandas as pd
import pm4py
from sklearn.model_selection import train_test_split
from nirdizati_light.encoding.common import get_encoded_df, EncodingType
from nirdizati_light.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from nirdizati_light.encoding.time_encoding import TimeEncodingType
from nirdizati_light.evaluation.common import evaluate_classifier
from nirdizati_light.explanation.common import ExplainerType, explain
from nirdizati_light.explanation.wrappers.dice_impressed import model_discovery
from nirdizati_light.pattern_discovery.common import discovery
from nirdizati_light.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from nirdizati_light.labeling.common import LabelTypes
from nirdizati_light.log.common import get_log, import_log_csv
from nirdizati_light.predictive_model.common import ClassificationMethods, get_tensor
from nirdizati_light.predictive_model.predictive_model import PredictiveModel, drop_columns
import random
from dataset_confs import DatasetConfs
import pickle
import dtreeviz
from declare4py.declare4py import Declare4Py
from declare4py.enums import TraceState
from datetime import datetime
from sklearn import tree
from nirdizati_light.pattern_discovery.utils.Alignment_Check import alignment_check
import itertools
import shutil

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
import category_encoders as ce


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def run_simple_pipeline(CONF=None, dataset_name=None):
    random.seed(CONF['seed'])
    np.random.seed(CONF['seed'])
    # dataset = CONF['data'].rpartition('/')[0].replace('datasets/', '')

    dataset_confs = DatasetConfs(dataset_name=dataset_name, where_is_the_file=CONF['data'])

    logger.debug('LOAD DATA')
    log = get_log(filepath=CONF['data'],dataset_confs=dataset_confs)
    logger.debug('Update EVENT ATTRIBUTES')

    logger.debug('ENCODE DATA')
    encoder, full_df = get_encoded_df(log=log, CONF=CONF)

    logger.debug('TRAIN PREDICTIVE MODEL')
    train_size = CONF['train_val_test_split'][0]
    val_size = CONF['train_val_test_split'][1]
    test_size = CONF['train_val_test_split'][2]
    if train_size + val_size + test_size != 1.0:
        raise Exception('Train-val-test split does not sum up to 1')

    train_df, val_df, test_df = np.split(full_df,
                                         [int(train_size * len(full_df)), int((train_size + val_size) * len(full_df))])


    encoder.decode(full_df)
    cols = [*dataset_confs.static_cat_cols.values(), *dataset_confs.dynamic_cat_cols.values()]
    ohe_encode = list(itertools.chain.from_iterable(cols))
    cols_to_encode = [col for feat in ohe_encode for col in full_df.columns if feat in col]
    enc_ohe = ce.one_hot.OneHotEncoder(
        cols=cols_to_encode, use_cat_names=True)
    enc_ohe.fit(full_df)
    encoder.encode(full_df)

    log_conf = CONF.copy()
    log_conf['feature_selection'] = EncodingType.COMPLEX.value
    complex_encoder, full_df_timestamps = get_encoded_df(log=log, CONF=log_conf)
    train_df_alignment, val_df_alignment, test_df_alignment = np.split(full_df_timestamps,
                                                                       [int(train_size * len(full_df_timestamps)),
                                                                        int((train_size + val_size) * len(
                                                                            full_df_timestamps))])

    complex_encoder.decode(full_df_timestamps)
    complex_encoder.decode(test_df_alignment)

    predictive_model = PredictiveModel(CONF, CONF['predictive_model'], train_df, val_df)
    if CONF['hyperparameter_optimisation']:
        predictive_model.model, predictive_model.config = retrieve_best_model(
            predictive_model,
            CONF['predictive_model'],
            max_evaluations=CONF['hyperparameter_optimisation_epochs'],
            target=CONF['hyperparameter_optimisation_target'], seed=CONF['seed']
        )
    logger.debug('EVALUATE PREDICTIVE MODEL')
    if predictive_model.model_type is ClassificationMethods.LSTM.value:
        probabilities = predictive_model.model.predict(get_tensor(CONF, drop_columns(test_df)))
        predicted = np.argmax(probabilities, axis=1)
        scores = np.amax(probabilities, axis=1)
    elif predictive_model.model_type not in (ClassificationMethods.LSTM.value):
        predicted = predictive_model.model.predict(drop_columns(test_df))
        scores = predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]

    actual = test_df['label']
    if predictive_model.model_type is ClassificationMethods.LSTM.value:
        actual = np.array(actual.to_list())

    initial_result = evaluate_classifier(actual, predicted, scores)
    model_path = 'results/process_models'
    logger.debug('COMPUTE EXPLANATION')
    if CONF['explanator'] is ExplainerType.DICE_IMPRESSED.value:
        impressed_pipeline = CONF['impressed_pipeline']
        test_df_correct = test_df[(test_df['label'] == predicted) & (test_df['label'] == 0)]
        method = 'oneshot'
        optimization = 'genetic'
        diversity = 1.0
        sparsity = 0.5
        proximity = 1.0
        timestamp = [*dataset_confs.timestamp_col.values()][0]
        neighborhood_size = 75
        hit_rate = None
        if CONF['feature_selection'] == EncodingType.COMPLEX.value:
            dynamic_cols = list(itertools.chain(
                dataset_confs.activity_col.values(),
                itertools.chain.from_iterable(dataset_confs.dynamic_cat_cols.values()),
                itertools.chain.from_iterable(dataset_confs.dynamic_num_cols.values()),
                [timestamp]
            ))
        else:
            dynamic_cols = [*dataset_confs.activity_col.values()] + [timestamp]

        for x in range(len(test_df_correct.iloc[:50, :])):
            query_instance = test_df_correct.iloc[x, :].to_frame().T
            case_id = query_instance.iloc[0, 0]
            query_instance = query_instance.drop(columns=['trace_id'])
            if CONF['feature_selection'] == EncodingType.SIMPLE_TRACE.value:
                output_path = 'results/simple_trace_results_3_objectives/'
                #output_path = 'results/simple_trace_results_4_objectives/'
            elif CONF['feature_selection'] == EncodingType.COMPLEX.value:
                output_path = 'results/complex_results_3_objectives/'
                #output_path = 'results/complex_results_4_objectives/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            discovery_path = output_path + '%s_discovery_%s_%s_%s' % (
            dataset, impressed_pipeline, CONF['seed'], case_id)

            columns = drop_columns(train_df).columns
            features_to_vary = [column for column in columns if 'Timestamp' not in column]
            if len(features_to_vary) == 0:
                features_to_vary = [column for column in columns if 'time' not in column]

            timestamps = [col for col in full_df_timestamps.iloc[query_instance.index].columns if 'timestamp' in col]
            if len(timestamps) == 0:
                timestamps = [col for col in full_df_timestamps.iloc[query_instance.index].columns if
                              'Timestamp' in col]
            df = full_df_timestamps.loc[query_instance.index][timestamps].reset_index()
            timestamps_query = pd.DataFrame(np.repeat(df.values, neighborhood_size * 2, axis=0))
            timestamps_query.columns = df.columns
            timestamps_query.drop(columns=['index'], inplace=True)
            time_start = datetime.now()
            synth_log, x_eval, label_list = explain(CONF, predictive_model, encoder=encoder, cf_df=full_df.iloc[:, 1:],
                                                    query_instance=query_instance,
                                                    method=method, optimization=optimization,
                                                    timestamp_col_name=timestamp,
                                                    model_path=model_path, random_seed=CONF['seed'],
                                                    neighborhood_size=neighborhood_size
                                                    , sparsity_weight=sparsity,
                                                    diversity_weight=diversity, proximity_weight=proximity,
                                                    features_to_vary=features_to_vary,
                                                    impressed_pipeline=impressed_pipeline,
                                                    dynamic_cols=dynamic_cols, timestamps=timestamps_query)
            time_cf = (datetime.now() - time_start).total_seconds()

            logger.debug('RUN IMPRESSED DISCOVERY AND DECISION TREE PIPELINE')
            if impressed_pipeline:
                discovery_type = 'auto'
                case_id_col = 'case:concept:name'
                activity = 'concept:name'
                outcome = 'case:label'
                outcome_type = 'binary'
                delta_time = -1
                max_gap = CONF['prefix_length'] // 3
                max_extension_step = 2
                testing_percentage = 0.2
                factual_outcome = query_instance['label'].values[0]
                likelihood = 'likelihood'
                encoding = True
                discovery_algorithm = 'impressed'
                extension_style = 'Pareto'
                model = 'DT'
                pattern_extension_strategy = 'activities'  # 'activities', 'attributes'
                aggregation_style = 'all'  # 'all', 'none', 'pareto', 'mix
                frequency_type = 'relative'  # 'absolute', 'relative'
                distance_style = 'all'  # 'case' or 'all'
                data_dependency = 'independent'
                time_start = datetime.now()
                trace_encoding = CONF['feature_selection']
                only_event_attributes = False

                train_X, test_X = discovery(discovery_algorithm, synth_log, discovery_path, discovery_type, case_id_col,
                                            activity, timestamp, outcome,
                                            outcome_type, delta_time,
                                            max_gap, max_extension_step, factual_outcome, likelihood, encoding,
                                            testing_percentage, extension_style, data_dependency,
                                            model, pattern_extension_strategy, aggregation_style, frequency_type,
                                            distance_style, trace_encoding, only_event_attributes)
                test_ids = test_X['Case_ID'].unique()
                if 'BPIC17' in dataset:
                    synth_log['case:label'].replace({0: 'deviant', 1: 'regular'}, inplace=True)
                else:
                    synth_log['case:label'].replace({0: 'false', 1: 'true'}, inplace=True)
                time_discovery = (datetime.now() - time_start).total_seconds()

                #### Update trace attributes
                synth_log = synth_log.drop(columns=['likelihood'])
                event_log_pred = pm4py.convert_to_event_log(synth_log)
                cols = [*dataset_confs.static_cat_cols.values(), *dataset_confs.static_num_cols.values()]
                to_remove = list(itertools.chain.from_iterable(cols))


                pm4py.write_xes(event_log_pred, output_path + '/synthetic_log.xes')
                for i in range(len(event_log_pred)):
                    for x in to_remove:
                        event_log_pred[i].attributes.update({x: event_log_pred[i][1][x]
                                                             })
                        for j in range(len(event_log_pred[i])):
                            del event_log_pred[i][j]._dict[x]
                _, synth_df = get_encoded_df(log=event_log_pred, CONF=CONF, encoder=encoder)
                encoder.decode(synth_df)

                test = synth_df[synth_df['trace_id'].isin(test_ids)]
                encoder.encode(test)

                synth_df = enc_ohe.transform(synth_df)
                train_X = train_X.rename(columns={'Case_ID': 'trace_id', 'Outcome': 'label'})
                test_X = test_X.rename(columns={'Case_ID': 'trace_id', 'Outcome': 'label'})

                if discovery_type == 'interactive':
                    train_X = train_X.rename(columns={'case:concept:name': 'trace_id', 'case:label': 'label'})
                    test_X = test_X.rename(columns={'case:concept:name': 'trace_id', 'case:label': 'label'})
                synth_df_subset = synth_df.drop(
                    columns=[col for col in synth_df.columns if 'prefix' in col] + ['label'])
                try:
                    train_X['label'] = train_X['label'].astype(int)
                    test_X['label'] = test_X['label'].astype(int)
                except:
                    print('Not possible to convert to int')
                update_train_X = pd.merge(synth_df_subset, train_X, on='trace_id', how='left')
                update_train_X = update_train_X.dropna()
                update_train_X['label'] = update_train_X['label'].map(int)
                update_test_X = pd.merge(synth_df_subset, test_X, on='trace_id', how='left')
                update_test_X = update_test_X.dropna()
                update_test_X['label'] = update_test_X['label'].map(int)
                #### Update trace attributes

                DT_CONF = CONF.copy()
                DT_CONF['predictive_model'] = ClassificationMethods.DT.value
                DT_CONF['hyperparameter_optimisation_target'] = HyperoptTarget.F1.value
                print('df columns', update_train_X.columns)
                glass_box = PredictiveModel(DT_CONF, DT_CONF['predictive_model'], update_train_X, update_test_X)
                if DT_CONF['hyperparameter_optimisation']:
                    glass_box.model, glass_box.config = retrieve_best_model(
                        glass_box,
                        DT_CONF['predictive_model'],
                        max_evaluations=DT_CONF['hyperparameter_optimisation_epochs'],
                        target=DT_CONF['hyperparameter_optimisation_target'], seed=DT_CONF['seed']
                    )
                glass_box_result = evaluate_classifier(update_test_X['label'], glass_box.model.predict(
                    np.array(drop_columns(update_test_X))),
                    glass_box.model.predict_proba(np.array(drop_columns(update_test_X))))
                local_fidelity = glass_box_result['accuracy']
                print('Local fidelity',local_fidelity)

                logger.debug("EVALUATE GLOBAL GLASS-BOX MODEL")
                encoder.decode(test_df)
                test_df_dummy = enc_ohe.transform(test_df)
                test_df_dummy_subset = test_df_dummy.drop(columns=[col for col in test_df_dummy.columns if 'prefix' in col]+['label'])
                stub_cols = ['prefix', timestamp]+dynamic_cols
                test_log_df = pd.wide_to_long(test_df_alignment, stubnames=dynamic_cols, i='trace_id',
                                              j='order', sep='_', suffix=r'\w+').reset_index()
                start_time = datetime.now()
                impressed_test_df = alignment_check(log_df=test_log_df,case_id='trace_id',timestamp=timestamp,activity='prefix',
                                                    outcome='label',pattern_folder=discovery_path,delta_time=delta_time)
                time_alignment = (datetime.now() - start_time).total_seconds()
                impressed_test_df.drop(columns=[timestamp, 'prefix'], inplace=True)
                update_impressed_test_df = pd.merge(impressed_test_df, test_df_dummy_subset,
                                                    on='trace_id', how='left')

                update_impressed_test_df.dropna(inplace=True)
                update_impressed_test_df = update_impressed_test_df[(drop_columns(update_test_X).columns)]
                global_preds = glass_box.model.predict(update_impressed_test_df)
                global_probs = glass_box.model.predict_proba(update_impressed_test_df)
                encoder.encode(test_df)
                predicted = predictive_model.model.predict(drop_columns(test_df))
                pred_evaluate_glassbox = evaluate_classifier(predicted, global_preds.astype(int), global_probs)
                real_evaluate_glassbox = evaluate_classifier(actual, global_preds.astype(int), global_probs)
                global_fidelity = pred_evaluate_glassbox['accuracy'] if pred_evaluate_glassbox['accuracy'] > real_evaluate_glassbox['accuracy'] else real_evaluate_glassbox['accuracy']

                if (local_fidelity > 0.9)  |  (global_fidelity > 0.8):
                    viz = dtreeviz.model(glass_box.model,
                                         drop_columns(update_train_X),
                                         update_train_X['label'],
                                         feature_names=drop_columns(update_train_X).columns,
                                         class_names=['false', 'true'],

                                         )
                    v = viz.view(orientation="LR", scale=2, label_fontsize=5.5)
                    v.save(
                        output_path + 'decision_trees' + '/' + '%s_impressed_encoding_%s_%s_%s' % (
                            dataset, case_id, CONF['prefix_length'],data_dependency+'_complex_index') + '.svg')

                shutil.rmtree(discovery_path)
            else:
                max_extension_step = 0
                testing_percentage = 0.2
                synth_log.drop(columns=['likelihood'], inplace=True)
                event_log_pred = pm4py.convert_to_event_log(synth_log)
                cols = [*dataset_confs.static_cat_cols.values(), *dataset_confs.static_num_cols.values()]
                to_remove = list(itertools.chain.from_iterable(cols))
                for i in range(len(event_log_pred)):
                    for x in to_remove:
                        event_log_pred[i].attributes.update({x: event_log_pred[i][1][x]
                                                             })
                        for j in range(len(event_log_pred[i])):
                            del event_log_pred[i][j]._dict[x]

                frequency_conf = CONF.copy()
                frequency_conf['feature_selection'] = EncodingType.FREQUENCY.value
                frequency_encoder, frequency_full_df = get_encoded_df(log=log, CONF=frequency_conf)
                _, synth_df = get_encoded_df(log=event_log_pred, CONF=frequency_conf, encoder=frequency_encoder)
                _, synth_df_trace = get_encoded_df(log=event_log_pred, CONF=CONF, encoder=encoder)
                encoder.decode(synth_df_trace)
                synth_df_trace = enc_ohe.transform(synth_df_trace)

                frequency_encoder.decode(synth_df)
                synth_df_subset = synth_df_trace.drop(
                    columns=[col for col in synth_df_trace.columns if 'prefix' in col] + ['label'])
                synth_df = pd.merge(synth_df_subset, synth_df, on='trace_id', how='left')

                train, test = train_test_split(synth_df, test_size=testing_percentage, random_state=42,
                                               stratify=synth_df['label'])
                train_dt, val_dt = train_test_split(train, test_size=testing_percentage, random_state=42,
                                                    stratify=train['label'])
                DT_CONF = CONF.copy()
                DT_CONF['predictive_model'] = ClassificationMethods.DT.value
                DT_CONF['hyperparameter_optimisation_target'] = HyperoptTarget.F1.value
                train = train.rename(str, axis="columns")
                test = test.rename(str, axis="columns")
                train_dt = train_dt.rename(str, axis="columns")
                val_dt = val_dt.rename(str, axis="columns")

                glass_box = PredictiveModel(DT_CONF, DT_CONF['predictive_model'], train_dt, val_dt)
                if DT_CONF['hyperparameter_optimisation']:
                    glass_box.model, glass_box.config = retrieve_best_model(
                        glass_box,
                        DT_CONF['predictive_model'],
                        max_evaluations=1,
                        target=DT_CONF['hyperparameter_optimisation_target'], seed=DT_CONF['seed']
                    )

                local_evaluate_glassbox = evaluate_classifier(test['label'],
                                                              glass_box.model.predict(drop_columns(test)), scores)
                local_fidelity = local_evaluate_glassbox['accuracy']
                print('Local fidelity', local_fidelity)

                time_discovery = 0
                time_alignment = 0

                original_test_df = frequency_full_df[frequency_full_df['trace_id'].isin(test_df['trace_id'])]
                frequency_encoder.decode(original_test_df)

                test_df_subset = test_df.copy()
                encoder.decode(test_df_subset)
                test_df_subset = enc_ohe.transform(test_df_subset)
                test_df_subset = test_df_subset.drop(
                    columns=[col for col in test_df_subset.columns if 'prefix' in col] + ['label'])
                original_test_df = pd.merge(test_df_subset, original_test_df, on='trace_id', how='left')
                diff = set(synth_df.columns.tolist()) - set(original_test_df.columns.tolist())
                original_test_df[list(diff)] = 0
                original_test_df = original_test_df[drop_columns(test).columns]
                global_preds = glass_box.model.predict(original_test_df)
                global_probs = glass_box.model.predict_proba(original_test_df)
                global_evaluate_glassbox = evaluate_classifier(predicted, global_preds, global_probs)

                global_fidelity = global_evaluate_glassbox['accuracy']
                print('Global fidelity', global_fidelity)

                if (local_fidelity > 0.9) | (global_fidelity > 0.8):
                    viz = dtreeviz.model(glass_box.model,
                                         drop_columns(train),
                                         train['label'],
                                         feature_names=drop_columns(train).columns,
                                         class_names=['false', 'true'],

                                         )
                    v = viz.view(orientation="LR", scale=2, label_fontsize=5.5)
                    v.save(
                        output_path + '/decision_trees' + '/' + '%s_baseline_%s_%s' % (
                            dataset, case_id, CONF['prefix_length']) + '.svg')

            logger.info('RESULT')
            logger.info('INITIAL', initial_result)
            logger.info('Done, cheers!')
            import json
            results = {}
            results['dataset'] = dataset
            results['case_id'] = case_id
            results['prefix_length'] = CONF['prefix_length']
            results['impressed_pipeline'] = impressed_pipeline
            # Add the other parameters
            results['local_fidelity'] = local_fidelity
            results['global_fidelity'] = global_fidelity
            results['time_discovery'] = time_discovery
            results['time_cf'] = time_cf
            results['time_alignment'] = time_alignment
            results['neighborhood_size'] = neighborhood_size
            results['encoding'] = CONF['feature_selection']
            x_eval['impressed_pipeline'] = impressed_pipeline
            x_eval['extension_step'] = max_extension_step
            x_eval['seed'] = CONF['seed']
            results['pattern_extension_strategy'] = pattern_extension_strategy
            results['aggregation_style'] = aggregation_style
            results['frequency_type'] = frequency_type
            results['distance_style'] = distance_style
            results['data_dependency'] = data_dependency+'_complex_index'
            try:
                results['number_of_patterns'] = impressed_test_df.shape[1]
                results['extension_style'] = extension_style
            except:
                results['number_of_patterns'] = 0
                results['extension_style'] = extension_style
                # results['pareto_only'] = False
            results['extension_step'] = max_extension_step
            results['seed'] = CONF['seed']
            res_df = pd.DataFrame(results, index=[0])

            if not os.path.isfile(output_path + '%s_results_impressed.csv' % (dataset)):
                res_df.to_csv(output_path + '%s_results_impressed.csv' % (dataset), index=False)
            else:
                res_df.to_csv(output_path + '%s_results_impressed.csv' % (dataset), mode='a', index=False, header=False)

            try:
                x_eval['number_of_patterns'] = Impressed_X.shape[1]
                x_eval['extension_style'] = extension_style
            except:
                x_eval['number_of_patterns'] = 0
                x_eval['extension_style'] = extension_style
            x_eval['local_fidelity'] = local_fidelity
            x_eval['global_fidelity'] = global_fidelity
            x_eval = pd.DataFrame(x_eval, index=[0])
            filename_results = output_path + 'cf_eval_%s_impressed_%s.csv' % (dataset, impressed_pipeline)
            if not os.path.isfile(filename_results):
                x_eval.to_csv(filename_results, index=False)
            else:
                x_eval.to_csv(filename_results, mode='a', index=False, header=False)
            print(dataset, 'impressed_pipeline', impressed_pipeline,
                  'LOCAL FIDELITY', local_fidelity, 'GLOBAL FIDELITY', global_fidelity, 'time_discovery',
                  time_discovery, 'time_cf', time_cf,
                  'time_alignment', time_alignment, 'neighborhood_size', neighborhood_size, 'number_of_patterns')


if __name__ == '__main__':
    dataset_list = {
        # 'synthetic_data': [3, 5, 7, 9],
        # 'bpic2012_O_ACCEPTED-COMPLETE': [20,25,30,35],
        # 'bpic2012_O_CANCELLED-COMPLETE':[20,25,30,35],
        # 'bpic2012_O_DECLINED-COMPLETE':[20,25,30,35],
        # 'sepsis_cases_1':[13],
        # 'sepsis_cases_2':[5,9,13,16],
        # 'sepsis_cases_4':[5,9,13,16],
        'BPIC17_O_ACCEPTED':[15,20,25,30]
        # 'BPIC17_O_CANCELLED':[15,20,25,30],
        # 'BPIC17_O_REFUSED':[15,20,25,30],

    }
    pipelines = [True]
    for dataset, prefix_lengths in dataset_list.items():
        for prefix_length in prefix_lengths:
            for pipeline in pipelines:
                if 'bpic2012' in dataset:
                    seed = 48
                elif 'sepsis' in dataset:
                    seed = 56
                else:
                    seed = 48
                print(os.path.join('datasets', dataset, 'full.xes'))
                CONF = {
                    'data': os.path.join('datasets', dataset, 'full.csv'),
                    'train_val_test_split': [0.7, 0.15, 0.15],
                    'output': os.path.join('..', 'output_data'),
                    'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
                    'prefix_length': prefix_length,
                    'padding': True,  # TODO, why use of padding?
                    'feature_selection': EncodingType.COMPLEX.value,
                    'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                    'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
                    'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                    'predictive_model': ClassificationMethods.XGBOOST.value,  # RANDOM_FOREST, LSTM, PERCEPTRON
                    'explanator': ExplainerType.DICE_IMPRESSED.value,  # SHAP, LRP, ICE, DICE
                    'threshold': 13,
                    'top_k': 10,
                    'hyperparameter_optimisation': True,  # TODO, this parameter is not used
                    'hyperparameter_optimisation_target': HyperoptTarget.AUC.value,
                    'hyperparameter_optimisation_epochs': 20,
                    'time_encoding': TimeEncodingType.NONE.value,
                    'target_event': None,
                    'seed': seed,
                    'impressed_pipeline': True,
                }

                run_simple_pipeline(CONF=CONF, dataset_name=dataset)
