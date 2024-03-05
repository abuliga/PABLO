import warnings
import os
from datetime import datetime
import dice_ml
import numpy as np
import pandas as pd
from scipy.spatial.distance import _validate_vector
from scipy.spatial.distance import cdist, pdist
from scipy.stats import median_abs_deviation
from pm4py import convert_to_event_log,write_xes,filter_event_attribute_values,convert_to_dataframe
from declare4py.declare4py import Declare4Py
from declare4py.enums import TraceState
from nirdizati_light.predictive_model.common import ClassificationMethods, get_tensor


warnings.filterwarnings("ignore", category=UserWarning)

path_cf = '../experiments/cf4impressed/impressed_datasets/'
single_prefix = ['loreley','loreley_complex']

def dice_impressed(CONF, predictive_model, cf_df, encoder, query_instance, method, support,optimization,
                 timestamp_col_name,model_path,neighborhood_size,random_seed=None,diversity_weight=None,sparsity_weight=None,proximity_weight=None,
                   features_to_vary=None,impressed_pipeline=None,dynamic_cols=None,timestamps=None):
    features_names = cf_df.columns.values[:-1]
    feature_selection = CONF['feature_selection']
    dataset = ''.join(CONF['data'].split('/')[2:3]).replace('.xes', '')
    categorical_features,continuous_features,cat_feature_index,cont_feature_index = split_features(cf_df.iloc[:,:-1], encoder)

    if CONF['feature_selection'] == 'loreley':
        query_instance = query_instance[query_instance['prefix'] != 0]
    if CONF['feature_selection'] == 'frequency':
        ratio_cont = 1
    else:
        ratio_cont = len(continuous_features)/len(categorical_features)
    time_start = datetime.now()
    if predictive_model.model_type == ClassificationMethods.LSTM.value:
        encoder.decode(cf_df)
        d = dice_ml.Data(dataframe=cf_df, continuous_features=continuous_features,
                         outcome_name='label')
        m = dice_model(predictive_model,func="ohe-min-max"
         )
        dice_query_instance = dice_ml.Dice(d, m,encoder=encoder, method='genetic')
    else:
        d = dice_ml.Data(dataframe=cf_df, continuous_features=continuous_features,categorical_features=categorical_features, outcome_name='label')
        m = dice_model(predictive_model)
        dice_query_instance = dice_ml.Dice(d, m, encoder=encoder, method=optimization)
    time_train = (datetime.now() - time_start).total_seconds()
    model_path = model_path +'_' + str(support) + '/'

    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print("Directory '%s' created successfully" % model_path)
    except OSError as error:
        print("Directory '%s' can not be created" % model_path)
    d4py = Declare4Py()
    try:
        if not os.path.exists(model_path+dataset+'_'+str(CONF['prefix_length'])+'.decl'):
            print('Do model discovery')
            model_discovery(CONF, encoder, cf_df, dataset, features_names,
                                              d4py, model_path, support, timestamp_col_name)
    except OSError as error:
        print("File '%s' can not be created" % (model_path+dataset+'_'+str(CONF['prefix_length'])+'.decl'))

    #print(datetime.now(), dataset, black_box, test_id, len(index_test_instances),
    #      '%.2f' % (test_id+1 / len(index_test_instances)))
    cf_list_all = list()
    cf_flip_all = list()
    cf_same_all = list()
    desired_cfs_all = list()

    x = query_instance.iloc[:,:-1]
    k = neighborhood_size
    if predictive_model.model_type == ClassificationMethods.LSTM.value:
        predicted_outcome = np.argmax(predictive_model.model.predict(get_tensor(CONF, x))[0])

    elif predictive_model.model_type == ClassificationMethods.XGBOOST.value:
        cat_cols = [col for col in x.columns if 'prefix' in col]
        x[cat_cols] = x[cat_cols].astype('category')
        predicted_outcome = predictive_model.model.predict(x.values.reshape(1, -1))[0]

    else:
        predicted_outcome = predictive_model.model.predict(x.values.reshape(1, -1))[0]
    total_traces = 0
    time_start_i = datetime.now()

    if method == 'iterative':
        while total_traces <  k:
            batch_size = np.random.choice([10,15,20,25,30])

            dice_result_flip = dice_query_instance.generate_counterfactuals(x,encoder=encoder, desired_class=int(1-predicted_outcome),
                                                                       verbose=False,
                                                                       posthoc_sparsity_algorithm='linear',
                                                                       total_CFs=batch_size,dataset=dataset+'_'+str(CONF['prefix_length']),
                                                                            proximity_weight=proximity_weight,
                                                                            diversity_weight=diversity_weight,sparsity_weight=sparsity_weight,
                                                                            features_to_vary=features_to_vary,random_seed=random_seed)

            dice_result_same = dice_query_instance.generate_counterfactuals(x,encoder=encoder, desired_class=int(predicted_outcome),
                                                                       verbose=False,
                                                                       posthoc_sparsity_algorithm='linear',
                                                                       total_CFs=batch_size,dataset=dataset+'_'+str(CONF['prefix_length']),
                                                                            proximity_weight=proximity_weight,
                                                                            diversity_weight=diversity_weight,
                                                                            sparsity_weight=sparsity_weight,features_to_vary=features_to_vary,
                                                                            random_seed=random_seed)

            generated_cfs_flip = dice_result_flip.cf_examples_list[0].final_cfs_df
            cf_list_flip = np.array(generated_cfs_flip).astype('float64')
            generated_cfs_same = dice_result_same.cf_examples_list[0].final_cfs_df
            cf_list_same = np.array(generated_cfs_same).astype('float64')
            if optimization == 'random':
                cf_list_same = cf_list_same[:, :-1]
                cf_list_flip = cf_list_flip[:, :-1]
            elif optimization == 'genetic':
                cf_list_same = cf_list_same[:, :-1]
                cf_list_flip = cf_list_flip[:, :-1]
            elif optimization == 'genetic_conformance':
                cf_list_same = cf_list_same[:, :-1]
                cf_list_flip = cf_list_flip[:, :-1]
            elif optimization == 'multi_objective_genetic':
                cf_list_same = cf_list_same[:, :-1]
                cf_list_flip = cf_list_flip[:, :-1]

            cf_flip_all.extend(cf_list_flip)
            cf_same_all.extend(cf_list_same)
            total_traces += batch_size
            print('Total traces generated',total_traces,'/',k)
    else:
        dice_result_flip = dice_query_instance.generate_counterfactuals(x, encoder=encoder,
                                                                        desired_class=int(1 - predicted_outcome),
                                                                        verbose=False,
                                                                        posthoc_sparsity_algorithm='linear',
                                                                        total_CFs=k,
                                                                        dataset=dataset + '_' + str(
                                                                            CONF['prefix_length']),
                                                                        #proximity_weight=proximity_weight,
                                                                        #diversity_weight=diversity_weight,
                                                                        #sparsity_weight=sparsity_weight,
                                                                        random_seed=random_seed)

        dice_result_same = dice_query_instance.generate_counterfactuals(x, encoder=encoder,
                                                                        desired_class=int(predicted_outcome),
                                                                        verbose=False,
                                                                        posthoc_sparsity_algorithm='linear',
                                                                        total_CFs=k,
                                                                        dataset=dataset + '_' + str(
                                                                            CONF['prefix_length']),
                                                                        #proximity_weight=proximity_weight,
                                                                        #diversity_weight=diversity_weight,
                                                                        #sparsity_weight=sparsity_weight,
                                                                        random_seed=random_seed
                                                                        )
        generated_cfs_flip = dice_result_flip.cf_examples_list[0].final_cfs_df
        cf_list_flip = np.array(generated_cfs_flip).astype('float64')
        generated_cfs_same = dice_result_same.cf_examples_list[0].final_cfs_df
        cf_list_same = np.array(generated_cfs_same).astype('float64')
        if optimization == 'random':
            cf_list_same = cf_list_same[:, :-1]
            cf_list_flip = cf_list_flip[:, :-1]
        elif optimization == 'genetic':
            cf_list_same = cf_list_same[:, :-1]
            cf_list_flip = cf_list_flip[:, :-1]
        elif optimization == 'genetic_conformance':
            cf_list_same = cf_list_same[:, :-1]
            cf_list_flip = cf_list_flip[:, :-1]
        elif optimization == 'multi_objective_genetic':
            cf_list_same = cf_list_same[:, :-1]
            cf_list_flip = cf_list_flip[:, :-1]

        cf_flip_all = cf_list_flip
        cf_same_all = cf_list_same
    # function to decode cf from train_df and show it decoded before adding to list
    #TODO: ADD COLUMNS FOR METRICS FROM DICE_RESULT
    y_pred = predictive_model.model.predict(x.values.reshape(1, -1))[0]
    time_test = (datetime.now() - time_start_i).total_seconds()

    df_conf_same = pd.DataFrame(data=cf_same_all, columns=features_names)
    likelihood_same = np.amax(predictive_model.model.predict_proba(cf_same_all),axis=1)
    label_same = predictive_model.model.predict(cf_same_all)
    
    df_conf_flip = pd.DataFrame(data=cf_flip_all, columns=features_names)
    likelihood_flip = np.amax(predictive_model.model.predict_proba(cf_flip_all),axis=1)
    label_flip = predictive_model.model.predict(cf_flip_all)
    cf_list = np.concatenate([cf_flip_all,cf_same_all])
    try:
        x_eval = evaluate_cf_list(cf_list, x.values.reshape(1, -1), cont_feature_index, cat_feature_index, df=cf_df,
                              nr_of_cfs=k, y_pred=y_pred, predictive_model=predictive_model,
                              query_instances=query_instance, continuous_features=continuous_features,
                              categorical_features=categorical_features, ratio_cont=ratio_cont
                              )
    except:
        print('Error evaluating cf_list')
    df_cf = pd.concat([df_conf_same,df_conf_flip])

    sat_score = conformance_score(CONF, encoder, df=df_cf, dataset=dataset, features_names=features_names,
                                  d4py=d4py, query_instance=x, model_path=model_path,
                                  timestamp_col_name=timestamp_col_name)
    x_eval['dataset'] = dataset
    x_eval['model'] = predictive_model.model_type
    x_eval['desired_nr_of_cfs'] = k
    x_eval['time_train'] = time_train
    x_eval['time_test'] = time_test
    x_eval['runtime'] = time_train + time_test
    x_eval['method'] = method
    x_eval['explainer'] = CONF['explanator']
    x_eval['prefix_length'] = CONF['prefix_length']
#    x_eval['heuristic'] = heuristic
    x_eval['optimization'] = optimization
    x_eval['conformance_score_decl']  = sat_score
    encoder.decode(df_cf)

    df_cf['label'] = np.concatenate([label_same,label_flip])
    df_cf['likelihood'] = np.concatenate([likelihood_same,likelihood_flip])
    cf_list_all = np.concatenate([cf_flip_all,cf_same_all])
    
    df_cf = df_cf.drop_duplicates().reset_index(drop=True)
    df_cf = df_cf[df_cf['likelihood'] > 0.5]

    try:
        if not os.path.exists(path_cf):
            os.makedirs(path_cf)
            print("Directory '%s' created successfully" % path_cf)
    except OSError as error:
        print("Directory '%s' can not be created" % path_cf)
    df_cf.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(df_cf)), 1)[0] + 1)
    df_cf['Case ID'] = 'Case'+df_cf['Case ID'].astype(str)
    if impressed_pipeline:
        df_cf = pd.merge(timestamps, df_cf, left_index=True, right_index=True)

    dynamic_cols[0] = 'prefix'

    long_data = pd.wide_to_long(df_cf, stubnames=dynamic_cols, i='Case ID',
                                j='order', sep='_', suffix=r'\w+')
    #timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')
    long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)

    long_data_sorted = long_data_sorted[long_data_sorted['prefix'] != '0']

    try:
        long_data_sorted[timestamp_col_name] = pd.to_datetime(long_data_sorted[timestamp_col_name], utc=True)
    except:
        print('Error converting timestamp column to datetime')
    long_data_sorted.drop(columns=['order'], inplace=True)
    columns_to_rename = {'Case ID': 'case:concept:name','prefix': 'concept:name','label': 'case:label'}
    columns_to_rename.update({'prefix': 'concept:name'})
    long_data_sorted.rename(columns=columns_to_rename, inplace=True)
    long_data_sorted['case:label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
    long_data_sorted.replace('0', 'other', inplace=True)
    event_log = convert_to_event_log(long_data_sorted)
    event_log = filter_event_attribute_values(event_log,'concept:name',['other'],level='event',retain=False)
    event_log = convert_to_dataframe(event_log)
    if 'BPIC17' in dataset:
    #    event_log['case:label'] = event_log['case:label'].replace({0: 'deviant', 1: 'regular'})
        df_cf['label'] = df_cf['label'].replace({0: 'deviant', 1: 'regular'})
    else:
    #    event_log['case:label'] = event_log['case:label'].replace({0:'false', 1:'true'})
        df_cf['label'] = df_cf['label'].replace({0:'false', 1:'true'})

    label_list = df_cf['label']

    try:
        event_log_pos = event_log[event_log['case:label'] == 'true'].reset_index(drop=True)
        event_log_neg = event_log[event_log['case:label'] == 'false'].reset_index(drop=True)
        event_log.to_csv(filename_cf, index=False)
        event_log_pos.to_csv(filename_cf_pos,index=False)
        event_log_neg.to_csv(filename_cf_neg,index=False)
    except:
        print('Cannot convert to event log, saving dataframe')
        #event_log = df_cf
    return event_log,x_eval,label_list
def dice_model(predictive_model,func=None):
    if predictive_model.model_type is ClassificationMethods.RANDOM_FOREST.value:
        m = dice_ml.Model(model=predictive_model.model, backend='sklearn')
    elif predictive_model.model_type is ClassificationMethods.PERCEPTRON.value:
        m = dice_ml.Model(model=predictive_model.model, backend='sklearn')
    elif predictive_model.model_type is ClassificationMethods.MLP.value:
        m = dice_ml.Model(model=predictive_model.model, backend='sklearn')
    elif predictive_model.model_type is ClassificationMethods.XGBOOST.value:
        m = dice_ml.Model(model=predictive_model.model, backend='sklearn')
    else:
        m = dice_ml.Model(model=predictive_model.model, backend='TF2',func=func)
    return m

def split_features(df, encoder):
    categorical_features = [col for col in df.columns if col in list(encoder._label_dict.keys())]
    cat_feature_index = [df.columns.get_loc(c) for c in categorical_features if c in df]
    continuous_features = [col for col in df.columns if col in list(encoder._numeric_encoder.keys())]
    cont_feature_index = [df.columns.get_loc(c) for c in continuous_features if c in df]
    return categorical_features,continuous_features,cat_feature_index,cont_feature_index

def evaluate_cf_list(cf_list, query_instance, cont_feature_index,cat_feature_index,df, y_pred,nr_of_cfs,
                     predictive_model, query_instances, continuous_features, categorical_features, ratio_cont):
    nbr_features = query_instance.shape[1]
    if cf_list.size > 4:
        nbr_cf_ = len(cf_list)
        nbr_features = cf_list.shape[1]
        plausibility_sum = plausibility(query_instance, predictive_model, cf_list,nr_of_cfs, y_pred,
                                        cont_feature_index,cat_feature_index, df, ratio_cont
                                       )
        plausibility_max_nbr_cf_ = plausibility_sum / nr_of_cfs
        plausibility_nbr_cf_ = plausibility_sum / nbr_cf_
        distance_l2_ = continuous_distance(query_instance, cf_list, cont_feature_index, metric='euclidean', X=df)
        distance_mad_ = continuous_distance(query_instance, cf_list, cont_feature_index, metric='mad', X=df)
        distance_j_ = categorical_distance(query_instance.astype('int'), cf_list, cat_feature_index, metric='jaccard')
        distance_h_ = categorical_distance(query_instance.astype('int'), cf_list, cat_feature_index, metric='hamming')
        distance_l2j_ = distance_l2j(query_instance.astype('int'), cf_list, cont_feature_index, cat_feature_index)
        distance_l1j_ = distance_l1j(query_instance.astype('int'), cf_list, cont_feature_index, cat_feature_index)
        distance_mh_ = distance_mh(query_instance.astype('int'), cf_list, cont_feature_index, cat_feature_index, df)

        distance_l2_min_ = continuous_distance(query_instance.astype('int'), cf_list, cont_feature_index, metric='euclidean', X=df,
                                               agg='min')
        distance_mad_min_ = continuous_distance(query_instance.astype('int'), cf_list, cont_feature_index, metric='mad', X=df, agg='min')
        distance_j_min_ = categorical_distance(query_instance.astype('int'), cf_list, cat_feature_index, metric='jaccard', agg='min')
        distance_h_min_ = categorical_distance(query_instance.astype('int'), cf_list, cat_feature_index, metric='hamming', agg='min')
        distance_l2j_min_ = distance_l2j(query_instance.astype('int'), cf_list, cont_feature_index, cat_feature_index,
                                         agg='min')
        distance_l1j_min_ = distance_l1j(query_instance.astype('int'), cf_list, cont_feature_index, cat_feature_index,
                                         agg='min')
        distance_mh_min_ = distance_mh(query_instance.astype('int'), cf_list, cont_feature_index, cat_feature_index, df,agg='min')

        distance_l2_max_ = continuous_distance(query_instance.astype('int'), cf_list, cont_feature_index, metric='euclidean', X=df, agg='max')
        distance_mad_max_ = continuous_distance(query_instance.astype('int'), cf_list, cont_feature_index, metric='mad', X=df, agg='max')
        distance_j_max_ = categorical_distance(query_instance.astype('int'), cf_list, cat_feature_index, metric='jaccard', agg='max')
        distance_h_max_ = categorical_distance(query_instance.astype('int'), cf_list, cat_feature_index, metric='hamming', agg='max')
        distance_l2j_max_ = distance_l2j(query_instance.astype('int'), cf_list, cont_feature_index, cat_feature_index, agg='max')
        distance_l1j_max_ = distance_l1j(query_instance.astype('int'), cf_list, cont_feature_index, cat_feature_index, agg='max')

        distance_mh_max_ = distance_mh(query_instance.astype('int'), cf_list, cont_feature_index, cat_feature_index, X=df, agg='max')
        avg_nbr_changes_per_cf_ = avg_nbr_changes_per_cf(query_instance.astype('int'), cf_list, continuous_features)
        avg_nbr_changes_ = avg_nbr_changes(query_instance.astype('int'), cf_list, nbr_features, continuous_features)
        if len(cf_list) > 1:
            diversity_l2_ = continuous_diversity(cf_list, cont_feature_index, metric='euclidean', X=df)
            diversity_mad_ = continuous_diversity(cf_list, cont_feature_index, metric='mad', X=df)
            diversity_j_ = categorical_diversity(cf_list, cat_feature_index, metric='jaccard')
            diversity_h_ = categorical_diversity(cf_list, cat_feature_index, metric='hamming')
            diversity_l2j_ = diversity_l2j(cf_list, cont_feature_index, cat_feature_index)
            diversity_mh_ = diversity_mh(cf_list, cont_feature_index, cat_feature_index, df)

            diversity_l2_min_ = continuous_diversity(cf_list, cont_feature_index, metric='euclidean', X=df, agg='min')
            diversity_mad_min_ = continuous_diversity(cf_list, cont_feature_index, metric='mad', X=df, agg='min')
            diversity_j_min_ = categorical_diversity(cf_list, cat_feature_index, metric='jaccard', agg='min')
            diversity_h_min_ = categorical_diversity(cf_list, cat_feature_index, metric='hamming', agg='min')
            diversity_l2j_min_ = diversity_l2j(cf_list, cont_feature_index, cat_feature_index, agg='min')
            diversity_mh_min_ = diversity_mh(cf_list, cont_feature_index, cat_feature_index, df, agg='min')

            diversity_l2_max_ = continuous_diversity(cf_list, cont_feature_index, metric='euclidean', X=None, agg='max')
            diversity_mad_max_ = continuous_diversity(cf_list, cont_feature_index, metric='mad', X=df, agg='max')
            diversity_j_max_ = categorical_diversity(cf_list, cat_feature_index, metric='jaccard', agg='max')
            diversity_h_max_ = categorical_diversity(cf_list, cat_feature_index, metric='hamming', agg='max')
            diversity_l2j_max_ = diversity_l2j(cf_list, cont_feature_index, cat_feature_index, agg='max')
            diversity_mh_max_ = diversity_mh(cf_list, cont_feature_index, cat_feature_index, df, agg='max')

        else:
            diversity_l2_ = 0.0
            diversity_mad_ = 0.0
            diversity_j_ = 0.0
            diversity_h_ = 0.0
            diversity_l2j_ = 0.0
            diversity_mh_ = 0.0

            diversity_l2_min_ = 0.0
            diversity_mad_min_ = 0.0
            diversity_j_min_ = 0.0
            diversity_h_min_ = 0.0
            diversity_l2j_min_ = 0.0
            diversity_mh_min_ = 0.0

            diversity_l2_max_ = 0.0
            diversity_mad_max_ = 0.0
            diversity_j_max_ = 0.0
            diversity_h_max_ = 0.0
            diversity_l2j_max_ = 0.0
            diversity_mh_max_ = 0.0

        count_diversity_cont_ = count_diversity(cf_list, cont_feature_index, nbr_features, cont_feature_index)
        count_diversity_cate_ = count_diversity(cf_list, cat_feature_index, nbr_features, cont_feature_index)
        count_diversity_all_ = count_diversity_all(cf_list, nbr_features, cont_feature_index)
        res = {  'generated_cfs': nr_of_cfs,
                'implausibility_sum': plausibility_sum,
                'implausibility_max_nbr_cf': plausibility_max_nbr_cf_,
                'implausibility_nbr_cf': plausibility_nbr_cf_,
                'distance_l2': distance_l2_,
                'distance_mad': distance_mad_,
                'distance_j': distance_j_,
                'distance_h': distance_h_,
                'distance_l2j': distance_l2j_,
                'distance_l1j':distance_l1j_,
                'distance_mh': distance_mh_,

                'distance_l2_min': distance_l2_min_,
                'distance_mad_min': distance_mad_min_,
                'distance_j_min': distance_j_min_,
                'distance_h_min': distance_h_min_,
                'distance_l2j_min': distance_l2j_min_,
                'distance_l1j_min': distance_l1j_min_,
                'distance_mh_min': distance_mh_min_,

                'distance_l2_max': distance_l2_max_,
                'distance_mad_max': distance_mad_max_,
                'distance_j_max': distance_j_max_,
                'distance_h_max': distance_h_max_,
                'distance_l2j_max': distance_l2j_max_,
                'distance_l1j_max':distance_l1j_max_,
                'distance_mh_max': distance_mh_max_,

                'diversity_l2': diversity_l2_,
                'diversity_mad': diversity_mad_,
                'diversity_j': diversity_j_,
                'diversity_h': diversity_h_,
                'diversity_l2j': diversity_l2j_,
                'diversity_mh': diversity_mh_,

                'diversity_l2_min': diversity_l2_min_,
                'diversity_mad_min': diversity_mad_min_,
                'diversity_j_min': diversity_j_min_,
                'diversity_h_min': diversity_h_min_,
                'diversity_l2j_min': diversity_l2j_min_,
                'diversity_mh_min': diversity_mh_min_,

                'diversity_l2_max': diversity_l2_max_,
                'diversity_mad_max': diversity_mad_max_,
                'diversity_j_max': diversity_j_max_,
                'diversity_h_max': diversity_h_max_,
                'diversity_l2j_max': diversity_l2j_max_,
                'diversity_mh_max': diversity_mh_max_,

                'count_diversity_cont': count_diversity_cont_,
                'count_diversity_cate': count_diversity_cate_,
                'count_diversity_all': count_diversity_all_,
                'avg_nbr_changes_per_cf':avg_nbr_changes_per_cf_,
                'avg_nbr_changes': avg_nbr_changes_}
    else:
        res = {
            'generated_cfs': 0,
            'distance_l2': np.nan,
            'distance_mad': np.nan,
            'distance_j': np.nan,
            'distance_h': np.nan,
            'distance_l2j': np.nan,
            'distance_l1j':np.nan,
            'distance_mh': np.nan,
            'distance_l2_min': np.nan,
            'distance_mad_min': np.nan,
            'distance_j_min': np.nan,
            'distance_h_min': np.nan,
            'distance_l2j_min': np.nan,
            'distance_l1j_min':np.nan,
            'distance_mh_min': np.nan,
            'distance_l2_max': np.nan,
            'distance_mad_max': np.nan,
            'distance_j_max': np.nan,
            'distance_h_max': np.nan,
            'distance_l2j_max': np.nan,
            'distance_l1j_max':np.nan,
            'distance_mh_max': np.nan,
            'avg_nbr_changes_per_cf': np.nan,
            'avg_nbr_changes': np.nan,
            'diversity_l2': np.nan,
            'diversity_mad': np.nan,
            'diversity_j': np.nan,
            'diversity_h': np.nan,
            'diversity_l2j': np.nan,
            'diversity_mh': np.nan,
            'diversity_l2_min': np.nan,
            'diversity_mad_min': np.nan,
            'diversity_j_min': np.nan,
            'diversity_h_min': np.nan,
            'diversity_l2j_min': np.nan,
            'diversity_mh_min': np.nan,
            'diversity_l2_max': np.nan,
            'diversity_mad_max': np.nan,
            'diversity_j_max': np.nan,
            'diversity_h_max': np.nan,
            'diversity_l2j_max': np.nan,
            'diversity_mh_max': np.nan,
            'count_diversity_cont': np.nan,
            'count_diversity_cate': np.nan,
            'count_diversity_all': np.nan,

            'implausibility_sum': 0.0,
            'implausibility_max_nbr_cf': 0.0,
            'implausibility_nbr_cf': 0.0,
            'sat_score': 0.0
        }
    return res

def continuous_diversity(cf_list, cont_feature_index, metric='euclidean', X=None, agg=None):
    if metric == 'mad':
        mad = median_abs_deviation(X.iloc[:, cont_feature_index], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)
        dist = pdist(cf_list[:, cont_feature_index], metric=_mad_cityblock)
    else:
        dist = pdist(cf_list[:, cont_feature_index], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)

def categorical_diversity(cf_list, cat_feature_index, metric='jaccard', agg=None):

    dist = pdist(cf_list[:, cat_feature_index], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)

def diversity_mh(cf_list, cont_feature_index, cat_feature_index, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, cont_feature_index, metric='mad', X=X, agg=agg)
    dist_cate = categorical_diversity(cf_list, cat_feature_index, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

def count_diversity(cf_list, features, nbr_features, cont_feature_index):
    nbr_cf = cf_list.shape[0]
    nbr_changes = 0
    for i in range(nbr_cf):
        for j in range(i+1, nbr_cf):
            for k in features:
                if cf_list[i][k] != cf_list[j][k]:
                    nbr_changes += 1 if j in cont_feature_index else 0.5
    return nbr_changes / (nbr_cf * nbr_cf * nbr_features)



# piu alto e' meglio conta variet' tra cf
def count_diversity_all(cf_list, nbr_features, cont_feature_index):
    return count_diversity(cf_list, range(cf_list.shape[1]), nbr_features, cont_feature_index)

def continuous_distance(query_instance, cf_list, cont_feature_index, metric='euclidean', X=None, agg=None):
    if metric == 'mad':
        mad = median_abs_deviation(X.iloc[:, cont_feature_index], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)
        dist = cdist(np.array(query_instance,dtype='float')[:, cont_feature_index], cf_list[:, cont_feature_index], metric=_mad_cityblock)
    else:
        dist = cdist(np.array(query_instance,dtype='float')[:, cont_feature_index], cf_list[:, cont_feature_index], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)

def mad_cityblock(u, v, mad):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()

def categorical_distance(query_instance, cf_list, cat_feature_index, metric='jaccard', agg=None):
    dist = cdist(query_instance.reshape(1, -1)[:, cat_feature_index], cf_list[:, cat_feature_index], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)

def euclidean_jaccard(query_instance, A, cont_feature_index, cat_feature_index, ratio_cont=None):
    nbr_features = A.shape[1]
    dist_cont = cdist(query_instance.reshape(1, -1)[:, cont_feature_index], A[:, cont_feature_index], metric='euclidean')
    dist_cate = cdist(query_instance.reshape(1, -1)[:, cat_feature_index], A[:, cat_feature_index], metric='jaccard')
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def distance_l2j(query_instance, cf_list, cont_feature_index, cat_feature_index, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(query_instance, cf_list, cont_feature_index, metric='euclidean', X=None, agg=agg)
    dist_cate = categorical_distance(query_instance, cf_list, cat_feature_index, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

def distance_l1j(query_instance, cf_list, cont_feature_index, cat_feature_index, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(query_instance, cf_list, cont_feature_index, metric='cityblock', X=None, agg=agg)
    dist_cate = categorical_distance(query_instance, cf_list, cat_feature_index, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

def distance_mh(query_instance, cf_list, cont_feature_index, cat_feature_index, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(query_instance, cf_list, cont_feature_index, metric='mad', X=X, agg=agg)
    dist_cate = categorical_distance(query_instance, cf_list, cat_feature_index, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

def categorical_diversity(cf_list, cat_feature_index, metric='jaccard', agg=None):
    dist = pdist(cf_list[:, cat_feature_index], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)

def diversity_l2j(cf_list, cont_feature_index, cat_feature_index, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, cont_feature_index, metric='euclidean', X=None, agg=agg)
    dist_cate = categorical_diversity(cf_list, cat_feature_index, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

def diversity_mh(cf_list, cont_feature_index, cat_feature_index, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, cont_feature_index, metric='mad', X=X, agg=agg)
    dist_cate = categorical_diversity(cf_list, cat_feature_index, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

def nbr_changes_per_cf(x, cf_list, continuous_features):
    x = np.array(x).ravel()
    nbr_features = cf_list.shape[1] - 1 #exclude label
    nbr_changes = np.zeros(len(cf_list))
    for i, cf in enumerate(cf_list):
        cf = cf[:-1]
        for j in range(nbr_features):
            if cf[j] != x[j]:
                nbr_changes[i] += 1 if j in continuous_features else 0.5
    return nbr_changes

def avg_nbr_changes_per_cf(x, cf_list, continuous_features):
    return np.mean(nbr_changes_per_cf(x, cf_list, continuous_features))

def avg_nbr_changes(x, cf_list, nbr_features, continuous_features):
    val = np.sum(nbr_changes_per_cf(x, cf_list, continuous_features))
    nbr_cf, _ = cf_list.shape
    return val / (nbr_cf * nbr_features)

def plausibility(query_instance, predictive_model, cf_list,nr_of_cfs,
                 y_pred, continuous_features, categorical_features, df, ratio_cont):
    distances = []
    full_df = df
    for cf in cf_list:
        #X_y = full_df[full_df['label'] == y_pred]
        X_y = full_df
        # neigh_dist = exp.cdist(x.reshape(1, -1), X_test_y)
        neigh_dist = distance_mh(np.array(query_instance).astype('int'), X_y.to_numpy().astype('int'), continuous_features,
                        categorical_features, df.astype('int'), ratio_cont)
        idx_neigh = np.argsort(neigh_dist)[0]
        # closest_idx = closest_idx = idx_neigh[0]
        # closest = X_test_y[closest_idx]
        closest = X_y.to_numpy()[idx_neigh]
        d = distance_mh(cf.reshape(1,-1).astype('int'), closest.reshape(1, -1).astype('int'), continuous_features,
                        categorical_features, df.astype('int'), ratio_cont)
        distances.append(d)
    sum_dist = sum(distances)
    return sum_dist

def conformance_score(CONF, encoder, df, dataset, features_names, d4py, query_instance, model_path, timestamp_col_name):
    d4py.parse_decl_model(model_path=model_path+dataset+'_'+str(CONF['prefix_length'])+'.decl')
    #d4py.parse_decl_model(model_path=model_path+dataset+'.decl')
    df = pd.DataFrame(df, columns=features_names)
    query_instance_to_decode = pd.DataFrame(np.array(query_instance, dtype=float),
                                            columns=features_names)

    encoder.decode(query_instance_to_decode)
    encoder.decode(df)
    df.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(df)), 1)[0] + 1)
    df.insert(loc=1, column='label', value=1)
    query_instance_to_decode.insert(loc=0, column='Case ID',
                                    value=np.divmod(np.arange(len(query_instance_to_decode)), 1)[0] + 1)
    query_instance_to_decode.insert(loc=1, column='label', value=1)

    df['Case ID'] = df['Case ID'].astype(str)
    query_instance_to_decode['Case ID'] = query_instance_to_decode['Case ID'].astype(str)
    long_data = pd.wide_to_long(df, stubnames=['prefix'], i='Case ID',
                                j='order', sep='_', suffix=r'\w+')
    long_query_instance = pd.wide_to_long(query_instance_to_decode, stubnames=['prefix'], i='Case ID',
                                          j='order', sep='_', suffix=r'\w+')
    long_query_instance_sorted = long_query_instance.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
    timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')
    timestamps_query = pd.date_range('1/1/2011', periods=len(long_query_instance_sorted), freq='H')
    long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
    long_data_sorted[timestamp_col_name] = timestamps
    long_query_instance_sorted[timestamp_col_name]= timestamps_query
    long_data_sorted['label'].replace({1: 'regular'}, inplace=True)
    long_data_sorted.drop(columns=['order'], inplace=True)
    columns_to_rename = {'Case ID': 'case:concept:name'}
    columns_to_rename.update({'prefix': 'concept:name'})
    long_data_sorted.rename(columns=columns_to_rename, inplace=True)
    long_data_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
    long_data_sorted.replace('0', 'other', inplace=True)
    long_query_instance_sorted.rename(columns=columns_to_rename, inplace=True)
    long_query_instance_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
    long_query_instance_sorted.replace('0', 'other', inplace=True)
    event_log = convert_to_event_log(long_data_sorted)
    query_log = convert_to_event_log(long_query_instance_sorted)
    d4py.load_xes_log(event_log)
    model_check_res = d4py.conformance_checking(consider_vacuity=True)
    d4py.load_xes_log(query_log)
    model_check_query = d4py.conformance_checking(consider_vacuity=True)
    query_patterns = {
        constraint
        for trace, patts in model_check_query.items()
        for constraint, checker in patts.items()
        if checker.state == TraceState.SATISFIED
    }

    model_check_res = {
        k: {
            constraint: checker
            for constraint, checker in v.items()
            if checker.state != TraceState.VIOLATED and constraint in query_patterns
        }
        for k, v in model_check_res.items()
    }

    conformance_score = [len(v) / len(query_patterns) for v in model_check_res.values() ]
    avg_conformance = np.mean(conformance_score)
    print('Average conformance score', np.mean(conformance_score))
    return avg_conformance

def model_discovery(CONF, encoder, df, dataset, features_names, d4py, model_path, support, timestamp_col_name):
    df = pd.DataFrame(df, columns=features_names)
    encoder.decode(df)
    df.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(df)), 1)[0] + 1)
    df['Case ID'] = df['Case ID'].astype(str)
    df.insert(loc=1, column='label', value=1)
    long_data = pd.wide_to_long(df, stubnames=['prefix'], i='Case ID',
                                j='order', sep='_', suffix=r'\w+')
    timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')
    long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
    long_data_sorted[timestamp_col_name] = timestamps
    long_data_sorted['label'].replace({1: 'regular'}, inplace=True)
    long_data_sorted.drop(columns=['order'], inplace=True)
    columns_to_rename = {'Case ID': 'case:concept:name'}
    columns_to_rename.update({'prefix': 'concept:name'})
    long_data_sorted.rename(columns=columns_to_rename, inplace=True)
    long_data_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
    long_data_sorted.replace('0', 'other', inplace=True)
    long_data_sorted.replace(0.0, 'other', inplace=True)
    long_data_sorted.replace(0, 'other', inplace=True)
    event_log = convert_to_event_log(long_data_sorted)
    d4py.load_xes_log(event_log)
    d4py.compute_frequent_itemsets(min_support=support, len_itemset=2)
    d4py.discovery(consider_vacuity=False, max_declare_cardinality=2)
    discovered = d4py.filter_discovery(min_support=support, output_path=model_path+dataset+'_'+str(CONF['prefix_length'])+'.decl')

columns = ['dataset', 'model', 'method', 'optimization','prefix_length','idx', 'desired_nr_of_cfs','generated_cfs', 'time_train','time_test',
           'runtime','distance_l2', 'distance_mad', 'distance_j', 'distance_h','distance_l1j', 'distance_l2j', 'distance_mh',
           'distance_l2_min', 'distance_mad_min', 'distance_j_min', 'distance_h_min','distance_l1j_min', 'distance_l2j_min',
           'distance_mh_min', 'distance_l2_max', 'distance_mad_max', 'distance_j_max', 'distance_h_max',
           'distance_l1j_max','distance_l2j_max', 'distance_mh_max', 'diversity_l2',
           'diversity_mad', 'diversity_j', 'diversity_h', 'diversity_l2j', 'diversity_mh', 'diversity_l2_min',
           'diversity_mad_min', 'diversity_j_min', 'diversity_h_min', 'diversity_l2j_min', 'diversity_mh_min',
           'diversity_l2_max', 'diversity_mad_max', 'diversity_j_max', 'diversity_h_max', 'diversity_l2j_max',
           'diversity_mh_max', 'count_diversity_cont', 'count_diversity_cate', 'count_diversity_all',
            'avg_nbr_changes_per_cf','avg_nbr_changes','implausibility_sum',
            'implausibility_max_nbr_cf','implausibility_nbr_cf','sat_score']
