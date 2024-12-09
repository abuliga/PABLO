# PAttern-Based LOcal explanations (PABLO) for Outcome-Based Predictive Process Monitoring

This repository provides the implementation for generating pattern-ware local explanations for Outcome-Base PPM,
using the PABLO framework introduced in the paper.



# Usage
To rerun the experiments presented in the paper, we first need to download and unzip the datasets archive file found at this link:https://drive.google.com/file/d/1Jr4RM4dRlbi-aXAxNaifRVGLDOnhlIdN/view?usp=sharing.
Make sure that the datasets are placed within the project PABLO folder.
Afterwards, a Conda/virtualenv environment has to be setup, then the requirements can be installed by opening a Terminal instance,
going to the PABLO folder and running following command: ```pip install -e .```

The results can be obtained by running the ```python impressed_pipeline_complex.py```.
To adjust the datasets to be run, or the prefix lengths considered, one can comment out the dataset list and/or prefix lengths
found at the end of each script file:
```if __name__ == '__main__':
    dataset_list = {
        'BPIC11_f1':[10,15,20,25],
       'BPIC11_f2':[10,15,20,25],
        'BPIC11_f3':[10,15,20,25],
       'BPIC11_f4':[10,15,20,25],
         'bpic2012_O_ACCEPTED-COMPLETE': [20,25,30,35],
        'bpic2012_O_CANCELLED-COMPLETE':[20,25,30,35],
         'bpic2012_O_DECLINED-COMPLETE':[20,25,30,35],
        'sepsis_cases_1':[7,9,13,16],
        'sepsis_cases_2':[7,9,13,16],
        'sepsis_cases_4':[7,9,13,16],
       'BPIC17_O_ACCEPTED':[15,20,25,30],
         'BPIC17_O_CANCELLED':[15,20,25,30],
        'BPIC17_O_REFUSED':[15,20,25,30],
        "Production": [7, 11, 15, 19]

    }
    pipelines = [True]
    methods = ['independent','independent_complex_index','dependent','dependent_True']
    for dataset, prefix_lengths in dataset_list.items():
        for prefix_length in prefix_lengths:
            for method in methods:
                if 'bpic2012' in dataset:
                    seed = 48
                elif 'sepsis' in dataset:
                    seed = 56
                else:
                    seed = 48
                print(os.path.join('datasets', dataset, 'full.xes'))
                CONF = {
                    'data': os.path.join('datasets', dataset, 'full.xes'),
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
                    'discovery_method':method
                }

                run_simple_pipeline(CONF=CONF, dataset_name=dataset)
```
The CONF file also allows one to the different predictive models, found in the code, along with different seeds
for the random number generators. The pipeline argument specifies whether the PABLO method is used, or the baseline method.
The results are saved in the results folder, that is created at runtime when running the scripts.

Each of the discovery methods refer to the 4 different encodings proposed in the paper, namely:
- independent: the Control-Flow Patterns (CFP) encoding
- independent_complex_index: the Concatenate ALL (CALL) encoding
- dependent: the Full Context Patterns (FCP) encoding
- dependent_True: the Dynamic Context Pattern (DCP) encoding
