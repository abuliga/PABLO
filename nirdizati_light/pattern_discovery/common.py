from enum import Enum

from nirdizati_light.pattern_discovery.wrappers.impressed_wrapper import impressed_wrapper

class PatternDiscoveryType(Enum):
    IMPRESSED = 'impressed'


def discovery(discovery_algorithm,log,output_path,discovery_type,case_id,activity,timestamp,outcome,outcome_type,delta_time,
                      max_gap,max_extension_step,factual_outcome,likelihood,encoding,testing_percentage,extension_style, data_dependency,
              model,pattern_extension_strategy,aggregation_style,frequency_type, distance_style):
    if discovery_algorithm is PatternDiscoveryType.IMPRESSED.value:
        return impressed_wrapper(log,output_path,discovery_type,case_id,activity,timestamp,outcome,outcome_type,delta_time,
                      max_gap,max_extension_step,factual_outcome,likelihood,encoding,testing_percentage,extension_style,data_dependency,
                                 model,pattern_extension_strategy,aggregation_style,frequency_type, distance_style)
   