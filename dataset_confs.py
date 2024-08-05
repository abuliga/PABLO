import os


class DatasetConfs:

    def __init__(self, dataset_name, where_is_the_file='',data=False):
        if dataset_name in ["traffic_fines_%s" % formula for formula in range(1, 3)]:
            #### Traffic fines settings ####
            dataset = dataset_name
            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity"}
            self.resource_col = {dataset: "Resource"}
            self.timestamp_col = {dataset: "Complete Timestamp"}
            self.label_col = {dataset: "label"}
            self.pos_label = {dataset: "deviant"}
            self.neg_label = {dataset: "regular"}

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]}
            self.static_cat_cols = {dataset: ["article", "vehicleClass"]}
            self.dynamic_num_cols = {dataset: ["expense", "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr",
                                         "month", "weekday", "hour", "open_cases"]}
            self.static_num_cols = {dataset: ["amount", "points"]}
        elif dataset_name in ["sepsis_cases_1","sepsis_cases_2","sepsis_cases_3","sepsis_cases_4","sepsis_cases_5","sepsis"]:
            #### Sepsis Cases settings ####
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            self.case_id_col = {dataset: "case:concept:name"}
            self.activity_col = {dataset: "concept:name"}
            self.resource_col = {dataset: "org:group"}
            self.timestamp_col = {dataset: "time:timestamp"}
            self.label_col = {dataset: "case:label"}
            self.pos_label = {dataset: False}
            self.neg_label = {dataset: True}

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["org:group"]}  # i.e. event attributes
            self.static_cat_cols = {dataset: ['case:Diagnose', 'case:DiagnosticArtAstrup', 'case:DiagnosticBlood', 'case:DiagnosticECG',
                                        'case:DiagnosticIC', 'case:DiagnosticLacticAcid', 'case:DiagnosticLiquor',
                                        'case:DiagnosticOther', 'case:DiagnosticSputum', 'case:DiagnosticUrinaryCulture',
                                        'case:DiagnosticUrinarySediment', 'case:DiagnosticXthorax', 'case:DisfuncOrg',
                                        'case:Hypotensie', 'case:Hypoxie', 'case:InfectionSuspected', 'case:Infusion', 'case:Oligurie',
                                        'case:SIRSCritHeartRate', 'case:SIRSCritLeucos', 'case:SIRSCritTachypnea',
                                        'case:SIRSCritTemperature',
                                        'case:SIRSCriteria2OrMore'] } # i.e. case attributes that are known from the start
            self.dynamic_num_cols = {dataset: ["timesincelastevent","Leucocytes","CRP","LacticAcid"]} # ,"month", "weekday", "hour" , "timesincecasestart", "event_nr", "open_cases" , "timesincemidnight",
            self.static_num_cols = {dataset: ['case:Age']}
        elif dataset_name is "Production":
            #### Production log settings ####
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity"}
            self.resource_col = {dataset: "Resource"}
            self.timestamp_col = {dataset: "Complete Timestamp"}
            self.label_col = {dataset: "label"}
            self.neg_label = {dataset: "regular"}
            self.pos_label = {dataset: "deviant"}

            # features for classifier
            self.static_cat_cols = {dataset: ["Part_Desc_", "Rework"]}
            self.static_num_cols = {dataset: ["Work_Order_Qty"]}
            self.dynamic_cat_cols = {dataset: ["Activity", "Resource", "Report_Type", "Resource.1"]}
            self.dynamic_num_cols = {dataset: ["Qty_Completed", "Qty_for_MRB", "activity_duration", "hour", "weekday", "month",
                                         "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]}
        elif dataset_name in ['BPIC17_O_CANCELLED', 'BPIC17_O_ACCEPTED', 'BPIC17_O_REFUSED']:
            #### BPIC2017 settings ####
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity"}
            self.resource_col = {dataset: 'org:resource'}
            self.timestamp_col = {dataset: 'time:timestamp'}
            self.label_col = {dataset: "label"}
            self.neg_label = {dataset: "regular"}
            self.pos_label = {dataset: "deviant"}

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["Activity", 'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition',
                                         "Accepted", "Selected"]}
            self.static_cat_cols = {dataset: ['ApplicationType', 'LoanGoal']}
            self.dynamic_num_cols = {dataset: ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount',
                                         'CreditScore', "timesincelastevent", "timesincecasestart", "timesincemidnight",
                                         "event_nr", "month", "weekday", "hour", "open_cases","RequestedAmount","@@index"]}
            self.static_num_cols = {dataset: ['RequestedAmount']}
        elif dataset_name in [ "hospital_billing_%s" % (i) for i in range(1, 7) ]:
            #### Hospital billing settings ####
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity"}
            self.resource_col = {dataset: "Resource"}
            self.timestamp_col = {dataset: "Complete Timestamp"}
            self.label_col = {dataset: "label"}
            self.neg_label = {dataset: "regular"}
            self.pos_label = {dataset: "deviant"}

            # if i == 1:
            #     neg_label[dataset] = "deviant"
            #     pos_label[dataset] = "regular"

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["Activity", 'Resource', 'actOrange', 'actRed', 'blocked', 'caseType', 'diagnosis',
                                         'flagC', 'flagD', 'msgCode', 'msgType', 'state',
                                         'version'] } # , 'isCancelled', 'isClosed', 'closeCode']
            self.static_cat_cols = {dataset: ['speciality']}
            self.dynamic_num_cols = {dataset: ['msgCount', "timesincelastevent", "timesincecasestart", "timesincemidnight",
                                         "event_nr", "month", "weekday", "hour"] } # , "open_cases"]
            self.static_num_cols = {dataset: []}
            #
            # if i == 1:  # label is created based on isCancelled attribute
            #     dynamic_cat_cols[dataset] = [col for col in dynamic_cat_cols[dataset] if col != "isCancelled"]
            # elif i == 2:
            #     dynamic_cat_cols[dataset] = [col for col in dynamic_cat_cols[dataset] if col != "isClosed"]
        elif dataset_name in ['bpic2012','bpic2012_O_ACCEPTED-COMPLETE','bpic2012_O_CANCELLED-COMPLETE','bpic2012_O_DECLINED-COMPLETE','full_triple_pattern.xes','full_complex_direct_pattern.xes', 'full_double_pattern_symmetrical_0.8.xes', 'full_direct_follow.xes', 'full_eventually_follow.xes', 'full.xes',
                              'full_triple_pattern_0.8.xes','full_triple_direct_pattern.xes','full_complex_concurrent_pattern.xes','full_a_a_pattern.xes','full_a_b_pattern_pareto.xes','full_a_b_c_pattern.xes'
                              'full_complex_direct_both_pattern.xes','']:
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "concept:name"}
            self.resource_col = {dataset: "Resource"}
            self.timestamp_col = {dataset: "Complete Timestamp"}
            self.label_col = {dataset: "label"}
            self.neg_label = {dataset: "regular"}
            self.pos_label = {dataset: "deviant"}

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["Resource", "lifecycle:transition"]}
            self.static_cat_cols = {dataset: []}
            self.dynamic_num_cols = {dataset: ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent",
                                         "timesincecasestart", "event_nr", "open_cases"]}
            self.static_num_cols = {dataset: ['AMOUNT_REQ']}
        elif dataset_name in [ "BPIC15_%s_f%s" % (municipality, formula) for municipality in range(1, 6) for formula in range(1, 3)]:
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            # filename[dataset] = os.path.join(logs_dir, "BPIC15_%s_f%s.csv" % (municipality, formula))

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity"}
            self.resource_col = {dataset: "org:resource"}
            self.timestamp_col = {dataset: "time:timestamp"}
            self.label_col = {dataset: "label"}
            self.pos_label = {dataset: "deviant"}
            self.neg_label = {dataset: "regular"}

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["Activity", "monitoringResource", "question", "org:resource"]}
            self.static_cat_cols = {dataset: ["Responsible_actor"]}
            self.dynamic_num_cols = {dataset: ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent",
                                         "timesincecasestart", "event_nr", "open_cases"]}
            self.static_num_cols = {dataset: ["SUMleges", 'Aanleg (Uitvoeren werk of werkzaamheid)', 'Bouw',
                                        'Brandveilig gebruik (vergunning)', 'Gebiedsbescherming',
                                        'Handelen in strijd met regels RO', 'Inrit/Uitweg', 'Kap',
                                        'Milieu (neutraal wijziging)', 'Milieu (omgevingsvergunning beperkte milieutoets)',
                                        'Milieu (vergunning)', 'Monument', 'Reclame', 'Sloop']}
            if dataset_name in ['BPIC15_3_f2', 'BPIC15_5_f2']:
                self.dynamic_num_cols[dataset].append('Flora en Fauna')
            if dataset_name in ['BPIC15_1_f2','BPIC15_2_f2','BPIC15_3_f2','BPIC15_5_f2']:
                self.dynamic_num_cols[dataset].append('Brandveilig gebruik (melding)')
                self.dynamic_num_cols[dataset].append('Milieu (melding)')
            if dataset_name in ['BPIC15_5_f2']:
                self.dynamic_num_cols[dataset].append('Integraal')
        elif dataset_name in ["BPIC11_f%s" % formula for formula in range(1, 5)]:
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity code"}
            self.resource_col = {dataset: "Producer code"}
            self.timestamp_col = {dataset: "time:timestamp"}
            self.label_col = {dataset: "label"}
            self.pos_label = {dataset: "deviant"}
            self.neg_label = {dataset: "regular"}

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["Activity", "Producer code", "Section", "Specialism code.1", "group"]}
            self.static_cat_cols = {dataset: ["Diagnosis", "Treatment code", "Diagnosis code", "Specialism code",
                                        "Diagnosis Treatment Combination ID"]}
            self.dynamic_num_cols = {dataset: ["Number of executions", "hour", "weekday", "month", "timesincemidnight",
                                         "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]}
            self.static_num_cols = {dataset: ["Age"]}
        elif dataset_name == 'synthetic_data':
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity"}
            self.resource_col = {dataset: "org:resource"}
            self.timestamp_col = {dataset: "time:timestamp"}
            self.label_col = {dataset: "label"}
            self.pos_label = {dataset: "deviant"}
            self.neg_label = {dataset: "regular"}

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["Activity","PClaims", "CType", "ClType", "lifecycle:transition", "org:resource"]}
            self.static_cat_cols = {dataset: []}
            self.dynamic_num_cols = {dataset: ["Age", "ClaimValue"]} 
            self.static_num_cols = {dataset: []}
        elif dataset_name in ['synthetic_bank_accepted','synthetic_bank_declined','synthetic_bank_cancelled']:
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity"}
            self.resource_col = {dataset: "org:resource"}
            self.timestamp_col = {dataset: "time:timestamp"}
            self.label_col = {dataset: "label"}
            self.neg_label = {dataset: "regular"}
            self.pos_label = {dataset: "deviant"}

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["Activity", "org:resource","lifecycle:transition","case"]}
            self.static_cat_cols = {dataset: []}
            self.dynamic_num_cols = {dataset: []}
            self.static_num_cols = {dataset: ['amount']}
        elif dataset_name == 'legal_complaints':
            #### Sepsis Cases settings ####
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset: ''}

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity"}
            self.resource_col = {dataset: "org:group"}
            self.timestamp_col = {dataset: "time:timestamp"}
            self.label_col = {dataset: "label"}
            self.pos_label = {dataset: "deviant"}
            self.neg_label = {dataset: "regular"}
        elif dataset_name in ['bpi2012_W_Two_TS','bpi2012_W_One_TS']:
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity"}
            self.resource_col = {dataset: "Resource"}
            self.timestamp_col = {dataset: "time:timestamp"}
            self.label_col = {dataset: "label"}
            self.neg_label = {dataset: "regular"}
            self.pos_label = {dataset: "deviant"}

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["prefix", "org:resource","lifecycle:transition"]}
            self.static_cat_cols = {dataset: []}
            self.dynamic_num_cols = {dataset: ["start:timestamp"]}
            self.static_num_cols = {dataset: ['AMOUNT_REQ']}
        elif dataset_name in ["sepsis_cases_1_start"]:
            #### Sepsis Cases settings ####
            dataset = dataset_name
            if where_is_the_file != '':
                self.filename = {dataset: where_is_the_file}
            else:
                self.filename = {dataset:''}

            self.case_id_col = {dataset: "Case ID"}
            self.activity_col = {dataset: "Activity"}
            self.resource_col = {dataset: "org:group"}
            self.timestamp_col = {dataset: "time:timestamp"}
            self.label_col = {dataset: "label"}
            self.pos_label = {dataset: "deviant"}
            self.neg_label = {dataset: "regular"}

            # features for classifier
            self.dynamic_cat_cols = {dataset: ["Activity", 'org:group',"org:resource"]}  # i.e. event attributes
            self.static_cat_cols = {dataset: ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                                        'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                                        'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                                        'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                                        'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                                        'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                                        'SIRSCritTemperature',
                                        'SIRSCriteria2OrMore'] } # i.e. case attributes that are known from the start
            self.dynamic_num_cols = {dataset: [
                "start:timestamp"
                                               ]}
            self.static_num_cols = {dataset: ['Age']}