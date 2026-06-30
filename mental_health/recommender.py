import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score

class VASuicide:
    def __init__(self,mental_health_descriptor_path,input_dict):
        self.mental_health_feature_descriptor = pd.read_excel(mental_health_descriptor_path)
        if 'feature_names' not in input_dict.keys():
            input_dict.update({'feature_names':['id', 'case', 'Alprazolam12', 'Alprazolam24', 'als12', 'als24', 'ami12', 'ami24', 'amp12', 'amp24', 'analgesic12', 'analgesic24', 'anticonvulsant12', 'anticonvulsant24', 'antidep12', 'antidep24', 'antipsy12', 'antipsy24', 'anyattempt1', 'anyattempt12', 'anyattempt18', 'anyattempt2', 'anyattempt24', 'anyattempt3', 'anyattempt6', 'anybipol12', 'anybipol24', 'AnyEDvisits_prior1', 'AnyEDvisits_prior12', 'AnyEDvisits_prior18', 'AnyEDvisits_prior2', 'AnyEDvisits_prior24', 'AnyEDvisits_prior3', 'AnyEDvisits_prior6', 'AnyIPSub_prior12', 'AnyIPSub_prior24', 'AnyMHBed_prior12', 'AnyMHBed_prior24', 'AnyMHDisPrior12Mos', 'AnyMHDisPrior1Mos', 'AnyMHDisPrior24Mos', 'AnyMHDisPrior3Mos', 'AnyMHDisPrior6Mos', 'anyMHdx12', 'anyMHdx24', 'anyMHtx12', 'anyMHtx24', 'AnyPain12', 'AnyPain24', 'AnyResBed_prior12', 'AnyResBed_prior24', 'AnyResSub_prior12', 'AnyResSub_prior24', 'AnyUCvisits_prior1', 'AnyUCvisits_prior12', 'AnyUCvisits_prior18', 'AnyUCvisits_prior2', 'AnyUCvisits_prior24', 'AnyUCvisits_prior3', 'AnyUCvisits_prior6', 'AnyUsein10MoPrior', 'AnyUsein11MoPrior', 'AnyUsein12MoPrior', 'AnyUsein13MoPrior', 'AnyUsein14MoPrior', 'AnyUsein15MoPrior', 'AnyUsein16MoPrior', 'AnyUsein17MoPrior', 'AnyUsein18MoPrior', 'AnyUsein19MoPrior', 'AnyUsein1MoPrior', 'AnyUsein20MoPrior', 'AnyUsein21MoPrior', 'AnyUsein22MoPrior', 'AnyUsein23MoPrior', 'AnyUsein24MoPrior', 'AnyUsein2MoPrior', 'AnyUsein3MoPrior', 'AnyUsein4MoPrior', 'AnyUsein5MoPrior', 'AnyUsein6MoPrior', 'AnyUsein7MoPrior', 'AnyUsein8MoPrior', 'AnyUsein9MoPrior', 'apnea12', 'apnea24', 'ARB12', 'ARB24', 'arth12', 'arth24', 'auto12', 'auto24', 'backpain12', 'backpain24', 'bipolI12', 'bipolI24', 'bipolII12', 'bipolII24', 'blind12', 'blind24', 'ca_head12', 'ca_head24', 'ca_prost12', 'ca_prost24', 'ca12', 'ca24', 'cad12', 'cad24', 'cancer12', 'cancer24', 'cess12', 'cess24', 'chronic12', 'chronic24', 'Clonazepam12', 'Clonazepam24', 'conc12', 'conc24', 'copd12', 'copd24', 'CurrIPSub', 'CurrMHBed', 'CurrResBed', 'CurrResSub', 'cush12', 'cush24', 'cva12', 'cva24', 'cvd12', 'cvd24', 'dementia12', 'dementia24', 'depr12', 'depr24', 'dm12', 'dm24', 'dt12', 'dt24', 'dysthymia12', 'dysthymia24', 'ep12', 'ep24', 'fib12', 'fib24', 'FirstUse1Yr', 'FirstUse2Yr', 'FirstUse3Yr', 'FirstUse4Yr', 'gu12', 'gu24', 'ha12', 'ha24', 'hc12', 'hc24', 'hear12', 'hear24', 'hemi12', 'hemi24', 'highrisk', 'hiv12', 'hiv24', 'homeless12', 'homeless24', 'hyp12', 'hyp24', 'lagca12', 'lagca24', 'Lorazepam12', 'Lorazepam24', 'mig12', 'mig24', 'Mirtazepam12', 'Mirtazepam24', 'moodst12', 'moodst24', 'ms12', 'ms24', 'neuro12', 'neuro24', 'opioid12', 'opioid24', 'OthAnxDis12', 'OthAnxDis24', 'OthPsych12', 'OthPsych24', 'par12', 'par24', 'Persond12', 'Persond24', 'psy12', 'psy24', 'ptsd12', 'ptsd24', 'ra12', 'ra24', 'rd12', 'rd24', 'relax12', 'relax24', 'schizo12', 'schizo24', 'sci12', 'sci24', 'sedative_anxiolytic12', 'sedative_anxiolytic24', 'Sildenafil12', 'Sildenafil24', 'sle12', 'sle24', 'sleep12', 'sleep24', 'Statin12', 'Statin24', 'sud12', 'sud24', 'symptom12', 'symptom24', 'tbi12', 'tbi24', 'TCA12', 'TCA24', 'tension12', 'tension24', 'thy12', 'thy24', 'tobacco12', 'tobacco24', 'topical12', 'topical24', 'Trazodone12', 'Trazodone24', 'use12', 'use24', 'vet12', 'vet24', 'vision12', 'vision24', 'Zolpidem12', 'Zolpidem24', 'age', 'Attempt1', 'Attempt12', 'Attempt18', 'Attempt2', 'Attempt24', 'Attempt3', 'Attempt6', 'change_divide', 'change_Sq', 'change_subtract', 'CumDaysUsein12MoPrior', 'CumDaysUsein18MoPrior', 'CumDaysUsein1MoPrior', 'CumDaysUsein24MoPrior', 'CumDaysUsein2MoPrior', 'CumDaysUsein3MoPrior', 'CumDaysUsein6MoPrior', 'CumIPDaysUsein12MoPrior', 'CumIPDaysUsein18MoPrior', 'CumIPDaysUsein1MoPrior', 'CumIPDaysUsein24MoPrior', 'CumIPDaysUsein2MoPrior', 'CumIPDaysUsein3MoPrior', 'CumIPDaysUsein6MoPrior', 'CumIPMHDaysUsein12MoPrior', 'CumIPMHDaysUsein18MoPrior', 'CumIPMHDaysUsein1MoPrior', 'CumIPMHDaysUsein24MoPrior', 'CumIPMHDaysUsein2MoPrior', 'CumIPMHDaysUsein3MoPrior', 'CumIPMHDaysUsein6MoPrior', 'CumOPDaysUsein12MoPrior', 'CumOPDaysUsein18MoPrior', 'CumOPDaysUsein1MoPrior', 'CumOPDaysUsein24MoPrior', 'CumOPDaysUsein2MoPrior', 'CumOPDaysUsein3MoPrior', 'CumOPDaysUsein6MoPrior', 'CumOPMHDaysUsein12MoPrior', 'CumOPMHDaysUsein18MoPrior', 'CumOPMHDaysUsein1MoPrior', 'CumOPMHDaysUsein24MoPrior', 'CumOPMHDaysUsein2MoPrior', 'CumOPMHDaysUsein3MoPrior', 'CumOPMHDaysUsein6MoPrior', 'DaysLastED', 'DaysLastIPsub', 'DaysLastMHbed', 'DaysLastResbed', 'DaysLastResSub', 'DaysUsein10MoPrior', 'DaysUsein11MoPrior', 'DaysUsein12MoPrior', 'DaysUsein13MoPrior', 'DaysUsein14MoPrior', 'DaysUsein15MoPrior', 'DaysUsein16MoPrior', 'DaysUsein17MoPrior', 'DaysUsein18MoPrior', 'DaysUsein19MoPrior', 'DaysUsein1MoPrior', 'DaysUsein20MoPrior', 'DaysUsein21MoPrior', 'DaysUsein22MoPrior', 'DaysUsein23MoPrior', 'DaysUsein24MoPrior', 'DaysUsein2MoPrior', 'DaysUsein3MoPrior', 'DaysUsein4MoPrior', 'DaysUsein5MoPrior', 'DaysUsein6MoPrior', 'DaysUsein7MoPrior', 'DaysUsein8MoPrior', 'DaysUsein9MoPrior', 'dob', 'EDvisits_prior1', 'EDvisits_prior12', 'EDvisits_prior18', 'EDvisits_prior2', 'EDvisits_prior24', 'EDvisits_prior3', 'EDvisits_prior6', 'FirstUse5yr', 'IPDaysUsein10MoPrior', 'IPDaysUsein11MoPrior', 'IPDaysUsein12MoPrior', 'IPDaysUsein13MoPrior', 'IPDaysUsein14MoPrior', 'IPDaysUsein15MoPrior', 'IPDaysUsein16MoPrior', 'IPDaysUsein17MoPrior', 'IPDaysUsein18MoPrior', 'IPDaysUsein19MoPrior', 'IPDaysUsein1MoPrior', 'IPDaysUsein20MoPrior', 'IPDaysUsein21MoPrior', 'IPDaysUsein22MoPrior', 'IPDaysUsein23MoPrior', 'IPDaysUsein24MoPrior', 'IPDaysUsein2MoPrior', 'IPDaysUsein3MoPrior', 'IPDaysUsein4MoPrior', 'IPDaysUsein5MoPrior', 'IPDaysUsein6MoPrior', 'IPDaysUsein7MoPrior', 'IPDaysUsein8MoPrior', 'IPDaysUsein9MoPrior', 'IPMHDaysUsein10MoPrior', 'IPMHDaysUsein11MoPrior', 'IPMHDaysUsein12MoPrior', 'IPMHDaysUsein13MoPrior', 'IPMHDaysUsein14MoPrior', 'IPMHDaysUsein15MoPrior', 'IPMHDaysUsein16MoPrior', 'IPMHDaysUsein17MoPrior', 'IPMHDaysUsein18MoPrior', 'IPMHDaysUsein19MoPrior', 'IPMHDaysUsein1MoPrior', 'IPMHDaysUsein20MoPrior', 'IPMHDaysUsein21MoPrior', 'IPMHDaysUsein22MoPrior', 'IPMHDaysUsein23MoPrior', 'IPMHDaysUsein24MoPrior', 'IPMHDaysUsein2MoPrior', 'IPMHDaysUsein3MoPrior', 'IPMHDaysUsein4MoPrior', 'IPMHDaysUsein5MoPrior', 'IPMHDaysUsein6MoPrior', 'IPMHDaysUsein7MoPrior', 'IPMHDaysUsein8MoPrior', 'IPMHDaysUsein9MoPrior', 'IPsub_prior12', 'IPsub_prior24', 'LastEDVisit', 'MHbed_prior12', 'MHbed_prior24', 'mhdisprior10mo', 'mhdisprior11mo', 'mhdisprior12mo', 'mhdisprior13mo', 'mhdisprior14mo', 'mhdisprior15mo', 'mhdisprior16mo', 'mhdisprior17mo', 'mhdisprior18mo', 'mhdisprior19mo', 'mhdisprior1mo', 'mhdisprior20mo', 'mhdisprior21mo', 'mhdisprior22mo', 'mhdisprior23mo', 'mhdisprior24mo', 'mhdisprior2mo', 'mhdisprior3mo', 'mhdisprior4mo', 'mhdisprior5mo', 'mhdisprior6mo', 'mhdisprior7mo', 'mhdisprior8mo', 'mhdisprior9mo', 'month', 'Month_Ind', 'OPDaysUsein10MoPrior', 'OPDaysUsein11MoPrior', 'OPDaysUsein12MoPrior', 'OPDaysUsein13MoPrior', 'OPDaysUsein14MoPrior', 'OPDaysUsein15MoPrior', 'OPDaysUsein16MoPrior', 'OPDaysUsein17MoPrior', 'OPDaysUsein18MoPrior', 'OPDaysUsein19MoPrior', 'OPDaysUsein1MoPrior', 'OPDaysUsein20MoPrior', 'OPDaysUsein21MoPrior', 'OPDaysUsein22MoPrior', 'OPDaysUsein23MoPrior', 'OPDaysUsein24MoPrior', 'OPDaysUsein2MoPrior', 'OPDaysUsein3MoPrior', 'OPDaysUsein4MoPrior', 'OPDaysUsein5MoPrior', 'OPDaysUsein6MoPrior', 'OPDaysUsein7MoPrior', 'OPDaysUsein8MoPrior', 'OPDaysUsein9MoPrior', 'OPMHDaysUsein10MoPrior', 'OPMHDaysUsein11MoPrior', 'OPMHDaysUsein12MoPrior', 'OPMHDaysUsein13MoPrior', 'OPMHDaysUsein14MoPrior', 'OPMHDaysUsein15MoPrior', 'OPMHDaysUsein16MoPrior', 'OPMHDaysUsein17MoPrior', 'OPMHDaysUsein18MoPrior', 'OPMHDaysUsein19MoPrior', 'OPMHDaysUsein1MoPrior', 'OPMHDaysUsein20MoPrior', 'OPMHDaysUsein21MoPrior', 'OPMHDaysUsein22MoPrior', 'OPMHDaysUsein23MoPrior', 'OPMHDaysUsein24MoPrior', 'OPMHDaysUsein2MoPrior', 'OPMHDaysUsein3MoPrior', 'OPMHDaysUsein4MoPrior', 'OPMHDaysUsein5MoPrior', 'OPMHDaysUsein6MoPrior', 'OPMHDaysUsein7MoPrior', 'OPMHDaysUsein8MoPrior', 'OPMHDaysUsein9MoPrior', 'percentserviceconnect', 'prior0to3', 'prior4to6', 'race', 'race2', 'region', 'Resbed_prior12', 'Resbed_prior24', 'ResSub_prior12', 'ResSub_prior24', 'serviceconnectedgroup', 'UCvisits_prior1', 'UCvisits_prior12', 'UCvisits_prior18', 'UCvisits_prior2', 'UCvisits_prior24', 'UCvisits_prior3', 'UCvisits_prior6', 'weight_pm', 'white', 'YearsSinceFirstUse']})
        self.feature_names =input_dict['feature_names']
        self.df = pd.DataFrame(input_dict['dataset'][0], columns=[input_dict['feature_names'][0]])
        for i in range(len(self.feature_names)):
            self.df[input_dict['feature_names'][i]] = input_dict['dataset'][i]        
            self.train_params = {
            'continous_features':['age', 'Attempt1', 'Attempt12', 'Attempt18', 'Attempt2', 'Attempt24', 'Attempt3', 'Attempt6', 'change_divide', 'change_Sq', 'change_subtract', 'CumDaysUsein12MoPrior', 'CumDaysUsein18MoPrior', 'CumDaysUsein1MoPrior', 'CumDaysUsein24MoPrior', 'CumDaysUsein2MoPrior', 'CumDaysUsein3MoPrior', 'CumDaysUsein6MoPrior', 'CumIPDaysUsein12MoPrior', 'CumIPDaysUsein18MoPrior', 'CumIPDaysUsein1MoPrior', 'CumIPDaysUsein24MoPrior', 'CumIPDaysUsein2MoPrior', 'CumIPDaysUsein3MoPrior', 'CumIPDaysUsein6MoPrior', 'CumIPMHDaysUsein12MoPrior', 'CumIPMHDaysUsein18MoPrior', 'CumIPMHDaysUsein1MoPrior', 'CumIPMHDaysUsein24MoPrior', 'CumIPMHDaysUsein2MoPrior', 'CumIPMHDaysUsein3MoPrior', 'CumIPMHDaysUsein6MoPrior', 'CumOPDaysUsein12MoPrior', 'CumOPDaysUsein18MoPrior', 'CumOPDaysUsein1MoPrior', 'CumOPDaysUsein24MoPrior', 'CumOPDaysUsein2MoPrior', 'CumOPDaysUsein3MoPrior', 'CumOPDaysUsein6MoPrior', 'CumOPMHDaysUsein12MoPrior', 'CumOPMHDaysUsein18MoPrior', 'CumOPMHDaysUsein1MoPrior', 'CumOPMHDaysUsein24MoPrior', 'CumOPMHDaysUsein2MoPrior', 'CumOPMHDaysUsein3MoPrior', 'CumOPMHDaysUsein6MoPrior', 'DaysLastED', 'DaysLastIPsub', 'DaysLastMHbed', 'DaysLastResbed', 'DaysLastResSub', 'DaysUsein10MoPrior', 'DaysUsein11MoPrior', 'DaysUsein12MoPrior', 'DaysUsein13MoPrior', 'DaysUsein14MoPrior', 'DaysUsein15MoPrior', 'DaysUsein16MoPrior', 'DaysUsein17MoPrior', 'DaysUsein18MoPrior', 'DaysUsein19MoPrior', 'DaysUsein1MoPrior', 'DaysUsein20MoPrior', 'DaysUsein21MoPrior', 'DaysUsein22MoPrior', 'DaysUsein23MoPrior', 'DaysUsein24MoPrior', 'DaysUsein2MoPrior', 'DaysUsein3MoPrior', 'DaysUsein4MoPrior', 'DaysUsein5MoPrior', 'DaysUsein6MoPrior', 'DaysUsein7MoPrior', 'DaysUsein8MoPrior', 'DaysUsein9MoPrior', 'dob', 'EDvisits_prior1', 'EDvisits_prior12', 'EDvisits_prior18', 'EDvisits_prior2', 'EDvisits_prior24', 'EDvisits_prior3', 'EDvisits_prior6', 'FirstUse5yr', 'IPDaysUsein10MoPrior', 'IPDaysUsein11MoPrior', 'IPDaysUsein12MoPrior', 'IPDaysUsein13MoPrior', 'IPDaysUsein14MoPrior', 'IPDaysUsein15MoPrior', 'IPDaysUsein16MoPrior', 'IPDaysUsein17MoPrior', 'IPDaysUsein18MoPrior', 'IPDaysUsein19MoPrior', 'IPDaysUsein1MoPrior', 'IPDaysUsein20MoPrior', 'IPDaysUsein21MoPrior', 'IPDaysUsein22MoPrior', 'IPDaysUsein23MoPrior', 'IPDaysUsein24MoPrior', 'IPDaysUsein2MoPrior', 'IPDaysUsein3MoPrior', 'IPDaysUsein4MoPrior', 'IPDaysUsein5MoPrior', 'IPDaysUsein6MoPrior', 'IPDaysUsein7MoPrior', 'IPDaysUsein8MoPrior', 'IPDaysUsein9MoPrior', 'IPMHDaysUsein10MoPrior', 'IPMHDaysUsein11MoPrior', 'IPMHDaysUsein12MoPrior', 'IPMHDaysUsein13MoPrior', 'IPMHDaysUsein14MoPrior', 'IPMHDaysUsein15MoPrior', 'IPMHDaysUsein16MoPrior', 'IPMHDaysUsein17MoPrior', 'IPMHDaysUsein18MoPrior', 'IPMHDaysUsein19MoPrior', 'IPMHDaysUsein1MoPrior', 'IPMHDaysUsein20MoPrior', 'IPMHDaysUsein21MoPrior', 'IPMHDaysUsein22MoPrior', 'IPMHDaysUsein23MoPrior', 'IPMHDaysUsein24MoPrior', 'IPMHDaysUsein2MoPrior', 'IPMHDaysUsein3MoPrior', 'IPMHDaysUsein4MoPrior', 'IPMHDaysUsein5MoPrior', 'IPMHDaysUsein6MoPrior', 'IPMHDaysUsein7MoPrior', 'IPMHDaysUsein8MoPrior', 'IPMHDaysUsein9MoPrior', 'IPsub_prior12', 'IPsub_prior24', 'LastEDVisit', 'MHbed_prior12', 'MHbed_prior24', 'mhdisprior10mo', 'mhdisprior11mo', 'mhdisprior12mo', 'mhdisprior13mo', 'mhdisprior14mo', 'mhdisprior15mo', 'mhdisprior16mo', 'mhdisprior17mo', 'mhdisprior18mo', 'mhdisprior19mo', 'mhdisprior1mo', 'mhdisprior20mo', 'mhdisprior21mo', 'mhdisprior22mo', 'mhdisprior23mo', 'mhdisprior24mo', 'mhdisprior2mo', 'mhdisprior3mo', 'mhdisprior4mo', 'mhdisprior5mo', 'mhdisprior6mo', 'mhdisprior7mo', 'mhdisprior8mo', 'mhdisprior9mo', 'month', 'Month_Ind', 'OPDaysUsein10MoPrior', 'OPDaysUsein11MoPrior', 'OPDaysUsein12MoPrior', 'OPDaysUsein13MoPrior', 'OPDaysUsein14MoPrior', 'OPDaysUsein15MoPrior', 'OPDaysUsein16MoPrior', 'OPDaysUsein17MoPrior', 'OPDaysUsein18MoPrior', 'OPDaysUsein19MoPrior', 'OPDaysUsein1MoPrior', 'OPDaysUsein20MoPrior', 'OPDaysUsein21MoPrior', 'OPDaysUsein22MoPrior', 'OPDaysUsein23MoPrior', 'OPDaysUsein24MoPrior', 'OPDaysUsein2MoPrior', 'OPDaysUsein3MoPrior', 'OPDaysUsein4MoPrior', 'OPDaysUsein5MoPrior', 'OPDaysUsein6MoPrior', 'OPDaysUsein7MoPrior', 'OPDaysUsein8MoPrior', 'OPDaysUsein9MoPrior', 'OPMHDaysUsein10MoPrior', 'OPMHDaysUsein11MoPrior', 'OPMHDaysUsein12MoPrior', 'OPMHDaysUsein13MoPrior', 'OPMHDaysUsein14MoPrior', 'OPMHDaysUsein15MoPrior', 'OPMHDaysUsein16MoPrior', 'OPMHDaysUsein17MoPrior', 'OPMHDaysUsein18MoPrior', 'OPMHDaysUsein19MoPrior', 'OPMHDaysUsein1MoPrior', 'OPMHDaysUsein20MoPrior', 'OPMHDaysUsein21MoPrior', 'OPMHDaysUsein22MoPrior', 'OPMHDaysUsein23MoPrior', 'OPMHDaysUsein24MoPrior', 'OPMHDaysUsein2MoPrior', 'OPMHDaysUsein3MoPrior', 'OPMHDaysUsein4MoPrior', 'OPMHDaysUsein5MoPrior', 'OPMHDaysUsein6MoPrior', 'OPMHDaysUsein7MoPrior', 'OPMHDaysUsein8MoPrior', 'OPMHDaysUsein9MoPrior', 'percentserviceconnect', 'prior0to3', 'prior4to6', 'race', 'race2', 'region', 'Resbed_prior12', 'Resbed_prior24', 'ResSub_prior12', 'ResSub_prior24', 'serviceconnectedgroup', 'UCvisits_prior1', 'UCvisits_prior12', 'UCvisits_prior18', 'UCvisits_prior2', 'UCvisits_prior24', 'UCvisits_prior3', 'UCvisits_prior6', 'weight_pm', 'white', 'YearsSinceFirstUse'],
            'binary_features':['case', 'Alprazolam12', 'Alprazolam24', 'als12', 'als24', 'ami12', 'ami24', 'amp12', 'amp24', 'analgesic12', 'analgesic24', 'anticonvulsant12', 'anticonvulsant24', 'antidep12', 'antidep24', 'antipsy12', 'antipsy24', 'anyattempt1', 'anyattempt12', 'anyattempt18', 'anyattempt2', 'anyattempt24', 'anyattempt3', 'anyattempt6', 'anybipol12', 'anybipol24', 'AnyEDvisits_prior1', 'AnyEDvisits_prior12', 'AnyEDvisits_prior18', 'AnyEDvisits_prior2', 'AnyEDvisits_prior24', 'AnyEDvisits_prior3', 'AnyEDvisits_prior6', 'AnyIPSub_prior12', 'AnyIPSub_prior24', 'AnyMHBed_prior12', 'AnyMHBed_prior24', 'AnyMHDisPrior12Mos', 'AnyMHDisPrior1Mos', 'AnyMHDisPrior24Mos', 'AnyMHDisPrior3Mos', 'AnyMHDisPrior6Mos', 'anyMHdx12', 'anyMHdx24', 'anyMHtx12', 'anyMHtx24', 'AnyPain12', 'AnyPain24', 'AnyResBed_prior12', 'AnyResBed_prior24', 'AnyResSub_prior12', 'AnyResSub_prior24', 'AnyUCvisits_prior1', 'AnyUCvisits_prior12', 'AnyUCvisits_prior18', 'AnyUCvisits_prior2', 'AnyUCvisits_prior24', 'AnyUCvisits_prior3', 'AnyUCvisits_prior6', 'AnyUsein10MoPrior', 'AnyUsein11MoPrior', 'AnyUsein12MoPrior', 'AnyUsein13MoPrior', 'AnyUsein14MoPrior', 'AnyUsein15MoPrior', 'AnyUsein16MoPrior', 'AnyUsein17MoPrior', 'AnyUsein18MoPrior', 'AnyUsein19MoPrior', 'AnyUsein1MoPrior', 'AnyUsein20MoPrior', 'AnyUsein21MoPrior', 'AnyUsein22MoPrior', 'AnyUsein23MoPrior', 'AnyUsein24MoPrior', 'AnyUsein2MoPrior', 'AnyUsein3MoPrior', 'AnyUsein4MoPrior', 'AnyUsein5MoPrior', 'AnyUsein6MoPrior', 'AnyUsein7MoPrior', 'AnyUsein8MoPrior', 'AnyUsein9MoPrior', 'apnea12', 'apnea24', 'ARB12', 'ARB24', 'arth12', 'arth24', 'auto12', 'auto24', 'backpain12', 'backpain24', 'bipolI12', 'bipolI24', 'bipolII12', 'bipolII24', 'blind12', 'blind24', 'ca_head12', 'ca_head24', 'ca_prost12', 'ca_prost24', 'ca12', 'ca24', 'cad12', 'cad24', 'cancer12', 'cancer24', 'cess12', 'cess24', 'chronic12', 'chronic24', 'Clonazepam12', 'Clonazepam24', 'conc12', 'conc24', 'copd12', 'copd24', 'CurrIPSub', 'CurrMHBed', 'CurrResBed', 'CurrResSub', 'cush12', 'cush24', 'cva12', 'cva24', 'cvd12', 'cvd24', 'dementia12', 'dementia24', 'depr12', 'depr24', 'dm12', 'dm24', 'dt12', 'dt24', 'dysthymia12', 'dysthymia24', 'ep12', 'ep24', 'fib12', 'fib24', 'FirstUse1Yr', 'FirstUse2Yr', 'FirstUse3Yr', 'FirstUse4Yr', 'gu12', 'gu24', 'ha12', 'ha24', 'hc12', 'hc24', 'hear12', 'hear24', 'hemi12', 'hemi24', 'highrisk', 'hiv12', 'hiv24', 'homeless12', 'homeless24', 'hyp12', 'hyp24', 'lagca12', 'lagca24', 'Lorazepam12', 'Lorazepam24', 'mig12', 'mig24', 'Mirtazepam12', 'Mirtazepam24', 'moodst12', 'moodst24', 'ms12', 'ms24', 'neuro12', 'neuro24', 'opioid12', 'opioid24', 'OthAnxDis12', 'OthAnxDis24', 'OthPsych12', 'OthPsych24', 'par12', 'par24', 'Persond12', 'Persond24', 'psy12', 'psy24', 'ptsd12', 'ptsd24', 'ra12', 'ra24', 'rd12', 'rd24', 'relax12', 'relax24', 'schizo12', 'schizo24', 'sci12', 'sci24', 'sedative_anxiolytic12', 'sedative_anxiolytic24', 'Sildenafil12', 'Sildenafil24', 'sle12', 'sle24', 'sleep12', 'sleep24', 'Statin12', 'Statin24', 'sud12', 'sud24', 'symptom12', 'symptom24', 'tbi12', 'tbi24', 'TCA12', 'TCA24', 'tension12', 'tension24', 'thy12', 'thy24', 'tobacco12', 'tobacco24', 'topical12', 'topical24', 'Trazodone12', 'Trazodone24', 'use12', 'use24', 'vet12', 'vet24', 'vision12', 'vision24', 'Zolpidem12', 'Zolpidem24']
        }
        self.models = {}
    def split(self,df):
        mental_health_features = self.mental_health_feature_descriptor[self.mental_health_feature_descriptor['Mental illness']==1]
        mental_health_indicators = mental_health_features['Name'].to_list()
        mental_health_continous_indicators = []
        mental_health_binary_indicators = []

        for ind in mental_health_indicators:
            if ind in self.train_params['continous_features']:
                mental_health_continous_indicators.append(ind)
            else:
                mental_health_binary_indicators.append(ind)
        mhfd_tt = self.mental_health_feature_descriptor.dropna(subset=['TrainTarget'])
        drop_features = mhfd_tt[mhfd_tt['TrainTarget']<0]['Name'].to_list()
        self.train_features = mhfd_tt[mhfd_tt['TrainTarget']>0]['Name'].to_list()
        # df = self.df.drop(labels=drop_features,axis=1)
        Y = df[self.train_features]
        X = df.drop(self.train_features,axis=1)
        return X,Y

    def train(self,models_dir,device='gpu',backend='xgboost'):
        Path(models_dir).mkdir(parents='True',exist_ok='True')
        X,Y = self.split(self.df)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.2)
        t0 = time.perf_counter()
        train_log = {}
        self.models = {}
        if backend == 'xgboost':
            for i,target_feature in enumerate(self.train_features):

                print(f'{i}. Training {target_feature}')
                if device=='gpu':
                    model = xgb.XGBClassifier(tree_method='gpu_hist',n_estimators=30)
                else:
                    model = xgb.XGBClassifier(n_estimators=30)

                # model = xgb.XGBClassifier(n_estimators=5)
                tik = time.perf_counter()
                model = model.fit(Xtrain,Ytrain[target_feature])
                # models[target_feature] = model
                # joblib.dump(model,'models/'+target_feature+'.joblib')
                model.save_model(os.path.join(models_dir,target_feature+'.xgb'))
                tok = time.perf_counter()

                y_true = Ytest[target_feature]
                y_pred = model.predict(Xtest)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                jaccard = jaccard_score(y_true, y_pred)

                print('Test Accuracy: ',acc)
                # xgb.plot_importance(model,max_num_features=20)
                # plt.show()
                print(f'Time taken: {tok-tik}, Total Time: {tok-t0}')
                train_log[target_feature] = {
                    'Test Accuracy': acc,
                    'Time taken': tok-tik,
                    'Test F1 score': f1,
                    'Test Precision': precision,
                    'Test Recall': recall,
                    'Test Jaccard score': jaccard
                    
                }

        return train_log


    def __call__(self,models_dir):
        X,_ = self.split(self.df)
        output= {}
        for model_name in os.listdir(models_dir):
            if not model_name.endswith('.xgb'):
                continue
            model_path = os.path.join(models_dir,model_name)
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            target_feature = model_name[:-4]
            Y = model.predict_proba(X)
            # Y = model.predict(X)
            # output[target_feature] = Y
            output[target_feature] = Y[:,1].tolist()
        return output

            
