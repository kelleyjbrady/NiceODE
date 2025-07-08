import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy

@dataclass
class InitValsPd:

    model_coeff:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = str))
    log_name:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = str))
    model_coeff_dep_var:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = str))
    population_coeff:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = bool))
    model_error:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = bool))
    init_val:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = np.float64))
    model_coeff_lower_bound:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = np.float64))
    model_coeff_upper_bound:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = np.float64))
    allometric:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = bool))
    allometric_norm_value:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = np.float64))
    subject_level_intercept:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = bool))
    subject_level_intercept_name:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = str))
    subject_level_intercept_sd_init_val:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = np.float64))
    subject_level_intercept_init_vals_column_name:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = np.float64))
    subject_level_intercect_sd_lower_bound:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = np.float64))
    subject_level_intercect_sd_upper_bound:np.ndarray = field(default_factory=lambda: np.empty((0,),dtype = np.float64))
    

    def df(self):
        df = pd.DataFrame({c:self.__dict__[c] for c in self.__dataclass_fields__ })
        return df.copy()

@dataclass
class InitValsPdCols:

    model_coeff:str = "model_coeff" 
    log_name:str = "log_name" 
    model_coeff_dep_var:str = "model_coeff_dep_var" 
    population_coeff:str = "population_coeff" 
    model_error:str = "model_error" 
    init_val:str = "init_val" 
    model_coeff_lower_bound:str = "model_coeff_lower_bound" 
    model_coeff_upper_bound:str = "model_coeff_upper_bound" 
    allometric:str = "allometric" 
    allometric_norm_value:str = "allometric_norm_value" 
    subject_level_intercept:str = "subject_level_intercept" 
    subject_level_intercept_name:str = "subject_level_intercept_name" 
    subject_level_intercept_sd_init_val:str = "subject_level_intercept_sd_init_val" 
    subject_level_intercept_init_vals_column_name:str = "subject_level_intercept_init_vals_column_name" 
    subject_level_intercect_sd_lower_bound:str = "subject_level_intercect_sd_lower_bound" 
    subject_level_intercect_sd_upper_bound:str = "subject_level_intercect_sd_upper_bound" 
    fix_param_value:str = "fix_param_value"
    fix_subject_level_effect:str = "fix_subject_level_effect"
    
    def __post_init__(self, ):
        self.pd_dtypes = [
        pd.StringDtype(),
        pd.StringDtype(), 
        pd.StringDtype(), 
        pd.BooleanDtype(), 
        pd.BooleanDtype(), 
        pd.Float64Dtype(), #init_val
        pd.Float64Dtype(), 
        pd.Float64Dtype(), 
        pd.BooleanDtype(), #allometric
        pd.Float64Dtype(), 
        pd.BooleanDtype(), 
        pd.StringDtype(), #subject_level_intercept_name
        pd.Float64Dtype(), 
        pd.StringDtype(), 
        pd.Float64Dtype(), 
        pd.Float64Dtype(), 
        pd.BooleanDtype(), 
        pd.BooleanDtype()  
        ]
    def validate_df_row(self,df_row_dict:dict):
        new_keys = [i for i in df_row_dict]
        template_keys = [i for i in self.__dataclass_fields__]
        
        new_df_key_in_template = np.isin(new_keys, template_keys)
        template_in_new_keys = np.isin(template_keys, new_keys)
        
        test1 = np.all(new_df_key_in_template)
        if not test1:
            extra_keys = new_keys[new_df_key_in_template == False]
            raise ValueError(f'`{extra_keys}` is/are provided but only {template_keys} should be provided')

        test2 = np.all(template_in_new_keys)
        if not test2:
            missing_keys = template_keys[template_in_new_keys == False]
            raise ValueError(f'`{missing_keys}` is/are not provided. {template_keys} should be provided.')
        
        test3 = new_keys == template_keys
        if not test3:
            new_df_row = {
                k:df_row_dict[k] for k in self.__dataclass_fields__
            }
            df_row_dict = deepcopy(new_df_row)
        
        return df_row_dict
            
