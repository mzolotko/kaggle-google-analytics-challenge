import numpy as np
import pandas as pd
import lightgbm as lgb
import click
import os, logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import get_logger
from src.configs import lgb_params
import json




@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('submission_filepath', type=click.Path(exists=True))
@click.argument('output_prediction_filepath', type=click.Path())
@click.argument('output_trained_filepath', type=click.Path())
def main(input_filepath, submission_filepath, 
         output_prediction_filepath, output_trained_filepath):
    """ Runs a script to train a model using the prepecified hyperparameters
        and using previously created X_train adn y; uses the trained model
        to make a prediction 
    """
    
    
    if os.path.isfile('training_logging.log'):
        os.remove('training_logging.log')
    logger = get_logger('training_logging.log')
    logger.info('Model training - process started')
    print('Model training - process started')
    
    for output_filepath in [output_prediction_filepath, output_trained_filepath]:
        for fil in os.listdir(output_filepath):
            os.remove(os.path.join(output_filepath, fil))
        
    X_train = pd.read_pickle(os.path.join(input_filepath, 'X_train_concat.zip'))
    X_pred = pd.read_pickle(os.path.join(input_filepath, 'X_pred.zip'))
    y_train = pd.read_pickle(os.path.join(input_filepath, 'y_concat.zip'))
    y_log = np.log1p(y_train)
    
    #  LightGBM requires an indication what features are categorical
    
    cat_feat = ['channelGrouping_MODE', 'device_browser_MODE', 
                'device_deviceCategory_MODE', 'device_operatingSystem_MODE', 
                'geoNetwork_city_MODE', 'referSocNetwork_MODE']
    
    dtrain = lgb.Dataset(X_train, y_log, free_raw_data=False, silent=False, 
                     categorical_feature= cat_feat)
    
    lgbm_model = lgb.train(params = lgb_params, 
                           train_set = dtrain,  
                           valid_sets = [dtrain],
                           categorical_feature=cat_feat, 
                           verbose_eval = 100
                           )
    
    print('Saving model')
    logger.info('Saving model')
    lgbm_model.save_model(os.path.join(output_trained_filepath, 'lgbm_model.txt'))
    print('Dumping model to JSON')
    logger.info('Dumping model to JSON')
    # dump model to JSON (and save to file)
    model_json = lgbm_model.dump_model()
    with open(os.path.join(output_trained_filepath, 'model.json'), 'w+') as f:
        json.dump(model_json, f, indent=4)
    
    y_log_pred = lgbm_model.predict(X_pred)
    y_log_pred_pos = y_log_pred.copy()
    y_log_pred_pos[y_log_pred_pos<0] = 0 # forecasts can turn out to be negative
    y_log_pred_2_months = np.log1p(np.expm1(y_log_pred_pos) * 0.6)  
    # 0.6 is an approximate way to scale down the total revenue from the actual 
    # forecast period (108 days) to two months (~60 days)
    subm = pd.read_csv(os.path.join(submission_filepath, 'sample_submission_v2.csv'), 
                       dtype = {'fullVisitorId': 'str'},  index_col = 'fullVisitorId')
    y_log_pred_2_months_df = pd.DataFrame(y_log_pred_2_months, index = X_pred.index)
    fin_subm = subm.join(y_log_pred_2_months_df)
    fin_subm.drop(['PredictedLogRevenue'], axis = 1, inplace = True)
    fin_subm.rename(columns = {0: 'PredictedLogRevenue'}, inplace = True)
    fin_subm.to_csv(os.path.join(output_prediction_filepath, 'fin_subm.csv'))
    
    
    logger.info('Process finished')
    print('Process finished')
    logging.shutdown()
    
    
if __name__ == '__main__':

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()