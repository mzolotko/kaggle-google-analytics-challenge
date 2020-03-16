# -*- coding: utf-8 -*-
import click
import os, logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
#from collections import Counter
import glob
import shutil
from src.utils import get_logger
#from pandas.io.json import json_normalize
#from ast import literal_eval




@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        unpacked data (saved in ../unpacked).
    

    
    
    """
   
    
    if os.path.isfile('concat_X_y_logging.log'):
        os.remove('concat_X_y_logging.log')
    logger = get_logger('concat_X_y_logging.log')
    logger.info('Concatenating X and y rolling windows - process started')
    print('Concatenating X and y rolling windows - process started')
    
    for fil in os.listdir(output_filepath):
        os.remove(os.path.join(output_filepath, fil))
    
    y_list = []
    X_list = []
    
    for fil in sorted(glob.glob(os.path.join(input_filepath, 'X_train*.*'))):
        X_frag = pd.read_pickle(fil)
        X_list.append(X_frag)
    X_concat = pd.concat(X_list, axis = 0)
    X_concat.to_pickle(os.path.join(output_filepath, 'X_train_concat.zip'))
    del X_list, X_concat
    
    
    
    for fil in sorted(glob.glob(os.path.join(input_filepath, 'y*.*'))):
        y_frag = pd.read_pickle(fil)
        y_list.append(y_frag)
    y_concat = pd.concat(y_list, axis = 0)
    y_concat.to_pickle(os.path.join(output_filepath, 'y_concat.zip'))
    del y_list, y_concat
    
    # X_pred should just be copied over
    shutil.copy(os.path.join(input_filepath, 'X_pred.zip'), output_filepath)
    

    logger.info('Process finished')
    print('Process finished')
    logging.shutdown()
    
if __name__ == '__main__':
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
