# -*- coding: utf-8 -*-
import click
import logging, os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
#from collections import Counter
import glob
#from pandas.io.json import json_normalize
#from ast import literal_eval




def get_logger(logfile):
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    if (logger_.hasHandlers()):
        logger_.handlers.clear()
    logger_.addHandler(fh)
    #logger_.addHandler(ch)

    return logger_





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
        X_frag = pd.read_pickle(os.path.join(input_filepath, fil))
        X_list.append(X_frag)
    X_concat = pd.concat(X_list, axis = 0)
    X_concat.to_pickle(os.path.join(output_filepath, 'X_train_concat.zip'))
    del X_list, X_concat
    
    
    
    for fil in sorted(glob.glob(os.path.join(input_filepath, 'y*.*'))):
        y_frag = pd.read_pickle(os.path.join(input_filepath, fil))
        y_list.append(y_frag)
    y_concat = pd.concat(y_list, axis = 0)
    y_concat.to_pickle(os.path.join(output_filepath, 'y_concat.zip'))
    del y_list, y_concat
    


    

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
