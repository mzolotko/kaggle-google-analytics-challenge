# -*- coding: utf-8 -*-
import click
import os, logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import glob
import shutil
from src.utils import get_logger




@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs a script to concatenate all parts of X_train and y 
        that were calculated in create_X_y_roll_windows
        for each time window separately.
        X_pred is just copied from input_filepath to output_filepath
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

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
