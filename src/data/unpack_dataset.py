# -*- coding: utf-8 -*-
import click
import os, logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from ast import literal_eval
from src.utils import get_logger
import time


def unpack(df, jsoncols):
    '''
    Parameters
    ----------
    df : initial df
    jsoncols : columns to unpack

    Returns
    -------
    df : dataframe with each jsoncol replaced by  new columns corresponding 
    to each json field in it
        
    The function unpacks jsoncols. Each jsoncol contains a string consisting of a json record where 
    json fields are the same for many rows, e.g. '{"browser": "Chrome", 
    "browserVersion": "not available in demo dataset",...}'. 
    '''
    for jc in jsoncols:
        # use index of df to enable joining (see below)
        flat_df = pd.DataFrame(df.pop(jc).apply(pd.io.json.loads).values.tolist(), index = df.index)
        flat_df.columns = ['{}_{}'.format(jc, c) for c in flat_df.columns]
        df = df.join(flat_df)
    return df

def unpack_2(df, jsoncols):
    '''
    Parameters
    ----------
    df : initial df
    jsoncols : columns to unpack

    Returns
    -------
    df : dataframe with each jsoncol replaced by  new columns corresponding 
    to each json field in it
    
    The function also unpacks json columns, but these columns are packed in a different way. 
    Each column value is a string of the pattern "[{'index': '4', 'value': 'APAC'}]"
    Some values are strings containing an empty list "[]"
    This is why literal evaluation comes into play.
     Written for one particular column (customDimensions)
    '''
    for jc in jsoncols:  # parse json
        flat_df = df[jc].apply(literal_eval).apply(pd.json_normalize).tolist()  
        # result is a list of dataframes, each has one row, row index=0 and columns 
        # 'index' and 'value' according to the given json format
        # cannot make a DataFrame from the list right away because some elements are empty dataframes
        # they would just collapsed, but they should be kept as NaN
        for i in range(len(flat_df)):
            if flat_df[i].empty:
                # just insert some dataframe with the correct format. 
                # The dfs will be concatenated, and if the corresponding df was empty, 
                # the values in the concatenated df will be replaced with NaN in the next step
                flat_df[i] = flat_df[i-1]  
            # next line makes a new index, otherwise it is 0 for all dfs
            flat_df[i] = pd.DataFrame(flat_df[i].values, columns=flat_df[i].columns, index=[i])
        flat_df = pd.concat(flat_df, sort=True)
        # use index of df to enable joining (see below)
        flat_df = pd.DataFrame(flat_df.values, columns=flat_df.columns, index=df.index)
        flat_df.columns = ['{}_{}'.format(jc, c) for c in flat_df.columns]
        df = df.join(flat_df)
        for col in flat_df.columns:
            df.loc[df[jc] == '[]', col] = np.nan
        df.drop(jc, axis=1, inplace=True)        
    return df

def unpack_3(df, jsoncols):
    '''
    
    Parameters
    ----------
    df : initial df
    jsoncols : columns to unpack

    Returns
    -------
    df : dataframe with each jsoncol replaced by  new columns corresponding 
    to each json field in it
    
    Another function for unpacking json columns packed in a third way: each column value is just 
    a dict. Written for one particular column  (trafficSource_adwordsClickInfo)
    '''
    for col in jsoncols :
        df = df.copy()
        flat_df = df.pop(col).apply(pd.Series)
        flat_df = pd.DataFrame(flat_df.values, columns = flat_df.columns, index = df.index)
        flat_df.columns = ['{}_{}'.format(col, c) for c in flat_df.columns]
        df = df.join(flat_df)
    return df  

def unpack_hits(df, cols_to_unpack, required_cols):
    '''
    df : initial df
    cols_to_unpack : columns to unpack
    required_cols: columns to be left in the returned df

    Returns
    -------
    df : dataframe with unpacked column only
    
    Another function for unpacking json columns packed in the following way:
    "[{'index': '4', 'value': 'APAC', ...}]" (similar to unpack_2), BUT
    the list can have more than one element - one for every "hit".
    In addition some values can themselves be jsons (dicts) or lists of jsons (dicts)
    Written for one particular column (hits).
    '''
    for jc in cols_to_unpack:  # parse json
        flat_df = df[jc].apply(literal_eval).apply(pd.json_normalize).tolist()  
        # in flat_df each element is a dataframe consisting of one or several rows
        # each row represents one hit. Each df represents one session of one user)
        for i in range(len(flat_df)):
            if flat_df[i].empty: # if the value in the original hist column is '[]',
                                 # this will result in an empty df
                # Create a DataFrame composed of NAs with all required columns and a multiindex (see below)
                flat_df[i] = pd.DataFrame(np.tile(np.nan,(1, len(required_cols))), columns = required_cols, 
                                        index = pd.MultiIndex.from_product([[i],[0]],  names=['ind_exter', 'ind_inter']))
            else:
                iterables = [[i], flat_df[i].index]
                # multiindex necessary because each cell contained more than one entry of several json fields
                m_index = pd.MultiIndex.from_product(iterables, names=['ind_exter', 'ind_inter'])
                
                flat_df[i] = pd.DataFrame(flat_df[i].values, columns = flat_df[i].columns, index = m_index)
                
        flat_df = pd.concat(flat_df, sort=True)
        # As a result we obtain a df with the number of rows equal to
        # the total number of hits, i.e.
        # the sum of hits for all individual sessions
        flat_df.columns = ['{}_{}'.format(jc, c) for c in flat_df.columns]
        flat_df = flat_df[required_cols]
        # we don't join the result with the initial df. We return it as is
        # and aggregate to the session level later
    return flat_df 

def count_cart(x):
    '''
    Parameters
    ----------
    x : Series

    Returns
    -------
    number of occurrences of "Add to Cart" in the series
    '''
    return (x == 'Add to Cart').sum()

def count_page(x):
    
    '''
    Parameters
    ----------
    x : Series

    Returns
    -------
    number of occurrences of "PAGE" in the series'''
    return (x == 'PAGE').sum()


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs the script to "unpack" the train and test datasets.
        Many columns in train and test have (possibly multilevel) json elements 
        rather than just string or numeric values. The scipt unpacks 
        more all json columns, but does not turn all json fields into columns.
        Some json fields are excluded in advance, some newly created columns
        are deleted after the creation (imitating the real-life situation where
        some columns can be declared irrelevant after viewing the logs).
        "Hits" columns contains information on several hist per each row (web-session).
        The data for all hits are aggregated for each session.
    """

    
    if os.path.isfile('unpack_logging.log'):
        os.remove('unpack_logging.log')
    logger = get_logger('unpack_logging.log')
    logger.info('Unpacking data in chunks and saving - process started')
    print('Unpacking data in chunks and saving - process started')
    
    for fil in os.listdir(output_filepath):
        os.remove(os.path.join(output_filepath, fil))

    jsoncols = ['device',  'geoNetwork',  'totals', 'trafficSource']
    cols_cust_dim = [ 'customDimensions']    
    col_adclick = ['trafficSource_adwordsClickInfo']
    # col_to_investigate are supposedly useless columns (with a single value throughout)
    col_to_investigate = ['device_browserSize', 'device_browserVersion', 
                          'device_flashVersion', 'device_language',
                          'device_mobileDeviceBranding',	'device_mobileDeviceInfo',	
                          'device_mobileDeviceMarketingName', 'device_mobileDeviceModel',	
                          'device_mobileInputSelector', 'device_operatingSystemVersion',	
                          'device_screenColors',	'device_screenResolution', 
                          'geoNetwork_latitude',	'geoNetwork_longitude',
                          'geoNetwork_networkLocation', 'trafficSource_adwordsClickInfo_adNetworkType',
                          'trafficSource_adwordsClickInfo_criteriaParameters',	
                          'trafficSource_adwordsClickInfo_isVideoAd', 
                          'trafficSource_adwordsClickInfo_page']
    
    # these are based on exploratory analysis of the columns where feasible,
    #  on the logs created below and also on the common sense
    col_to_drop = ['date', 'visitId', 'device_browserSize', 'device_browserVersion', 
                   'device_flashVersion', 'device_language', 'device_mobileDeviceBranding',	
                   'device_mobileDeviceInfo',	'device_mobileDeviceMarketingName', 
                   'device_mobileDeviceModel',	'device_mobileInputSelector',
                   'device_operatingSystemVersion',	'device_screenColors',	
                   'device_screenResolution', 'geoNetwork_latitude',	
                   'geoNetwork_longitude','geoNetwork_metro', 'geoNetwork_networkLocation',	
                   'geoNetwork_region',	'geoNetwork_subContinent',
                   'trafficSource_adwordsClickInfo_adNetworkType', 
                   'trafficSource_adwordsClickInfo_criteriaParameters',	
                   'trafficSource_adwordsClickInfo_gclId','trafficSource_adwordsClickInfo_isVideoAd', 
                   'trafficSource_adwordsClickInfo_page', 
                   'trafficSource_adwordsClickInfo_targetingCriteria', 
                   'trafficSource_adwordsClickInfo_slot',
                   'trafficSource_campaignCode', 
                   'geoNetwork_cityId' , 'socialEngagementType', 
                   'customDimensions_index', 'customDimensions_value',
                   'trafficSource_medium', 'trafficSource_referralPath',
                     'totals_visits',
                    'totals_transactions', 'visitNumber',
                    'geoNetwork_networkDomain',
                    'device_isMobile', 'geoNetwork_continent',
                    'geoNetwork_country', 'totals_newVisits'
                   ]
    
    # fields in the hits column to be left (according to sample exploratory analysis)
    hits_required_cols = [
                          'hits_contentGroup.contentGroup2', 
                          'hits_eventInfo.eventAction', 
                          'hits_promotionActionInfo.promoIsView',
                          'hits_referer',
                          'hits_social.socialNetwork', 
                          'hits_type']

    for file_name in ['train', 'test']:
        logger.info('Working on the {} dataset, all columns apart from hits'.format(file_name))
        print('Working on the {} dataset, all columns apart from hits'.format(file_name))
        # reading in chunks because train dataset is over 16 GB
        reader = pd.read_csv(os.path.join(input_filepath, file_name + '_v2.csv'), 
                       dtype = {'fullVisitorId': 'str'}, chunksize = 150000, 
                       usecols = lambda x: x not in ['hits'], index_col=False)
        
        # processing all columns except "hits"
        for i, chunk in enumerate(reader):
            logger.info('chunk number: {}'.format(i))
            print('chunk number: {}'.format(i))
            # drop first unnamed column, which is supposed to be the index
            # the range index was created automatically
            chunk.drop(columns=[chunk.columns[0]], inplace=True)
            df = unpack(chunk, jsoncols)
            df = unpack_2(df, cols_cust_dim)
            df = unpack_3(df, col_adclick)
            for col in col_to_investigate:
                if col in df.columns:
                    logger.info(df[col].value_counts())  
                    # to check if these columns can be of any interest,
                    # value_counts is self-explanatory, column name is printed automatically
            if 'trafficSource_adwordsClickInfo_targetingCriteria' in df.columns:
                logger.info('targetingCriteria NA mean: {:.2f}'.format(pd.isna(df['trafficSource_adwordsClickInfo_targetingCriteria']).mean()))
            df.drop(col_to_drop, axis =1, inplace = True, errors='ignore')
    
            # save each processed chunk of initial files without the hits column
            df.to_pickle(os.path.join(output_filepath, file_name + '_no_hits_{}.zip'.format(i)))
            
            
            # now concatenate each saved part
        without_hits_lst = []
        i = 0
        while os.path.isfile(os.path.join(output_filepath, file_name+ '_no_hits_{}.zip'.format(i))):
            nxt_fil = pd.read_pickle(os.path.join(output_filepath, file_name+ '_no_hits_{}.zip'.format(i)))
            logger.info('{}_no_hits_{}.zip: shape {}'.format(file_name, i, nxt_fil.shape))
            print('{}_no_hits_{}.zip: shape {}'.format(file_name, i, nxt_fil.shape))
            without_hits_lst.append(nxt_fil)
            i += 1
        conc_without_hits = pd.concat(without_hits_lst, axis=0,  ignore_index=True)
    
        # processing columns hits
        time_0 = time.time()
        reader = pd.read_csv(os.path.join(input_filepath, file_name + '_v2.csv'), 
                             dtype = {'fullVisitorId': 'str'}, chunksize = 40000, 
                             usecols = ['hits'], index_col=False)
        logger.info('Working on the {} dataset, the hits column'.format(file_name))
        print('Working on the {} dataset, the hits column'.format(file_name))
        
        for i, chunk in enumerate(reader):
            logger.info('chunk number: {}'.format(i))
            print('chunk number: {}'.format(i))
            hits = chunk[['hits']]

            unp_hits = unpack_hits(hits, hits.columns, hits_required_cols)
            logger.info('unpacking finished')
            print('unpacking finished, {} minutes elapsed'.format((time.time() - time_0) / 60))
          
            # aggregate hits data (that are on hit level) to the session level
            unp_hits_agg = unp_hits[[
                                     'hits_contentGroup.contentGroup2', 
                                     'hits_eventInfo.eventAction',
                                     'hits_promotionActionInfo.promoIsView',
                                     'hits_referer',
                                     'hits_social.socialNetwork', 
                                     'hits_type']].groupby(level=0).agg({
        
                                
                                'hits_contentGroup.contentGroup2':  'nunique',
                                'hits_eventInfo.eventAction': count_cart,
                                'hits_promotionActionInfo.promoIsView': 'count',
                                'hits_referer': 'first',
                                'hits_social.socialNetwork': 'first',
                                'hits_type': count_page
                                })
            logger.info('aggregation of hits data finished')
            print('aggregation of hits data finished, {} minutes elapsed'.format((time.time() - time_0) / 60))
            unp_hits_agg.to_pickle(os.path.join(output_filepath, 
                                                file_name + '_agg_hits_{}.zip'.format(i)))
        
        
        # concatenate each saved part 
        hits_lst = []
        i = 0
        while os.path.isfile(os.path.join(output_filepath, file_name+ '_agg_hits_{}.zip'.format(i))):
            nxt_fil = pd.read_pickle(os.path.join(output_filepath, file_name+ '_agg_hits_{}.zip'.format(i)))
            logger.info('{}_agg_hits_{}.zip: shape {}'.format(file_name, i, nxt_fil.shape))
            print('{}_agg_hits_{}.zip: shape {}'.format(file_name, i, nxt_fil.shape))
            hits_lst.append(nxt_fil)
            i += 1
        conc_hits = pd.concat(hits_lst, axis=0, ignore_index=True)
        
        # concatenate "no_hits" and "hits" parts for each of [train, test] together
        logger.info('Concatenating {}'.format(file_name))
        print('Concatenating {}'.format(file_name))
        hits_nothits = pd.concat([conc_without_hits, conc_hits], axis = 1)
        del conc_without_hits, conc_hits
        hits_nothits.to_pickle(os.path.join(output_filepath, file_name + '_conc.zip'))
    
    # finally, concatenate train_conc and test_conc and get just one file
    # we can concatenate them because in this competition 
    # the "test" dataset is effectively used for training
    train = pd.read_pickle(os.path.join(output_filepath, 'train_conc.zip'))
    test = pd.read_pickle(os.path.join(output_filepath, 'test_conc.zip'))
    logger.info('Concatenating train and test')
    print('Concatenating train and test')
    conc = pd.concat([train, test], ignore_index = True, axis = 0)
    conc.to_pickle(os.path.join(output_filepath, 'conc.zip'))
    logger.info('Process finished')
    print('Process finished')
    logging.shutdown()    
                
if __name__ == '__main__':

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
