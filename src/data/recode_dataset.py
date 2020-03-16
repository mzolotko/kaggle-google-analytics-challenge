# -*- coding: utf-8 -*-
import click
import os, logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from src.utils import get_logger
#from pandas.io.json import json_normalize
#from ast import literal_eval



def rename_cols(df):
    
    df = df.copy()
    df = df.rename(columns={
                          'hits_contentGroup.contentGroup2': 'count_ViewedCat', 
                          'hits_eventInfo.eventAction': 'count_AddCart',
                          'hits_promotionActionInfo.promoIsView' : 'count_PromoView', 
                          'hits_referer' : 'referURL',
                          'hits_social.socialNetwork' : 'referSocNetwork', 
                          'hits_type': 'countPageViews'})
    return df



########## convert to int
def fillna_convert(df):
    
    df = df.copy()
    for col in ['totals_bounces', 'trafficSource_isTrueDirect', 
                'count_AddCart', 'count_PromoView', 
                'count_ViewedCat', 'totals_sessionQualityDim',
               'totals_timeOnSite']:
        df[col].fillna(0, inplace = True)
        df[col] = df[col].astype('int16')
    
    #if 'totals_transactionRevenue' in df.columns:
    for col in ['totals_transactionRevenue', 'totals_totalTransactionRevenue']:
        df[col].fillna(0, inplace = True)
        df[col] = df[col].astype('int64')
        
    df['referSocNetwork'].fillna('(not set)', inplace = True)
    
    # first fillna with countPageViews (as a proxy for a number of page views 
    # calculated based on one of the "hits" fields), 
    # then if there are still some NAs, use hits counts
    df.loc[pd.isnull(df['totals_pageviews']), 'totals_pageviews'] = \
        df.loc[pd.isnull(df['totals_pageviews']), 'countPageViews']
    df.loc[pd.isnull(df['totals_pageviews']), 'totals_pageviews'] = \
        df.loc[pd.isnull(df['totals_pageviews']), 'totals_hits']
    
    # countPageViews is no longer needed
    df.drop(['countPageViews'], axis =1, inplace = True, errors='ignore')
    
    df['totals_hits'] = df['totals_hits'].astype('int32')
    df['totals_pageviews'] = df['totals_pageviews'].astype('int32')
        
    return df

def create_date_cols(df):
    
    df = df.copy()
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit = 's')
    df.index = df['visitStartTime']
    df = df.sort_index(level = 0)
    
    df['year'] = df['visitStartTime'].dt.year
    df['month']= df['visitStartTime'].dt.month
    #df['dayofyear'] = df['visitStartTime'].dt.dayofyear
    #df['weekofyear'] = df['visitStartTime'].dt.weekofyear
    #df['weekday'] = df['visitStartTime'].dt.weekday
    #df['hour'] = df['visitStartTime'].dt.hour
    return df

def gen_dummies_ratios(df):
  df = df.copy()
  
  user_id_analyt = df[df['referURL'].str.contains('analytics', na = False)]['fullVisitorId'].unique()  
  # indicator if the referrer URL has any relation to Google Analytics: 
  # hypothesis is that Google analytics users primary goal is not to buy google merchandise
  df['user_id_analyt'] = df['fullVisitorId'].isin(user_id_analyt).astype('int16')
  
  #df_by_fullVisId = df[['totals_totalTransactionRevenue', 'fullVisitorId']].groupby('fullVisitorId').sum()
  #user_id_paying = df_by_fullVisId[df_by_fullVisId['totals_totalTransactionRevenue']>0].index
  #df['paying_user'] = df['fullVisitorId'].isin(user_id_paying).astype('int16')
  
  #df['count_events'] = df['totals_hits'] - df['totals_pageviews']
  df['count_events'] = df.eval('totals_hits - totals_pageviews')
  
  
  #df['share_AddCart'] = df['count_AddCart'] / df['count_events']
  df['share_AddCart'] = df.eval('count_AddCart / count_events')
  df.loc[df['count_events'] == 0, 'share_AddCart'] = 0
  df.loc[df['share_AddCart']>1, 'share_AddCart'] = 1
  
  #df['share_PromoView'] = df['count_PromoView'] / df['totals_hits']
  df['share_PromoView'] = df.eval('count_PromoView / totals_hits')
  
  # after calculating ratios we don't need these two columns
  df.drop( ['count_PromoView', 'count_AddCart'], 
            axis = 1, inplace = True)
  # share of the number of total hits
  df.loc[df['totals_hits'] == 0, 'share_PromoView'] = 0
  
  #df['time_per_hit'] = df['totals_timeOnSite'] / df['totals_hits']
  df['time_per_hit'] = df.eval('totals_timeOnSite / totals_hits')
  
  return df


def recod(df):
    
    recod_col ={
    'trafficSource_campaign' : ['AW - Office',  'AW - Accessories', 'AW - Bags', 'AW - Google Brand', 'AW - Apparel', 
                        'AW - Dynamic Search Ads Whole Site', 'AW - Drinkware ' ],
    'trafficSource_source': ['(direct)', 'google', 'dfa', 'mail.google.com', 'hangouts.google.com', 'dealspotr.com', 
                         'connect.googleforwork.com', 'basecamp.com'],
    'trafficSource_adContent':   ['Full auto ad IMAGE ONLY'],
    'referURL': ['https://mall.googleplex.com/']
    }

    # the following values of the respective feature will be replaced with NaN
    recod_na = {
        'geoNetwork_city': ['not available in demo dataset', '(not set)'],
        'device_operatingSystem': ['(not set)'],
 
    } 
    
    
    df = df.copy()
    
	# recoding some values into NaN according to the above dict
    for col in recod_na.keys():
        df.loc[(df[col].isin(recod_na[col] )), col] = np.nan
    
	# recoding some values into 0/1 according to the above dict
    for col in recod_col.keys():
        df[ col + '_recod'] =  0
        df.loc[df[col].isin(recod_col[col] ), col + '_recod'] = 1
        
	# some further recoding (binarisation) of trafficSource_keyword
    df['trafficSource_keyword_recod'] = 0
    df.loc[df['trafficSource_keyword'].str.contains('google|merch', 
                                                    case  = False, na = False) , 
                                       'trafficSource_keyword_recod'] = 1
    
    df.drop(list(recod_col.keys()) + ['trafficSource_keyword'], 
            axis = 1, inplace = True)
    #df['visitStartTime'] = df['visitStartTime'].apply(lambda x: x.timestamp())
    
    return df


def factoris(df):
    '''
	Encoding of categorical features according to the recommendation from the LightGBM documentation:
	Categorical features must be encoded as non-negative integers (int) less than Int32.MaxValue (2147483647). 
	It is best to use a contiguous range of integers started from zero.
	'''
    df = df.copy()
    
    for col in (set(df.columns) - set(['fullVisitorId'])):
        if df[col].dtype == 'object':
            df[col], _ = pd.factorize(df[col])
            df.loc[df[col] == -1, col] = np.nan
            try:
                df[col] = df[col].astype('int32')
            except ValueError as e:  # NaN cannot be converted to int
                pass
                
    return df




@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        unpacked data (saved in ../unpacked).
    """
    #logger = logging.getLogger(__name__)
    #logger.info('making final data set from raw data')
    
    
    
    if os.path.isfile('recode_logging.log'):
        os.remove('recode_logging.log')
    logger = get_logger('recode_logging.log')
    logger.info('Recoding - process started')
    print('Recoding - process started')
    
    for fil in os.listdir(output_filepath):
        os.remove(os.path.join(output_filepath, fil))
    # dict for recoding: these columns will be converted to dummy variables, the listed values for the respective feature will correspond to 1, all others to 0

    
    
    conc = pd.read_pickle(os.path.join(input_filepath, 'conc.zip'))
    conc = rename_cols(conc)
    
    conc = fillna_convert(conc)
    conc = create_date_cols(conc)
    #print(conc.shape)
    conc = gen_dummies_ratios(conc)


    #conc['referSocNetwork'].fillna('(not set)', inplace = True)
    conc = recod(conc)
    conc = factoris(conc)
    conc.to_pickle(os.path.join(output_filepath, 'conc_recoded.zip'))

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
