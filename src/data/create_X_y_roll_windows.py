# -*- coding: utf-8 -*-
import click
import logging, os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from collections import Counter
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


def mapping_month_histRev(multiindex):
    
    # just to create a multiindex made of years and months 
    #agg_year_month = df.groupby(['year', 'month']).count()
    #actual monthly revenues from the GA demo account: 
    # 27 values from Aug 2016 to Oct 2018 incl. the latter
    #print(agg_year_month.shape)
    year_month_df = pd.DataFrame([184319, 132387, 158161, 557886, 682118, 
                                  461792, 354995, 398422, 437901, 348331, 
                                  364283, 402301, 432101, 380179, 444810, 
                                  558821, 554029, 432826, 338916, 534808, 
                                  610061, 481326, 210257, 127364, 108038, 
                                  119692, 115086],
                                 index=multiindex)
    # aggregate (take averages) of revenues by month (level 1), drop the first 
    # three values as they seem to be outliers, the rest of the values build 
    # quite a smooth line
    # mapping defined as month: average historical revenue
    monthly_ave_rev_mapping = year_month_df.iloc[3:,0].groupby(level = 1).mean()	
    return monthly_ave_rev_mapping



def mode(x):
    '''
	Returns the mode
	'''
    return Counter(x).most_common(1)[0][0]

def max_month(x):
    '''returns two latest month of the three (e.g. if [2,3,4] is supplied, (3,4) is returned). 
	   Problematic cases are 11,12,1 and 12,1,2; dealt with in the IF-clause  '''
    if (12 in x)&(1 in x):
        if 2 in x:
            month = 2
            month_lag1 = 1
        else:
            month = 1
            month_lag1 = 12
    else:
        month = max(x)
        month_lag1 = sorted(x,reverse=True)[1]
    return month, month_lag1
		

def agg_features(df):
    '''
	Aggregation of features by fullVisitorId, if there are not enough elements 
    to calculate st. deviation, assume it is 0
	'''
    feat_to_agg = {
        'channelGrouping': [mode, 'count'],
        'count_ViewedCat': 'mean',
        'device_browser': mode,
        'device_deviceCategory': mode,
        'device_operatingSystem': mode,
        'geoNetwork_city': mode,
        'referSocNetwork':  mode,
        'totals_bounces': 'mean',
        # 'totals_hits': 'mean',
        'trafficSource_campaign_recod': ['sum', 'mean'],
        'trafficSource_isTrueDirect': 'sum',
        'trafficSource_keyword_recod': ['sum', 'mean'],
        'trafficSource_source_recod': ['sum', 'mean'],
        'referURL_recod':  ['sum', 'mean'],
        'trafficSource_adContent_recod':  ['sum', 'mean'],
        'user_id_analyt': 'first',
        'share_AddCart': ['mean', 'std'],
        'share_PromoView': ['mean', 'std'],
        'totals_pageviews': ['mean', 'std'],
        'count_events': ['mean', 'std'],
        'time_per_hit':  ['mean', 'std'],
        'totals_sessionQualityDim': ['mean', 'std'],
        'totals_timeOnSite' :  ['mean', 'std']
}
   
    agg = df[list(feat_to_agg.keys()) + ['fullVisitorId']].groupby('fullVisitorId').agg(feat_to_agg)
    # flatten pandas MultiIndex
    agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in agg.columns.tolist()])  
    col_std = agg.columns[agg.columns.str.contains('_STD')]
    for col in col_std:
        agg[col].fillna(0, inplace = True)
    return agg
   
def reshapeRevDates(df, global_rev_var, num_last_revenues):
    '''
    Parameters
    ----------
    df : initial df
    global_rev_var : column used as the revenue metric (dataset has two,  
    according to the competition rules 'totals_transactionRevenue' )
    num_last_revenues: number of last revenues to be used as feautures

    Returns
    -------
    reshRevTime : dataframe containing (indexed by fullVisitorId) 
    three features describing last positive revenue figures for each visitor:
    - num_last_revenues last positive revenue figures,
    - time intervals since these revenues were earned (reference point: )
    - ratios of these two metrics (revenues weighted by the reverse of time)
    
    If fewer than num_last_revenues are available in a given time window,
    the remaining revenues are replaced by zeros,
    time intervals are replaced by 10000000 (arbitrary large number)
    '''
    
    win_last_day = df['visitStartTime'].iloc[-1]
    lst_idx = []
    lst_data = []
	# filter rows with positive revenue and group by fullVisitorId
    grouped_df = df[df[global_rev_var]>0][['fullVisitorId', global_rev_var,
                                           'visitStartTime']].groupby('fullVisitorId')
    for q, sub_df in grouped_df:
        
        # make last observation first, and take first num_last_revenues of them
        rev_seq = np.flip(sub_df[global_rev_var].values, axis = 0)[:num_last_revenues]  
        # calculate how much time elapsed from each revenue to last day of the time window
        #print('last day')
        #print(win_last_day)
        #print('flip')
        #print( np.flip(sub_df['visitStartTime'], 
        #                                       axis = 0)[:num_last_revenues])
        timedelta_seq = win_last_day - np.flip(sub_df['visitStartTime'], 
                                               axis = 0)[:num_last_revenues]
        timedelta_seq = timedelta_seq.apply(lambda x: x.total_seconds()).values / 86400
        lst_idx.append(q)
        # how many zeros are required to maintain the length equal to num_last_revenues
        len_padding_zeros =  np.max((0,num_last_revenues - len(rev_seq))) 
        
        # how the row in the below line is composed:
        # 10 entries for revenue: actual numbers, rest is zeros; 
        # 10 entries for relative times: actual numbers, rest are big positive numbers (10000000);
        # finally, one divided by the other, i.e. revenues are weighted by inverse times
        lst_data.append(np.concatenate([ rev_seq, np.repeat(0, len_padding_zeros), 
                                    timedelta_seq, np.repeat(10000000, len_padding_zeros),
                                   np.divide(rev_seq, timedelta_seq), np.repeat(0, len_padding_zeros)
                                   ]))
    reshRevTime = pd.DataFrame(np.vstack(lst_data), 
                               index = lst_idx, 
                               columns = 
     ['{}_{}'.format(variab, i+1) for 
      variab in ['rev', 'timeShift', 'weightRev'] for i in range(num_last_revenues)]) 
    # result: rev_1, rev_2 etc.
        
    return reshRevTime  
 


def generate_X_y(df, global_rev_var, num_last_revenues, start_day, monthly_ave_rev_mapping,
                 present_delta, future_delta, generate_y):

    present_tf = pd.Timedelta(present_delta, 'd') # present_tf means present time frame in days
    future_tf = pd.Timedelta(future_delta, 'd')  # future_tf means future time frame in days
    
    end_day = start_day + present_tf
    present_sample = df.query('(visitStartTime >= @start_day) & (visitStartTime < @end_day)')
    #present_sample = df[(df['visitStartTime']>= start_day.timestamp()) & (df['visitStartTime']< (start_day + present_tf).timestamp())]
    
    ##############################    X      ##################
    
    feat_agg_user = agg_features(present_sample)
    reshRevDate = reshapeRevDates(present_sample, global_rev_var, num_last_revenues)
    # since reshRevTime contains data only for users with positive revenues, it contains far fewer rows that feat_agg_user
    X = feat_agg_user.join(reshRevDate, how = 'left')
    # argument is a dict: fillna for revenues and weighted revenues with 0, times with 10000000
    X.fillna({**{'rev_{}'.format(i+1):0 for i in range(num_last_revenues)} , 
                   **{'timeShift_{}'.format(i+1):10000000 for i in range(num_last_revenues)},
                   **{'weightRev_{}'.format(i+1):0 for i in range(num_last_revenues)}}, 
                  inplace=True, axis= 0) 
    # we need as the regressors average historical revenue for the calendar months
    # that most well cover the "future" period - period when revenues will be earned
    # first we find out what months these are: by taking 3 most common
    # months in the dates of the revenue period
    future_date_range = pd.date_range(start=start_day+present_tf, end=start_day+present_tf+future_tf)
    three_most_relevant_months = Counter(future_date_range.month).most_common(3)
    # out of the three choose two last months
    two_last_relev_month = max_month([i for (i,j) in three_most_relevant_months])
    # average of the average revenues for these two months
    X['forecast_ave_rev'] = pd.Series(two_last_relev_month).map(monthly_ave_rev_mapping).mean()
    #print(full_X['forecast_ave_rev'].isna().mean())
    X['forecast_ave_rev'] = X['forecast_ave_rev'].astype('int32')
    
    if generate_y:  # only if we generate the training dataset, not the test one
    
        
        # actual cumulated revenues over the future period
        future_rev = df[[global_rev_var, 'fullVisitorId']]
        future_start_day = start_day + present_tf
        future_end_day = start_day + present_tf + future_tf
        future_rev = future_rev.query('visitStartTime >= @future_start_day &   \
                                visitStartTime < @future_end_day')
        future_rev = future_rev.groupby('fullVisitorId').sum()
                                #future_rev = df[(df['visitStartTime']>= (start_day + present_tf).timestamp()) & (df['visitStartTime']< (start_day + 
                                #    present_tf +future_tf).timestamp())][[global_rev_var, 'fullVisitorId']].groupby('fullVisitorId').sum()
        future_rev.rename(columns = {global_rev_var: 'future_rev'}, inplace = True)
        rev_present_ids = pd.DataFrame(index = present_sample['fullVisitorId'].unique())
        #rev_present_ids = pd.DataFrame(np.zeros(present_sample['fullVisitorId'].nunique()), 
        #                         columns = ['rev_in_fut'], index = present_sample['fullVisitorId'].unique() )
        # future revenues are first filled with zeros 
    
    
    
        # for indices present in present_sample join future revenues (for the IDs from the future set)
        y = rev_present_ids.join(future_rev, how='left')
        # if there is no corresponding revenue in the "future revenues" df,
        # then no revenue was earned - replace it with 0
        y = y.fillna(0).sort_index()
        #y_join.loc[(~pd.isna(y_join['valid'])), 'rev_in_fut']= y_join['valid'] 
        #y = y_join['rev_in_fut'].sort_index()
        #y_bin = (y>0).astype('int16')
    
        return y, X

    
    
    return X
	
#def generate_X_test(df, start_day, present_delta):
#    print('Generating X test')
#    print(start_day)
#    present_tf = pd.Timedelta(present_delta, 'd') # days
#    present_sample = df.query('(visitStartTime >= @start_day) & (visitStartTime < (@start_day + @present_tf))')
#    #present_sample = df[(df['visitStartTime']>= start_day.timestamp()) & (df['visitStartTime']< (start_day + present_tf).timestamp())]
#  
#      ##############################    X      ##################
#    
#    feat_agg_user = agg_features(present_sample)
#    reshRevDate = reshapeRevDates(present_sample, global_rev_var, num_last_revenues)
#	# since reshRevTime contains data only for users with positive revenues, it contains far fewer rows that feat_agg_user
#    full_X = feat_agg_user.join(reshRevDate, how = 'left')
#	# argument is a dict: fillna for revenues and weighted revenues with 0, times with 10000000
#    full_X.fillna({**{'rev_{}'.format(i+1):0 for i in range(num_last_revenues)} , 
#                   **{'timeShift_{}'.format(i+1):10000000 for i in range(num_last_revenues)},
#      **{'weightRev_{}'.format(i+1):0 for i in range(num_last_revenues)}}, inplace=True, axis= 0) 
#	# numbers of two last full months of the validation period
#    forecast_month_tuple = (12,1)
#	# average of the average revenues for these two months
#    full_X['forecast_ave_rev'] = pd.Series(forecast_month_tuple).map(monthly_ave_rev_mapping).mean()
  
#    return full_X
  
  
def reduce_mem_usage(df, num_last_revenues):
    '''convert to int just rev and timeShift
    '''
    for col in ['{}_{}'.format(variab, i+1) for 
                variab in ['rev', 'timeShift'] for i in range(num_last_revenues)]:
        df[col] = df[col].astype('int32')
    return df
	



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        unpacked data (saved in ../unpacked).
    
    Throughout the code we differentiate between 
    - present (timeframe, timedelta) - this refers to the "current" time period
      for which we have data
    - future (timeframe, timedelta) - this refers to the "future" time period
      for we have to  predict revenue
    These two time periods do not intersect.
    
    
    """
   
    
    if os.path.isfile('create_X_y_logging.log'):
        os.remove('create_X_y_logging.log')
    logger = get_logger('create_X_y_logging.log')
    logger.info('Creating X and y based on rolling windows - process started')
    print('Creating X and y based on rolling windows - process started')
    
    global_rev_var = 'totals_transactionRevenue'   
    num_last_revenues = 10
    present_delta = 500
    future_delta = 108
    num_windows = 8
    abs_first_day = pd.to_datetime('2016-08-01')
    abs_last_day = pd.to_datetime('2018-10-16')
    
    
    for fil in os.listdir(output_filepath):
        os.remove(os.path.join(output_filepath, fil))
    
    conc = pd.read_pickle(os.path.join(input_filepath, 'conc_recoded.zip'))
    # make a df which is guaranteed to have all month-year combinations
    # revenue was earned in all months
    
    month_year_pairs = pd.date_range(start=abs_first_day, end=abs_last_day, freq='MS' )
    month_year_index = pd.MultiIndex.from_arrays([month_year_pairs.year, 
                                                  month_year_pairs.month])
    #conc_light = conc[conc[global_rev_var]>0][['year', 'month']]
    monthly_ave_rev_mapping = mapping_month_histRev(month_year_index)
    
    # we need to use all latest data in full
    last_window_start = (abs_last_day - abs_first_day - 
                         pd.Timedelta(present_delta, 'd') - pd.Timedelta(future_delta, 'd')).days
    # given the number of windows figure out by how much the window start
    # has to be shifted
    window_shift = np.trunc(last_window_start/(num_windows - 1)).astype('int16')

    windows_start_list = [window_shift * i for i in range(num_windows - 1)] + [last_window_start ]

    for winshift in windows_start_list:
        start_day = abs_first_day + pd.Timedelta(winshift, 'd')
        logger.info('Generating X and y train')
        print('Generating X and y train')
        logger.info('starting day: {}-{}-{}'.format(start_day.year, start_day.month,
                                                    start_day.day) )
        print('starting day: ' , start_day)
        y, X = generate_X_y(df=conc, 
                                 global_rev_var=global_rev_var,
                                 num_last_revenues=num_last_revenues,
                                 start_day = abs_first_day + pd.Timedelta(winshift, 'd'), 
                                 monthly_ave_rev_mapping=monthly_ave_rev_mapping,
                                 present_delta = present_delta, 
                                 future_delta = future_delta,
                                 generate_y=True)
        #logger.info('Window shift: {}. y, full_x are generated'.format(winshift))
        y.to_pickle( os.path.join(output_filepath, 'y_{:0=4}.zip'.format(winshift)))
        del y
        logger.info('y_{:0=4} saved'.format(winshift))
        print('y_{:0=4} saved'.format(winshift))
        X = reduce_mem_usage(X, num_last_revenues)
        X.to_pickle(os.path.join(output_filepath, 'X_train_{:0=4}.zip'.format(winshift)))
        del X
        logger.info('X_train_{:0=4} saved'.format(winshift)) 
        print('X_train_{:0=4} saved'.format(winshift)) 

    logger.info('Generating X test')
    print('Generating X test')
    X_test = generate_X_y(df=conc, 
                                 global_rev_var=global_rev_var,
                                 num_last_revenues=num_last_revenues,
                                 start_day = abs_last_day + pd.Timedelta(1, 'd') - pd.Timedelta(present_delta, 'd'), 
                                 monthly_ave_rev_mapping=monthly_ave_rev_mapping,
                                 present_delta = present_delta, 
                                 future_delta = future_delta,
                                 generate_y=False)
    X_test = reduce_mem_usage(X_test, num_last_revenues)

    X_test.to_pickle(os.path.join(output_filepath, 'X_test.zip'.format(winshift)))
    logger.info('X_test saved'.format(winshift)) 
    print('X_test saved'.format(winshift)) 
    del X_test
    #gc.collect()
    del conc
    #gc.collect()

    #gc.collect()

    


    

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
