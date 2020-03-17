# -*- coding: utf-8 -*-
import click
import os, logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from collections import Counter
from src.utils import get_logger, mode, reduce_mem_usage, two_latest_months
from src.configs import global_params



def mapping_month_histRev(multiindex):
    '''
    Parameters
    ----------
    multiindex : multiindex consisting of year and month

    Returns
    -------
    monthly_ave_rev_mapping : dataframe that maps year/month combinations
    to actual historical average monthly revenues for Google merch store
    
    '''
    #actual monthly revenues from the GA demo account: 
    # 27 values from Aug 2016 to Oct 2018 incl. the latter
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


		

def agg_features(df):

    '''
    Parameters
    ----------
    df : initial df

    Returns
    -------
    agg : df aggregated by fullVisitorId.  If there are not enough elements 
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
    '''
    Parameters
    ----------
    df : initial df
    global_rev_var : column used as the revenue metric (dataset has two,  
    according to the competition rules 'totals_transactionRevenue' )
    num_last_revenues: number of last revenues to be used as feautures
    start_day: first day of the current time window
    monthly_ave_rev_mapping: mapping year/month to historical monthly revenue (function mapping_month_histRev)
    present_delta: length (in days) of "present" time window for the calculation of features
    future_delta: length (in days) of "future" time window for the calculation of future revenues
    generate_y: (bool) whether to generate y (not possible for actual prediction)


    Returns
    -------
    y: Series of y (future revenues for certain users for this time window)
    X: df of features (based on the "present" time period for this time window)
    
    y is not returned if generate_y=False (used for generation of X for predicition)

    '''
    present_tf = pd.Timedelta(present_delta, 'd') # present_tf means present time frame in days
    future_tf = pd.Timedelta(future_delta, 'd')  # future_tf means future time frame in days
    
    end_day = start_day + present_tf
    present_sample = df.query('(visitStartTime >= @start_day) & (visitStartTime < @end_day)')
    
    
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
    two_last_relev_month = two_latest_months([i for (i,j) in three_most_relevant_months])
    # average of the average revenues for these two months
    X['hist_monthly_rev'] = pd.Series(two_last_relev_month).map(monthly_ave_rev_mapping).mean()
    X['hist_monthly_rev'] = X['hist_monthly_rev'].astype('int32')
    
    if generate_y:  # only if we generate the training dataset, not the prediction one        
        # actual cumulated revenues over the future period
        future_rev = df[[global_rev_var, 'fullVisitorId']]
        future_start_day = start_day + present_tf
        future_end_day = start_day + present_tf + future_tf
        future_rev = future_rev.query('visitStartTime >= @future_start_day &   \
                                visitStartTime < @future_end_day')
        future_rev = future_rev.groupby('fullVisitorId').sum()
        future_rev.rename(columns = {global_rev_var: 'future_rev'}, inplace = True)
        rev_present_ids = pd.DataFrame(index = present_sample['fullVisitorId'].unique())
        # for indices present in present_sample join future revenues (for the IDs from the future set)
        y = rev_present_ids.join(future_rev, how='left')
        # if there is no corresponding revenue in the "future revenues" df,
        # then no revenue was earned - replace it with 0
        y = y.fillna(0).sort_index()
    
        return y, X
        
    return X
	

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs the script to generate features and target variables
         based on several time windows of equal sizes (rolling window).
         The resulting level of granularity is fullVisitorId
         Each column in X represents features related to one visitor
         aggregated over a certain period of time ("present" period). The corresponding y value
         is the revenue generated by this visitor during a certain future period of time.
    
        Throughout the code we differentiate between 
        - present (timeframe, timedelta) - this refers to the "current" time period
            used for feature calculation / aggregation
        - future (timeframe, timedelta) - this refers to the "future" time period
            used to calculate future revenue
        These two time periods do not intersect.
    
    """
    
    if os.path.isfile('create_X_y_logging.log'):
        os.remove('create_X_y_logging.log')
    logger = get_logger('create_X_y_logging.log')
    logger.info('Creating X and y based on rolling windows - process started')
    print('Creating X and y based on rolling windows - process started')
    
    
    global_rev_var = global_params['global_rev_var']
    num_last_revenues = global_params['num_last_revenues']
    present_delta = global_params['present_delta']
    future_delta = global_params['future_delta']
    num_windows = global_params['num_windows']
    abs_first_day = global_params['abs_first_day']
    abs_last_day = global_params['abs_last_day']
    
    
    for fil in os.listdir(output_filepath):
        os.remove(os.path.join(output_filepath, fil))
    
    conc = pd.read_pickle(os.path.join(input_filepath, 'conc_recoded.zip'))
    # make a df which is guaranteed to have all month-year combinations
    # revenue was earned in all months
    
    month_year_pairs = pd.date_range(start=abs_first_day, end=abs_last_day, freq='MS' )
    month_year_index = pd.MultiIndex.from_arrays([month_year_pairs.year, 
                                                  month_year_pairs.month])
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
        y.to_pickle( os.path.join(output_filepath, 'y_{:0=4}.zip'.format(winshift)))
        del y
        logger.info('y_{:0=4} saved'.format(winshift))
        print('y_{:0=4} saved'.format(winshift))
        X = reduce_mem_usage(X, num_last_revenues)
        X.to_pickle(os.path.join(output_filepath, 'X_train_{:0=4}.zip'.format(winshift)))
        del X
        logger.info('X_train_{:0=4} saved'.format(winshift)) 
        print('X_train_{:0=4} saved'.format(winshift)) 

    logger.info('Generating X prediction')
    print('Generating X prediction')
    X_pred = generate_X_y(df=conc, 
                                 global_rev_var=global_rev_var,
                                 num_last_revenues=num_last_revenues,
                                 start_day = abs_last_day + pd.Timedelta(1, 'd') - pd.Timedelta(present_delta, 'd'), 
                                 monthly_ave_rev_mapping=monthly_ave_rev_mapping,
                                 present_delta = present_delta, 
                                 future_delta = future_delta,
                                 generate_y=False)
    X_pred = reduce_mem_usage(X_pred, num_last_revenues)

    X_pred.to_pickle(os.path.join(output_filepath, 'X_pred.zip'.format(winshift)))
    logger.info('X_pred saved'.format(winshift)) 
    print('X_pred saved'.format(winshift)) 
    del X_pred
    del conc


    logger.info('Process finished')
    print('Process finished')
    logging.shutdown()
    

    
if __name__ == '__main__':

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
