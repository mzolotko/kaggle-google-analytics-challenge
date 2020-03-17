import logging
import pandas as pd
from collections import Counter



def get_logger(logfile):
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    if (logger_.hasHandlers()):
        logger_.handlers.clear()
    logger_.addHandler(fh)
    
    return logger_



def mode(x):
    '''
	Returns the mode of x
	'''
    return Counter(x).most_common(1)[0][0]


  
def reduce_mem_usage(df, num_last_revenues):

    '''
    Parameters
    ----------
    df : initial df
    num_last_revenues: number of last revenues to be used as feautures

    Returns
    -------
    df : df with  rev_0, rev_1... and timeShift_0, timeShift_1 ... columns 
    converted to int
    '''
    for col in ['{}_{}'.format(variab, i+1) for 
                variab in ['rev', 'timeShift'] for i in range(num_last_revenues)]:
        df[col] = df[col].astype('int32')
    return df


def two_latest_months(x):
       
    '''
    Parameters
    ----------
    x : list of three values (month numbers)

    Returns
    -------
    month, month_lag1 : tuple of the last and prelast of submitted month numbers

    The function returns two latest month of the submitted three 
    (e.g. if [2,3,4] is supplied, (3,4) is returned). 
	Problematic cases are 11,12,1 and 12,1,2; dealt with in the IF-clause 
    '''
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