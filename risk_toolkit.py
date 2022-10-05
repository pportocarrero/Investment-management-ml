import pandas as pd
import scipy.stats

def drawdown(return_series: pd.Series):
    '''
    Takes a time series of returns, then returns a Dataframe with columns for the wealth index, the previous peaks, and the percentage drawdown.
    :param return_series:
    :return: wealth index, previous peaks, drawdown
    '''

    wealth_index = 1000 * (1 + return_series).cumprod()

    previous_peaks = wealth_index.cummax()

    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame({
        'Wealth': wealth_index,
        'Previous peal': previous_peaks,
        'Drawdown': drawdowns
    })


def get_ffme_returns():
    '''
    Loads the Fama-French database for the returns of the large and small cap portfolios
    :return: returns of the small and large cap portfolios
    '''

    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', header=0, index_col=0, na_values=-99.99)

    returns = me_m[['Lo 10', 'Hi 10']]

    returns.columns = ['Small cap', 'Large cap']

    returns = returns / 100

    returns.index = pd.to_datetime(returns.index, format='%Y%m').to_period('M')

    return returns


def get_hf_returns():
    '''
    Loads and format EDHEC Hedge Fund Index Returns
    :return: Loads and format EDHEC Hedge Fund Index Returns
    '''

    hfi = pd.read_csv('data/edhec-hedgefundindices.csv', header=0, index_col=0, parse_dates=True)

    hfi = hfi / 100

    hfi.index = hfi.index.to_period('M')

    return hfi

def skewness(r):
    '''
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    :param r:
    :return:
    '''

    demeaned_r = r - r.mean()

    # use the population standard deviation, so set dof=0

    sigma_r = r.std(ddof=0)

    exp = (demeaned_r ** 3).mean()

    return exp / sigma_r ** 3

def kurtosis(r):
    '''

    :param r:
    :return:
    '''

    demeaned_r = r - r.mean()

    # use the population standard deviation, so set dof=0

    sigma_r = r.std(ddof=0)

    exp = (demeaned_r ** 4).mean()

    return exp / sigma_r ** 4

def is_normal(r, level=0.01):
    '''
    Applies the Jarque-Bera test to determine if a Series is normal or not.
    Test is applied at the level indicated (in numerical values)
    :param level:
    :param r:
    :return: True if the hypothesis of normality is accepted, False otherwise
    '''

    statistic, p_value = scipy.stats.jarque_bera(r)

    return p_value > level
