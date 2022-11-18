import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize

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
        'Previous peak': previous_peaks,
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


def get_ind_returns():

    '''
    Load and format the Ken French 30 Industry portfolios value weighted monthly returns
    '''

    ind = pd.read_csv('data/ind30_m_vw_rets.csv', header=0, index_col=0, parse_dates=True) / 100

    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')

    ind.columns = ind.columns.str.strip()

    return ind


def annualized_returns(r, periods_per_year):

    '''
    Annualizes returns
    :param r: database
    :param periods_per_year: number of periods in a year to estimate the annualized returns
    :return:
    '''

    compounded_growth = (1 + r).prod()

    n_periods = r.shape[0]

    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualized_volatility(r, periods_per_year):

    '''
    Annaualizes the volatility
    :param r:
    :param periods_per_year:
    :return:
    '''

    return r.std() * (periods_per_year ** 0.5)


def sharpe_ratio(r, risk_free_rate, periods_per_year):

    '''
    Estimates the annualized sharpe ratio of a set of returns
    :param r: returns
    :param risk_free_rate: risk free rate to use
    :param periods_per_year: number of periods per year
    :return: sharpe ratio
    '''

    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_return = r - rf_per_period

    annual_excess_return = annualized_returns(excess_return, periods_per_year)

    annual_volatility = annualized_volatility(r, periods_per_year)

    return annual_excess_return / annual_volatility


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


def semi_deviation(r):
    '''
    Returns the semideviation (negative semideviation of r).
    r must be a series or a Dataframe.
    '''

    excess = r - r.mean()  # We demean the returns

    excess_negative = excess[excess < 0]  # We take only the returns below the mean

    excess_negative_square = excess_negative**2  # We square the demeaned returns below the mean

    n_negative = (excess < 0).sum()  # number of returns under the mean

    return (excess_negative_square.sum() / n_negative)**0.5  # semideviation


def historic_var(r, level=5):
    '''
    Returns the historic VaR at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    :param r:
    :param level:
    :return:
    '''

    if isinstance(r, pd.DataFrame):

        return r.aggregate(historic_var, level=level)

    elif isinstance(r, pd.Series):

        return -np.percentile(r, level)

    else:
        raise TypeError('Expected r to be a Series or DataFrame')


def historic_cvar(r, level=5):
    '''
    Calculates the Conditional VaR of Series or DataFrame
    :param r:
    :param level:
    :return:
    '''

    if isinstance(r, pd.Series):

        is_beyond = r <= -historic_var(r, level=level)

        return -r[is_beyond].mean()

    elif isinstance(r, pd.DataFrame):

        return r.aggregate(historic_cvar, level=level)

    else:

        raise TypeError("Expected r to be a Series or DataFrame")


def gaussian_var(r, level=5, modified=False):
    '''
    Calculates the parametric (Gaussian) VaR of a Series or df.
    If modified is True, then the modified VaR is return using the Cornish-Fisher estimation
    :param r:
    :param level:
    :param modified:
    :return:
    '''

    # Calculate the Z score, assuming it's gaussian

    z = norm.ppf(level / 100)

    if modified:

        s = skewness(r)
        k = kurtosis(r)
        z = (z + (z**2 - 1) * s/6 + (z**3 -3 * z) * (k - 3)/24 - (2*z**3 - 5*z) * (s**2)/36)

    return -(r.mean() + z * r.std(ddof=0))


def portfolio_return(weights, returns):

    return weights.T @ returns


def portfolio_vol(weights, cov_matrix):

    return (weights.T @ cov_matrix @ weights) ** 0.5


def efficient_frontier_2_asset(n_points, expected_return, cov_matrix, style='.-'):

    '''
    Plots a 2 asset efficient frontier
    :param n_points:
    :param expected_return:
    :param covariance_matrix:
    :return:
    '''

    if expected_return.shape[0] != 2 or expected_return.shape[0] != 2:

        raise ValueError('This function can only plot a 2 asset frontier')

    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]

    rets = [portfolio_return(w, expected_return) for w in weights]

    vol = [portfolio_vol(w, cov_matrix) for w in weights]

    efficient_frontier_df = pd.DataFrame({
        'Returns': rets,
        'Volatility': vol
    })

    return efficient_frontier_df.plot.line(x='Volatility', y='Returns', style=style)


def minimize_vol(target_return, expected_return, cov_matrix):

    n = expected_return.shape[0]

    init_guess = np.repeat(1/n, n)  # This is the initial guess for the optimizer. It computes the same weights for all possibilities.

    bounds = ((0.0, 1.0),) * n  # This gives boundaries for the optimizer. Only going long on assets (between 0 to 100%) for every n possibility.

    weights_sum_to_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    return_is_target = {
        'type': 'eq',
        'args': (expected_return,),
        'fun': lambda weights, expected_return: target_return - portfolio_return(weights, expected_return)  # Lambda function is an anonymous function
    }

    weights = minimize(portfolio_vol, init_guess,
                       args=(cov_matrix,),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_one, return_is_target),
                       bounds=bounds
    )

    return weights.x


def max_sharpe_ratio(risk_free_rate, expected_return, cov_matrix):

    '''
    Returns the weights of the portfolio that gives the maximum sharpe ratio given the risk-free rate and expected returns and a covariance matrix.
    :param risk_free_rate:
    :param expected_return:
    :param cov_matrix:
    :return:
    '''

    n = expected_return.shape[0]

    init_guess = np.repeat(1/n, n)

    bounds = ((0.0, 1.0),) * n

    weights_sum_to_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }


    def neg_sharpe_ratio(weights, risk_free_rate, expected_return, cov_matrix):

        '''
        Returns the negative of the sharpe ratio, given weights
        :param weights:
        :param risk_free_rate:
        :param expected_return:
        :param cov_matrix:
        :return:
        '''

        r = portfolio_return(weights, expected_return)

        vol = portfolio_vol(weights, cov_matrix)

        return -(r - risk_free_rate) / vol

    weights = minimize(neg_sharpe_ratio, init_guess,
                       args=(risk_free_rate, expected_return, cov_matrix),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_one,),
                       bounds=bounds
                       )

    return weights.x

def optimal_weights(n_points, expected_return, cov_matrix):

    '''
    This is the list of weights to run the optimizer to minimize the vol.
    :param n_points:
    :param expected_return:
    :param cov_matrix:
    :return:
    '''

    target_returns = np.linspace(expected_return.min(), expected_return.max(), n_points)

    weights = [minimize_vol(target_return, expected_return, cov_matrix) for target_return in target_returns]

    return weights


def efficient_frontier_multi_asset(n_points, expected_return, cov_matrix, show_cml=False, risk_free_rate=0, style='.-'):

    '''
    Plots the multi-asset efficient frontier including the Capital Market Line.
    :param n_points:
    :param expected_return:
    :param cov_matrix:
    :param show_cml:
    :param risk_free_rate:
    :param style:
    :return:
    '''
    weights = optimal_weights(n_points, expected_return, cov_matrix)

    rets = [portfolio_return(w, expected_return) for w in weights]

    vol = [portfolio_vol(w, cov_matrix) for w in weights]

    efficient_frontier_df = pd.DataFrame({
        'Returns': rets,
        'Volatility': vol
    })

    ax = efficient_frontier_df.plot.line(x='Volatility', y='Returns', style=style)

    if show_cml:

        ax.set_xlim(left=0)

        weights_msr = max_sharpe_ratio(risk_free_rate, expected_return, cov_matrix)

        ret_msr = portfolio_return(weights_msr, expected_return)

        vol_msr = portfolio_vol(weights_msr, cov_matrix)

        # Add the Capital Market Line

        cml_x = [0, vol_msr]  # Both ends of the line in axis X

        cml_y = [risk_free_rate, ret_msr]

        ax.plot(cml_x, cml_y, color='red', marker='o', linestyle='dashed', markersize=12, linewidth=2)

    return ax

