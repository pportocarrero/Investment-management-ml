import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

def compound(r):

    return np.expm1(np.log1p(r).sum())

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

def get_ind_size():

    '''
    '''

    ind = pd.read_csv('data/ind30_m_size.csv', header=0, index_col=0)

    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')

    ind.columns = ind.columns.str.strip()

    return ind

def get_ind_nfirms():

    '''
    '''

    ind = pd.read_csv('data/ind30_m_nfirms.csv', header=0, index_col=0)

    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')

    ind.columns = ind.columns.str.strip()

    return ind

def get_total_market_index_returns():

    '''
    Load the 30 industry portfolio data and derive the returns of a cap-weighted total market index
    :return:
    '''

    ind_nfirms = get_ind_nfirms()

    ind_size = get_ind_size()

    ind_return = get_ind_returns()

    ind_mkt_cap = ind_nfirms * ind_size

    total_mkt_cap = ind_mkt_cap.sum(axis=1)

    ind_cap_weight = ind_mkt_cap.divide(total_mkt_cap, axis='rows')

    total_market_return = (ind_cap_weight * ind_return).sum(axis='columns')

    return total_market_return

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

def gmv(cov_matrix):

    '''
    Return the weight of the Global Minimum Volatility Portfolio given the covariance matrix
    :param cov_matrix:
    :return:
    '''

    n = cov_matrix.shape[0]

    return max_sharpe_ratio(0, np.repeat(1, n), cov_matrix)

def efficient_frontier_multi_asset(n_points, expected_return, cov_matrix, show_cml=False, risk_free_rate=0, show_ew=False, show_gmv=False, style='.-'):

    '''
    Plots the multi-asset efficient frontier including the Capital Market Line.
    :param show_gmv:
    :param show_ew:
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

    if show_ew:

        n = expected_return.shape[0]

        w_ew = np.repeat(1/n, n)

        r_ew = portfolio_return(w_ew, expected_return)

        vol_ew = portfolio_vol(w_ew, cov_matrix)

        ax.plot([vol_ew], [r_ew], color='green', marker='o', markersize=12)

    if show_gmv:

        w_gmv = gmv(cov_matrix)

        r_gmv = portfolio_return(w_gmv, expected_return)

        vol_gmv = portfolio_vol(w_gmv, cov_matrix)

        ax.plot([vol_gmv], [r_gmv], color='blue', marker='o', markersize=10)

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

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, risk_free_rate=0.03, drawdown=None):

    '''
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset.
    Returns a dictionary with Asset Value History, Risk Budget History, Risky Weight History
    :param risk_r:
    :param safe_r:
    :param m:
    :param start:
    :param floor:
    :param risk_free_rate:
    :return:
    '''

    # Set up the CPPI parameters

    dates = risky_r.index

    n_steps = len(dates)

    account_value = start

    floor_value = start * floor

    peak = account_value

    if isinstance(risky_r, pd.Series):

        risky_r = pd.DataFrame(risky_r, columns=['R'])

    if safe_r is None:

        safe_r = pd.DataFrame().reindex_like(risky_r)

        safe_r.values[:] = risk_free_rate / 12

    # Set up dataframes for saving intermediate values

    account_history = pd.DataFrame().reindex_like(risky_r)

    cushion_history = pd.DataFrame().reindex_like(risky_r)

    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):

        if drawdown is not None:

            peak = np.maximum(peak, account_value)

            floor_value = peak * (1 - drawdown)

        cushion = (account_value - floor_value) / account_value  # Risk budget

        risky_w = m * cushion

        risky_w = np.minimum(risky_w, 1)  # To limit the weight to 100% and no leverage

        risky_w = np.maximum(risky_w, 0)

        safe_w = 1 - risky_w

        risky_alloc = account_value * risky_w

        safe_alloc = account_value * safe_w

        # Update the account value for this time sstep

        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])

        # Save the values to look at the history and plot it

        cushion_history.iloc[step] = cushion

        risky_w_history.iloc[step] = risky_w

        account_history.iloc[step] = account_value

    risky_wealth = start * (1 + risky_r).cumprod()

    backtest_result = {
        'Wealth': account_history,
        'Risky Wealth': risky_wealth,
        'Risk Budget': cushion_history,
        'Risky Allocation': risky_w_history,
        'm': m,
        'start': start,
        'floor': floor,
        'risky_r': risky_r,
        'safe_r': safe_r
    }

    return backtest_result

def summary_stats(r, risk_free_rate=0.03):

    '''
    Returns a DataFrame that contains aggregated summary stats for the returns in the columns of r
    :param r:
    :param risk_free_rate:
    :return:
    '''

    ann_r = r.aggregate(annualized_returns, periods_per_year=12)
    ann_vol = r.aggregate(annualized_volatility, periods_per_year=12)
    ann_sharpe_ratio = r.aggregate(sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=12)
    draw_down = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(gaussian_var, modified=True)
    hist_cvar5 = r.aggregate(historic_cvar)

    df = pd.DataFrame({
        'Annualized Return': ann_r,
        'Annualized Volatility': ann_vol,
        'Skewness': skew,
        'Kurtosis': kurt,
        'Cornish-Fisher VaR (5%)': cf_var5,
        'Historic CVaR (5%)': hist_cvar5,
        'Sharpe Ratio': ann_sharpe_ratio,
        'Max Drawdown': draw_down
    })

    return df

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):

    '''
    Evolution of a stock price using a Geometric Brownian Motion Model
    :param n_years: number of years to simulate
    :param n_scenarios: number of scenarios to simulate
    :param mu: mean
    :param sigma: volatilty measure
    :param steps_per_year: number of steps to simulate per year
    :param s_0: initial price
    :return:
    '''

    dt = 1/steps_per_year

    n_steps = int(n_years * steps_per_year) + 1

    rets_plus_1 = np.random.normal(loc=(1 + mu) ** dt, scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios))

    rets_plus_1[0] = 1

    # Convert returns to prices

    prices = s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1

    return prices

def show_gbm(n_scenarios, mu, sigma, s_0=100, y_lim=100):
    
    '''
    Draw the results of a stock price evolution under a Geometric Brownian Motion Model
    '''
       
    prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
    
    ax = prices.plot(legend=False, color='indianred', alpha=0.5, linewidth=2, figsize=(12,6))
    
    ax.axhline(y=s_0, ls=':', color='black')
    
    y_lim = prices.values.max() * y_lim / 100
    
    ax.set_ylim(top=y_lim)
    
    # Draw a dot at the origin
    
    ax.plot(0, s_0, marker='o', color='darkred', alpha=0.2)
    
def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0, risk_free_rate=0.03, steps_per_year=12, y_max=100, s_0=100.0):
    
    '''
    Plot the results of a Monte Carlo Simulation of CPPI
    '''
    
    start = s_0
    
    sim_returns = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=steps_per_year)
    
    risky_r = pd.DataFrame(sim_returns)
    
    # Run the "back"-test
    
    btr = run_cppi(risky_r=pd.DataFrame(risky_r), risk_free_rate=risk_free_rate, m=m, start=start, floor=floor)
    
    wealth = btr['Wealth']
    
    # Calculate terminal wealth stats
    
    y_max = wealth.values.max() * y_max / 100
    
    terminal_wealth = wealth.iloc[-1]
    
    tw_mean = terminal_wealth.mean()
    
    tw_median = terminal_wealth.median()
    
    failure_mask = np.less(terminal_wealth, start * floor)  # It shows if terminal_wealth is less than the determineed floor and returns a boolean (true or false).
    
    n_failures = failure_mask.sum()
    
    p_fail = n_failures / n_scenarios
    
    e_shortfall = np.dot(terminal_wealth - start * floor, failure_mask) / n_failures if n_failures > 0 else 0.0
    
    # Plot it
    
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24,9))
    
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color='indianred')
    
    wealth_ax.axhline(y=start, ls=':', color='black')
    
    wealth_ax.axhline(y=start * floor, ls='--', color='red')
    
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    
    hist_ax.axhline(y=start, ls=':', color='black')
    
    hist_ax.axhline(y=tw_mean, ls=':', color='blue')
    
    hist_ax.axhline(y=tw_median, ls=':', color='purple')
    
    hist_ax.annotate(f'Mean: ${int(tw_mean)}', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=24)
    
    hist_ax.annotate(f'Median: ${int(tw_median)}', xy=(0.7, 0.85), xycoords='axes fraction', fontsize=24)
    
    if (floor > 0.01):
        
        hist_ax.axhline(y=start * floor, ls='--', color='red', linewidth=3)
        
        hist_ax.annotate(f'Violations: {n_failures} ({p_fail * 100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}', xy=(0.7, 0.7), xycoords='axes fraction', fontsize=24)


def discount(time, interest_rate, price=1):

    '''
    Compute the price of a pure discount bond that pays "p" dollars at time t, given interest rate r
    :param t: time in years
    :param r: interest rate in annual terms
    :return: present value
    '''

    return price / (1 + interest_rate) ** time

def present_value(liabilities, interest_rate):

    '''
    Computes the PV of a sequence of liabilities
    :param liabilities: index by the time, and the values are the amounts of each liability
    :param r: interest rate in annual terms
    :return: PV of the sequence
    '''

    dates = liabilities.index

    discounts = discount(dates, interest_rate)

    return (discounts * liabilities).sum()

def funding_ratio(assets, liabilities, interest_rate):

    '''
    Computes the funding ratio.
    :param assets:
    :param liabilitites:
    :param interest_rate:
    :return:
    '''

    return assets / present_value(liabilities, interest_rate)

def show_funding_ratio(assets, interest_rate):
    
    fr = funding_ratio(assets, liabilities, interest_rate)
    
    print(f'{fr*100:.2f}')