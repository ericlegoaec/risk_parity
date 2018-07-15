import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt
TOLERANCE = 1e-10


def _allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

    # It returns the risk of the weights distribution
    return portfolio_risk


def _assets_risk_contribution_to_allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) \
        / portfolio_risk

    # It returns the contribution of each asset to the risk of the weights
    # distribution
    return assets_risk_contribution


def _risk_budget_objective_error(weights, args):

    # The covariance matrix occupies the first position in the variable
    covariances = args[0]

    # The desired contribution of each asset to the portfolio risk occupies the
    # second position
    assets_risk_budget = args[1]

    # We convert the weights to a matrix
    weights = np.matrix(weights)

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = \
        _assets_risk_contribution_to_allocation_risk(weights, covariances)

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = \
        np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = \
        sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]

    # It returns the calculated error
    return error


def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})

    # Optimisation process in scipy
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})

    # Recover the weights from the optimised object
    weights = optimize_result.x

    # It returns the optimised weights
    return weights


def get_weights(yahoo_tickers=['GOOGL', 'AAPL', 'AMZN'],
                start_date=datetime.datetime(2016, 10, 31),
                end_date=datetime.datetime(2017, 10, 31)):

    # # We download the prices from Yahoo Finance
    # prices = pd.DataFrame([web.DataReader(t,
    #                                       'yahoo',
    #                                       start_date,
    #                                       end_date).loc[:, 'Adj Close']
    #                        for t in yahoo_tickers],
    #                       index=yahoo_tickers).T.asfreq('B').ffill()

    start_date = '2004-01-01'
    end_date = '2018-07-01'
    market_data = pd.read_csv(f'data/SPY {start_date} to {end_date}.csv')
    bond_data = pd.read_csv(f'data/AGG {start_date} to {end_date}.csv')
    market_close = market_data['Adj Close']
    bond_close = bond_data['Adj Close']

    market_close_stdev = np.std(market_close)
    market_close_var = np.var(market_close)
    market_close_mean = np.mean(market_close)
    market_close_coeff_of_variation = market_close_var / market_close_mean
    bond_close_stdev = np.std(bond_close)
    bond_close_var = np.var(bond_close)
    bond_close_mean = np.mean(bond_close)

    data = pd.concat([market_close, bond_close], axis=1, ignore_index=True)
    data_ret = data.pct_change()
    annual_ret = data_ret.mean() * 252
    covariances = data.cov().values
    # We calculate the covariance matrix

    # The desired contribution of each asset to the portfolio risk: we want all
    # asset to contribute equally
    assets_risk_budget = [1 / data.shape[1]] * data.shape[1]

    # Initial weights: equally weighted
    init_weights = [1 / data.shape[1]] * data.shape[1]

    # Optimisation process of weights
    weights = _get_risk_parity_weights(covariances, assets_risk_budget, init_weights)

    # Convert the weights to a pandas Series
    weights = pd.Series(weights, index=data.columns, name='weight')

    labels = ['SPY', 'AGG']
    plt.figure('Risk')
    plt.pie(assets_risk_budget, labels=labels, startangle=90, autopct='%.0f%%')
    plt.title('Target distribution of risk')
    plt.tight_layout()
    plt.show()

    # print out graph showing large holding in bonds (which are lower risk) where the risk is evenly divided by asset
    labels = ['SPY', 'AGG']
    plt.figure('Risk Dist')
    plt.pie(weights, labels=labels, startangle=90, autopct='%.0f%%')
    plt.title(f'Risk Distribution {start_date} to {end_date}')
    plt.tight_layout()
    plt.show()
    # It returns the optimised weights
    return weights


if __name__ == '__main__':
    print(get_weights())