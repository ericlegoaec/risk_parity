import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data, wb
import math, numpy
import numpy as np
import pandas_datareader.data as web
# import pandas.io.data as web

def get_data(ticker, start_date, end_date):
    return data.DataReader(ticker, 'yahoo', start_date, end_date)


def main():

    # Set the stock symbols, data source, and time range

    np.random.seed(1)

    # length of artificial time series
    n_obs = 1000

    # number of different assets
    n_assets = 4

    # sample Nx4 data series matrix
    artificial_returns = np.random.randn(n_obs, n_assets) + 0.05
    returns = artificial_returns

    # stocks = ['GOOGL', 'TM', 'KO', 'PEP']
    # numAssets = len(stocks)
    # source = 'yahoo'
    # start = '2010-01-01'
    # end = '2015-10-31'
    #
    # # Retrieve stock price data and save just the dividend adjusted closing prices
    #
    # for symbol in stocks:
    #     data[symbol] = web.DataReader(symbol, data_source=source, start=start, end=end)['Adj Close']

    # Calculate simple linear returns

    # returns = (data - data.shift(1)) / data.shift(1)

    # Calculate individual mean returns and covariance between the stocks

    meanDailyReturns = returns.mean()
    covMatrix = np.cov(returns.T)

    # Calculate expected portfolio performance

    weights = [0.5, 0.2, 0.2, 0.1]
    portReturn = np.sum(meanDailyReturns * weights)
    portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))


    # def calc_min_variance_portfolio(return_vector, stddev_vector, correlation_matrix, target_return):
    #     """
    #     Given return, variance, and correlation data on multiple assets and a target portfolio
    #     return, calculate the minimum variance portfolio that achieves the target_return, if possible.
    #
    #     return_vector: vector of returns
    #     stddev_vector: vector of standard deviations of returns
    #     correlation_matrix: correlation matrix
    #     target_return: target portfolio return
    #     returns: (portfolio allocations, portfolio variance)
    #
    #     Short positions are indicated as negative allocations
    #     """
    #     MU = return_vector
    #     R = correlation_matrix
    #     m = target_return
    #     S = numpy.matrix(numpy.diagflat(stddev_vector))
    #     COV = S * R * S
    #     ONE = numpy.matrix((1,) * COV.shape[0]).T
    #     A = ONE.T * COV.I * ONE
    #     a = float(A)
    #     B = MU.T * COV.I * ONE
    #     b = float(B)
    #     C = MU.T * COV.I * MU
    #     c = float(C)
    #     LAMBDA = (a * m - b) / (a * c - (b * b))
    #     GAMMA = ((c - b * m) / ((a * c) - (b * b)))
    #     WSTAR = COV.I * ((LAMBDA * MU) + (GAMMA * ONE))
    #     STDDEV = math.sqrt(WSTAR.T * COV * WSTAR)
    #     return WSTAR, STDDEV
    #
    # def calc_max_return_portfolio(return_vector, stddev_vector, correlation_matrix, target_variance):
    #     """
    #     Given return, variance, and correlation data on multiple assets and a target portfolio
    #     variance, calculate the maximum return portfolio that achieves the target variance, if possible.
    #
    #     return_vector: vector of returns
    #     stddev_vector: vector of standard deviations of returns
    #     correlation_matrix: correlation matrix
    #     target_variance: target portfolio variance
    #     returns: (portfolio allocations, portfolio variance, portfolio return)
    #
    #     Short positions are indicated as negative allocations
    #     """
    #     last_return = None
    #     last_allocation = None
    #     last_stddev = None
    #     target_return = float(min(return_vector))
    #     while target_return <= float(max(return_vector)):
    #         this_allocation, this_stddev = calc_min_variance_portfolio(return_vector, stddev_vector, correlation_matrix,
    #                                                                    target_return)
    #         if this_stddev > target_variance:
    #             return (last_allocation, last_stddev, last_return)
    #         last_allocation = this_allocation
    #         last_stddev = this_stddev
    #         last_return = target_return
    #         target_return += .0005  # TODO: linear search, not ideal (try Newton Raphson instead?)
    #     return (None, None, None)
    #
    # if __name__ == "__main__":
    #     return_vector = numpy.matrix((0.05, 0.07, 0.15, 0.27)).T
    #     stddev_vector = numpy.matrix((0.07, 0.12, 0.30, 0.60)).T
    #     correlation_matrix = numpy.matrix(((1.0, 0.8, 0.5, 0.4),
    #                                        (0.8, 1.0, 0.7, 0.5),
    #                                        (0.5, 0.7, 1.0, 0.8),
    #                                        (0.4, 0.5, 0.8, 1.0)))
    #
    #     target_return = .125
    #     allocations, stddev = calc_min_variance_portfolio(return_vector, stddev_vector,
    #                                                       correlation_matrix, target_return)
    #     print("scenario 1 - optimize portfolio for target return")
    #     print("target return: %.2f%%" % (target_return * 100.0))
    #     print("min variance portfolio:")
    #     print(allocations)
    #     print("portfolio std deviation: %.2f%%" % (stddev * 100.0))
    #
    #     print("-" * 40)
    #
    #     target_variance = .15
    #     allocations, stddev, rtn = calc_max_return_portfolio(return_vector, stddev_vector,
    #                                                          correlation_matrix, target_variance)
    #     print("scenario 2 - optimize portfolio for target variance")
    #     print("target variance: %.2f%%" % (target_variance * 100.0))
    #     print("max return:")
    #     print(allocations)
    #     print("portfolio std deviation: %.2f%%" % (stddev * 100.0))
    #     print("portfolio return: %.2f%%" % (rtn * 100.0))

if __name__ == '__main__':
    main()
