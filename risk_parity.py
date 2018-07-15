import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data, wb


def get_data(ticker, start_date, end_date):
    return data.DataReader(ticker, 'yahoo', start_date, end_date)


def main():
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
    bond_close_coeff_of_variation = bond_close_var / bond_close_mean

    total_coeff_of_variation = \
        market_close_coeff_of_variation + bond_close_coeff_of_variation

    risk_dist_market = \
        market_close_coeff_of_variation / total_coeff_of_variation
    risk_dist_bond = \
        bond_close_coeff_of_variation / total_coeff_of_variation

    sizes = [risk_dist_market, risk_dist_bond]
    labels = ['SPY', 'AGG']
    plt.figure('Risk Dist')
    plt.pie(sizes, labels=labels, startangle=90, autopct='%.0f%%')
    plt.title(f'Risk Distribution {start_date} to {end_date}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
