import numpy as np
import pandas as pd
import talib
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from google.cloud import storage
from config_entity import (SimulateStrategyConfig)

import warnings
warnings.filterwarnings('ignore')


class StrategySimulation:
    def __init__(self, config: SimulateStrategyConfig):
                #  df, rsi, prob_thres, LE1, LE2, SL, PP, window=5, starting_corpus=500000):
        
        self.config = config
        self.storage_client = storage.Client()

        self.df = pd.read_csv(self.config.data_path + '/df_pred_forSim.csv')
        self.df.Date = pd.to_datetime(self.df.Date).dt.date
        self.df.growth_future_5d = self.df.growth_future_5d - 1
        print (self.df.shape, self.df[self.df.split=='test'].shape)

        self.risk_free_rate = 0.08
        self.max_daily_investment = self.config.starting_corpus / 5

    def preprocess_data(self):
        # Ensure DataFrame is sorted by Date and Ticker
        self.df.sort_values(by=['Ticker', 'Date'], inplace=True)
        
        # Calculate max of high over next 5 days including the current day
        self.df['high_5day'] = self.df.groupby('Ticker')['High'].rolling(window=self.config.window, min_periods=1).max().shift(-self.config.window + 1).reset_index(level=0, drop=True)
        # Calculate min of low over next 5 days including the current day
        self.df['low_5day'] = self.df.groupby('Ticker')['Low'].rolling(window=self.config.window, min_periods=1).min().shift(-self.config.window + 1).reset_index(level=0, drop=True)
        # Calculate low of day 1 (current day)
        self.df['low_1day'] = self.df['Low']
        # Calculate 'low_before_high_5day'
        self.df['high_5day_date'] = self.df.groupby('Ticker')['High'].rolling(window=self.config.window, min_periods=1).apply(lambda x: x.idxmax()).shift(-self.config.window + 1).reset_index(level=0, drop=True)
        self.df['low_5day_date'] = self.df.groupby('Ticker')['Low'].rolling(window=self.config.window, min_periods=1).apply(lambda x: x.idxmin()).shift(-self.config.window + 1).reset_index(level=0, drop=True)
        self.df['low_before_high_5day'] = (self.df['low_5day_date'] < self.df['high_5day_date']).astype(int)
        # Calculate 'max_profit_5day': using 'high_5day' and curr day open, calculate percent change
        self.df['max_profit_5day'] = (self.df['high_5day'] - self.df['Open']) / self.df['Open']
        # Calculate 'stop_loss_5day': using 'low_5day' and curr day open, calculate percent change
        self.df['stop_loss_5day'] = (self.df['low_5day'] - self.df['Open']) / self.df['Open']
        # Calculate 50 day simple moving average (SMA50)
        self.df['SMA50'] = self.df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
        # Calculate 20 day simple moving average (SMA20)
        self.df['SMA20'] = self.df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        # Calculate 30 day standard deviation (Volatility30)
        self.df['Volatility30'] = self.df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=30, min_periods=1).std()) * np.sqrt(252)
        # Calculate RSI (14-day period by default)
        self.df['RSI'] = self.df.groupby('Ticker')['Adj Close'].transform(lambda x: talib.RSI(x, timeperiod=14))
        # Drop intermediate columns if necessary
        self.df.drop(columns=['high_5day_date', 'low_5day_date'], inplace=True)
        print("Indicators calculated: High 5-day, Low 5-day, Low 1-day, High/Low dates, SMA, Volatility, RSI.")

    def calculate_selection_criteria(self):
        self.df.sort_values(by=['Ticker', 'Date'], inplace=True)

        # Calculate stock selection criteria
        self.df['stock_sel'] = np.where((self.df['RSI'] < self.config.rsi) & 
                                        (self.df['Open'] > 20.0) &  ## avoid penny stocks
                                        (self.df['pred_xgp_rf_best_rank'] <= 7) &
                                        (self.df['pred_xgp_rf_best'] > self.config.prob_thres), 1, 0)
        # Calculate lower_entry criteria
        self.df['close_prev'] = self.df.groupby('Ticker')['Adj Close'].shift(1)
        self.df['low_entry'] = np.where((self.df['Open'] <= self.df['close_prev'] * (1 + self.config.LE1)), 1, 0)# & 
                                        # (self.df['Open'] <= self.df['close_prev'] * (1 + self.config.LE2)), 1, 0)
        # Calculate stop_loss criteria
        self.df['stop_loss'] = np.where(self.df['stop_loss_5day'] <= -self.config.SL, 1, 0)
        # Calculate take_profit criteria
        self.df['take_profit'] = np.where(self.df['max_profit_5day'] >= self.config.PP, 1, 0)
        print("Selection criteria calculated: stock selection, lower entry, stop loss, take profit.")

        # print(f"Number of selected stocks: {len(self.selected_stocks)}")


    def get_profit_booking_date(self, ticker):
        
        # Filter data for the specific ticker
        df_ticker = self.df[self.df['Ticker'] == ticker].copy()

        # Calculate the required shifts
        for i in range(0, self.config.window+1):
            df_ticker[f'high_shift_{i}'] = df_ticker['High'].shift(-i)
            df_ticker[f'low_shift_{i}'] = df_ticker['Low'].shift(-i)
            df_ticker[f'date_shift_{i}'] = df_ticker['Date'].shift(-i)

        # Calculate conditions for profit booking
        for i in range(0, self.config.window+1):
            df_ticker[f'profit_condition_{i}'] = np.where(
                (df_ticker['stock_sel'] == 1) & 
                (df_ticker['low_entry'] == 1) & 
                ((df_ticker[f'high_shift_{i}'] - df_ticker['Open']) / df_ticker['Open'] >= self.config.PP) ,
                df_ticker[f'date_shift_{i}'], np.nan)

        # Calculate conditions for loss booking
        for i in range(0, self.config.window+1):
            df_ticker[f'loss_condition_{i}'] = np.where(
                (df_ticker['stock_sel'] == 1) & 
                (df_ticker['low_entry'] == 1) & 
                ((df_ticker[f'low_shift_{i}'] - df_ticker['Open']) / df_ticker['Open'] <= -self.config.SL) ,
                df_ticker[f'date_shift_{i}'], np.nan)
            
        # Find the first valid profit booking date
        profit_booking_date_cols = [f'profit_condition_{i}' for i in range(0, self.config.window+1)]
        df_ticker['profit_booking_date'] = df_ticker[profit_booking_date_cols].bfill(axis=1).iloc[:, 0]

        # Find the first valid loss booking date
        loss_booking_date_cols = [f'loss_condition_{i}' for i in range(0, self.config.window+1)]
        df_ticker['loss_booking_date'] = df_ticker[loss_booking_date_cols].bfill(axis=1).iloc[:, 0]
        
        df_ticker.drop(['high_shift_0','low_shift_0','date_shift_0',
                        'high_shift_1','low_shift_1','date_shift_1',
                        'high_shift_2','low_shift_2','date_shift_2',
                        'high_shift_3','low_shift_3','date_shift_3',
                        'high_shift_4','low_shift_4','date_shift_4',
                        'high_shift_5','low_shift_5','date_shift_5',
                        'profit_condition_0','profit_condition_1','profit_condition_2','profit_condition_3','profit_condition_4','profit_condition_5',
                        'loss_condition_0','loss_condition_1','loss_condition_2','loss_condition_3','loss_condition_4','loss_condition_5'], 
                        axis=1, inplace=True)
        return df_ticker #[profit_booking_dates, loss_booking_dates]

    def apply_combined_weighting(self):

        # Ensure that the dataframe is sorted by Date
        self.df.sort_values(by=['Ticker', 'Date'], inplace=True)    
        sell_date = []
        for tckr in self.df.Ticker.unique():
            sellDate_df  = self.get_profit_booking_date(tckr)
            sell_date.append(sellDate_df)
        self.df = pd.concat(sell_date)

        # Filter the stocks that meet the selection criteria
        self.selected_stocks = self.df[self.df['stock_sel'] == 1].copy()
        self.selected_stocks = self.selected_stocks[self.selected_stocks.split=='test']
        self.selected_stocks = self.selected_stocks[self.selected_stocks.Date < self.selected_stocks.Date.max()-datetime.timedelta(7)]
        
        self.selected_stocks.sort_values(by=['Ticker', 'Date'], inplace=True)
        
        #calculate if profit/loss
        self.selected_stocks['is_profit'] = np.nan
        self.selected_stocks.loc[self.selected_stocks.low_entry==0, 'is_profit'] = np.nan
        self.selected_stocks.loc[(self.selected_stocks.low_entry==1)&   
                                (self.selected_stocks.take_profit==1), 'is_profit'] = 1
        self.selected_stocks.loc[(self.selected_stocks.low_entry==1)&
                                (self.selected_stocks.take_profit==0)&
                                (self.selected_stocks.growth_future_5d > 0), 'is_profit'] = 1
        self.selected_stocks.loc[(self.selected_stocks.low_entry==1)&
                                (self.selected_stocks.take_profit==0)&
                                (self.selected_stocks.growth_future_5d < 0), 'is_profit'] = 0
        self.selected_stocks.loc[(self.selected_stocks.low_entry==1)&
                                (self.selected_stocks.take_profit==0)&
                                (self.selected_stocks.stop_loss==1), 'is_profit'] = 0
        self.selected_stocks.loc[(self.selected_stocks.low_entry==1)&
                                (self.selected_stocks.stop_loss==1)&
                                (self.selected_stocks.low_before_high_5day==1), 'is_profit'] = 0
        
        #calculate sell date
        self.selected_stocks.loc[self.selected_stocks.is_profit.isna(),'sell_date'] = np.nan
        self.selected_stocks.loc[self.selected_stocks.is_profit==0,'sell_date'] = self.selected_stocks.loc[self.selected_stocks.is_profit==0,'loss_booking_date']
        self.selected_stocks.loc[self.selected_stocks.is_profit==1,'sell_date'] = self.selected_stocks.loc[self.selected_stocks.is_profit==1,'profit_booking_date']
        
        #calculate intraday dummy
        self.selected_stocks['intraday'] = 0
        self.selected_stocks.loc[self.selected_stocks.sell_date == self.selected_stocks.Date, 'intraday'] = 1
        
        self.selected_stocks.sell_date = self.selected_stocks.sell_date + datetime.timedelta(1) ## next purchase day (return to be apeended to investment)
        self.selected_stocks.loc[(self.selected_stocks.is_profit.isna())&
                                 (self.selected_stocks.low_entry==1),'sell_date'] = self.selected_stocks.loc[(self.selected_stocks.is_profit.isna())&
                                                                                                             (self.selected_stocks.low_entry==1),'Date'] + datetime.timedelta(5)

        daily_investments = []
        rollingInvestment = self.max_daily_investment
        rollInvdf = pd.DataFrame(columns=['Date','Ticker','sell_date','future_net_return'])
        # print (rollingInvestment)
        for date in sorted(self.selected_stocks['Date'].unique()):
            # print (date, rollingInvestment)
            daily_df = self.selected_stocks[self.selected_stocks['Date'] == date]
            if len(daily_df) >=1:
                combined_score = (daily_df['pred_xgp_rf_best'] / daily_df['Volatility30'])
                total_combined_score = combined_score.sum()
                #calculate weights
                daily_df['weights'] = combined_score / total_combined_score

                # calculate weighted investment
                daily_df['rollInv'] = rollInvdf.loc[rollInvdf.sell_date <= date, 'future_net_return'].sum()
                rollingInvestment = rollingInvestment + rollInvdf.loc[rollInvdf.sell_date <= date, 'future_net_return'].sum()
                daily_df['investment_prob_volt'] = daily_df['weights'] * (rollingInvestment)
                
                rollInvdf = rollInvdf[rollInvdf.sell_date > date]

                # calculate actual investment
                daily_df['count_buy'] = np.floor(daily_df['investment_prob_volt'] / daily_df['Open'])
                daily_df['investment_actual'] = daily_df['count_buy'] * daily_df['Open']
                
                if daily_df['investment_actual'].sum() < rollingInvestment:
                    buffer = rollingInvestment - daily_df['investment_actual'].sum() ### cases where total actual investement dont round up to previous day total return
                else:
                    pass

                #calculate gross return
                daily_df.loc[daily_df.low_entry==0, 'future_gross_return'] = 0
                daily_df.loc[(daily_df.low_entry==1)&   
                             (daily_df.take_profit==1), 'future_gross_return'] = daily_df.loc[(daily_df.low_entry==1)&
                                                                                              (daily_df.take_profit==1), 'investment_actual'] * (1 + self.config.PP)
                daily_df.loc[(daily_df.low_entry==1)&
                             (daily_df.take_profit==0), 'future_gross_return'] = daily_df.loc[(daily_df.low_entry==1)&
                                                                                              (daily_df.take_profit==0), 'investment_actual'] * (1 + daily_df.growth_future_5d)
                daily_df.loc[(daily_df.low_entry==1)&
                             (daily_df.take_profit==0)&
                             (daily_df.stop_loss==1), 'future_gross_return'] = daily_df.loc[(daily_df.low_entry==1)&
                                                                                            (daily_df.take_profit==0)&
                                                                                            (daily_df.stop_loss==1), 'investment_actual'] * (1 - self.config.SL)
                
                daily_df.loc[(daily_df.low_entry==1)&
                             (daily_df.stop_loss==1)&
                             (daily_df.low_before_high_5day==1), 'future_gross_return'] = daily_df.loc[(daily_df.low_entry==1)&
                                                                                                        (daily_df.stop_loss==1)&
                                                                                                        (daily_df.low_before_high_5day==1), 'investment_actual'] * (1 - self.config.SL)
                
                
                #calculate fee
                daily_df['fee'] = np.where(daily_df['intraday'] == 1, 0.07, 0.47)
                daily_df.loc[daily_df['future_gross_return'] == 0, 'fee'] = 0
                #calculate net return
                daily_df['future_net_return'] = daily_df['future_gross_return'] * (1 - daily_df['fee'] / 100)

                
                buffer = buffer + daily_df[daily_df.low_entry==0]['investment_actual'].sum() ## cases where entry was not done but investment allocated


                ## future net return realised on a future date to used as investement on that day
                rollInvdf = pd.concat([rollInvdf, daily_df.loc[daily_df.intraday==0, ['Date','Ticker','sell_date','future_net_return']]])

                # current day return to be used as investment for next day
                if daily_df['future_net_return'].sum() == 0:
                    pass
                else:
                    rollingInvestment = daily_df['future_net_return'].sum() + buffer - daily_df.loc[daily_df.intraday==0, 'future_net_return'].sum()



                daily_investments.append(daily_df)
        self.selected_stocks = pd.concat(daily_investments)
        self.selected_stocks['daily_return'] = self.selected_stocks['future_net_return'] / self.selected_stocks['investment_actual']
        
        print (f"NA return calc = {self.selected_stocks[self.selected_stocks.split=='test']['future_net_return'].isna().sum()}")
        
        print(f'============================================================================================')
        # print(f'Simulations params: {sim_params}')
        print(f" Count bids {len(self.selected_stocks[(self.selected_stocks['future_net_return'] != 0)])} in total, avg.bids per day {len(self.selected_stocks[(self.selected_stocks['future_net_return'] != 0)])/len(self.selected_stocks[(self.selected_stocks['future_net_return'] != 0)].Date.unique())}, number trading days: {len(self.selected_stocks[(self.selected_stocks['future_net_return'] != 0)].Date.unique())}")
        print(f" Total investment: {self.max_daily_investment}, Total future net return: {self.selected_stocks[self.selected_stocks.Date==self.selected_stocks.Date.max()].future_net_return.sum()}")
        stop_loss_filter = (self.selected_stocks.stop_loss==1)&(self.selected_stocks.daily_return < 1)&(self.selected_stocks.low_entry==1)
        print(f"  Stop loss events: count = {len(self.selected_stocks[stop_loss_filter])}, net loss = {self.selected_stocks[stop_loss_filter].future_net_return.sum()-self.selected_stocks[stop_loss_filter].investment_actual.sum()} ")
        take_profit_filter = (self.selected_stocks.take_profit==1)&(self.selected_stocks.daily_return > 1)&(self.selected_stocks.low_entry==1)
        print(f"  Take profit events: count = {len(self.selected_stocks[take_profit_filter])}, net profit = {self.selected_stocks[take_profit_filter].future_net_return.sum()-self.selected_stocks[take_profit_filter].investment_actual.sum()} ")
        print(f"  Start capital = {self.max_daily_investment}, Resulting capital: {self.selected_stocks[self.selected_stocks.Date==self.selected_stocks.Date.max()].future_net_return.sum()} ")
        print(f'============================================================================================')
    

    def calculate_return_metrics(self):
      
        # Calculate total return
        # total_return = (self.selected_stocks[self.selected_stocks.split=='test']['future_net_return'].sum() - 
        #                 self.selected_stocks[self.selected_stocks.split=='test']['investment_actual'].sum()) /  \
        #                     self.selected_stocks[self.selected_stocks.split=='test']['investment_actual'].sum()
        total_return = (self.selected_stocks[self.selected_stocks.Date==self.selected_stocks.Date.max()].future_net_return.sum() - self.max_daily_investment) / self.max_daily_investment

        # Calculate CAGR (Compound Annual Growth Rate)
        num_years = (self.selected_stocks[self.selected_stocks.split=='test']['Date'].max() - self.selected_stocks[self.selected_stocks.split=='test']['Date'].min()).days / 365.25
        cagr = (((self.selected_stocks[self.selected_stocks.Date==self.selected_stocks.Date.max()].future_net_return.sum()) / self.max_daily_investment) ** (1 / num_years)) - 1

        # Calculate daily returns
        daily_returns = self.selected_stocks[self.selected_stocks.split=='test']['daily_return']
        excess_daily_returns = (1 - daily_returns) - (self.risk_free_rate / 252)
        ann_std_dev = np.std(excess_daily_returns) * np.sqrt(252)

        annualised_returns = ((self.selected_stocks[self.selected_stocks.Date==self.selected_stocks.Date.max()].future_net_return.sum()/self.max_daily_investment)**(1/num_years)) - 1
        # Calculate Sharpe ratio
        # sharpe_ratio = np.mean(excess_daily_returns) / ann_std_dev
        sharpe_ratio =  annualised_returns / ann_std_dev

        print ({
            'total_return': total_return * 100,
            'CAGR': cagr * 100,
            'sharpe_ratio': sharpe_ratio
        })
    
    def run_simulation(self):
        self.preprocess_data()
        self.calculate_selection_criteria()
        self.apply_combined_weighting()
        self.calculate_return_metrics()

        self.selected_stocks[self.selected_stocks.Date >= (self.selected_stocks.Date.max() - datetime.timedelta(5))] \
            .drop(['investment_prob_volt','count_buy','investment_actual','future_gross_return','intraday','fee','future_net_return','daily_return'],axis=1) \
                .to_csv(self.config.root_dir + "/selectedStocks_lastWeek.csv")
        # self.selected_stocks.to_csv("df_sim.csv")