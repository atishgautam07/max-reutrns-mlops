import os
from tqdm import tqdm
import time 
import json
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
import pandas as pd
import urllib.request as request
from google.cloud import storage
from common import create_gcs_directories
from config_entity import (DataIngestionConfig)

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.storage_client = storage.Client()
        self.ticker_df = None
        self.indexes_df = None
        self.macro_df = None

    def create_gcs_paths(self, gcs_path):
        # bucket_name = gcs_path.split('/')[2]
        prefix = '/'.join(gcs_path.split('/')[3:])
        create_gcs_directories(self.config.bucketName, [prefix]) 

    def tickerList(self):
        
        # https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/
        usStocks = ['MSFT', 'AAPL', 'GOOG', 'NVDA', 'AMZN', 'META', 'BRK-B', 'LLY', 'AVGO','V', 'JPM']
        # https://companiesmarketcap.com/european-union/largest-companies-in-the-eu-by-market-cap/
        euStocks = ['NVO','MC.PA', 'ASML', 'RMS.PA', 'OR.PA', 'SAP', 'ACN', 'TTE', 'SIE.DE','IDEXY','CDI.PA']
        # # https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/
        # INDIA_STOCKS = ['RELIANCE.NS','TCS.NS','HDB','BHARTIARTL.NS','IBN','SBIN.NS','LICI.NS','INFY','ITC.NS','HINDUNILVR.NS','LT.NS']

        inStocks = self.getIndiaStocks().index.tolist()

        # allTickers = usStocks  + euStocks + inStocks
        allTickers = inStocks

        print (len(usStocks), len(euStocks), len(inStocks), len(allTickers))
        return {
        'allTickers' : allTickers,
        'usStocks' : usStocks,
        'euStocks' : euStocks,
        'inStocks' : inStocks
        }

    def _get_growth_df(self, df:pd.DataFrame, prefix:str)->pd.DataFrame:
        '''Help function to produce a df with growth columns'''
        for i in [1,3,7,30,90,365]:
            df['growth_'+prefix+'_'+str(i)+'d'] = df['Adj Close'] / df['Adj Close'].shift(i)
            GROWTH_KEYS = [k for k in df.keys() if k.startswith('growth')]
        return df[GROWTH_KEYS]
        
    def get_fundamental_data(self, ticker):
        try:
            stock = yf.Ticker(ticker)

            # Fetch financial data
            
            try:
                info = pd.DataFrame(stock.info).T
            except:
                info = pd.DataFrame([stock.info]).T
            financials = stock.financials
            # balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow

            # Calculate metrics
            try:
                promoter_holding = float(info.loc['heldPercentInsiders'][0] + info.loc['heldPercentInstitutions'][0])
            except:
                promoter_holding = np.nan
            try:
                revenue_growth = float(info.loc['revenueGrowth'][0])
            except:
                revenue_growth = np.nan
            try:
                earnings_growth = float(financials.loc['Net Income'].pct_change(periods=-1, fill_method=None).iloc[0])
            except:
                earnings_growth = np.nan
            try:
                op_profit_margin = float(info.loc['operatingMargins'][0])
            except:
                op_profit_margin = np.nan
            try:
                roe = float(info.loc['returnOnEquity'][0])
            except:
                roe = np.nan
            try:
                roa = float(info.loc['returnOnAssets'][0])
            except:
                roa = np.nan
            try:
                debt_to_equity = float(info.loc['debtToEquity'][0])
            except:
                debt_to_equity = np.nan
            try:
                pe_ratio = float(info.loc['trailingPE'][0])
            except:
                pe_ratio = np.nan
            try:
                pb_ratio = float(info.loc['priceToBook'][0])
            except:
                pb_ratio = np.nan
            try:
                opCashFlow = float(cashflow.loc['Operating Cash Flow'].pct_change(periods=-1, fill_method=None).iloc[0])
            except:
                opCashFlow = np.nan

            # Append data to the list
            # screener_data.append(
            return {'Ticker': ticker,
                    'Promoter holding percent': promoter_holding,
                    'Revenue Growth': revenue_growth,
                    'Earnings Growth': earnings_growth,
                    'Operating Profit Margin': op_profit_margin,
                    'ROE': roe,
                    'ROA': roa,
                    'Debt to Equity': debt_to_equity,
                    'P/E': pe_ratio,
                    'P/B': pb_ratio,
                    'Operating Cash Flow Change': opCashFlow
                    }
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

    def getIndiaStocks(self):
        '''Generate list of Indian stocks for analysis and filtering based on fndamentals'''
        print("Generate list of Indian stocks for analysis and filtering based on fndamentals")
        
        dftmp = pd.read_excel(self.config.source_dir + "/" + "MCAP28032024.xlsx") ## static file containing list of india tickers with market cap
        dftmp = dftmp[(dftmp['Market capitalization as on March 28, 2024\n(In lakhs)'] != '*Not available for trading as on March 28, 2024')]
        dftmp = dftmp[(dftmp['Market capitalization as on March 28, 2024\n(In lakhs)'] <= 5000000)&
                    (dftmp['Market capitalization as on March 28, 2024\n(In lakhs)'] >= 5000)]
        dftmp.Symbol = dftmp.Symbol + ".NS"
        listInit = dftmp.Symbol.unique().tolist()
        
        screener_data = []
        for ticker in listInit:
            screener_data.append(self.get_fundamental_data(ticker))

        # Convert the list to a DataFrame
        screener_df = pd.DataFrame(screener_data)
        
        ## criteria for filtering stock based on fundamentals
        filtered_df = screener_df[
            # ((screener_df['Promoter holding percent'] > 0.50)|(screener_df['Promoter holding percent'].isna())) &
            ((screener_df['Revenue Growth'] > 0.03)|(screener_df['Revenue Growth'].isna())) &
            ((screener_df['Earnings Growth'] > 0.10)|(screener_df['Earnings Growth'].isna())) &
            ((screener_df['Operating Profit Margin'] > 0.10)|(screener_df['Operating Profit Margin'].isna())) &
            ((screener_df['ROE'] > 0.10)|(screener_df['ROE'].isna())) &
            # ((screener_df['ROA'] > 0.10)|(screener_df['ROA'].isna())) &
            ((screener_df['Debt to Equity'] < 1.5)|(screener_df['Debt to Equity'].isna())) &
            # ((screener_df['P/E'] < 60)|(screener_df['P/E'].isna())) &
            # ((screener_df['P/B'] < 60)|(screener_df['P/B'].isna())) &
            ((screener_df['Operating Cash Flow Change'] > 0)|(screener_df['Operating Cash Flow Change'].isna()))
        ]

        # Print the filtered screener data
        print("Filtered Screener Data:")
        print(filtered_df.set_index('Ticker').dropna(how='all').shape)
        return (filtered_df.set_index('Ticker').dropna(how='all'))

    def fetch(self, min_date = None):
        '''Fetch all data from APIs'''

        print('Fetching Tickers info from YFinance')
        self.fetch_tickers(min_date=min_date)
        print('Fetching Indexes info from YFinance')
        self.fetch_indexes(min_date=min_date)
        print('Fetching Macro info from FRED (Pandas_datareader)')
        self.fetch_macro(min_date=min_date)
    
    def fetch_tickers(self, min_date=None):
        '''Fetch Tickers data from the Yfinance API'''
        if min_date is None:
            min_date = "1970-01-01"
        else:
            min_date = pd.to_datetime(min_date)   

        print ("Fetching ticker lists to download from yFinance. ")
        
        bucket = self.storage_client.bucket(self.config.bucketName)
        blob = bucket.blob('/'.join(self.config.source_dir.split('/')[3:]) + '/' + 'tickerDict.json')
        print (blob)
        if self.config.calcTickers:
            print("Removing old list of tickers.")
            try:
                blob.delete()
                print(f"File {blob} deleted.")
                print("Proceeding with list generation.")
            except :
                print("Ticker list not present, proceeding with list generation.")
                pass
        else:
            pass

        # if self.config.calcTickers:
        #     print("Removing old list of tickers.")
        #     try:
        #         os.remove(os.path.join(self.config.resPath, 'tickerDict.json'))
        #     except OSError:
        #         print("Ticker list not present, proceeding wiht list generation.")
        #         pass
        # else:
        #     pass
        print (blob.exists(self.storage_client))
        if blob.exists(self.storage_client): #os.path.exists(os.path.join(self.config.resPath, 'tickerDict.json')):
            print("Loading list of IN stocks")
            # with open(self.config.source_dir+'/'+'tickerDict.json') as f:
            #     tickerDict = json.loads(f)
            str_json = blob.download_as_text()
            tickerDict = json.loads(str_json)
            print("Loaded list of IN stocks")
        else:
            tickerDict = self.tickerList()
            print("Generating list of IN stock and dumping to location.")
            with open('tickerDict.json', 'w', encoding='utf-8') as f:
                json.dump(tickerDict, f, ensure_ascii=False, indent=4)
            with open('tickerDict.json', 'rb') as f:
                blob.upload_from_file(f)
            print(f"Generated list of IN stock and dumped to location. {self.config.source_dir}")

        ALL_TICKERS = tickerDict['allTickers']
        US_STOCKS = tickerDict['usStocks']
        EU_STOCKS = tickerDict['euStocks']
        INDIA_STOCKS = tickerDict['inStocks']


        print (len(ALL_TICKERS))
        print(f'Going download data for this tickers: {ALL_TICKERS[0:3]}')
        tq = tqdm(ALL_TICKERS)
        for i,ticker in enumerate(tq):
            tq.set_description(ticker)

            # Download stock prices from YFinance
            historyPrices = yf.download(tickers = ticker,
                                # period = "max",
                                start=min_date,
                                interval = "1d")

            # generate features for historical prices, and what we want to predict

            if ticker in US_STOCKS:
                historyPrices['ticker_type'] = 'US'
            elif ticker in EU_STOCKS:
                historyPrices['ticker_type'] = 'EU'
            elif ticker in INDIA_STOCKS:
                historyPrices['ticker_type'] = 'INDIA'
            else:
                historyPrices['ticker_type'] = 'ERROR'

            historyPrices['Ticker'] = ticker
            historyPrices['Year']= historyPrices.index.year
            historyPrices['Month'] = historyPrices.index.month
            historyPrices['Weekday'] = historyPrices.index.weekday
            historyPrices['Date'] = historyPrices.index.date

            # historical returns
            for i in [1,3,7,30,90,365]:
                historyPrices['growth_'+str(i)+'d'] = historyPrices['Adj Close'] / historyPrices['Adj Close'].shift(i)
            historyPrices['growth_future_5d'] = historyPrices['Adj Close'].shift(-5) / historyPrices['Adj Close']

            # Technical indicators
            # SimpleMovingAverage 10 days and 20 days
            historyPrices['SMA10']= historyPrices['Close'].rolling(10).mean()
            historyPrices['SMA20']= historyPrices['Close'].rolling(20).mean()
            historyPrices['growing_moving_average'] = np.where(historyPrices['SMA10'] > historyPrices['SMA20'], 1, 0)
            historyPrices['high_minus_low_relative'] = (historyPrices.High - historyPrices.Low) / historyPrices['Adj Close']

            # 30d rolling volatility : https://ycharts.com/glossary/terms/rolling_vol_30
            historyPrices['volatility'] =   historyPrices['Adj Close'].rolling(30).std() * np.sqrt(252)

            # what we want to predict
            historyPrices['is_positive_growth_5d_future'] = np.where(historyPrices['growth_future_5d'] > 1, 1, 0)

            # sleep 1 sec between downloads - not to overload the API server
            time.sleep(1)

            if self.ticker_df is None:
                self.ticker_df = historyPrices
            else:
                self.ticker_df = pd.concat([self.ticker_df, historyPrices], ignore_index=True)

    def fetch_indexes(self, min_date=None):
        '''Fetch Indexes data from the Yfinance API'''

        if min_date is None:
            min_date = "1970-01-01"
        else:
            min_date = pd.to_datetime(min_date)   
        
        # https://finance.yahoo.com/quote/^INDIAVIX/
        # India VIX
        indiaVIX = yf.download(tickers = "^INDIAVIX",
                            start = min_date,    
                            # period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # https://finance.yahoo.com/quote/^BSESN/
        # bse
        bse = yf.download(tickers = "^BSESN",
                            start = min_date,    
                            # period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # https://finance.yahoo.com/quote/^NSEI/
        # nse
        nse = yf.download(tickers = "^NSEI",
                            start = min_date,    
                            # period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # https://finance.yahoo.com/quote/^CNXCMDT/
        # nifty commodity
        cmdt = yf.download(tickers = "^CNXCMDT",
                            start = min_date,    
                            # period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # https://finance.yahoo.com/quote/%5EGDAXI/
        # DAX PERFORMANCE-INDEX
        dax_daily = yf.download(tickers = "^GDAXI",
                            start = min_date,    
                            # period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)
        
        # https://finance.yahoo.com/quote/%5EGSPC/
        # SNP - SNP Real Time Price. Currency in USD
        snp500_daily = yf.download(tickers = "^GSPC",
                        start = min_date,          
                        #  period = "max",
                        interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)
        
        # https://finance.yahoo.com/quote/%5EDJI?.tsrc=fin-srch
        # Dow Jones Industrial Average
        dji_daily = yf.download(tickers = "^DJI",
                        start = min_date,       
                        #  period = "max",
                        interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)
        
        # https://finance.yahoo.com/quote/EPI/history?p=EPI
        # WisdomTree India Earnings Fund (EPI) : NYSEArca - Nasdaq Real Time Price (USD)
        epi_etf_daily = yf.download(tickers = "EPI",
                        start = min_date,            
                        #  period = "max",
                        interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)
        
        # VIX - Volatility Index
        # https://finance.yahoo.com/quote/%5EVIX/
        vix = yf.download(tickers = "^VIX",
                            start = min_date, 
                            # period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)
        
        # GOLD
        # https://finance.yahoo.com/quote/GC%3DF
        gold = yf.download(tickers = "GC=F",
                        start = min_date,   
                        #  period = "max",
                        interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)
        
        # WTI Crude Oil
        # https://uk.finance.yahoo.com/quote/CL=F/
        crude_oil = yf.download(tickers = "CL=F",
                        start = min_date,          
                        #  period = "max",
                        interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # Brent Oil
        # WEB: https://uk.finance.yahoo.com/quote/BZ=F/
        brent_oil = yf.download(tickers = "BZ=F",
                                start = min_date,
                                # period = "max",
                                interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # Natural Gas
        # WEB: https://uk.finance.yahoo.com/quote/NG=F/
        natural_gas = yf.download(tickers = "NG=F",
                                start = min_date,
                                # period = "max",
                                interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # Coal
        # WEB: https://uk.finance.yahoo.com/quote/QCLN/
        coal = yf.download(tickers = "QCLN",
                                start = min_date,
                                # period = "max",
                                interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # Copper
        # WEB: https://uk.finance.yahoo.com/quote/HG=F/
        copper = yf.download(tickers = "HG=F",
                                start = min_date,
                                # period = "max",
                                interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # Aluminium
        # WEB: https://uk.finance.yahoo.com/quote/ALI=F/
        aluminium = yf.download(tickers = "ALI=F",
                                start = min_date,
                                # period = "max",
                                interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # BTC_USD
        # WEB: https://finance.yahoo.com/quote/BTC-USD/
        btc_usd =  yf.download(tickers = "BTC-USD",
                            start = min_date,
                            #  period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)
        
        # Ethereum
        # WEB: https://finance.yahoo.com/quote/ETH-USD/
        eth_usd =  yf.download(tickers = "ETH-USD",
                            start = min_date,
                            #  period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # Ripple
        # WEB: https://finance.yahoo.com/quote/XRP-USD/
        xrp_usd =  yf.download(tickers = "XRP-USD",
                            start = min_date,
                            #  period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # BCH_USD
        # WEB: https://finance.yahoo.com/quote/BCH-USD/
        bch_usd =  yf.download(tickers = "BCH-USD",
                            start = min_date,
                            #  period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # USD
        # WEB: https://finance.yahoo.com/quote/USDINR=X/
        usdinr =  yf.download(tickers = "USDINR=X",
                            start = min_date,
                            #  period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # GBP
        # WEB: https://finance.yahoo.com/quote/GBPINR=X/
        gbpinr =  yf.download(tickers = "GBPINR=X",
                            start = min_date,
                            #  period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # EUR
        # WEB: https://finance.yahoo.com/quote/EURINR=X/
        eurinr =  yf.download(tickers = "EURINR=X",
                            start = min_date,
                            #  period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # JPY
        # WEB: https://finance.yahoo.com/quote/JPYINR=X/
        jpyinr =  yf.download(tickers = "JPYINR=X",
                            start = min_date,
                            #  period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # AUD
        # WEB: https://finance.yahoo.com/quote/AUDINR=X/
        audinr =  yf.download(tickers = "AUDINR=X",
                            start = min_date,
                            #  period = "max",
                            interval = "1d")
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # Prepare to merge
        dax_daily_to_merge = self._get_growth_df(dax_daily, 'dax')
        snp500_daily_to_merge = self._get_growth_df(snp500_daily, 'snp500')
        dji_daily_to_merge = self._get_growth_df(dji_daily, 'dji')
        epi_etf_daily_to_merge = self._get_growth_df(epi_etf_daily, 'epi')
        vix_to_merge = vix.rename(columns={'Adj Close':'vix_adj_close'})[['vix_adj_close']]
        gold_to_merge = self._get_growth_df(gold, 'gold')
        crude_oil_to_merge = self._get_growth_df(crude_oil,'wti_oil')
        brent_oil_to_merge = self._get_growth_df(brent_oil,'brent_oil')
        btc_usd_to_merge = self._get_growth_df(btc_usd,'btc_usd')

        indiaVIX_to_merge = self._get_growth_df(indiaVIX,'indiaVIX')
        nse_to_merge = self._get_growth_df(nse,'nse')
        bse_to_merge = self._get_growth_df(bse,'bse')
        cmdt_to_merge = self._get_growth_df(cmdt,'cmdt')
        natural_gas_to_merge = self._get_growth_df(natural_gas,'natural_gas')
        coal_to_merge = self._get_growth_df(coal,'coal')
        copper_to_merge = self._get_growth_df(copper,'copper')
        aluminium_to_merge = self._get_growth_df(aluminium,'aluminium')
        eth_usd_to_merge = self._get_growth_df(eth_usd,'eth_usd')
        xrp_usd_to_merge = self._get_growth_df(xrp_usd,'xrp_usd')
        bch_usd_to_merge = self._get_growth_df(bch_usd,'bch_usd')
        usdinr_to_merge = self._get_growth_df(usdinr,'usdinr')
        gbpinr_to_merge = self._get_growth_df(gbpinr,'gbpinr')
        eurinr_to_merge = self._get_growth_df(eurinr,'eurinr')
        jpyinr_to_merge = self._get_growth_df(jpyinr,'jpyinr')
        audinr_to_merge = self._get_growth_df(audinr,'audinr')

        # Merging
        m2 = pd.merge(snp500_daily_to_merge,
                                dax_daily_to_merge,
                                left_index=True,
                                right_index=True,
                                how='left',
                                validate='one_to_one') \
            .merge(dji_daily_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(epi_etf_daily_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(vix_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(gold_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(crude_oil_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(brent_oil_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(btc_usd_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(indiaVIX_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(nse_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(bse_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(cmdt_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(natural_gas_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(coal_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(copper_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(aluminium_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(eth_usd_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(xrp_usd_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(bch_usd_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(usdinr_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(gbpinr_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(eurinr_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(jpyinr_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one') \
            .merge(audinr_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one')   

        self.indexes_df = m2

    def fetch_macro(self, min_date=None):
        '''Fetch Macro data from FRED (using Pandas datareader)'''

        if min_date is None:
            min_date = "1970-01-01"
        else:
            min_date = pd.to_datetime(min_date)

        # Real Potential Gross Domestic Product (GDPPOT), Billions of Chained 2012 Dollars, QUARTERLY
        # https://fred.stlouisfed.org/series/GDPPOT
        gdppot = pdr.DataReader("GDPPOT", "fred", start=min_date)
        gdppot['gdppot_us_yoy'] = gdppot.GDPPOT/gdppot.GDPPOT.shift(4)-1
        gdppot['gdppot_us_qoq'] = gdppot.GDPPOT/gdppot.GDPPOT.shift(1)-1
        # sleep 1 sec between downloads - not to overload the API server
        time.sleep(1)

        # # "Core CPI index", MONTHLY
        # https://fred.stlouisfed.org/series/CPILFESL
        # The "Consumer Price Index for All Urban Consumers: All Items Less Food & Energy"
        # is an aggregate of prices paid by urban consumers for a typical basket of goods, excluding food and energy.
        # This measurement, known as "Core CPI," is widely used by economists because food and energy have very volatile prices.
        cpilfesl = pdr.DataReader("CPILFESL", "fred", start=min_date)
        cpilfesl['cpi_core_yoy'] = cpilfesl.CPILFESL/cpilfesl.CPILFESL.shift(12)-1
        cpilfesl['cpi_core_mom'] = cpilfesl.CPILFESL/cpilfesl.CPILFESL.shift(1)-1    
        time.sleep(1)

        # Fed rate https://fred.stlouisfed.org/series/FEDFUNDS
        fedfunds = pdr.DataReader("FEDFUNDS", "fred", start=min_date)
        time.sleep(1)


        # https://fred.stlouisfed.org/series/DGS1
        dgs1 = pdr.DataReader("DGS1", "fred", start=min_date)
        time.sleep(1)

        # https://fred.stlouisfed.org/series/DGS5
        dgs5 = pdr.DataReader("DGS5", "fred", start=min_date)
        time.sleep(1)

        # https://fred.stlouisfed.org/series/DGS10
        dgs10 = pdr.DataReader("DGS10", "fred", start=min_date)
        time.sleep(1)

        # gold volatility
        gold_volatility = pdr.DataReader("GVZCLS", "fred", start=min_date)
        time.sleep(1)


        gdppot_to_merge = gdppot[['gdppot_us_yoy','gdppot_us_qoq']]
        cpilfesl_to_merge = cpilfesl[['cpi_core_yoy','cpi_core_mom']]


        # Merging - start from daily stats (dgs1)
        m2 = pd.merge(dgs1,
                    dgs5,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one')
        
        m2['Date'] = m2.index

        # gdppot_to_merge is Quarterly (but m2 index is daily)
        m2['Quarter'] = m2.Date.dt.to_period('Q').dt.to_timestamp()

        m3 = pd.merge(m2,
                    gdppot_to_merge,
                    left_on='Quarter',
                    right_index=True,
                    how='left',
                    validate='many_to_one')

        # gdppot_to_merge is Quarterly
        # m3.index = pd.to_datetime(m3.index) # Ensure the index is a DatetimeIndex
        m3['Month'] = m2.Date.dt.to_period('M').dt.to_timestamp()

        m4 = pd.merge(m3,
                    fedfunds,
                    left_on='Month',
                    right_index=True,
                    how='left',
                    validate='many_to_one')
        
        m5 = pd.merge(m4,
                    cpilfesl_to_merge,
                    left_on='Month',
                    right_index=True,
                    how='left',
                    validate='many_to_one')
        
        m6 = pd.merge(m5,
                    dgs10,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one')
        
        m7 = pd.merge(m6,
                    gold_volatility,
                    left_index=True,
                    right_index=True,
                    how='left',
                    validate='one_to_one')
        
        fields_to_fill = ['cpi_core_yoy',	'cpi_core_mom','FEDFUNDS','DGS1','DGS5','DGS10','GVZCLS']
        # Fill missing values in selected fields with the last defined value
        for field in fields_to_fill:
            m7[field] = m7[field].ffill()

        self.macro_df =  m7      

    def persist(self):
        '''Save dataframes to files in a GCS directory 'dir' '''
        print (f"Save dataframes to files in a GCS {self.config.bucketName} ")
        # os.makedirs(self.config.root_dir, exist_ok=True)
        self.create_gcs_paths(self.config.root_dir)

        file_name = 'tickers_df.parquet'
        # if os.path.exists(file_name):
        #     os.remove(file_name)
        self.ticker_df.to_parquet(self.config.root_dir+"/"+file_name, compression='brotli')
    
        file_name = 'indexes_df.parquet'
        # if os.path.exists(file_name):
        #     os.remove(file_name)
        self.indexes_df.to_parquet(self.config.root_dir+"/"+file_name, compression='brotli')
    
        file_name = 'macro_df.parquet'
        # if os.path.exists(file_name):
        #     os.remove(file_name)
        self.macro_df.to_parquet(self.config.root_dir+"/"+file_name, compression='brotli')

    def load(self):
        """Load files from the GCS bucket"""
        print (f"Load files from the GCS bucket {self.config.bucketName}")
        self.ticker_df = pd.read_parquet(self.config.root_dir + '/tickers_df.parquet')
        self.macro_df = pd.read_parquet(self.config.root_dir + '/macro_df.parquet')
        self.indexes_df = pd.read_parquet(self.config.root_dir + '/indexes_df.parquet')