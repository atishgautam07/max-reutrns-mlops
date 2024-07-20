import os
from tqdm import tqdm
import talib
import pickle
import numpy as np
import pandas as pd
from google.cloud import storage
from common import create_gcs_directories
from config_entity import DataTransformationConfig


class DataTransformation:
        def __init__(self, config: DataTransformationConfig):
                self.config = config
                self.storage_client = storage.Client()

                # read initial dfs from repo
                """Load files from the GCS bucket"""
                print (f"Load files from the GCS bucket {self.config.bucketName}")
                self.tickers_df = pd.read_parquet(self.config.data_path + '/tickers_df.parquet')
                self.macro_df = pd.read_parquet(self.config.data_path + '/macro_df.parquet')
                self.indexes_df = pd.read_parquet(self.config.data_path + '/indexes_df.parquet')

                # init transformed_df
                self.transformed_df = None

        def create_gcs_paths(self, gcs_path):
                # bucket_name = gcs_path.split('/')[2]
                prefix = '/'.join(gcs_path.split('/')[3:])
                create_gcs_directories(self.config.bucketName, [prefix]) 

        def transform(self):
                '''Transform all dataframes from repo'''

                # Transform initial tickers_df to one with Tech indicators
                self._transform_tickers()

                # merge tickers (tech.indicators) with macro_df and indexes_df 
                self._merge_tickers_macro_indexes_df()

                # truncate all data before 2000
                self.transformed_df = self.transformed_df[self.transformed_df.Date>='2000-01-01']

        def _transform_tickers(self):
                '''Transform tickers dataframes from repo'''

                # TaLib needs inputs of a datatype 'Double' 
                self.tickers_df['Volume'] = self.tickers_df['Volume']*1.0

                for key in ['Open','High','Low','Close','Volume','Adj Close']:
                        self.tickers_df.loc[:,key] = self.tickers_df[key].astype('double')

                merged_df = None

                # supress warnings
                pd.options.mode.chained_assignment = None  # default='warn'

                tickers = tqdm(self.tickers_df.Ticker.unique())
                for ticker in tickers:
                        tickers.set_description(ticker)
                        filter = (self.tickers_df.Ticker == ticker)
                        current_ticker_df = self.tickers_df[filter]
                        df_current_ticker_momentum_indicators = self._get_talib_momentum_indicators(current_ticker_df)
                        df_current_ticker_volume_indicators = self._get_talib_volatility_cycle_price_indicators(current_ticker_df)
                        df_current_ticker_pattern_indicators = self._get_talib_pattern_indicators(current_ticker_df)

                        # need to have same 'utc' time on both sides of merges
                        # https://stackoverflow.com/questions/73964894/you-are-trying-to-merge-on-datetime64ns-utc-and-datetime64ns-columns-if-yo
                        current_ticker_df['Date']= pd.to_datetime(current_ticker_df['Date'], utc=True)
                        df_current_ticker_momentum_indicators['Date']= pd.to_datetime(df_current_ticker_momentum_indicators['Date'], utc=True)
                        df_current_ticker_volume_indicators['Date']= pd.to_datetime(df_current_ticker_volume_indicators['Date'], utc=True)
                        df_current_ticker_pattern_indicators['Date']= pd.to_datetime(df_current_ticker_pattern_indicators['Date'], utc=True)
                        
                        # merge to one df
                        m1 = pd.merge(current_ticker_df, df_current_ticker_momentum_indicators.reset_index(), how = 'left', on = ["Date","Ticker"], validate = "one_to_one")
                        m2 = pd.merge(m1, df_current_ticker_volume_indicators.reset_index(), how = 'left', on = ["Date","Ticker"], validate = "one_to_one")
                        m3 = pd.merge(m2, df_current_ticker_pattern_indicators.reset_index(), how = 'left', on = ["Date","Ticker"], validate = "one_to_one")
                        # m3 = current_ticker_df

                        if merged_df is None:
                                merged_df = m3
                        else:
                                merged_df = pd.concat([merged_df,m3], ignore_index = False)

                self.transformed_df = merged_df    

        def _get_talib_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:

                momentum_df = None
                # ADX - Average Directional Movement Index
                talib_momentum_adx = talib.ADX(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
                # ADXR - Average Directional Movement Index Rating
                talib_momentum_adxr = talib.ADXR(df.High.values, df.Low.values, df.Close.values, timeperiod=14 )
                # APO - Absolute Price Oscillator
                talib_momentum_apo = talib.APO(df.Close.values, fastperiod=12, slowperiod=26, matype=0 )
                # AROON - Aroon
                talib_momentum_aroon = talib.AROON(df.High.values, df.Low.values, timeperiod=14 )
                # talib_momentum_aroon[0].size
                # talib_momentum_aroon[1].size
                # AROONOSC - Aroon Oscillator
                talib_momentum_aroonosc = talib.AROONOSC(df.High.values, df.Low.values, timeperiod=14)
                # BOP - Balance of Power
                # https://school.stockcharts.com/doku.php?id=technical_indicators:balance_of_power
                #calculate open prices as shifted closed prices from the prev day
                # open = df.Last.shift(1)
                talib_momentum_bop = talib.BOP(df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CCI - Commodity Channel Index
                talib_momentum_cci = talib.CCI(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
                # CMO - Chande Momentum Oscillator
                talib_momentum_cmo = talib.CMO(df.Close.values, timeperiod=14)
                # DX - Directional Movement Index
                talib_momentum_dx = talib.DX(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
                # MACD - Moving Average Convergence/Divergence
                talib_momentum_macd, talib_momentum_macdsignal, talib_momentum_macdhist = talib.MACD(df.Close.values, fastperiod=12, \
                                                                                                        slowperiod=26, signalperiod=9)
                # MACDEXT - MACD with controllable MA type
                talib_momentum_macd_ext, talib_momentum_macdsignal_ext, talib_momentum_macdhist_ext = talib.MACDEXT(df.Close.values, \
                                                                                                                fastperiod=12, \
                                                                                                                fastmatype=0, \
                                                                                                                slowperiod=26, \
                                                                                                                slowmatype=0, \
                                                                                                                signalperiod=9, \
                                                                                                                signalmatype=0)
                # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
                talib_momentum_macd_fix, talib_momentum_macdsignal_fix, talib_momentum_macdhist_fix = talib.MACDFIX(df.Close.values, \
                                                                                                                        signalperiod=9)
                # MFI - Money Flow Index
                talib_momentum_mfi = talib.MFI(df.High.values, df.Low.values, df.Close.values, df.Volume.values, timeperiod=14)
                # MINUS_DI - Minus Directional Indicator
                talib_momentum_minus_di = talib.MINUS_DM(df.High.values, df.Low.values, timeperiod=14)
                # MOM - Momentum
                talib_momentum_mom = talib.MOM(df.Close.values, timeperiod=10)
                # PLUS_DI - Plus Directional Indicator
                talib_momentum_plus_di = talib.PLUS_DI(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
                # PLUS_DM - Plus Directional Movement
                talib_momentum_plus_dm = talib.PLUS_DM(df.High.values, df.Low.values, timeperiod=14)
                # PPO - Percentage Price Oscillator
                talib_momentum_ppo = talib.PPO(df.Close.values, fastperiod=12, slowperiod=26, matype=0)
                # ROC - Rate of change : ((price/prevPrice)-1)*100
                talib_momentum_roc = talib.ROC(df.Close.values, timeperiod=10)
                # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
                talib_momentum_rocp = talib.ROCP(df.Close.values, timeperiod=10)
                # ROCR - Rate of change ratio: (price/prevPrice)
                talib_momentum_rocr = talib.ROCR(df.Close.values, timeperiod=10)
                # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
                talib_momentum_rocr100 = talib.ROCR100(df.Close.values, timeperiod=10)
                # RSI - Relative Strength Index
                talib_momentum_rsi = talib.RSI(df.Close.values, timeperiod=14)
                # STOCH - Stochastic
                talib_momentum_slowk, talib_momentum_slowd = talib.STOCH(df.High.values, df.Low.values, df.Close.values, \
                                                                        fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
                # STOCHF - Stochastic Fast
                talib_momentum_fastk, talib_momentum_fastd = talib.STOCHF(df.High.values, df.Low.values, df.Close.values, \
                                                                        fastk_period=5, fastd_period=3, fastd_matype=0)
                # STOCHRSI - Stochastic Relative Strength Index
                talib_momentum_fastk_rsi, talib_momentum_fastd_rsi = talib.STOCHRSI(df.Close.values, timeperiod=14, \
                                                                                        fastk_period=5, fastd_period=3, fastd_matype=0)
                # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
                talib_momentum_trix = talib.TRIX(df.Close.values, timeperiod=30)
                # ULTOSC - Ultimate Oscillator
                talib_momentum_ultosc = talib.ULTOSC(df.High.values, df.Low.values, df.Close.values, timeperiod1=7, timeperiod2=14, timeperiod3=28)
                # WILLR - Williams' %R
                talib_momentum_willr = talib.WILLR(df.High.values, df.Low.values, df.Close.values, timeperiod=14)

                momentum_df =   pd.DataFrame(
                {
                        # assume here multi-index <dateTime, ticker>
                        # 'datetime': df.index.get_level_values(0),
                        # 'ticker': df.index.get_level_values(1) ,

                        # old way with separate columns
                        'Date': df.Date.values,
                        'Ticker': df.Ticker,

                        'adx': talib_momentum_adx,
                        'adxr': talib_momentum_adxr,
                        'apo': talib_momentum_apo,
                        'aroon_1': talib_momentum_aroon[0] ,
                        'aroon_2': talib_momentum_aroon[1],
                        'aroonosc': talib_momentum_aroonosc,
                        'bop': talib_momentum_bop,
                        'cci': talib_momentum_cci,
                        'cmo': talib_momentum_cmo,
                        'dx': talib_momentum_dx,
                        'macd': talib_momentum_macd,
                        'macdsignal': talib_momentum_macdsignal,
                        'macdhist': talib_momentum_macdhist,
                        'macd_ext': talib_momentum_macd_ext,
                        'macdsignal_ext': talib_momentum_macdsignal_ext,
                        'macdhist_ext': talib_momentum_macdhist_ext,
                        'macd_fix': talib_momentum_macd_fix,
                        'macdsignal_fix': talib_momentum_macdsignal_fix,
                        'macdhist_fix': talib_momentum_macdhist_fix,
                        'mfi': talib_momentum_mfi,
                        'minus_di': talib_momentum_minus_di,
                        'mom': talib_momentum_mom,
                        'plus_di': talib_momentum_plus_di,
                        'dm': talib_momentum_plus_dm,
                        'ppo': talib_momentum_ppo,
                        'roc': talib_momentum_roc,
                        'rocp': talib_momentum_rocp,
                        'rocr': talib_momentum_rocr,
                        'rocr100': talib_momentum_rocr100,
                        'rsi': talib_momentum_rsi,
                        'slowk': talib_momentum_slowk,
                        'slowd': talib_momentum_slowd,
                        'fastk': talib_momentum_fastk,
                        'fastd': talib_momentum_fastd,
                        'fastk_rsi': talib_momentum_fastk_rsi,
                        'fastd_rsi': talib_momentum_fastd_rsi,
                        'trix': talib_momentum_trix,
                        'ultosc': talib_momentum_ultosc,
                        'willr': talib_momentum_willr,
                }
                )
                return momentum_df

        def _get_talib_volatility_cycle_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:

                # TA-Lib Volume indicators
                # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/volume_indicators.md
                # AD - Chaikin A/D Line
                talib_ad = talib.AD(df.High.values, df.Low.values, df.Close.values, df.Volume.values)
                # ADOSC - Chaikin A/D Oscillator
                talib_adosc = talib.ADOSC(
                        df.High.values, df.Low.values, df.Close.values, df.Volume.values, fastperiod=3, slowperiod=10)
                # OBV - On Balance Volume
                talib_obv = talib.OBV(
                        df.Close.values, df.Volume.values)

                # TA-Lib Volatility indicators
                # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/volatility_indicators.md
                # ATR - Average True Range
                talib_atr = talib.ATR(
                        df.High.values, df.Low.values, df.Close.values, timeperiod=14)
                # NATR - Normalized Average True Range
                talib_natr = talib.NATR(
                        df.High.values, df.Low.values, df.Close.values, timeperiod=14)
                # OBV - On Balance Volume
                talib_obv = talib.OBV(
                        df.Close.values, df.Volume.values)
                        
                # TA-Lib Cycle Indicators
                # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/cycle_indicators.md
                # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
                talib_ht_dcperiod = talib.HT_DCPERIOD(df.Close.values)
                # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
                talib_ht_dcphase = talib.HT_DCPHASE(df.Close.values)
                # HT_PHASOR - Hilbert Transform - Phasor Components
                talib_ht_phasor_inphase, talib_ht_phasor_quadrature = talib.HT_PHASOR(
                        df.Close.values)
                # HT_SINE - Hilbert Transform - SineWave
                talib_ht_sine_sine, talib_ht_sine_leadsine = talib.HT_SINE(
                        df.Close.values)
                # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
                talib_ht_trendmode = talib.HT_TRENDMODE(df.Close.values)

                # TA-Lib Price Transform Functions
                # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/price_transform.md
                # AVGPRICE - Average Price
                talib_avgprice = talib.AVGPRICE(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # MEDPRICE - Median Price
                talib_medprice = talib.MEDPRICE(df.High.values, df.Low.values)
                # TYPPRICE - Typical Price
                talib_typprice = talib.TYPPRICE(
                        df.High.values, df.Low.values, df.Close.values)
                # WCLPRICE - Weighted Close Price
                talib_wclprice = talib.WCLPRICE(
                        df.High.values, df.Low.values, df.Close.values)

                volume_volatility_cycle_price_df = pd.DataFrame(
                        {'Date': df.Date.values,
                        'Ticker': df.Ticker,
                        # TA-Lib Volume indicators
                        'ad': talib_ad,
                        'adosc': talib_adosc,
                        'obv': talib_obv,
                        # TA-Lib Volatility indicators
                        'atr': talib_atr,
                        'natr': talib_natr,
                        'obv': talib_obv,
                        # TA-Lib Cycle Indicators
                        'ht_dcperiod': talib_ht_dcperiod,
                        'ht_dcphase': talib_ht_dcphase,
                        'ht_phasor_inphase': talib_ht_phasor_inphase,
                        'ht_phasor_quadrature': talib_ht_phasor_quadrature,
                        'ht_sine_sine': talib_ht_sine_sine,
                        'ht_sine_leadsine': talib_ht_sine_leadsine,
                        'ht_trendmod': talib_ht_trendmode,
                        # TA-Lib Price Transform Functions
                        'avgprice': talib_avgprice,
                        'medprice': talib_medprice,
                        'typprice': talib_typprice,
                        'wclprice': talib_wclprice,
                        }
                        )

                # Need a proper date type
                volume_volatility_cycle_price_df['Date'] = pd.to_datetime(
                        volume_volatility_cycle_price_df['Date'])

                return volume_volatility_cycle_price_df

        def _get_talib_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
                # TA-Lib Pattern Recognition indicators
                # https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/pattern_recognition.md
                # Nice article about candles (pattern recognition) https://medium.com/analytics-vidhya/recognizing-over-50-candlestick-patterns-with-python-4f02a1822cb5

                # CDL2CROWS - Two Crows
                talib_cdl2crows = talib.CDL2CROWS(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDL3BLACKCROWS - Three Black Crows
                talib_cdl3blackrows = talib.CDL3BLACKCROWS(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDL3INSIDE - Three Inside Up/Down
                talib_cdl3inside = talib.CDL3INSIDE(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDL3LINESTRIKE - Three-Line Strike
                talib_cdl3linestrike = talib.CDL3LINESTRIKE(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDL3OUTSIDE - Three Outside Up/Down
                talib_cdl3outside = talib.CDL3OUTSIDE(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDL3STARSINSOUTH - Three Stars In The South
                talib_cdl3starsinsouth = talib.CDL3STARSINSOUTH(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDL3WHITESOLDIERS - Three Advancing White Soldiers
                talib_cdl3whitesoldiers = talib.CDL3WHITESOLDIERS(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLABANDONEDBABY - Abandoned Baby
                talib_cdlabandonedbaby = talib.CDLABANDONEDBABY(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
                # CDLADVANCEBLOCK - Advance Block
                talib_cdladvancedblock = talib.CDLADVANCEBLOCK(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLBELTHOLD - Belt-hold
                talib_cdlbelthold = talib.CDLBELTHOLD(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLBREAKAWAY - Breakaway
                talib_cdlbreakaway = talib.CDLBREAKAWAY(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLCLOSINGMARUBOZU - Closing Marubozu
                talib_cdlclosingmarubozu = talib.CDLCLOSINGMARUBOZU(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLCONCEALBABYSWALL - Concealing Baby Swallow
                talib_cdlconcealbabyswall = talib.CDLCONCEALBABYSWALL(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLCOUNTERATTACK - Counterattack
                talib_cdlcounterattack = talib.CDLCOUNTERATTACK(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLDARKCLOUDCOVER - Dark Cloud Cover
                talib_cdldarkcloudcover = talib.CDLDARKCLOUDCOVER(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
                # CDLDOJI - Doji
                talib_cdldoji = talib.CDLDOJI(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLDOJISTAR - Doji Star
                talib_cdldojistar = talib.CDLDOJISTAR(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLDRAGONFLYDOJI - Dragonfly Doji
                talib_cdldragonflydoji = talib.CDLDRAGONFLYDOJI(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLENGULFING - Engulfing Pattern
                talib_cdlengulfing = talib.CDLENGULFING(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)

                # CDLEVENINGDOJISTAR - Evening Doji Star
                talib_cdleveningdojistar = talib.CDLEVENINGDOJISTAR(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
                # CDLEVENINGSTAR - Evening Star
                talib_cdleveningstar = talib.CDLEVENINGSTAR(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
                # CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines
                talib_cdlgapsidesidewhite = talib.CDLGAPSIDESIDEWHITE(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLGRAVESTONEDOJI - Gravestone Doji
                talib_cdlgravestonedoji = talib.CDLGRAVESTONEDOJI(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLHAMMER - Hammer
                talib_cdlhammer = talib.CDLHAMMER(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLHANGINGMAN - Hanging Man
                talib_cdlhangingman = talib.CDLHANGINGMAN(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLHARAMI - Harami Pattern
                talib_cdlharami = talib.CDLHARAMI(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLHARAMICROSS - Harami Cross Pattern
                talib_cdlharamicross = talib.CDLHARAMICROSS(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLHIGHWAVE - High-Wave Candle
                talib_cdlhighwave = talib.CDLHIGHWAVE(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLHIKKAKE - Hikkake Pattern
                talib_cdlhikkake = talib.CDLHIKKAKE(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLHIKKAKEMOD - Modified Hikkake Pattern
                talib_cdlhikkakemod = talib.CDLHIKKAKEMOD(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)

                # CDLHOMINGPIGEON - Homing Pigeon
                talib_cdlhomingpigeon = talib.CDLHOMINGPIGEON(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLIDENTICAL3CROWS - Identical Three Crows
                talib_cdlidentical3crows = talib.CDLIDENTICAL3CROWS(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLINNECK - In-Neck Pattern
                talib_cdlinneck = talib.CDLINNECK(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLINVERTEDHAMMER - Inverted Hammer
                talib_cdlinvertedhammer = talib.CDLINVERTEDHAMMER(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLKICKING - Kicking
                talib_cdlkicking = talib.CDLKICKING(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu
                talib_cdlkickingbylength = talib.CDLKICKINGBYLENGTH(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLLADDERBOTTOM - Ladder Bottom
                talib_cdlladderbottom = talib.CDLLADDERBOTTOM(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLLONGLEGGEDDOJI - Long Legged Doji
                talib_cdllongleggeddoji = talib.CDLLONGLEGGEDDOJI(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLLONGLINE - Long Line Candle
                talib_cdllongline = talib.CDLLONGLINE(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLMARUBOZU - Marubozu
                talib_cdlmarubozu = talib.CDLMARUBOZU(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLMATCHINGLOW - Matching Low
                talib_cdlmatchinglow = talib.CDLMATCHINGLOW(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)

                # CDLMATHOLD - Mat Hold
                talib_cdlmathold = talib.CDLMATHOLD(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
                # CDLMORNINGDOJISTAR - Morning Doji Star
                talib_cdlmorningdojistar = talib.CDLMORNINGDOJISTAR(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
                # CDLMORNINGSTAR - Morning Star
                talib_cdlmorningstar = talib.CDLMORNINGSTAR(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
                # CDLONNECK - On-Neck Pattern
                talib_cdlonneck = talib.CDLONNECK(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLPIERCING - Piercing Pattern
                talib_cdlpiercing = talib.CDLPIERCING(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLRICKSHAWMAN - Rickshaw Man
                talib_cdlrickshawman = talib.CDLRICKSHAWMAN(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLRISEFALL3METHODS - Rising/Falling Three Methods
                talib_cdlrisefall3methods = talib.CDLRISEFALL3METHODS(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLSEPARATINGLINES - Separating Lines
                talib_cdlseparatinglines = talib.CDLSEPARATINGLINES(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLSHOOTINGSTAR - Shooting Star
                talib_cdlshootingstar = talib.CDLSHOOTINGSTAR(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLSHORTLINE - Short Line Candle
                talib_cdlshortline = talib.CDLSHORTLINE(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLSPINNINGTOP - Spinning Top
                talib_cdlspinningtop = talib.CDLSPINNINGTOP(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)

                # CDLSTALLEDPATTERN - Stalled Pattern
                talib_cdlstalledpattern = talib.CDLSTALLEDPATTERN(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLSTICKSANDWICH - Stick Sandwich
                talib_cdlsticksandwich = talib.CDLSTICKSANDWICH(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLTAKURI - Takuri (Dragonfly Doji with very long lower shadow)
                talib_cdltakuru = talib.CDLTAKURI(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLTASUKIGAP - Tasuki Gap
                talib_cdltasukigap = talib.CDLTASUKIGAP(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLTHRUSTING - Thrusting Pattern
                talib_cdlthrusting = talib.CDLTHRUSTING(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLTRISTAR - Tristar Pattern
                talib_cdltristar = talib.CDLTRISTAR(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLUNIQUE3RIVER - Unique 3 River
                talib_cdlunique3river = talib.CDLUNIQUE3RIVER(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLUPSIDEGAP2CROWS - Upside Gap Two Crows
                talib_cdlupsidegap2crows = talib.CDLUPSIDEGAP2CROWS(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)
                # CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods
                talib_cdlxsidegap3methods = talib.CDLXSIDEGAP3METHODS(
                        df.Open.values, df.High.values, df.Low.values, df.Close.values)

                pattern_indicators_df = pd.DataFrame(
                        {'Date': df.Date.values,
                        'Ticker': df.Ticker,
                        # TA-Lib Pattern Recognition indicators
                        'cdl2crows': talib_cdl2crows,
                        'cdl3blackrows': talib_cdl3blackrows,
                        'cdl3inside': talib_cdl3inside,
                        'cdl3linestrike': talib_cdl3linestrike,
                        'cdl3outside': talib_cdl3outside,
                        'cdl3starsinsouth': talib_cdl3starsinsouth,
                        'cdl3whitesoldiers': talib_cdl3whitesoldiers,
                        'cdlabandonedbaby': talib_cdlabandonedbaby,
                        'cdladvancedblock': talib_cdladvancedblock,
                        'cdlbelthold': talib_cdlbelthold,
                        'cdlbreakaway': talib_cdlbreakaway,
                        'cdlclosingmarubozu': talib_cdlclosingmarubozu,
                        'cdlconcealbabyswall': talib_cdlconcealbabyswall,
                        'cdlcounterattack': talib_cdlcounterattack,
                        'cdldarkcloudcover': talib_cdldarkcloudcover,
                        'cdldoji': talib_cdldoji,
                        'cdldojistar': talib_cdldojistar,
                        'cdldragonflydoji': talib_cdldragonflydoji,
                        'cdlengulfing': talib_cdlengulfing,
                        'cdleveningdojistar': talib_cdleveningdojistar,
                        'cdleveningstar': talib_cdleveningstar,
                        'cdlgapsidesidewhite': talib_cdlgapsidesidewhite,
                        'cdlgravestonedoji': talib_cdlgravestonedoji,
                        'cdlhammer': talib_cdlhammer,
                        'cdlhangingman': talib_cdlhangingman,
                        'cdlharami': talib_cdlharami,
                        'cdlharamicross': talib_cdlharamicross,
                        'cdlhighwave': talib_cdlhighwave,
                        'cdlhikkake': talib_cdlhikkake,
                        'cdlhikkakemod': talib_cdlhikkakemod,
                        'cdlhomingpigeon': talib_cdlhomingpigeon,
                        'cdlidentical3crows': talib_cdlidentical3crows,
                        'cdlinneck': talib_cdlinneck,
                        'cdlinvertedhammer': talib_cdlinvertedhammer,
                        'cdlkicking': talib_cdlkicking,
                        'cdlkickingbylength': talib_cdlkickingbylength,
                        'cdlladderbottom': talib_cdlladderbottom,
                        'cdllongleggeddoji': talib_cdllongleggeddoji,
                        'cdllongline': talib_cdllongline,
                        'cdlmarubozu': talib_cdlmarubozu,
                        'cdlmatchinglow': talib_cdlmatchinglow,
                        'cdlmathold': talib_cdlmathold,
                        'cdlmorningdojistar': talib_cdlmorningdojistar,
                        'cdlmorningstar': talib_cdlmorningstar,
                        'cdlonneck': talib_cdlonneck,
                        'cdlpiercing': talib_cdlpiercing,
                        'cdlrickshawman': talib_cdlrickshawman,
                        'cdlrisefall3methods': talib_cdlrisefall3methods,
                        'cdlseparatinglines': talib_cdlseparatinglines,
                        'cdlshootingstar': talib_cdlshootingstar,
                        'cdlshortline': talib_cdlshortline,
                        'cdlspinningtop': talib_cdlspinningtop,
                        'cdlstalledpattern': talib_cdlstalledpattern,
                        'cdlsticksandwich': talib_cdlsticksandwich,
                        'cdltakuru': talib_cdltakuru,
                        'cdltasukigap': talib_cdltasukigap,
                        'cdlthrusting': talib_cdlthrusting,
                        'cdltristar': talib_cdltristar,
                        'cdlunique3river': talib_cdlunique3river,
                        'cdlupsidegap2crows': talib_cdlupsidegap2crows,
                        'cdlxsidegap3methods': talib_cdlxsidegap3methods
                        }
                        )

                        # Need a proper date type

                pattern_indicators_df['Date'] = pd.to_datetime(
                        pattern_indicators_df['Date'])

                return pattern_indicators_df

        def _merge_tickers_macro_indexes_df(self):
                if self.transformed_df is None:
                        return

                # assuming non-None transformed_df
                self.macro_df['Date']= pd.to_datetime(self.macro_df['Date'], utc=True)
                self.indexes_df['Date']= pd.to_datetime(self.indexes_df.index, utc=True)

                self.macro_df.set_index('Date', inplace=True)
                self.indexes_df.set_index('Date', inplace=True)

                self.transformed_df = pd.merge(self.transformed_df,
                        self.macro_df,
                        how='left',
                        left_on='Date',
                        right_index=True,
                        validate = "many_to_one"
                        )

                self.transformed_df = pd.merge(self.transformed_df,
                        self.indexes_df,
                        how='left',
                        left_on='Date',
                        right_index=True,
                        validate = "many_to_one"
                        )

        def persist(self):
                '''Save dataframes to files in a GCS 'dir' '''
                print (f"Save files to GCS bucket {self.config.root_dir}")
                # os.makedirs(self.config.root_dir, exist_ok=True)      
                self.create_gcs_paths(self.config.root_dir)

                file_name = 'transformed_df.parquet'
                # if os.path.exists(file_name):
                #         os.remove(file_name)
                self.transformed_df.to_parquet(self.config.root_dir+"/"+file_name, compression='brotli')

        def load(self):
                """Load files from the GCS bucket"""
                print (f"Load files from the GCS bucket {self.config.root_dir}")
                self.transformed_df = pd.read_parquet(self.config.root_dir + '/transformed_df.parquet')              
                
        def _define_feature_sets(self):
                self.GROWTH = [g for g in self.transformed_df if (g.find('growth_')==0)&(g.find('future')<0)]
                self.OHLCV = ['Open','High','Low','Close','Adj Close','Volume']
                self.CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']
                self.TO_PREDICT = [g for g in self.transformed_df.keys() if (g.find('future')>=0)]
                self.MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS',
                        'DGS1', 'DGS5', 'DGS10']
                self.CUSTOM_NUMERICAL = ['vix_adj_close','SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative','volatility', 'ln_volume']
                
                # artifacts from joins and/or unused original vars
                self.TO_DROP = ['Year','Date','Month_x', 'Month_y', 'index', 'Quarter','index_x','index_y'] + self.CATEGORICAL + self.OHLCV

                # All Supported Ta-lib indicators: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/funcs.md
                self.TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1','aroon_2', 'aroonosc',
                'bop', 'cci', 'cmo','dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
                'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
                'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
                'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
                'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
                'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
                'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
                'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']
                self.TECHNICAL_PATTERNS =  [g for g in self.transformed_df.keys() if g.find('cdl')>=0]
                
                self.NUMERICAL = self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS + \
                        self.CUSTOM_NUMERICAL + self.MACRO
                
                # CHECK: NO OTHER INDICATORS LEFT
                self.OTHER = [k for k in self.transformed_df.keys() if k not in self.OHLCV + self.CATEGORICAL + self.NUMERICAL + self.TO_DROP + self.TO_PREDICT]
                return

        def _define_dummies(self):
                # dummy variables can't be generated from Date and numeric variables ==> convert to STRING (to define groups for Dummies)
                # self.transformed_df.loc[:,'Month'] = self.transformed_df.Month_x.dt.strftime('%B')
                self.transformed_df.loc[:,'Month'] = self.transformed_df.Month_x.astype(str)
                self.transformed_df['Weekday'] = self.transformed_df['Weekday'].astype(str)  

                # Generate dummy variables (no need for bool, let's have int32 instead)
                dummy_variables = pd.get_dummies(self.transformed_df[self.CATEGORICAL], dtype='int32')
                self.df_full = pd.concat([self.transformed_df, dummy_variables], axis=1)
                # get dummies names in a list
                self.DUMMIES = dummy_variables.keys().to_list()

        def _perform_temporal_split(self, df:pd.DataFrame, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
                """
                Splits a DataFrame into three buckets based on the temporal order of the 'Date' column.

                Args:
                        df (DataFrame): The DataFrame to split.
                        min_date (str or Timestamp): Minimum date in the DataFrame.
                        max_date (str or Timestamp): Maximum date in the DataFrame.
                        train_prop (float): Proportion of data for training set (default: 0.7).
                        val_prop (float): Proportion of data for validation set (default: 0.15).
                        test_prop (float): Proportion of data for test set (default: 0.15).

                Returns:
                        DataFrame: The input DataFrame with a new column 'split' indicating the split for each row.
                """
                # Define the date intervals
                train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
                val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

                # Assign split labels based on date ranges
                split_labels = []
                for date in df['Date']:
                        if date <= train_end:
                                split_labels.append('train')
                        elif date <= val_end:
                                split_labels.append('validation')
                        else:
                                split_labels.append('test')

                # Add 'split' column to the DataFrame
                df['split'] = split_labels

                return df

        def _define_dataframes_for_ML(self):

                features_list = self.NUMERICAL+ self.DUMMIES
                # What we're trying to predict?
                to_predict = 'is_positive_growth_5d_future'

                self.train_df = self.df_full[self.df_full.split.isin(['train'])].copy(deep=True)
                self.valid_df = self.df_full[self.df_full.split.isin(['validation'])].copy(deep=True)
                self.train_valid_df = self.df_full[self.df_full.split.isin(['train','validation'])].copy(deep=True)
                self.test_df =  self.df_full[self.df_full.split.isin(['test'])].copy(deep=True)

                # Separate numerical features and target variable for training and testing sets
                self.X_train = self.train_df[features_list+[to_predict]]
                self.X_valid = self.valid_df[features_list+[to_predict]]
                self.X_train_valid = self.train_valid_df[features_list+[to_predict]]
                self.X_test = self.test_df[features_list+[to_predict]]
                # this to be used for predictions and join to the original dataframe new_df
                self.X_all =  self.df_full[features_list+[to_predict]].copy(deep=True)

                # Clean from +-inf and NaNs:

                self.X_train = self._clean_dataframe_from_inf_and_nan(self.X_train)
                self.X_valid = self._clean_dataframe_from_inf_and_nan(self.X_valid)
                self.X_train_valid = self._clean_dataframe_from_inf_and_nan(self.X_train_valid)
                self.X_test = self._clean_dataframe_from_inf_and_nan(self.X_test)
                self.X_all = self._clean_dataframe_from_inf_and_nan(self.X_all)


                self.y_train = self.X_train[to_predict]
                self.y_valid = self.X_valid[to_predict]
                self.y_train_valid = self.X_train_valid[to_predict]
                self.y_test = self.X_test[to_predict]
                self.y_all =  self.X_all[to_predict]

                # remove y_train, y_test from X_ dataframes
                del self.X_train[to_predict]
                del self.X_valid[to_predict]
                del self.X_train_valid[to_predict]
                del self.X_test[to_predict]
                del self.X_all[to_predict]

                print (f"Saving intermediary dfs to GSC {self.config.root_dir} ")
                self.X_all.to_parquet(self.config.root_dir + '/' + 'X_all.parquet', compression='brotli')
                pd.DataFrame(self.y_all).to_parquet(self.config.root_dir + '/' + 'y_all.parquet', compression='brotli')
                self.X_train_valid.to_parquet(self.config.root_dir + '/' + 'X_train_valid.parquet', compression='brotli')
                pd.DataFrame(self.y_train_valid).to_parquet(self.config.root_dir + '/' + 'y_train_valid.parquet', compression='brotli')
                self.X_test.to_parquet(self.config.root_dir + '/' + 'X_test.parquet', compression='brotli')
                pd.DataFrame(self.y_test).to_parquet(self.config.root_dir + '/' + 'y_test.parquet', compression='brotli')
                self.X_valid.to_parquet(self.config.root_dir + '/' + 'X_valid.parquet', compression='brotli')
                pd.DataFrame(self.y_valid).to_parquet(self.config.root_dir + '/' + 'y_valid.parquet', compression='brotli')
                self.df_full.to_parquet(self.config.root_dir + '/' + 'dfOrigData.parquet', compression='brotli')

                # with open(os.path.join(self.config.root_dir,"dfsForModel.pickle"),"wb") as f:
                #         pickle.dump((self.X_all, self.y_all, 
                #                      self.X_train_valid, self.y_train_valid,
                #                      self.X_train, self.y_train,
                #                      self.X_test, self.y_test,
                #                      self.X_valid, self.y_valid), f)
                        
                # with open(os.path.join(self.config.root_dir,"dfOrigData.pickle"),"wb") as f:
                #         pickle.dump(self.df_full, f)

                print(f'length: X_train {self.X_train.shape},  X_validation {self.X_valid.shape}, X_test {self.X_test.shape}')
                print(f'  X_train_valid = {self.X_train_valid.shape},  all combined: X_all {self.X_all.shape}')     

        def _clean_dataframe_from_inf_and_nan(self, df:pd.DataFrame):
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.fillna(0, inplace=True)
                return df   

        def prepare_dataframe(self):
                
                print("Prepare the dataframe: define feature sets, add dummies, temporal split")
                self.transformed_df['ln_volume'] = self.transformed_df.Volume.apply(lambda x: np.log(x) if x >0 else np.nan)
                # self.transformed_df['Date'] = pd.to_datetime(self.transformed_df['Date']).dt.strftime('%Y-%m-%d')

                self._define_feature_sets()
                # get dummies and df_full
                print ("get dummies and df_full")
                self._define_dummies()
                
                # temporal split
                print ("temporal split")
                min_date_df = self.df_full.Date.min()
                max_date_df = self.df_full.Date.max()
                self._perform_temporal_split(self.df_full, min_date=min_date_df, max_date=max_date_df)

                # define dataframes for ML
                print ("define dataframes for ML")
                self._define_dataframes_for_ML()

                return
