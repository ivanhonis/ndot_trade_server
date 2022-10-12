import datetime
import time
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager, Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
import requests
from threading import Thread



import numpy as np
import pandas as pd
import pandas_ta as ta
import asyncio


class ntade:

    def __init__(self):
        self.api_key = "DAqss9T987L0ruIbVEW9rBEFDD2sKxEKBvpvDVUJfdjijzqPqBgD8semkNF2I5Ul"
        self.api_secret = "3C1203CjVU3J0djfqG62QUSA2sFJJwWnHAmd7gd7t87OoOJJbx7NCnFV7PXx4Wpk"
        self.binance_client = Client(self.api_key, self.api_secret)

    def get_klines_mth2(self, symbol, last_minutes):
        i_now = datetime.datetime.now()
        to_dt = i_now
        from_dt = i_now - datetime.timedelta(days=int(last_minutes))
    
        i_df = pd.DataFrame(None)
        try:
            klines = self.binance_client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE,
                                                               str(from_dt), str(to_dt))
        except Exception as e:
            print(e)
        else:

            i_df = pd.DataFrame(klines, columns=('Date',
                                                 'Open', 'High', 'Low', 'Close',
                                                 'Volume',
                                                 'Close time',
                                                 'Quote_asset_volume',
                                                 'Number_of_trades',
                                                 'Taker_buy_base_asset_volume',
                                                 'Taker_buy_quote_asset_volume',
                                                 'Ignore'))

            i_df = i_df.drop(['Close time', 'Ignore'], axis=1)

            i_df['Date'] = i_df['Date'].astype(np.int64)
            i_df['Date'] = i_df['Date'] / 1000
            i_df['Date'] = i_df['Date'].astype(int)
            i_df['t'] = i_df['Date']

            i_df['Open'] = i_df['Open'].astype(float)
            i_df['High'] = i_df['High'].astype(float)
            i_df['Low'] = i_df['Low'].astype(float)
            i_df['Close'] = i_df['Close'].astype(float)
            i_df['Volume'] = i_df['Volume'].astype(float)

            i_df['Quote_asset_volume'] = i_df['Quote_asset_volume'].astype(float)
            i_df['Number_of_trades'] = i_df['Number_of_trades'].astype(float)
            i_df['Taker_buy_base_asset_volume'] = i_df['Taker_buy_base_asset_volume'].astype(float)
            i_df['Taker_buy_quote_asset_volume'] = i_df['Taker_buy_quote_asset_volume'].astype(float)
            i_df['Date'] = pd.to_datetime(i_df.Date, unit='s')
            i_df['ohlc4'] = round(((i_df['Open'] + i_df['High'] + i_df['Low'] + i_df['Close']) / 4), 8)

        return i_df

    def add_tech(self, symbol, tech_indicator):
        if tech_indicator == "PRICE_DIFF":
            self.case_set_back(symbol)
            nddf[symbol]["Low_DIFF"] = (nddf[symbol]["Low"] / nddf[symbol]["Low"].shift(1)) - 1
            nddf[symbol]["High_DIFF"] = (nddf[symbol]["High"] / nddf[symbol]["High"].shift(1)) - 1
            nddf[symbol]["Open_DIFF"] = (nddf[symbol]["Open"] / nddf[symbol]["Open"].shift(1)) - 1
            nddf[symbol]["Close_DIFF"] = (nddf[symbol]["Close"] / nddf[symbol]["Close"].shift(1)) - 1
            nddf[symbol]["ohlc4_DIFF"] = (nddf[symbol]["ohlc4"] / nddf[symbol]["ohlc4"].shift(1)) - 1
            nddf[symbol]["Volume_DIFF"] = (nddf[symbol]["Volume"] / nddf[symbol]["Volume"].shift(1)) - 1
        
            nddf[symbol]["Quote_asset_volume_DIFF"] = (nddf[symbol]["Quote_asset_volume"] / nddf[symbol][
                "Quote_asset_volume"].shift(1)) - 1
            nddf[symbol]["Number_of_trades_DIFF"] = (nddf[symbol]["Number_of_trades"] / nddf[symbol][
                "Number_of_trades"].shift(1)) - 1
            nddf[symbol]["Taker_buy_base_asset_volume_DIFF"] = (nddf[symbol]["Taker_buy_base_asset_volume"] /
                                                                nddf[symbol]["Taker_buy_base_asset_volume"].shift(
                                                                    1)) - 1
            nddf[symbol]["Taker_buy_quote_asset_volume_DIFF"] = (nddf[symbol]["Taker_buy_quote_asset_volume"] /
                                                                 nddf[symbol]["Taker_buy_quote_asset_volume"].shift(
                                                                     1)) - 1
        
            # vol diff esetén előjön hogy np.inf készül amikor 0 forgaromlór 100000 ra ugrik, ezeket kicserélem
            # a legnagyobb hisztorikus számra
            self.inf_to_max(symbol, "Volume_DIFF")
            self.inf_to_max(symbol, "Quote_asset_volume_DIFF")
            self.inf_to_max(symbol, "Number_of_trades_DIFF")
            self.inf_to_max(symbol, "Taker_buy_base_asset_volume_DIFF")
            self.inf_to_max(symbol, "Taker_buy_quote_asset_volume_DIFF")
        
            nddf[symbol]["Low_R_OHLC4"] = nddf[symbol]["Low"] / nddf[symbol]["ohlc4"]
            nddf[symbol]["High_R_OHLC4"] = nddf[symbol]["High"] / nddf[symbol]["ohlc4"]
            self.set_dt_order(symbol)
    
        elif tech_indicator == "MACD":
            nddf[symbol].ta.macd(append=True)
            self.set_dt_order(symbol)
    
        elif tech_indicator == "MACD_LONG":
            df_copy = nddf[symbol].copy()
            df_copy.ta.macd(fast=26, slow=48, signal=12, append=True)
            nddf[symbol]['MACDh_26_48_12'] = df_copy['MACDh_26_48_12']
            del df_copy
            self.set_dt_order(symbol)
    
        elif tech_indicator == "CCI21":
            nddf[symbol].ta.cci(length=21, append=True)
            self.set_dt_order(symbol)
    
        elif tech_indicator == "MA_X":
            nddf[symbol].ta.sma(close=nddf[symbol].ta.ohlc4(), length=5, append=True)
            nddf[symbol].ta.sma(close=nddf[symbol].ta.ohlc4(), length=8, append=True)
            nddf[symbol].ta.sma(close=nddf[symbol].ta.ohlc4(), length=13, append=True)
            nddf[symbol].ta.sma(close=nddf[symbol].ta.ohlc4(), length=26, append=True)
            nddf[symbol].ta.sma(close=nddf[symbol].ta.ohlc4(), length=52, append=True)
        
            nddf[symbol].ta.ema(close=nddf[symbol].ta.ohlc4(), length=5, append=True)
            nddf[symbol].ta.ema(close=nddf[symbol].ta.ohlc4(), length=8, append=True)
            nddf[symbol].ta.ema(close=nddf[symbol].ta.ohlc4(), length=13, append=True)
            nddf[symbol].ta.ema(close=nddf[symbol].ta.ohlc4(), length=26, append=True)
            nddf[symbol].ta.ema(close=nddf[symbol].ta.ohlc4(), length=52, append=True)
        
            nddf[symbol]["SMA_5_DIFF"] = (nddf[symbol]["SMA_5"] / nddf[symbol]["SMA_5"].shift(1)) - 1
            nddf[symbol]["SMA_8_DIFF"] = (nddf[symbol]["SMA_8"] / nddf[symbol]["SMA_8"].shift(1)) - 1
            nddf[symbol]["SMA_13_DIFF"] = (nddf[symbol]["SMA_13"] / nddf[symbol]["SMA_13"].shift(1)) - 1
            nddf[symbol]["SMA_26_DIFF"] = (nddf[symbol]["SMA_26"] / nddf[symbol]["SMA_26"].shift(1)) - 1
            nddf[symbol]["SMA_52_DIFF"] = (nddf[symbol]["SMA_52"] / nddf[symbol]["SMA_52"].shift(1)) - 1
        
            nddf[symbol]["EMA_5_DIFF"] = (nddf[symbol]["EMA_5"] / nddf[symbol]["EMA_5"].shift(1)) - 1
            nddf[symbol]["EMA_8_DIFF"] = (nddf[symbol]["EMA_8"] / nddf[symbol]["EMA_8"].shift(1)) - 1
            nddf[symbol]["EMA_13_DIFF"] = (nddf[symbol]["EMA_13"] / nddf[symbol]["EMA_13"].shift(1)) - 1
            nddf[symbol]["EMA_26_DIFF"] = (nddf[symbol]["EMA_26"] / nddf[symbol]["EMA_26"].shift(1)) - 1
            nddf[symbol]["EMA_52_DIFF"] = (nddf[symbol]["EMA_52"] / nddf[symbol]["EMA_52"].shift(1)) - 1
        
            self.set_dt_order(symbol)
    
        elif tech_indicator == "RSI14":
            nddf[symbol].ta.rsi(append=True)
            self.set_dt_order(symbol)
    
        elif tech_indicator == "ADX8":
            nddf[symbol].ta.adx(length=8, append=True)
            nddf[symbol]["adx_inc"] = nddf[symbol]["DMN_8"] < nddf[symbol]["DMP_8"]
            nddf[symbol]["ADX_8_ONE"] = 0
        
            def set_adx8_one(row):
                if row["adx_inc"]:
                    return row["ADX_8"]
                else:
                    return row["ADX_8"] * -1
        
            nddf[symbol] = nddf[symbol].assign(ADX_8_ONE=nddf[symbol].apply(set_adx8_one, axis=1))
            nddf[symbol] = nddf[symbol].drop(
                ['adx_inc'
                 ], axis=1, errors='ignore')
        
            self.set_dt_order(symbol)
    
        elif tech_indicator == "VWAP":
            nddf[symbol].set_index(pd.DatetimeIndex(nddf[symbol]["Date"]), inplace=True, verify_integrity=True)
            nddf[symbol].ta.vwap(append=True)
            nddf[symbol].reset_index(drop=True, inplace=True)
            nddf[symbol]["VWAP_D_R_OHLC4"] = nddf[symbol]["VWAP_D"] / nddf[symbol]["ohlc4"]
            self.set_dt_order(symbol)

    def case_set_back(self, symbol):
        # a pandas ta elállítgatja a neveket, ezért minden
        # hívás után szépen vissza állítom a neveket :)
        nddf[symbol] = nddf[symbol].rename(columns={"open": "Open",
                                                    "close": "Close",
                                                    "low": "Low",
                                                    "high": "High",
                                                    "volume": "Volume",
                                                    "date": "Date"
                                                    }, errors='ignore')

    def is_datetime_ordered(self, symbol):
        i_df = nddf[symbol].copy()
        i_df = i_df.set_index("Date")
        return i_df.index[0] < i_df.index[-1]

    def set_dt_order(self, symbol):
        """ rendezi időben az index oszlopot újra íraja kiszűri a duplikációt"""
        self.case_set_back(symbol)
        nddf[symbol] = self.i_df_dt_order(nddf[symbol])
        if not self.is_datetime_ordered(symbol):
            print("nddf->set_dt_order : Datetime order ERROR", False, True, "red")
    
    def i_df_dt_order(self, df):
        if not str(df.index.name) == "None":
            df.reset_index(drop=False, inplace=True)
        if len(df['Date']) - len(df['Date'].drop_duplicates()) > 0:
            print(f"Duplicates: {len(df['Date']) - len(df['Date'].drop_duplicates())}")
            df.drop_duplicates('Date', keep='last', inplace=True)
        df.sort_values(by=['Date'], inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def inf_to_max(self, symbol, field):
        vd = np.array(nddf[symbol][field])
        vdx = np.where(vd == np.inf)[0]
        vd[vdx] = 0
        vd_max = np.nanmax(vd)
        vd[vdx] = vd_max
        nddf[symbol][field] = vd
        
    def add_all_tech(self, symbol):
        self.add_tech(symbol, "PRICE_DIFF")
        self.add_tech(symbol, "MACD")
        self.add_tech(symbol, "MACD_LONG")
        self.add_tech(symbol, "RSI14")
        self.add_tech(symbol, "WVAP")

    def start_stream(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.asyc_websocket())
        loop.close()

    async def asyc_websocket(self):
        traded_symbols = ["BTCUSDT"]
        i_socket_list = []
        for sp in traded_symbols:
            i_socket_list.append(sp.lower() + '@kline_1m')
    
        client = await AsyncClient.create()
        bm = BinanceSocketManager(client)
        ts = bm.multiplex_socket(i_socket_list)
    
        async with ts as tscm:
            while True:
                res = await tscm.recv()
                print(res['data']['k']['x'])
        # ez sosem fog lefutni mert a szervernek nincs leállítási funkciója csak kilövöm éskész
        await client.close_connection()


if __name__ == "__main__":
    nddf = {}
    nt = ntade()
    symbol = "BTCUSDT"
    nddf[symbol] = nt.get_klines_mth2(symbol, 1)
    print(datetime.datetime.now())
    nt.add_all_tech(symbol)
    print(datetime.datetime.now())
    nt.start_stream()
    while True:
        time.sleep(30)
        pass

