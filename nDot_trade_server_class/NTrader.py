import datetime
import sys
import os
import time
from binance import AsyncClient, BinanceSocketManager, Client
from binance.exceptions import BinanceAPIException
from binance.enums import *


from pathlib import Path  # dataset config beolvasóhoz kell
from json import loads as json_loads
from re import sub as re_sub

import threading

import numpy as np
import pandas as pd
import pandas_ta
import asyncio
from nDot_trade_server_class.NFolderManager import NFolderManager


class NTrader:

	def __init__(self, params):
		self.all_used_cores = params['all_used_cores']
		self.core = params['core']
		self.shared_memory_name = params['shared_memory_name']
		self.slots_fname = params['slots_fname']
		self.nddf = {}
		self.api_key = "DAqss9T987L0ruIbVEW9rBEFDD2sKxEKBvpvDVUJfdjijzqPqBgD8semkNF2I5Ul"
		self.api_secret = "3C1203CjVU3J0djfqG62QUSA2sFJJwWnHAmd7gd7t87OoOJJbx7NCnFV7PXx4Wpk"
		self.binance_client = Client(self.api_key, self.api_secret)
		self.async_client = ""  # AsyncClient.create()
		self.bm = ""  # BinanceSocketManager(client)
		self.ts = ""  # bm.multiplex_socket(i_socket_list)

		self.ai = {}
		self.tfm_mp = NFolderManager()
		self.main()

	def main(self):
		print("main start")
		self.order(symbol="BTCUSDT", side="BUY", amount=25)

		# self.update_ai()
		# symbols = self.get_symbols()
		# for symbol in symbols:
		# 	self.nddf[symbol] = self.get_klines(symbol)
		# 	self.set_dt_order(symbol)
		# 	self.nddf[symbol] = self.nddf[symbol].tail(100)
		# 	self.set_dt_order(symbol)
		# 	print(self.nddf[symbol])
		#
		# data_stream = threading.Thread(target=self.start_stream, args=[])
		# data_stream.start()
		# while True:
		# 	# self.update_sockets()
		# 	time.sleep(15)
		# 	# self.update_ai()

	def order(self, symbol, side, amount):
		try:
			# order = self.binance_client.order_market(symbol=symbol,
			# 										 side=side,
			# 										 quantity=None,
			# 										 quoteOrderQty=amount,
			# 										 stopPrice=str(19100))

			# order = self.binance_client.create_order(symbol=symbol,
			# 										 side=SIDE_BUY,
			# 										 type=ORDER_TYPE_STOP_LOSS_LIMIT,
			# 										 timeInForce=TIME_IN_FORCE_GTC,
			# 										 quantity=str(0.0013),
			# 										 price=str(19105.00000000),
			# 										 stopPrice=str(19090.00000000))

			exchange_info = self.binance_client.get_exchange_info()
			# print(exchange_info['symbols'][0])
			selected_symbol = {}
			for sy in exchange_info['symbols']:
				if sy['symbol'] == "BTCUSDT":
					selected_symbol = sy
					break
			# print(selected_symbol)

			tick_size = float(selected_symbol['filters'][0]['tickSize'])

			ticker = self.binance_client.get_order_book(symbol="BTCUSDT", limit=1)
			# print(ticker)
			best_ask = float(ticker['asks'][0][0])
			enter_price = best_ask + (tick_size * 100)
			stop_price = enter_price - (tick_size * 500)
			quantity = 25 / enter_price
			quantity = round(quantity, 4)

			print(best_ask)
			print(enter_price)
			print(stop_price)
			print(quantity)

			order = self.binance_client.create_order(symbol=symbol,
													 side=SIDE_BUY,
													 type=ORDER_TYPE_TAKE_PROFIT_LIMIT,
													 timeInForce=TIME_IN_FORCE_GTC,
													 quantity=str(quantity),
													 price=str(enter_price),
													 stopPrice=str(stop_price),
													 trailingDelta=str(100)
													 )

			print(order)
		except BinanceAPIException as e:
			print(e.status_code)
			print(e.message)

	@staticmethod
	def log(text):
		print(f"{datetime.datetime.now()} -> {text}")

	def update_sockets(self):
		pass

	def get_dataset_config(self, config_file_path):

		file = Path(config_file_path)
		if file.exists():
			ok = True

			# ez nem vizsgálja, hogy létezik e file, ezt hívás előtt kell

			def clear_string(i_contents, space=True):
				if space:
					i_contents = i_contents.replace(" ", "")
				i_contents = i_contents.replace('\n', '').replace('\r', '')
				i_contents = re_sub('<.*?>', '', i_contents)
				if i_contents[-2:] == ",]":
					i_contents = i_contents[:-2] + "]"
				return i_contents

			with open(config_file_path) as f:
				contents = f.read()
			contents = contents.split(";")
			description = clear_string(contents[0], space=False)

			try:
				dataset_config = json_loads(clear_string(contents[1]))
			except ValueError:
				dataset_config = {}
				ok = False

			try:
				original_fields = json_loads(clear_string(contents[2]))
			except ValueError:
				original_fields = []
				ok = False

			try:
				contras = json_loads(clear_string(contents[3]))
			except ValueError:
				contras = []
				ok = False

			try:
				indexes = json_loads(clear_string(contents[4]))
			except ValueError:
				indexes = []
				ok = False

			try:
				trade = json_loads(clear_string(contents[5]))
			except ValueError:
				trade = {}
				ok = False

		else:
			self.log("Project config file is not found.")
			ok = False
			description = ""
			dataset_config = {}
			original_fields = []
			contras = []
			indexes = []
			trade = {}

		return ok, description, dataset_config, original_fields, contras, indexes, trade

	def update_ai(self):
		slots = np.load(self.slots_fname)

		def get_project_update_str(i_project):
			i_tf_name = self.tfm_mp.slots_dir + f"/{i_project}/nDot_TF_MODEL_{i_project}.h5"
			i_minmax_name = self.tfm_mp.slots_dir + f"/{i_project}/nDot_MinMaxScaler_{i_project}.pickle"
			i_config_name = self.tfm_mp.slots_dir + f"/{i_project}/nDot_PRO_{i_project}.txt"
			if os.path.exists(i_tf_name) and os.path.exists(i_minmax_name) and os.path.exists(i_config_name):
				dtf = os.path.getmtime(i_tf_name)
				dminmax = os.path.getmtime(i_minmax_name)
				dconfig = os.path.getmtime(i_config_name)
				i_udstr = str(dtf) + str(dminmax) + str(dconfig)
			else:
				i_udstr = ""
			return i_udstr

		def get_project_fnames(i_project):
			i_tf_name = self.tfm_mp.slots_dir + f"/{i_project}/nDot_TF_MODEL_{i_project}.h5"
			i_minmax_name = self.tfm_mp.slots_dir + f"/{i_project}/nDot_MinMaxScaler_{i_project}.pickle"
			i_config_name = self.tfm_mp.slots_dir + f"/{i_project}/nDot_PRO_{i_project}.txt"
			return i_tf_name, i_minmax_name, i_config_name

		for project in slots:
			tf_name, minmax_name, config_name = get_project_fnames(project)
			if os.path.exists(tf_name) and os.path.exists(minmax_name) and os.path.exists(config_name):
				udstr = get_project_update_str(project)
				if project not in self.ai or self.ai[project]["update_str"] != udstr:
					ok, description, dataset_config, original_fields, contras, indexes, trade = self.get_dataset_config(config_name)
					self.ai[project] = {"original_fields": original_fields,
										"contras": contras,
										"indexes": indexes,
										"trade": trade,
										"tf": "tf",
										"minmax": "minmax",
										"update_str": udstr,
										"last_update": datetime.datetime.now()}
			else:
				try:
					del self.ai[project]
				except:
					pass

		for project in self.ai:
			if project not in slots:
				del self.ai[project]

	def get_klines(self, symbol):

		i_df = pd.DataFrame(None)
		try:
			klines = self.binance_client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE,
															   "1 day ago UTC")
		except Exception as e:
			print(e)
		else:

			i_df = pd.DataFrame(klines, columns=['Date',
												 'Open', 'High', 'Low', 'Close',
												 'Volume',
												 'Close time',
												 'Quote_asset_volume',
												 'Number_of_trades',
												 'Taker_buy_base_asset_volume',
												 'Taker_buy_quote_asset_volume',
												 'Ignore'])

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

	@staticmethod
	def get_techs():
		i_ret = ['PRICE_DIFF',
				 'MACD',
				 'MACD_LONG',
				 'CCI21',
				 'MA_X',
				 'RSI14',
				 'ADX8',
				 'VWAP']
		return i_ret

	def add_tech(self, symbol, tech_indicator):
		if tech_indicator == "PRICE_DIFF":
			self.case_set_back(symbol)
			self.nddf[symbol]["Low_DIFF"] = (self.nddf[symbol]["Low"] / self.nddf[symbol]["Low"].shift(1)) - 1
			self.nddf[symbol]["High_DIFF"] = (self.nddf[symbol]["High"] / self.nddf[symbol]["High"].shift(1)) - 1
			self.nddf[symbol]["Open_DIFF"] = (self.nddf[symbol]["Open"] / self.nddf[symbol]["Open"].shift(1)) - 1
			self.nddf[symbol]["Close_DIFF"] = (self.nddf[symbol]["Close"] / self.nddf[symbol]["Close"].shift(1)) - 1
			self.nddf[symbol]["ohlc4_DIFF"] = (self.nddf[symbol]["ohlc4"] / self.nddf[symbol]["ohlc4"].shift(1)) - 1
			self.nddf[symbol]["Volume_DIFF"] = (self.nddf[symbol]["Volume"] / self.nddf[symbol]["Volume"].shift(1)) - 1

			self.nddf[symbol]["Quote_asset_volume_DIFF"] = (self.nddf[symbol]["Quote_asset_volume"] / self.nddf[symbol][
				"Quote_asset_volume"].shift(1)) - 1
			self.nddf[symbol]["Number_of_trades_DIFF"] = (self.nddf[symbol]["Number_of_trades"] / self.nddf[symbol][
				"Number_of_trades"].shift(1)) - 1
			self.nddf[symbol]["Taker_buy_base_asset_volume_DIFF"] = (self.nddf[symbol]["Taker_buy_base_asset_volume"] /
																self.nddf[symbol]["Taker_buy_base_asset_volume"].shift(
																	1)) - 1
			self.nddf[symbol]["Taker_buy_quote_asset_volume_DIFF"] = (self.nddf[symbol]["Taker_buy_quote_asset_volume"] /
																 self.nddf[symbol]["Taker_buy_quote_asset_volume"].shift(
																	 1)) - 1

			# vol diff esetén előjön hogy np.inf készül amikor 0 forgaromlór 100000 ra ugrik, ezeket kicserélem
			# a legnagyobb hisztorikus számra
			self.inf_to_max(symbol, "Volume_DIFF")
			self.inf_to_max(symbol, "Quote_asset_volume_DIFF")
			self.inf_to_max(symbol, "Number_of_trades_DIFF")
			self.inf_to_max(symbol, "Taker_buy_base_asset_volume_DIFF")
			self.inf_to_max(symbol, "Taker_buy_quote_asset_volume_DIFF")

			self.nddf[symbol]["Low_R_OHLC4"] = self.nddf[symbol]["Low"] / self.nddf[symbol]["ohlc4"]
			self.nddf[symbol]["High_R_OHLC4"] = self.nddf[symbol]["High"] / self.nddf[symbol]["ohlc4"]
			self.set_dt_order(symbol)

		elif tech_indicator == "MACD":
			self.nddf[symbol].ta.macd(append=True)
			self.set_dt_order(symbol)

		elif tech_indicator == "MACD_LONG":
			df_copy = self.nddf[symbol].copy()
			df_copy.ta.macd(fast=26, slow=48, signal=12, append=True)
			self.nddf[symbol]['MACDh_26_48_12'] = df_copy['MACDh_26_48_12']
			del df_copy
			self.set_dt_order(symbol)

		elif tech_indicator == "CCI21":
			self.nddf[symbol].ta.cci(length=21, append=True)
			self.set_dt_order(symbol)

		elif tech_indicator == "MA_X":
			self.nddf[symbol].ta.sma(close=self.nddf[symbol].ta.ohlc4(), length=5, append=True)
			self.nddf[symbol].ta.sma(close=self.nddf[symbol].ta.ohlc4(), length=8, append=True)
			self.nddf[symbol].ta.sma(close=self.nddf[symbol].ta.ohlc4(), length=13, append=True)
			self.nddf[symbol].ta.sma(close=self.nddf[symbol].ta.ohlc4(), length=26, append=True)
			self.nddf[symbol].ta.sma(close=self.nddf[symbol].ta.ohlc4(), length=52, append=True)

			self.nddf[symbol].ta.ema(close=self.nddf[symbol].ta.ohlc4(), length=5, append=True)
			self.nddf[symbol].ta.ema(close=self.nddf[symbol].ta.ohlc4(), length=8, append=True)
			self.nddf[symbol].ta.ema(close=self.nddf[symbol].ta.ohlc4(), length=13, append=True)
			self.nddf[symbol].ta.ema(close=self.nddf[symbol].ta.ohlc4(), length=26, append=True)
			self.nddf[symbol].ta.ema(close=self.nddf[symbol].ta.ohlc4(), length=52, append=True)

			self.nddf[symbol]["SMA_5_DIFF"] = (self.nddf[symbol]["SMA_5"] / self.nddf[symbol]["SMA_5"].shift(1)) - 1
			self.nddf[symbol]["SMA_8_DIFF"] = (self.nddf[symbol]["SMA_8"] / self.nddf[symbol]["SMA_8"].shift(1)) - 1
			self.nddf[symbol]["SMA_13_DIFF"] = (self.nddf[symbol]["SMA_13"] / self.nddf[symbol]["SMA_13"].shift(1)) - 1
			self.nddf[symbol]["SMA_26_DIFF"] = (self.nddf[symbol]["SMA_26"] / self.nddf[symbol]["SMA_26"].shift(1)) - 1
			self.nddf[symbol]["SMA_52_DIFF"] = (self.nddf[symbol]["SMA_52"] / self.nddf[symbol]["SMA_52"].shift(1)) - 1

			self.nddf[symbol]["EMA_5_DIFF"] = (self.nddf[symbol]["EMA_5"] / self.nddf[symbol]["EMA_5"].shift(1)) - 1
			self.nddf[symbol]["EMA_8_DIFF"] = (self.nddf[symbol]["EMA_8"] / self.nddf[symbol]["EMA_8"].shift(1)) - 1
			self.nddf[symbol]["EMA_13_DIFF"] = (self.nddf[symbol]["EMA_13"] / self.nddf[symbol]["EMA_13"].shift(1)) - 1
			self.nddf[symbol]["EMA_26_DIFF"] = (self.nddf[symbol]["EMA_26"] / self.nddf[symbol]["EMA_26"].shift(1)) - 1
			self.nddf[symbol]["EMA_52_DIFF"] = (self.nddf[symbol]["EMA_52"] / self.nddf[symbol]["EMA_52"].shift(1)) - 1

			self.set_dt_order(symbol)

		elif tech_indicator == "RSI14":
			self.nddf[symbol].ta.rsi(append=True)
			self.set_dt_order(symbol)

		elif tech_indicator == "ADX8":
			self.nddf[symbol].ta.adx(length=8, append=True)
			self.nddf[symbol]["adx_inc"] = self.nddf[symbol]["DMN_8"] < self.nddf[symbol]["DMP_8"]
			self.nddf[symbol]["ADX_8_ONE"] = 0

			def set_adx8_one(row):
				if row["adx_inc"]:
					return row["ADX_8"]
				else:
					return row["ADX_8"] * -1

			self.nddf[symbol] = self.nddf[symbol].assign(ADX_8_ONE=self.nddf[symbol].apply(set_adx8_one, axis=1))
			self.nddf[symbol] = self.nddf[symbol].drop(
				['adx_inc'
				 ], axis=1, errors='ignore')

			self.set_dt_order(symbol)

		elif tech_indicator == "VWAP":
			self.nddf[symbol].set_index(pd.DatetimeIndex(self.nddf[symbol]["Date"]), inplace=True, verify_integrity=True)
			self.nddf[symbol].ta.vwap(append=True)
			self.nddf[symbol].reset_index(drop=True, inplace=True)
			self.nddf[symbol]["VWAP_D_R_OHLC4"] = self.nddf[symbol]["VWAP_D"] / self.nddf[symbol]["ohlc4"]
			self.set_dt_order(symbol)

	def case_set_back(self, symbol):
		# a pandas ta elállítgatja a neveket, ezért minden
		# hívás után szépen vissza állítom a neveket :)
		self.nddf[symbol] = self.nddf[symbol].rename(columns={"open": "Open",
													"close": "Close",
													"low": "Low",
													"high": "High",
													"volume": "Volume",
													"date": "Date"},
													 errors='ignore')

	def is_datetime_ordered(self, symbol):
		i_df = self.nddf[symbol].copy()
		i_df = i_df.set_index("Date")
		return i_df.index[0] < i_df.index[-1]

	def set_dt_order(self, symbol):
		""" rendezi időben az index oszlopot újra íraja kiszűri a duplikációt"""
		self.case_set_back(symbol)
		self.nddf[symbol] = self.i_df_dt_order(self.nddf[symbol])
		if not self.is_datetime_ordered(symbol):
			print("nddf->set_dt_order : Datetime order ERROR")

	@staticmethod
	def i_df_dt_order(df):
		if not str(df.index.name) == "None":
			df.reset_index(drop=False, inplace=True)
		if len(df['Date']) - len(df['Date'].drop_duplicates()) > 0:
			print(f"Duplicates: {len(df['Date']) - len(df['Date'].drop_duplicates())}")
			df.drop_duplicates('Date', keep='last', inplace=True)
		df.sort_values(by=['Date'], inplace=True, ascending=True)
		df.reset_index(drop=True, inplace=True)
		return df

	def inf_to_max(self, symbol, field):
		vd = np.array(self.nddf[symbol][field])
		vdx = np.where(vd == np.inf)[0]
		vd[vdx] = 0
		vd_max = np.nanmax(vd)
		vd[vdx] = vd_max
		self.nddf[symbol][field] = vd

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

	@staticmethod
	def get_socket_name_kline_1m(symbol):
		return symbol.lower() + '@kline_1m'

	def get_symbols(self):
		traded_symbols = []
		for project in self.ai:
			traded_symbols.append(self.ai[project]['trade']['symbol'])
		return traded_symbols

	async def asyc_websocket(self):
		traded_symbols = self.get_symbols()

		i_socket_list = []
		for symbol in traded_symbols:
			i_socket_list.append(self.get_socket_name_kline_1m(symbol))

		self.async_client = await AsyncClient.create()
		self.bm = BinanceSocketManager(self.async_client)
		self.ts = self.bm.multiplex_socket(i_socket_list)

		async with self.ts as tscm:
			while True:
				res = await tscm.recv()
				# print(res)
				if res['data']['k']['x']:
					start_dt = datetime.datetime.now()
					await self.add_data(res)
					print(datetime.datetime.now() - start_dt)

		# ebből a ciklusból soha nem jön ki

	async def add_data(self, res):
		symbol = res["data"]["s"]
		resk = res["data"]["k"]
		i_df = {"Date": resk["t"],
					"Open": resk["o"],
					"Close": resk["c"],
					"High": resk["h"],
					"Low": resk["l"],
					"Volume": resk["v"],
					"Number_of_trades": resk["n"],
					"Quote_asset_volume": resk["q"],
					"Taker_buy_base_asset_volume": resk["V"],
					"Taker_buy_quote_asset_volume": resk["Q"]}

		i_df['Date'] = int(i_df['Date'])
		i_df['Date'] = int(i_df['Date'] / 1000)
		i_df['t'] = i_df['Date']

		i_df['Open'] = float(i_df['Open'])
		i_df['High'] = float(i_df['High'])
		i_df['Low'] = float(i_df['Low'])
		i_df['Close'] = float(i_df['Close'])
		i_df['Volume'] = float(i_df['Volume'])

		i_df['Quote_asset_volume'] = float(i_df['Quote_asset_volume'])
		i_df['Number_of_trades'] = float(i_df['Number_of_trades'])
		i_df['Taker_buy_base_asset_volume'] = float(i_df['Taker_buy_base_asset_volume'])
		i_df['Taker_buy_quote_asset_volume'] = float(i_df['Taker_buy_quote_asset_volume'])
		i_df['Date'] = pd.to_datetime(i_df['Date'], unit='s')
		i_df['ohlc4'] = round(((i_df['Open'] + i_df['High'] + i_df['Low'] + i_df['Close']) / 4), 8)
		self.nddf[symbol] = pd.concat([self.nddf[symbol], pd.DataFrame(i_df, index=[0])], ignore_index=True)
		self.set_dt_order(symbol)
		self.add_all_tech(symbol)

		print(self.nddf[symbol].tail(2).T)

		# {
		# 	"e": "kline",  # Event type
		# 	"E": 123456789,  # Event time
		# 	"s": "BNBBTC",  # Symbol
		# 	"k": {
		# 		"t": 123400000,  # Kline start time
		# 		"T": 123460000,  # Kline close time
		# 		"s": "BNBBTC",  # Symbol
		# 		"i": "1m",  # Interval
		# 		"f": 100,  # First trade ID
		# 		"L": 200,  # Last trade ID
		# 		"o": "0.0010",  # Open price
		# 		"c": "0.0020",  # Close price
		# 		"h": "0.0025",  # High price
		# 		"l": "0.0015",  # Low price
		# 		"v": "1000",  # Base asset volume
		# 		"n": 100,  # Number of trades
		# 		"x": false,  # Is this kline closed?
		# 		"q": "1.0000",  # Quote asset volume
		# 		"V": "500",  # Taker buy base asset volume
		# 		"Q": "0.500",  # Taker buy quote asset volume
		# 		"B": "123456"  # Ignore
		# 	}
		# }



