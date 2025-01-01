import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import argparse
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

class SMCTradingBot:
    def __init__(self, api_key, secret, timeframe='1h'):
        self.api_key = api_key
        self.secret = secret
        self.timeframe = timeframe
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True,
        })
        print("Initializing and loading market data...")
        self.exchange.load_markets()  # Load markets during initialization
        self.precision = {}

    def _fetch_symbol_precision(self, symbol):
        """Fetch symbol precision for price and volume."""
        if not self.exchange.markets:
            print("Loading market data...")
            self.exchange.load_markets()
        market = self.exchange.market(symbol)
        print(market)
        self.precision['price'] = int(market['precision']['price'])  # Ensure it's an integer
        self.precision['volume'] = int(market['precision']['amount'])  # Ensure it's an integer

    def fetch_symbol_precision(self, symbol):
        """Fetch symbol precision for price and volume."""
        if not self.exchange.markets:
            print("Loading market data...")
            self.exchange.load_markets()
        market = self.exchange.market(symbol)
        try:
            # Safely convert precision values
            self.precision['price'] = int(-np.log10(float(market['precision']['price'])))
            self.precision['volume'] = int(-np.log10(float(market['precision']['amount'])))
        except ValueError:
            # Default precision handling if conversion fails
            self.precision['price'] = 6  # Default to 6 decimals
            self.precision['volume'] = 2  # Default to 2 decimals

    def fetch_historical_data(self, symbol, limit=200):
        """Fetch historical OHLCV data."""
        since = self.exchange.milliseconds() - limit * self.exchange.parse_timeframe(self.timeframe) * 1000
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['datetime'] = mdates.date2num(df['timestamp'])  # Convert to matplotlib date format
        return df

    def identify_market_structure(self, df):
        """Identify market structure with BoS and ChoCh."""
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))

        df['break_of_structure'] = (df['close'] > df['high'].shift(1))
        df['change_of_character'] = (df['close'] < df['low'].shift(1))
        return df

    def detect_liquidity_zones(self, df):
        """Detect support and resistance zones."""
        df['internal_liquidity_zone'] = df['low'].rolling(window=3).min()
        df['external_liquidity_zone'] = df['high'].rolling(window=3).max()
        return df

    def detect_order_blocks(self, df):
        """Detect bullish and bearish order blocks."""
        df['bullish_order_block'] = df['low'].where((df['break_of_structure']) & (df['close'] > df['open']))
        df['bearish_order_block'] = df['high'].where((df['break_of_structure']) & (df['close'] < df['open']))
        return df

    def detect_fvg(self, df):
        """Detect Fair Value Gaps (FVG)."""
        df['fvg_upper'] = df['high'].shift(1)
        df['fvg_lower'] = df['low'].shift(-1)
        df['fvg'] = (df['fvg_upper'] - df['fvg_lower']) > 0
        return df

    def calculate_entry_exit(self, df, risk_reward_ratio=3):
        """Calculate potential entry, stop-loss, and take-profit levels."""
        df['entry'] = df['close']
        df['stop_loss'] = df['low'] - (df['high'] - df['low']) * 0.5  # Example calculation
        df['take_profit'] = df['entry'] + (df['entry'] - df['stop_loss']) * risk_reward_ratio

        # Filter opportunities with approximately 3:1 R:R
        df['valid_trade'] = (df['take_profit'] - df['entry']) / (df['entry'] - df['stop_loss']) >= 3
        return df

    def predict_future_price(self, df):
        """Predict future price movement using linear regression."""
        df['time_index'] = np.arange(len(df))
        X = df[['time_index']].values
        y = df['close'].values

        model = LinearRegression()
        model.fit(X, y)

        future_index = np.array([[len(df) + 1]])
        predicted_price = model.predict(future_index)[0]
        price_precision = max(0, self.precision.get('price', 6))  # Default to 6 decimals if undefined
        return round(predicted_price, price_precision)


    def _calculate_deal_parameters(self, df, predicted_price):
        """Calculate LONG and SHORT deal parameters based on predicted price."""
        current_price = df['close'].iloc[-1]
        atr = (df['high'] - df['low']).rolling(window=14).mean().iloc[-1]  # Average True Range

        # LONG Deal Parameters
        long_entry = round(current_price, self.precision['price'])
        long_stop_loss = round(current_price - atr, self.precision['price'])
        long_take_profit = round(predicted_price if predicted_price > current_price else current_price + atr * 3,
                                 self.precision['price'])

        # SHORT Deal Parameters
        short_entry = round(current_price, self.precision['price'])
        short_stop_loss = round(current_price + atr, self.precision['price'])
        short_take_profit = round(predicted_price if predicted_price < current_price else current_price - atr * 3,
                                  self.precision['price'])

        return {
            'long': {
                'entry': long_entry,
                'stop_loss': long_stop_loss,
                'take_profit': long_take_profit
            },
            'short': {
                'entry': short_entry,
                'stop_loss': short_stop_loss,
                'take_profit': short_take_profit
            }
        }

    def visualize_data(self, df):
        """Visualize market structure, liquidity zones, order blocks, and entry/exit points."""
        plt.figure(figsize=(16, 8))

        # Candlestick chart setup
        ohlc = df[['datetime', 'open', 'high', 'low', 'close']].copy()
        candlestick_ohlc(plt.gca(), ohlc.values, width=0.02, colorup='green', colordown='red')
        plt.gca().xaxis_date()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        # Plot liquidity zones
        plt.fill_between(df['datetime'], df['internal_liquidity_zone'], df['internal_liquidity_zone'],
                         color='blue', alpha=0.2, label='Internal Liquidity Zone')
        plt.fill_between(df['datetime'], df['external_liquidity_zone'], df['external_liquidity_zone'],
                         color='orange', alpha=0.2, label='External Liquidity Zone')

        # Highlight valid LONG and SHORT zones
        valid_trades = df[df['valid_trade']]
        long_trades = valid_trades[valid_trades['entry'] > valid_trades['stop_loss']]
        short_trades = valid_trades[valid_trades['entry'] < valid_trades['stop_loss']]

        if not long_trades.empty:
            last_long = long_trades.iloc[-1]
            plt.axhspan(last_long['stop_loss'], last_long['take_profit'],
                        color='green', alpha=0.3, label='LONG Zone')

        if not short_trades.empty:
            last_short = short_trades.iloc[-1]
            plt.axhspan(last_short['take_profit'], last_short['stop_loss'],
                        color='red', alpha=0.3, label='SHORT Zone')

        # Highlight bullish and bearish order blocks
        bullish_blocks = df.dropna(subset=['bullish_order_block'])
        bearish_blocks = df.dropna(subset=['bearish_order_block'])

        if not bullish_blocks.empty:
            for _, row in bullish_blocks.iterrows():
                plt.axhspan(row['bullish_order_block'], row['bullish_order_block'] + 0.005, 
                            color='lightgreen', alpha=0.4, label='Bullish Order Block')
                break

        if not bearish_blocks.empty:
            for _, row in bearish_blocks.iterrows():
                plt.axhspan(row['bearish_order_block'] - 0.005, row['bearish_order_block'], 
                            color='lightcoral', alpha=0.4, label='Bearish Order Block')
                break

        plt.legend()
        plt.title('Market Structure, Liquidity Zones, Order Blocks, and Entry Points')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()

    def calculate_deal_parameters(self, df, predicted_price):
        """Calculate LONG and SHORT deal parameters based on predicted price."""
        current_price = df['close'].iloc[-1]
        atr = (df['high'] - df['low']).rolling(window=14).mean().iloc[-1]  # Average True Range

        # Avoid division by zero or NaN
        if pd.isna(current_price) or pd.isna(atr):
            raise ValueError("Invalid current price or ATR for deal calculations.")

        # LONG Deal Parameters
        long_entry = round(current_price, self.precision['price'])
        long_stop_loss = round(current_price - atr, self.precision['price'])
        long_take_profit = round(predicted_price if predicted_price > current_price else current_price + atr * 3,
                                 self.precision['price'])

        # SHORT Deal Parameters
        short_entry = round(current_price, self.precision['price'])
        short_stop_loss = round(current_price + atr, self.precision['price'])
        short_take_profit = round(predicted_price if predicted_price < current_price else current_price - atr * 3,
                                  self.precision['price'])

        return {
            'long': {
                'entry': long_entry,
                'stop_loss': long_stop_loss,
                'take_profit': long_take_profit
            },
            'short': {
                'entry': short_entry,
                'stop_loss': short_stop_loss,
                'take_profit': short_take_profit
            }
        }

    def run(self, symbol):
        """Run the bot to analyze and trade."""
        print(f"Fetching data for {symbol} on {self.timeframe} timeframe...")
        self.fetch_symbol_precision(symbol)  # Fetch symbol precision
        print(f"Precision for {symbol}: {self.precision}")

        df = self.fetch_historical_data(symbol)
        if df.empty:
            print("No historical data fetched. Exiting...")
            return

        print("Identifying market structure...")
        df = self.identify_market_structure(df)
        print("Detecting liquidity zones...")
        df = self.detect_liquidity_zones(df)
        print("Detecting order blocks...")
        df = self.detect_order_blocks(df)
        print("Detecting Fair Value Gaps (FVG)...")
        df = self.detect_fvg(df)
        print("Calculating entry and exit points...")
        df = self.calculate_entry_exit(df)
        print(f"Symbol precision {self.precision['price']}")
        print("Predicting future price...")
        predicted_price = self.predict_future_price(df)
        print(f"Predicted price: {predicted_price:.{self.precision['price']}f}")

        print("Calculating deal parameters...")
        deal_parameters = self.calculate_deal_parameters(df, predicted_price)
        print("Deal Parameters:")
        print(f"LONG: Entry = {deal_parameters['long']['entry']:.{self.precision['price']}f}, Stop Loss = {deal_parameters['long']['stop_loss']:.{self.precision['price']}f}, Take Profit = {deal_parameters['long']['take_profit']:.{self.precision['price']}f}")
        print(f"SHORT: Entry = {deal_parameters['short']['entry']:.{self.precision['price']}f}, Stop Loss = {deal_parameters['short']['stop_loss']:.{self.precision['price']}f}, Take Profit = {deal_parameters['short']['take_profit']:.{self.precision['price']}f}")

        print("Visualizing data...")
        self.visualize_data(df)



# Usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMC Trading Bot')
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTC/USDT)')
    parser.add_argument('--timeframe', type=str, choices=['15m', '1h', '4h', '1d'], default='1h', help='Timeframe for analysis')
    args = parser.parse_args()


    bot = SMCTradingBot(api_key, secret, timeframe=args.timeframe)
    bot.run(args.symbol)
