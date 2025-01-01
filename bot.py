import json
import os
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import argparse
from datetime import datetime, timedelta
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import mplcursors

def overlap(start1, end1, start2, end2):
    """how much does the range (start1, end1) overlap with (start2, end2)"""
    return max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)

class SMCTradingBot:
    def __init__(self, secrets_file, timeframe='1h', features=None, martingale=0.01, stake=300, symbol='BTCUSDT'):
        self.secrets = self.load_secrets(secrets_file)
        self.api_key = self.secrets.get("api_key")
        self.secret = self.secrets.get("secret")
        self.timeframe = timeframe
        self.martingale = martingale
        self.stake = stake
        self.symbol = symbol
        self.exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True,
        })
        self.precision = {}
        self.features = features if features else []

    @staticmethod
    def load_secrets(secrets_file):
        """Load secrets from a JSON file."""
        try:
            with open(secrets_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Secrets file '{secrets_file}' not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in secrets file: {e}")

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

    def fetch_current_price(self, symbol):
        """Fetch the current market price for the symbol."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None

    def mark_fractals(self, df):
        """Mark fractal swing highs and lows."""
        n = 3
        # df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        # df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))

        df['swing_high'] = (df['high'] == df['high'].rolling(n+n+1, center=True).max())
        df['swing_low'] = (df['low'] == df['low'].rolling(n+n+1, center=True).min())
        return df

    def detect_structure(self, df):
        """Detect LL, LH, HL, HH only where price direction changes."""
        structure = []
        last_high = None
        last_low = None
        last_direction = None

        for i in range(1, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]

            if last_high is None or last_low is None:
                last_high = current_high
                last_low = current_low
                continue

            if current_high > last_high:
                if last_direction != 'up':
                    structure.append(('HH', i))
                    last_direction = 'up'
                last_high = current_high

            elif current_low < last_low:
                if last_direction != 'down':
                    structure.append(('LL', i))
                    last_direction = 'down'
                last_low = current_low

            elif current_high < last_high:
                if last_direction == 'up':
                    structure.append(('LH', i))
                    last_direction = 'down'
                last_high = current_high

            elif current_low > last_low:
                if last_direction == 'down':
                    structure.append(('HL', i))
                    last_direction = 'up'
                last_low = current_low

        df['structure'] = None
        for label, index in structure:
            df.at[index, 'structure'] = label

        # Detect BOS and POB
        df['BOS'] = False
        df['POB'] = False
        for i in range(len(df)):
            if df['structure'].iloc[i] == 'HH':
                # Look for the next candle where close > current high
                future_candles = df.iloc[i + 1:]
                for j, future_row in future_candles.iterrows():
                    if future_row['close'] > df['high'].iloc[i]:
                        df.at[j, 'BOS'] = True
                        break

            if df['structure'].iloc[i] == 'LL':
                # Look for the next candle where close < current low
                future_candles = df.iloc[i + 1:]
                for j, future_row in future_candles.iterrows():
                    if future_row['close'] < df['low'].iloc[i]:
                        df.at[j, 'POB'] = True
                        break

        return df

    def detect_bos_refined(self, df):
        """Detect valid BoS (Break of Structure) lines with price reversal."""
        bos_lines = []  # Store valid BoS lines
        last_LL = None
        last_HH = None

        for i in range(1, len(df)):
            current_close = df['close'].iloc[i]
            current_close = df['open'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]

            # Bullish BoS: Ensure price reversal before breakout
            if last_LL and current_close > last_LL['high']:
                # Check if there was a price reversal
                reversal = False
                for j in range(last_LL['index'] + 1, i):  # Check candles in between
                    if df['close'].iloc[j] < df['open'].iloc[j] and df['close'].iloc[j - 1] < df['open'].iloc[j - 1]:  # Bullish reversal
                        reversal = True
                        break
                if reversal:
                    bos_lines.append({
                        'start_x': last_LL['timestamp'],
                        'start_y': last_LL['high'],
                        'end_x': df['timestamp'].iloc[i],
                        'end_y': last_LL['high'],  # Horizontal line
                        'direction': 'bullish'
                    })
                last_LL = None  # Reset after BoS

            # Bearish BoS: Ensure price reversal before breakout
            if last_HH and current_close < last_HH['low']:
                # Check if there was a price reversal
                reversal = False
                for j in range(last_HH['index'] + 1, i):  # Check candles in between
                    if df['close'].iloc[j] > df['open'].iloc[j] and df['close'].iloc[j - 1] > df['open'].iloc[j - 1]:  # Bullish reversal
                        reversal = True
                        break
                if reversal:
                    bos_lines.append({
                        'start_x': last_HH['timestamp'],
                        'start_y': last_HH['low'],
                        'end_x': df['timestamp'].iloc[i],
                        'end_y': last_HH['low'],  # Horizontal line
                        'direction': 'bearish'
                    })
                last_HH = None  # Reset after BoS

            # Update last_LL and last_HH based on structure
            if current_low < (last_LL['low'] if last_LL else float('inf')):
                last_LL = {'timestamp': df['timestamp'].iloc[i], 'low': current_low, 'high': current_high, 'index': i}

            if current_high > (last_HH['high'] if last_HH else float('-inf')):
                last_HH = {'timestamp': df['timestamp'].iloc[i], 'high': current_high, 'low': current_low, 'index': i}

        return bos_lines

    def plot_bos_lines(self, ax, bos_lines, added_labels):
        """Plot horizontal BoS lines on the chart."""
        for line in bos_lines:
            color = 'green' if line['direction'] == 'bullish' else 'red'
            ax.hlines(line['start_y'], mdates.date2num(line['start_x']), mdates.date2num(line['end_x']),
                    colors=color, linestyles='dashed', linewidth=1)
            added_labels.add(f"BoS ({line['direction']})")

    def draw_fvg_and_ob_rectangles(self, df, ax, added_labels):
        """Draw Fair Value Gaps (FVG) and Order Blocks (OB) on the chart."""

        # Detect FVG: A gap between the high of the previous candle and low of the next candle
        for i in range(1, len(df) - 1):
            prev_high = df['high'].iloc[i - 1]
            next_low = df['low'].iloc[i + 1]
            current_low = df['low'].iloc[i]
            current_high = df['high'].iloc[i]

            # FVG Condition
            if prev_high < next_low:
                # Draw FVG rectangle
                rect = patches.Rectangle(
                    (df['datetime'].iloc[i], prev_high),  # Bottom-left corner
                    width=df['datetime'].iloc[i + 1] - df['datetime'].iloc[i],  # Width based on time
                    height=next_low - prev_high,  # Height based on price gap
                    linewidth=1,
                    edgecolor='black',
                    facecolor='orange',
                    alpha=0.4,
                    label="FVG" if 'FVG' not in added_labels else ""
                )
                ax.add_patch(rect)
                added_labels.add("FVG")

            # OB Condition: A consolidation zone (for example, large-bodied candles)
            if abs(current_high - current_low) > (df['close'].mean() * 0.005):  # Simplified OB condition
                rect = patches.Rectangle(
                    (df['datetime'].iloc[i], current_low),
                    width=df['datetime'].iloc[i + 1] - df['datetime'].iloc[i],
                    height=current_high - current_low,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='red',
                    alpha=0.4,
                    label="OB" if 'OB' not in added_labels else ""
                )
                ax.add_patch(rect)
                added_labels.add("OB")

    def detect_and_merge_fvg(self, df):
        """
        Detect and merge contiguous Fair Value Gaps (FVG) into single areas.

        Args:
            df (pd.DataFrame): The DataFrame containing OHLC data.

        Returns:
            list: A list of merged FVG areas, each represented as a dictionary.
        """
        fvg_areas = []
        current_fvg = None

        for i in range(2, len(df)):
            # Check for Bullish FVG (low of current > high of two candles before)
            if df['low'].iloc[i] > df['high'].iloc[i - 2]:
                if current_fvg is None:
                    # Start a new FVG
                    current_fvg = {
                        'start': df['datetime'].iloc[i - 2],
                        'end': df['datetime'].iloc[-1],
                        'bottom': df['low'].iloc[i - 2],
                        'top': df['high'].iloc[i],
                        'direction': 'bullish',
                    }
                # else:
                #     # Extend the current FVG
                #     current_fvg['end'] = df['datetime'].iloc[i]
                #     current_fvg['top'] = max(current_fvg['top'], df['high'].iloc[i])

            # Check for Bearish FVG (high of current < low of two candles before)
            elif df['high'].iloc[i] < df['low'].iloc[i - 2]:
                if current_fvg is None:
                    # Start a new FVG
                    current_fvg = {
                        'start': df['datetime'].iloc[i - 2],
                        'end': df['datetime'].iloc[-1],
                        'bottom': df['low'].iloc[i],
                        'top': df['high'].iloc[i - 2],
                        'direction': 'bearish',
                    }
                # else:
                #     # Extend the current FVG
                #     current_fvg['end'] = df['datetime'].iloc[i]
                #     current_fvg['bottom'] = min(current_fvg['bottom'], df['low'].iloc[i])

            else:
                # Finalize the current FVG if conditions are no longer met
                if current_fvg is not None:
                    fvg_areas.append(current_fvg)
                    current_fvg = None

        # Append the last FVG if it exists
        if current_fvg is not None:
            fvg_areas.append(current_fvg)

        return fvg_areas

    def visualize_fvg(self, ax, fvg_areas, added_labels):
        """
        Visualize merged Fair Value Gaps (FVG) as rectangles.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to draw.
            fvg_areas (list): A list of merged FVG areas to visualize.
        """
        for fvg in fvg_areas:
            color = 'green' if fvg['direction'] == 'bullish' else 'red'
            rect = plt.Rectangle(
                (fvg['start'], fvg['bottom']),  # Bottom-left corner
                fvg['end'] - fvg['start'],  # Width (time range)
                fvg['top'] - fvg['bottom'],  # Height (price range)
                color=color,
                alpha=0.2
                # label=f"{'Bullish' if fvg['direction'] == 'bullish' else 'Bearish'} FVG",
            )
            ax.add_patch(rect)
            added_labels.add(f"{'Bullish' if fvg['direction'] == 'bullish' else 'Bearish'} FVG")
            ax.add_patch(rect)
        
    def calculate_trendlines(self, df):
        """Calculate trendlines based on price movement."""
        trend_segments = []
        current_trend = None
        start_idx = 0

        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                if current_trend != 'up':
                    if current_trend is not None:
                        trend_segments.append((start_idx, i - 1, current_trend))
                    current_trend = 'up'
                    start_idx = i - 1
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                if current_trend != 'down':
                    if current_trend is not None:
                        trend_segments.append((start_idx, i - 1, current_trend))
                    current_trend = 'down'
                    start_idx = i - 1

        # Append the last trend segment
        if current_trend is not None:
            trend_segments.append((start_idx, len(df) - 1, current_trend))

        return trend_segments

    def invalidate_fvg(self, df, i, low_band, high_band):
        end_time = df['datetime'].iloc[-1]
        for j in range(i + 2, len(df)):
            if overlap(low_band, high_band, df['low'].iloc[j], df['high'].iloc[j]):
                # print(f"invalidating block at {df['datetime'].iloc[j]} with high {df['high'].iloc[j]} and low {df['low'].iloc[j]} for band {low_band} {high_band}")
                end_time = df['datetime'].iloc[j]
                return end_time
        return end_time

    def detect_fvg(self, df):
        """Detect Fair Value Gaps (FVG)."""
        fvg_zones = []

        for i in range(2, len(df) - 2):
            prev_high = df['high'].iloc[i - 1]
            prev_low = df['low'].iloc[i - 1]
            next_low = df['low'].iloc[i + 1]
            next_high = df['high'].iloc[i + 1]

            # Check for bullish FVG
            if prev_high < next_low:
                # Check for invalidating mother candle
                high_band = next_low
                low_band = prev_high
                mother_candle = df.iloc[i - 2]
                if mother_candle['open'] < mother_candle['close'] and mother_candle['high'] > next_low:
                    continue

                # Default end_time to the last candle
                end_time = self.invalidate_fvg(df, i, low_band, high_band)

                fvg_zones.append({
                    'type': 'bullish',
                    'start_time': df['datetime'].iloc[i + 1],
                    'end_time': end_time,
                    'low': low_band,
                    'high': high_band
                })

            # Check for bearish FVG
            elif prev_low > next_high:
                # Check for invalidating mother candle
                high_band = prev_low
                low_band = next_high
                mother_candle = df.iloc[i - 2]
                if mother_candle['open'] > mother_candle['close'] and mother_candle['low'] < next_high:
                    continue

                # Default end_time to the last candle
                end_time = self.invalidate_fvg(df, i, low_band, high_band)

                fvg_zones.append({
                    'type': 'bearish',
                    'start_time': df['datetime'].iloc[i],
                    'end_time': end_time,
                    'low': low_band,
                    'high': high_band
                })

        return fvg_zones



    def detect_liquidity(self, df):
        """Detect unconsumed liquidity and filter for validity."""
        high_liquidity = []
        low_liquidity = []


        # Step 1: Collect potential liquidity levels
        for i in range(len(df)):  # Process from old to most recent
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            high_liquidity[:] = [item for item in high_liquidity if item[1] > current_high]
            low_liquidity[:] = [item for item in low_liquidity if item[1] < current_low]
            high_liquidity.append((df['datetime'].iloc[i], current_high))
            low_liquidity.append((df['datetime'].iloc[i], current_low))
        
        return high_liquidity, low_liquidity
    
    def display_grid(self, grid):
        """Display the calculated grid in a tabular format."""
        print("Calculated Grid Parameters:")
        print(f"{'Level':<6} {'Price':<28} {'Amount':<12} {'Qty':<16}")
        for i, (price, amount, qty) in enumerate(grid):
            print(f"{i + 1:<6} {price:<28.12f} {amount:<12.2f} {qty:<16.6f}")

    def suggest_grid_parameters(self, symbol, df, max_price_range=50, max_grid_size=12):
        """
        Suggest optimal grid parameters for the symbol based on historical volatility.
        
        Args:
            symbol (str): The trading symbol (e.g., BTC/USDT).
            df (DataFrame): Historical OHLCV data.
            max_price_range (float): Maximum price range in percentage for the grid.
            max_grid_size (int): Maximum number of grid levels.
            
        Returns:
            dict: Suggested grid parameters including grid size, price range, and distribution factor.
        """
        # Calculate historical price volatility
        # df['price_change'] = (df['high'] - df['low']) / df['low'] * 100  # Intraday price change in percentage
        # average_volatility = df['price_change'].mean()
        df['returns'] = df['price_change'] = (df['high'] - df['low']) / df['low'] * 100
        df['volatility'] = df['returns'].rolling(window=3).std() * np.sqrt(252)
        average_volatility = df['volatility'].mean()
        print(f"Average volatility for {symbol}: {average_volatility:.2f}%")

        # Calculate optimal price range based on volatility
        suggested_price_range = min(average_volatility * 1.2, max_price_range)  # Cap the range at max_price_range

        # Suggest grid size based on price range
        suggested_grid_size = max(3, min(int(suggested_price_range // 2), max_grid_size))  # At least 3 levels, up to max_grid_size

        # Suggest distribution factor based on grid size (smaller grids get tighter spacing)
        suggested_distribution_factor = 1 + (suggested_grid_size / max_grid_size) * 1.5  # Scale logarithmic factor

        return {
            'symbol': symbol,
            'grid_size': suggested_grid_size,
            'price_range': suggested_price_range,
            'distribution_factor': suggested_distribution_factor,
        }
    
    def calculate_grid_params(self, symbol, total_stake, grid_size, price_range, distribution_factor, martingale_factor):
        """
        Calculate grid parameters with stake distribution and logarithmic price spacing.
        Ensures prices stay non-negative and within the specified price range.
        """
        # Fetch current price
        market_price = self.fetch_current_price(symbol)
        if not market_price:
            print("Failed to fetch market price. Grid calculation aborted.")
            return []

        prices = self.calculate_logarithmic_grid_down(market_price, price_range, grid_size, distribution_factor)

        # Calculate weights for martingale and normalize them
        base_weight = 1  # Starting weight
        weights = [base_weight * (1 + martingale_factor) ** i for i in range(grid_size)]
        normalized_weights = [weight / sum(weights) for weight in weights]  # Normalize weights

        # Calculate order amounts
        order_amounts = [total_stake * weight for weight in normalized_weights]
        order_qty = [amount / price for amount, price  in zip(order_amounts, prices)]
        order_entry = [market_price * amount / price for amount, price  in zip(order_amounts, prices)]

        # Combine prices and amounts
        grid = list(zip(prices, order_amounts, order_qty))
        return grid

    def calculate_logarithmic_grid_down(self, entry_price, price_range, grid_size, distribution_factor):
        """
        Calculate a logarithmic grid of prices below the entry price.

        Args:
            entry_price (float): The starting price for the grid.
            price_range (float): The total price range for the grid as a percentage.
            grid_size (int): The number of grid levels.
            distribution_factor (float): The factor controlling the logarithmic spacing.

        Returns:
            list: A list of grid prices below the entry price.
        """
        # Ensure grid_size is greater than 1 to avoid division by zero
        if grid_size <= 1:
            raise ValueError("Grid size must be greater than 1.")
        
        # Calculate the logarithmic distribution of the steps
        log_steps = [np.log1p(distribution_factor * i) for i in range(1, grid_size + 1)]
        total_log = sum(log_steps)
        log_ratios = [step / total_log for step in log_steps]

        # Calculate price steps
        price_step = entry_price * (price_range / 100)  # Convert percentage to absolute range
        grid_prices = [
            round(entry_price - price_step * sum(log_ratios[:i]), self.precision['price'])
            for i in range(1, grid_size + 1)
        ]

        return grid_prices

    def visualize_data(self, df):
        """Visualize candlestick chart with selected features."""
        fig, ax = plt.subplots(figsize=(16, 8))  # Use a single Axes object

        # Candlestick chart setup
        ohlc = df[['datetime', 'open', 'high', 'low', 'close']].copy()
        candlestick_ohlc(ax, ohlc.values, width=0.02, colorup='green', colordown='red')
        current_price = self.fetch_current_price(self.symbol)

        # Ensure x-axis uses datetime format
        ax.xaxis_date()  # Converts x-axis to date format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Custom format
        price_range = df['high'].max() - df['low'].min()
        y_tick_spacing = price_range / 24  # Adjust for finer grid (e.g., 20 divisions of the price range)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing))
        ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.hlines(y=current_price, xmin=df['datetime'].iloc[0], xmax=df['datetime'].iloc[-1], colors='grey', linestyles='solid', linewidth=0.5, label='Current price')
        # Apply selected features
        added_labels = set()
        if 'fractals' in self.features:
            fractals = df[df['swing_high'] | df['swing_low']]
            for _, row in fractals.iterrows():
                if row['swing_high'] and 'Swing High' not in added_labels:
                    ax.plot(row['datetime'], row['high'], marker='^', color='blue', label='Swing High')
                    added_labels.add('Swing High')
                elif row['swing_high']:
                    ax.plot(row['datetime'], row['high'], marker='^', color='blue')

                if row['swing_low'] and 'Swing Low' not in added_labels:
                    ax.plot(row['datetime'], row['low'], marker='v', color='orange', label='Swing Low')
                    added_labels.add('Swing Low')
                elif row['swing_low']:
                    ax.plot(row['datetime'], row['low'], marker='v', color='orange')

        if 'trendlines' in self.features:
            print("Calculating trendlines...")
            trend_segments = self.calculate_trendlines(df)
            for start_idx, end_idx, trend in trend_segments:
                color = 'green' if trend == 'up' else 'red'
                ax.plot(df['datetime'].iloc[start_idx:end_idx + 1],
                         df['close'].iloc[start_idx:end_idx + 1],
                         color=color, linewidth=2, label=f"{'Uptrend' if trend == 'up' else 'Downtrend'}" if f"{'Uptrend' if trend == 'up' else 'Downtrend'}" not in added_labels else None)
                added_labels.add(f"{'Uptrend' if trend == 'up' else 'Downtrend'}")

        if 'structure' in self.features:
            print("Marking structure...")
            for _, row in df.iterrows():
                if row['structure'] == 'HH':
                    plt.text(row['datetime'], row['high'], 'HH', color='red', fontsize=8, weight='bold')
                elif row['structure'] == 'HL':
                    plt.text(row['datetime'], row['low'], 'HL', color='green', fontsize=8, weight='bold')
                elif row['structure'] == 'LH':
                    plt.text(row['datetime'], row['high'], 'LH', color='purple', fontsize=8, weight='bold')
                elif row['structure'] == 'LL':
                    plt.text(row['datetime'], row['low'], 'LL', color='blue', fontsize=8, weight='bold')

            # Draw BOS and POB lines
            for i, row in df[df['BOS']].iterrows():
                # Get the previous LH for BOS
                prev_lh = df[(df['structure'] == 'LH') & (df.index < i)]
                if not prev_lh.empty:
                    bos_start_idx = prev_lh.index[-1]
                    plt.hlines(
                        y=row['close'],
                        xmin=df['datetime'].iloc[bos_start_idx],
                        xmax=row['datetime'],
                        colors='pink',
                        linestyles='--',
                        label='BOS' if 'BOS' not in added_labels else ''
                    )
                    added_labels.add('BOS')

            for i, row in df[df['POB']].iterrows():
                # Get the previous HL for POB
                prev_hl = df[(df['structure'] == 'HL') & (df.index < i)]
                if not prev_hl.empty:
                    pob_start_idx = prev_hl.index[-1]
                    plt.hlines(
                        y=row['close'],
                        xmin=df['datetime'].iloc[pob_start_idx],
                        xmax=row['datetime'],
                        colors='cyan',
                        linestyles='--',
                        label='POB' if 'POB' not in added_labels else ''
                    )
                    added_labels.add('POB')

        if 'fvg_ob' in self.features:
            # Draw FVG and OB rectangles
            self.draw_fvg_and_ob_rectangles(df, ax, added_labels)
        
        if 'liquidity' in self.features:
            print("Detecting unconsumed liquidity...")
            high_liquidity, low_liquidity = self.detect_liquidity(df)

            print(f"Liquidity above the current price:")
            for time, price in high_liquidity:
                print(f"price level {price} detected at {time}")
                ax.hlines(y=price, linewidth=0.5, xmin=time, xmax=df['datetime'].iloc[-1], colors='green', linestyles='solid', label='Liquidity High' if 'Liquidity High' not in added_labels else None)
                added_labels.add('Liquidity High')

            print(f"Liquidity below the current price:")
            for time, price in low_liquidity:
                print(f"price level {price} detected at {time}")
                ax.hlines(y=price, linewidth=0.5, xmin=time, xmax=df['datetime'].iloc[-1], colors='red', linestyles='solid', label='Liquidity Low' if 'Liquidity Low' not in added_labels else None)
                added_labels.add('Liquidity Low')

        if 'fvg' in self.features:
            fvg_areas = self.detect_fvg(df)
            for fvg in fvg_areas:
                color = 'lightgreen' if fvg['type'] == 'bullish' else 'lightcoral'
                ax.add_patch(plt.Rectangle(
                    (fvg['start_time'], fvg['low']),
                    fvg['end_time'] - fvg['start_time'],
                    fvg['high'] - fvg['low'],
                    color=color, alpha=0.1
                ))
                added_labels.add('FVG')

        if 'bos' in self.features:
            bos_points = self.detect_bos_refined(df)
            self.plot_bos_lines(ax, bos_points, added_labels)

        # Add interactivity for hover
        cursor = mplcursors.cursor(ax, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            # Find the closest data point
            x = sel.target[0]
            y = sel.target[1]
            datetime_val = mdates.num2date(x).strftime('%Y-%m-%d %H:%M')
            sel.annotation.set(text=f"Time: {datetime_val}\nPrice: {y:.2f}")
            sel.annotation.get_bbox_patch().set_alpha(0.8)
            
        plt.legend()
        plt.title(f'Candlestick Chart with Features for {self.symbol}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()

    def run(self, symbol):
        """Run the bot to analyze and visualize."""
        print(f"Fetching data for {symbol} on {self.timeframe} timeframe...")
        self.fetch_symbol_precision(symbol)  # Fetch symbol precision
        df = self.fetch_historical_data(symbol)

        if 'fractals' in self.features:
            print("Marking fractals...")
            df = self.mark_fractals(df)

        if 'structure' in self.features:
            print("Detecting market structure...")
            df = self.detect_structure(df)
        
        if 'grid' in self.features:
            print("Suggesting grid parameters...")
            grid_params = self.suggest_grid_parameters(symbol, df)
            print("Suggested Grid Parameters:")
            print(f"Grid Size: {grid_params['grid_size']}")
            print(f"Price Range: {grid_params['price_range']:.2f}%")
            print(f"Logarithmic Distribution Factor: {grid_params['distribution_factor']:.2f}")
            grid = self.calculate_grid_params(symbol, self.stake, grid_params['grid_size'],  grid_params['price_range'], grid_params['distribution_factor'], self.martingale)
            self.display_grid(grid)
        
        print("Visualizing data...")
        self.visualize_data(df)
# Usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMC Trading Bot')
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTC/USDT)')
    parser.add_argument('--timeframe', type=str, choices=['1w', '1h', '4h', '1d'], default='1h', help='Timeframe for analysis')
    parser.add_argument('--features', type=str, nargs='+', default=[], help='Features to include (e.g., fractals, trendlines, structure)')
    parser.add_argument('--stake', type=int, required=False, default=300, help='Stake limit')
    parser.add_argument('--martingale', type=float, required=False, default=0.01, help='Martingale')
    parser.add_argument('--secrets', type=str, default='secrets.json', help='Path to the secrets file')
    args = parser.parse_args()

    
    bot = SMCTradingBot(secrets_file=args.secrets, timeframe=args.timeframe, features=args.features, martingale=args.martingale, stake=args.stake, symbol=args.symbol)
    bot.run(args.symbol)
