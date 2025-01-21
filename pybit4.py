import streamlit as st
import pandas as pd
import requests
import logging
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)

st.title('Signal Detection with Bybit ')

# Sidebar Widgets
st.sidebar.title("Settings")

# Select the time frame for data retrieval
timeframe = st.sidebar.selectbox("Select Time Frame", options=['1', '3', '5', '15', '30', '60', '4H', 'D', 'W', 'M'], index=5)

# Checkbox to filter signals
signal_filters = st.sidebar.multiselect(
    "Select Signals to Display",
    options=["Bullish", "Bearish", "Side After Rally", "Side After Drop", "Ready for Rally", "Ready for Drop", "Neutral"],
    default=["Bullish", "Bearish","Neutral"]
)

# Custom formatting function
def price_format(val):
    if isinstance(val, (int, float)):  
        if val > 0.1:
            return '{:,.4f}'.format(val)
        elif val > 0.01:
            return '{:,.4f}'.format(val)
        elif val > 0.0001:
            return '{:,.6f}'.format(val)
        elif val > 0.000001:
            return '{:,.8f}'.format(val)
        elif val > 0.00000001:
            return '{:,.10f}'.format(val)
        elif val > 0.0000000001:
            return '{:,.12f}'.format(val)
        else:
            return '{:,.15f}'.format(val)
    return "N/A"

# Function to fetch data with error handling
@st.cache_data
def fetch_data(url, params, timeout=10):
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return None

# Function to get valid trading pairs from Bybit
def get_valid_trading_pairs():
    url = 'https://api.bybit.com/v5/market/instruments-info'
    params = {'category': 'spot'}
    data = fetch_data(url, params)
    if data and 'result' in data and 'list' in data['result']:
        return [item['symbol'] for item in data['result']['list']]
    return []

# Function to fetch historical data from Bybit
def get_historical_data(symbol, interval=timeframe):
    url = 'https://api.bybit.com/v5/market/kline'
    params = {'category': 'spot', 'symbol': symbol, 'interval': interval, 'limit': 200}
    data = fetch_data(url, params)
    if data and 'result' in data and 'list' in data['result']:
        df = pd.DataFrame(data['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_'])
        df['start_time'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
        df['close'] = df['close'].astype(float)
        return df[['start_time', 'close']]
    return pd.DataFrame()

# Function to calculate TEMA
def calculate_tema(series, period):
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

# Function to analyze signals
def analyze_signals(data, signal_filters):
    signals = []

    for symbol, group in data.groupby('Coin'):
        group = group.sort_values('start_time')
        group['BLUE'] = calculate_tema(group['close'], 25)
        group['YELLOW'] = calculate_tema(group['close'], 98)
        group['RED'] = group['close'].ewm(span=51, adjust=False).mean()

        group['Signal'] = group.apply(lambda row:
            'Bullish' if row['YELLOW'] > row['RED'] and row['BLUE'] > row['YELLOW'] > row['RED'] else
            'Side After Rally' if row['YELLOW'] > row['RED'] and row['YELLOW'] > row['BLUE'] > row['RED'] else
            'Ready for Drop' if row['YELLOW'] > row['RED'] and row['YELLOW'] > row['RED'] > row['BLUE'] else
            'Ready for Rally' if row['YELLOW'] < row['RED'] and row['BLUE'] > row['RED'] > row['YELLOW'] else
            'Side After Drop' if row['YELLOW'] < row['RED'] and row['RED'] > row['BLUE'] > row['YELLOW'] else                         
            'Bearish' if row['YELLOW'] < row['RED'] and row['RED'] > row['YELLOW'] > row['BLUE'] else
            'Neutral', axis=1)

        group['Days Since Signal Change'] = (group['Signal'] != group['Signal'].shift()).cumsum()
        last_row = group.iloc[-1]
        signals.append({
            'Coin': symbol,
            'Current Close': last_row['close'],
            'BLUE': last_row['BLUE'],
            'YELLOW': last_row['BLUE'],
            'RED': last_row['RED'],
            'Signal': last_row['Signal'],
            'Days Since Change': last_row['Days Since Signal Change']
        })

    # Filter signals based on the selected options in the sidebar
    filtered_signals = [s for s in signals if s['Signal'] in signal_filters]

    return pd.DataFrame(filtered_signals)

# Function to apply color styling to the DataFrame
def apply_colors(row):
    # Define a list to hold style definitions for each column
    styles = [''] * len(row)  # Initialize with no style

    # Color the row based on signal condition
    if row['Current Close'] > max(row['BLUE'], row['YELLOW'], row['RED']):
        styles[0] = 'background-color: green; color: white;'  # Apply green to the first column (Current Close)
    elif row['Current Close'] < min(row['BLUE'], row['YELLOW'], row['RED']):
        styles[0] = 'background-color: red; color: white;'  # Apply red to the first column (Current Close)
    
    return styles


# Main application
valid_pairs = get_valid_trading_pairs()

if valid_pairs:
    # Display the total number of coins available
    st.sidebar.write(f"Total number of valid trading pairs: {len(valid_pairs)}")

    with st.spinner("Fetching and analyzing data..."):
        all_data = {}

        def fetch_and_process(symbol):
            data = get_historical_data(symbol, timeframe)
            if not data.empty:
                return symbol, data
            return symbol, pd.DataFrame()

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(fetch_and_process, valid_pairs))  # No limit on coins

        for symbol, df in results:
            if not df.empty:
                all_data[symbol] = df

        if all_data:
            combined_data = pd.concat(all_data.values(), keys=all_data.keys(), names=['Coin', 'Index']).reset_index(level=0)
            signals_df = analyze_signals(combined_data, signal_filters)

            # Check if the 'Current Close' column exists before applying the format
            if 'Current Close' in signals_df.columns:
                signals_df['Current Close'] = signals_df['Current Close'].apply(price_format)
            if 'BLUE' in signals_df.columns:
                signals_df['BLUE'] = signals_df['BLUE'].apply(price_format)
            if 'YELLOW' in signals_df.columns:
                signals_df['YELLOW'] = signals_df['YELLOW'].apply(price_format)
            if 'RED' in signals_df.columns:
                signals_df['RED'] = signals_df['RED'].apply(price_format)
            

            if not signals_df.empty:
                styled_df = signals_df.style.apply(apply_colors, axis=1)  # Apply row-wise styling
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.write("No signals found.")
        else:
            st.write("No data available.")
else:
    st.write("Unable to fetch Bybit data.")


