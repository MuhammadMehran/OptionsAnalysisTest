# %%writefile app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import mysql.connector

def get_intraday_data(symbol):
    df = pd.DataFrame()
    df['Symbol'] = ['Current Price', 'Volume', "Today's Highest Price",
                    "Today's Lowest Price", 'Percentage Difference %' ]
    ticker = yf.Ticker(symbol)
    history = ticker.history()
    current = round(history['Close'][-1],2)
    volume = int(history['Volume'][-1])
    high = round(history['High'][-1],2)
    low = round(history['Low'][-1],2)
    df[symbol] = [current, volume, high, low, round(100 * (high - low ) / low,2)]
    return df

def relative_performance(symbol):
    if symbol != 'SPY':
        spy_data = yf.download('SPY', period="2d")['Close']
        symbol_data = yf.download(symbol, period='2d')['Close']

        # Check if either DataFrame is empty
        if spy_data.empty or symbol_data.empty:
            st.warning(f"No data available for {symbol} or 'SPY' in the specified period.")
            return

        combined_df = pd.concat([spy_data, symbol_data], axis=1)
        combined_df.columns = ['SPY', f'{symbol}']
        combined_df['SPY'] = combined_df['SPY'].pct_change() * 100
        combined_df[f'{symbol}'] = combined_df[f'{symbol}'].pct_change() * 100
        combined_df = combined_df.round(3)
        combined_df = combined_df.dropna()

        fig2, ax2 = plt.subplots(figsize=(10, 6))

        combined_df.plot(kind='bar', rot=0, width=0.9, ax=ax2)
        ax2.set_xticklabels(combined_df.index.format())
        ax2.set_ylabel('Change (%)')
        for container in ax2.containers:
            ax2.bar_label(container)
        ax2.set_xticklabels(combined_df.index.format())
        plt.title(f'Relative Performance to SPY for {symbol}', fontsize=18, fontweight='bold')
        plt.tight_layout()

        return fig2



def options_chain(symbol):
    '''Utility Method to Get Options for Stock'''
    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt =  pd.concat([opt.calls, opt.puts])
        opt['expirationDate'] = e
        options = pd.concat([options, opt])

    options = options.reset_index(drop=True)
    options['expirationDate'] = pd.to_datetime(
        options['expirationDate'])
    options['dte'] = (options['expirationDate'] -
                      dt.datetime.today()).dt.days / 365  # Fix this line

    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)

    # drop original ask column
    options = options.drop('ask', axis=1)
    # change lastprice column to ask to ask column
    options = options.rename(columns={'lastPrice': 'ask'})

    options[['bid', 'ask', 'strike']] = options[[
        'bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / \
        2  # Calculate the midpoint of the bid-ask

    # Drop unnecessary and meaningless columns
    options = options.drop(
        columns=['contractSize', 'currency', 'change', 'lastTradeDate'])

    return options


def get_put_call_ratio(data, volume=False):
    '''Utility Method to get PUT CALL Data'''
    dates = np.sort(data['expirationDate'].unique())

    if dates[0].astype('datetime64[D]') == np.datetime64('today'):
        exp_date = dates[1]
    else:
        exp_date = dates[0]

    cur = data[data['expirationDate'] == exp_date]

    if volume:
        ratio = cur[cur['CALL'] == False]['volume'].sum(
        ) / cur[cur['CALL'] == True]['volume'].sum()
    else:
        ratio = cur[cur['CALL'] == False]['openInterest'].sum(
        ) / cur[cur['CALL'] == True]['openInterest'].sum()

    if str(ratio).lower() == 'nan':
        ratio = 0
    return exp_date.astype('datetime64[D]'), round(ratio, 2)


def get_max_strike(data, exp_date, call=True):
    curr = data[data['expirationDate'] == exp_date]
    curr = curr[curr['CALL'] == call]
    max_volume = curr[curr['volume'] == curr['volume'].max()].iloc[0]
    max_volume_strike = max_volume['strike']
    max_volume_value = max_volume['volume']
    return max_volume_strike, max_volume_value


def get_expected_move(stock, expiration_date):
    '''Utility Method to apply ATM for expected move'''
    ticker = yf.Ticker(stock)
    current_price = ticker.history(period='1d')['Close'][0]

    data = options_chain(stock)
    dates = np.sort(data['expirationDate'].unique())

    closest = data[data['expirationDate'] == expiration_date]

    closest['abs'] = abs(current_price - closest['strike'])
    closest = closest.sort_values('abs')
    move = (closest[closest['CALL'] == True]['ask'].iloc[0] +
            closest[closest['CALL'] == False]['ask'].iloc[0]) * 1.25
    return current_price, move


def calculate_options_metrics(symbol):
    df = options_chain(symbol)
    exp_date, interst_ratio = get_put_call_ratio(df)
    exp_date, volume_ratio = get_put_call_ratio(df, volume=True)

    max_strike, max_volume = get_max_strike(df, exp_date)
    max_strike_put, max_volume_put = get_max_strike(df, exp_date, call=False)

    current_price, move = get_expected_move(symbol, exp_date)

    data = {
        'Symbol': [symbol],
        'Current Price': [current_price],
        'Expiration Date': [str(exp_date)],
        'Put/Call Open Interest Ratio': [interst_ratio],
        'Put/Call Volume Ratio': [volume_ratio],
        'Most Tradeable CALL Strike': [max_strike],
        'Strike CALL Volume': [max_volume],
        'Most Tradeable PUT Strike': [max_strike_put],
        'Strike PUT Volume': [max_volume_put],
        'Expected Move (±)': [move],
        'Expected Price (+)': [current_price + move],
        'Expected Price (-)': [current_price - move]
    }

    today_date = dt.datetime.now().strftime('%Y-%m-%d')
    today_data = yf.download(symbol, start=today_date, interval="1m")

    # Create a figure for the minute-by-minute price graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(today_data.index, today_data['Close'], label='1-Min Interval Price', color='blue')

# Add lines or annotations for the expected move
    ax.axhline(current_price + move, linestyle='--', color='green', label='Expected Price (+)')
    ax.axhline(current_price - move, linestyle='--', color='red', label='Expected Price (-)')

# Customize the graph
    ax.set_title(f"{symbol} 1-Min Interval & Expected Move Until {exp_date}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()
    

# Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    transposed = pd.DataFrame(data).T
    return transposed, fig


def add_email_to_db(email):
    mydb = mysql.connector.connect(
        host=st.secrets["DATABASE_HOST"],
        user=st.secrets["DATABASE_USERNAME"],
        password=st.secrets["DATABASE_PASSWORD"],
        database=st.secrets["DATABASE_DB"],
        port=st.secrets["DATABASE_PORT"]
    )

    mycursor = mydb.cursor()

    sql = "INSERT INTO login (email) VALUES (%s)"
    val = (email, )
    mycursor.execute(sql, val)

    mydb.commit()

# Streamlit app
st.set_page_config(layout='centered', page_icon='📉', page_title='Option Chain Analysis')
st.markdown("<h2 style='text-align: center; font-family: Arial, sans-serif; font-size: 72px; font-weight: bold;'>Options Analysis 1.0</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>by Szymanski</h2>", unsafe_allow_html=True)



st.session_state.user_email = None

placeholder = st.empty()
with placeholder.container():
    email = st.text_input("Please enter email")
    if email:
        if '@' not in email:
            st.error('Please enter valid email address')
        else:
            add_email_to_db(email)
            st.session_state.user_email = email
            placeholder.empty()

if st.session_state.user_email:
    st.success(f'Welcome {st.session_state.user_email}!')
    # Input for stock symbol
    symbol = st.text_input("Enter a stock symbol (e.g., AAPL):")

    if symbol:
        st.markdown("<h1 style='text-align: center;'>Intraday Data</h2>", unsafe_allow_html=True)
        intraday_data = get_intraday_data(symbol)
        
        st.dataframe(intraday_data.set_index('Symbol'), use_container_width=True)
        
        

        if symbol != 'SPY':
            st.markdown("<h1 style='text-align: left;'>Relative Performance to SPY</h1>", unsafe_allow_html=True)
            fig2 = relative_performance(symbol)
            st.pyplot(fig2)

        st.markdown("<h1 style='text-align: center;'>Option Chain Analysis</h2>", unsafe_allow_html=True)
        disclosure = """
        Expected Stock Move: This is a quick estimation of price move of the underlying using the At-the-Money (ATM) options straddles with closest expiration date within
        1 standard deviation. Each strike is placed at the 84% probability. When added together,
        this means there is a 68% chance the underlying will stay between the strikes.

        Please allow a moment for a table & chart to load... 
        """
        st.write(disclosure)

        options_metrics, fig = calculate_options_metrics(symbol)
        st.dataframe(options_metrics, use_container_width=True)
        st.pyplot(fig)

        st.markdown("<h3 style='text-align: center;'>Disclaimer</h2>", unsafe_allow_html=True)

content = """
The content is solely for educational and informational purposes. It is not trading or investment advice or a recommendation, nor is it intended to be.
This should not be viewed as a licensed financial advisor, registered investment advisor, or a registered broker-dealer.
No responsibility for any errors, omissions, or representations in this content.
Any mistake, error, or discrepancy that is discovered may be brought to my attention, and appropriate efforts will be made to correct
it to the greatest extent possible.

If you have questions or suggestions for improvement, please don't hesistate to reach out: szymanskiresearch@protonmail.com


"""

st.write(content)