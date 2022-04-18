import alpaca_trade_api as tradeapi
import api_credentials
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
import time

unique_minutes = []


def alpaca_trader(ticker, polarity):
    '''
    Approximate price of the stock, then calculates a value equivalent to 1% of the current portfolio value divided 
    by the current price of the stock, which is then multiplied by the polarity score to determine the quantity of 
    shares to buy. If the buying power is exceeded by this amount, then the quantity is decremented by one share 
    until the expense is affordable for the account to make the purchase or the quantity of shares to buy is zero.

    Before selling shares of a stock, the bot needs to determine if it even owns any of that stock to avoid throwing 
    an error by trying to sell something it does not own. If it does own that stock, it then decides to sell the 
    quantity of shares in the same way that it determines the number of shares to buy, using a combination of polarity 
    score, portfolio value, and current approximate price per share. If that quantity is greater than the number of 
    shares currently owned, then the bot simply sells all of that stock.

    With all this math and fact checking, there is still room for error because the current stock price is always 
    an approximation since traders are buying and selling stock at various prices within milliseconds of each other. 
    Thus, the order is placed within a try-except block and marked as a valid trade once complete, inspired by how 
    mutual exclusion locks work with parallel programming systems. If an error is thrown because the trade expense 
    is suddenly too expensive within milliseconds, the bot decrements the quantity of shares to buy by one and tries 
    again. If the quantity decreases to zero before it becomes affordable, then the transaction is marked as “skipped”, 
    terminates the trading process, and exits the function. 
    '''
    global unique_minutes
    ALPACA_ENDPOINT_URL = api_credentials.ALPACA_ENDPOINT_URL
    ALPACA_API_KEY = api_credentials.ALPACA_API_KEY
    ALPACA_SECRET_KEY = api_credentials.ALPACA_SECRET_KEY
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY,
                        ALPACA_ENDPOINT_URL, api_version='v2')
    account = api.get_account()
    if account.trading_blocked:
        print('Account is currently restricted from trading.')
    market_clock = api.get_clock()
    minute = int(str(market_clock.timestamp)[14:16])
    frequency = 10  # minutes
    if minute < 2 and len(unique_minutes) == 6:
        unique_minutes = []
    with open('portfolio_performance.txt', 'a') as f:
        # Write to file every {frequency} minutes
        if minute % frequency == 0:
            if minute not in unique_minutes:
                unique_minutes.append(minute)
                f.write(
                    f"Equity: {account.equity}, Time Stamp: {market_clock.timestamp} \n")

    ### Past Attempts to get bars working

    # now = pd.Timestamp.now(tz='America/New_York')#.floor('1min')
    # yesterday = (now - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    # today = (now - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    # thirty_minutes_ago = (now - pd.Timedelta(minutes=30))#.strftime('%Y-%m-%d')
    # fifteen_minutes_ago = (now - pd.Timedelta(minutes=15))#.strftime('%Y-%m-%d')
    # print(ticker, yesterday, today, thirty_minutes_ago, fifteen_minutes_ago, now)
    # barset = api.get_bars(ticker[1:], TimeFrame.Day, yesterday, today, limit=1).df
    # open_price = float(str(barset.open.iloc[0]).split()[0])
    # close_price = float(str(barset.close.iloc[0]).split()[0])

    # barset = api.get_barset(ticker[1:], 'day', limit=1)
    # open_price = barset[ticker[1:]][0].o
    # close_price = barset[ticker[1:]][0].c

    now = pd.Timestamp.now(tz='America/New_York')
    yesterday = (now - pd.Timedelta(days=1))
    bars = api.get_bars(ticker[1:], TimeFrame.Day,
                   start=yesterday.isoformat(),
                   end=None,
                   limit=2
                   ).df
    open_price = float(str(bars.open.iloc[0]).split()[0])
    close_price = float(str(bars.close.iloc[0]).split()[0])
    approximate_price_per_share = (open_price + close_price)/2
    
    # Determine how many shares to buy based on the price of the stock.
    # Currently allowing for 1% of portfolio per trade.
    shares_per_polarity_point = (
        float(account.portfolio_value) * 0.01) // approximate_price_per_share

    with open('stock_trading_decisions.txt', 'a') as f:
        msg = f"Time Stamp: {market_clock.timestamp} \n"
        print(msg)
        f.write(msg)
        
        if market_clock.is_open:
            if polarity > 0:
                side = "buy"
                qty = polarity*shares_per_polarity_point
                expense = approximate_price_per_share * qty
                # If buying power is limited, then decrease quantity of shares until transaction amount is lower than buying power
                while expense > float(account.buying_power):
                    qty -= 1
                    expense = approximate_price_per_share * qty
            else:
                side = "sell"
                polarity *= -1
                qty = polarity*shares_per_polarity_point

                # Check how many shares I currently own, if any
                # try except because an error is thrown if zero shares are owned.
                try:
                    pos_qty = float(api.get_position(ticker[1:]).qty)
                except Exception as exception:
                    if exception.__str__() == 'position does not exist':
                        pos_qty = 0
                if qty > pos_qty:
                    qty = pos_qty

            # only perform a trade if trading more than 0 shares
            if qty > 0:
                # Sometimes the prices change and throw a buying power error. Decrease qty until satisfied.
                invalid = True
                skipped = False
                while invalid:
                    try:
                        if qty == 0:
                            skipped = True
                            break
                        # market: buy or sell at market price, opposed to a limit order.
                        # time_in_force: only keep order open until end of the day
                        order = api.submit_order(
                            symbol=ticker[1:], qty=qty, side=side, type="market", time_in_force="day")
                        invalid = False
                    except Exception as exception:
                        if exception.__str__() == 'insufficient buying power':
                            qty -= 1
                if not skipped:
                    if order.status == 'accepted':
                        msg = f"Success! Order placed to {order.side} {order.qty} shares of {ticker}. \n"
                        print(msg)
                        f.write(msg)
                    else:
                        msg = f"Trade failed. Alpaca account status: {account.status}. \n"
                        print(msg)
                        f.write(msg)
                else:
                    msg = f"Transaction prices changed during processing. Either not enough buying power or insufficient shares to sell. Skipping. \n"
                    print(msg)
                    f.write(msg)
                time.sleep(3)
            else:
                if side == "buy":
                    msg = f"You don't have enough buying power to buy {ticker}. Skipping. \n"
                    print(msg)
                else:
                    msg = f"You do not own any shares of {ticker} to sell. Skipping. \n"
                    print(msg)
            time.sleep(3)
        else:
            msg = f"No orders were made because the stock market is currently closed for trading. \n"
            print(msg)
        time.sleep(3)

    return account.equity, market_clock.timestamp, msg