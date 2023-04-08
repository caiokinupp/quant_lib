import pandas as pd
import numpy as np

def moving_average(value_series, periods = 20, ma_type = 'sma'):
    """
    Method to calculate SMA (Simple Moving Average) or 
    EMA (Exponential Moving average) from a series of values.
    
    Parameters
    ----------
    value_series : pandas.Series
        A pandas Series ordered from oldest to newest.
    periods : integer
        Periods of the calculation (default 20)
    ma_type : string
        sma or ema (default 'sma')

    Returns
    -------
    moving_average : pandas.Series
        A new Series with calculated moving average
    """
    
    if ma_type == 'sma':
        return value_series.rolling(periods).mean()
    elif ma_type == 'ema':
        return value_series.ewm(span = periods, min_periods = periods, adjust = False).mean()


def z_score(value_series, periods = 20):
    """
    Method to calculate the Z-Score.
    Is the distance from price to a MA in standard deviations.
    
    Parameters
    ----------
    value_series : pandas.Series
        A pandas Series ordered from oldest to newest.
    periods : integer
        Periods of the calculation (default 20)
    ma_type : string
        sma or ema (default 'sma')

    Returns
    -------
    z_score : pandas.Series
        A new Series with calculated z-score
    """
    
    sma = value_series.rolling(periods).mean()
    # Calculating distance between price and the MA
    distance_price_ma = value_series - sma
    # Calculating the moving standard deviation
    price_std = value_series.rolling(periods).std()

    return distance_price_ma/price_std


def high_low(value_series):
    """
    Method to define whether two sequence values is going up or down
    0 -> Lateral
    1 -> High
    -1 -> Low
    
    Parameters
    ----------
    value_series : pandas.Series
        A pandas Series ordered from oldest to newest.

    Returns
    -------
    directions : pandas.Series
        Series of values that define whether is going up or down
    """
    
    lst_directions = []
    
    for i in range(len(value_series)):
        if i == 0:
            lst_directions.append(0)
        else:
            if value_series[i] > value_series[i-1]:
                lst_directions.append(1)
            elif value_series[i] < value_series[i-1]:
                lst_directions.append(-1)
            elif value_series[i] == value_series[i-1]:
                lst_directions.append(0)
            else:
                lst_directions.append(np.nan)
    
    return np.array(lst_directions)


def sequence_counter(value_series):
    """
    Method to count sequence of periods in the same direction.

    Parameters
    ----------
    value_series : pandas.Series
        A pandas Series ordered from oldest to newest.
    
    Returns
    -------
    sequence : pandas.Series
        Sequence size on same direction
    """
    
    directions = high_low(value_series)
    lst_sequences = []
    
    for i in range(len(directions)):
        if i == 0:
            sequence_counter = 1
            lst_sequences.append(sequence_counter)
        else:
            if directions[i] == directions[i-1]:
                sequence_counter = sequence_counter + 1
                lst_sequences.append(sequence_counter)
            else:
                sequence_counter = 1
                lst_sequences.append(sequence_counter)
    
    return np.array(lst_sequences) * np.array(directions)


def candle_body_proportion(open, high, low, close):
    """
    Method to calculate proportion of the candle's body and the shadows

    Parameters
    ----------
    open, high, low, close : pandas.Series
        A pandas Series ordered from oldest to newest of the candle's prices.
    
    Returns
    -------
    proportions : pandas.Series
        Proportions of candle's body
    """
    if close >= open:
        shadow_sup = high - close
        body = close - open
        shadow_inf = open - low
        total = shadow_sup + body + shadow_inf
        if total == 0:
            return [1.0, 1.0, 1.0]
        shadow_sup = shadow_sup/total
        body = body/total
        shadow_inf = shadow_inf/total
        
        return [shadow_sup, body, shadow_inf]
    
    else:
        shadow_sup = high - open
        body = open - close
        shadow_inf = close - low
        total = shadow_sup + body + shadow_inf
        if total == 0:
            return [1.0, 1.0, 1.0]
        shadow_sup = shadow_sup/total
        body = body/total
        shadow_inf = shadow_inf/total

        return [shadow_sup, body, shadow_inf]


def candle_color(open, close):
    """
    Method to calculate the direction of the candles

    Parameters
    ----------
    open, close : pandas.Series
        A pandas Series ordered from oldest to newest of the candle's prices.
    
    Returns
    -------
    candle_direction : pandas.Series
        Candle's direction
    """
    return np.where(close > open, 1, -1)


def month_of_quarter(month_number):
    """
    Method to calculate if a particular month is the first, second or third of a quarter

    Parameters
    ----------
    month_number : Integer
        An integer that represents a month number
    
    Returns
    -------
    candle_direction : Integer
        Month 1, 2 or 3 of a quarter
    """
    if month_number in [1, 4, 7, 10]:
        return 1
    elif month_number in [2, 5, 8, 11]:
        return 2
    elif month_number in [3, 6, 9, 12]:
        return 3


def RSI(value_series, periods = 14):
    """
    Method to calculate RSI indicator

    Parameters
    ----------
    value_series : pandas.Series
        A pandas Series ordered from oldest to newest.
    periods : integer
        Periods of the calculation (default 14)
    
    Returns
    -------
    rsi : pandas.Series
        Calculated RSI indicator
    """
    df_rsi = pd.DataFrame(data = {'close':value_series})

    # Establish gains and losses for each day
    df_rsi['variation'] = df_rsi.diff()
    df_rsi = df_rsi[1:]
    df_rsi['gain'] = np.where(df_rsi['variation'] > 0, df_rsi['variation'], 0)
    df_rsi['loss'] = np.where(df_rsi['variation'] < 0, df_rsi['variation'], 0)

    # Calculate simple averages so we can initialize the classic averages
    df_rsi['avg_gain'] = df_rsi['gain'].rolling(periods).mean()
    df_rsi['avg_loss'] = df_rsi['loss'].abs().rolling(periods).mean()


    for i in range(periods, len(df_rsi['avg_gain'])):
        df_rsi['avg_gain'][i] = (df_rsi['avg_gain'][i - 1] * (periods - 1) + df_rsi['gain'][i]) / periods
        df_rsi['avg_loss'][i] = (df_rsi['avg_loss'][i - 1] * (periods - 1) + df_rsi['loss'].abs()[i]) / periods

    # Calculate the RSI
    df_rsi['rs'] = df_rsi['avg_gain'] / df_rsi['avg_loss']
    df_rsi['rsi'] = 100 - (100 / (1 + df_rsi['rs']))

    return df_rsi['rsi']
