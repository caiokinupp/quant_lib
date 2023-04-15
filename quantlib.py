import pandas as pd
import numpy as np

def simple_moving_average(value_series, periods = 20):
    """
    Method to calculate SMA (Simple Moving Average) from a series of values.
    
    Args:
        value_series (pandas.Series):  A pandas Series ordered from oldest to 
            newest.
        periods (int): Periods of the calculation (default 20).

    Returns:
        pandas.Series: A new Series with calculated moving average.
    """
    
    return value_series.rolling(periods).mean()

def moving_median(value_series, periods = 20):
    """
    Method to calculate moving median from a series of values.
    
    Args:
        value_series (pandas.Series): A pandas Series ordered from oldest to 
            newest.
        periods (int): Periods of the calculation (default 20).

    Returns:
        pandas.Series: A new Series with calculated moving median.
    """
    
    return value_series.rolling(periods).median()

def moving_std(value_series, periods = 20):
    """
    Method to calculate moving standard deviation from a series of values.
    
    Args:
        value_series (pandas.Series): A pandas Series ordered from oldest to 
            newest.
        periods (int): Periods of the calculation (default 20).

    Returns:
        pandas.Series: A new Series with calculated moving standard deviation.
    """
    
    return value_series.rolling(periods).std()

def z_score(value_series, periods = 20):
    """
    Method to calculate the Z-Score. Is the distance from price to a SMA in 
    standard deviations.
    
    Args:
        value_series (pandas.Series): A pandas Series ordered from oldest to 
            newest.
        periods (int): Periods of the calculation (default 20).

    Returns:
        pandas.Series: A new Series with calculated z-score.
    """
    
    sma = simple_moving_average(value_series, periods)
    
    # Calculating distance between price and the MA
    distance_price_sma = value_series - sma

    # Calculating the moving standard deviation
    moving_stds = moving_std(value_series, periods)

    return distance_price_sma/moving_stds

def median_z_score(value_series, periods = 20):
    """
    Method to calculate the Z-Score. Is the distance from price to a SMA in 
    standard deviations.
    
    Args:
        value_series (pandas.Series): A pandas Series ordered from oldest to 
            newest.
        periods (int): Periods of the calculation (default 20).

    Returns:
        pandas.Series: A new Series with calculated z-score.
    """
    
    mm = moving_median(value_series, periods)

    # Calculating distance between price and the MA
    distance_price_mm = value_series - mm

    # Calculating the moving standard deviation
    moving_stds = moving_std(value_series, periods)

    return distance_price_mm/moving_stds


def high_low(value_series):
    """
    Method to define whether the current value is bigger or lower than the 
    previous value.
    0 -> Lateral
    1 -> High
    -1 -> Low
    
    Args:
        value_series (pandas.Series): A pandas Series ordered from oldest to 
            newest.

    Returns:
        pandas.Series: Series of values that define whether is going up or 
            down.
    """
    
    shifted_series = value_series.shift(1)
    
    series_diff = value_series - shifted_series
    
    series_diff[series_diff > 0] = 1
    series_diff[series_diff < 0] = -1
    series_diff[series_diff == 0] = 0
    
    return series_diff


def sequence_counter(value_series):
    """
    Method to count sequence of periods in the same direction.

    Args:
        value_series (pandas.Series): A pandas Series ordered from oldest to 
            newest.
    
    Returns:
        pandas.Series: Sequence size on same direction.
    """
    values_directions = high_low(value_series)
    shifted_values_directions = values_directions.shift(1)
    # Comparing if has direction change
    has_direction_change = values_directions.ne(shifted_values_directions)
    # Calculate cumsum of direction change to obtain the references for groupby
    group_references = has_direction_change.cumsum()
    # Group by positions where have direction change
    grouped_values = values_directions.groupby(group_references)

    # Cumulative count to calculate each sequence (starting by 1)
    return (grouped_values.cumcount())+1


def candle_body_proportion(open, high, low, close):
    """
    Method to calculate proportion of the candle's body and the shadows.

    Args:
        open, high, low, close (pandas.Series): A pandas Series ordered from 
            oldest to newest of the candle's prices.
    
    Returns:
        pandas.Series: Proportions of candle's body and shadows.
    """
    
    df_body = pd.DataFrame({'open': open, 'close': close})
    max_body = df_body.max(axis=1)
    min_body = df_body.min(axis=1)
    
    candle_range = high-low
    
    top_shadow = (high-max_body)/candle_range
    body = (max_body-min_body)/candle_range
    bottom_shadow = (min_body-low)/candle_range
    
    # Replacing inf values by 1 
    # (Inf values are result from division by 0)
    top_shadow.replace([np.inf, -np.inf, np.nan], 1, inplace=True)
    body.replace([np.inf, -np.inf, np.nan], 1, inplace=True)
    bottom_shadow.replace([np.inf, -np.inf, np.nan], 1, inplace=True)
    
    return top_shadow, body, bottom_shadow


def candle_color(open, close):
    """
    Method to calculate the direction of the candles.

    Args:
    open, close (pandas.Series): A pandas Series ordered from oldest to newest 
        of the candle's prices.
    
    Returns:
        pandas.Series: Candle's direction.
    """
    return np.where(close > open, 1, -1)


def rsi(value_series, periods = 15):
    """
    Method to calculate RSL indicator.

    Args:
    value_series (pandas.Series): A pandas Series ordered from oldest to 
        newest.
    periods (int): Periods of the calculation (default 15).
    
    Returns:
        pandas.Series: Calculated RSI indicator.
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
        df_rsi['avg_gain'][i] = df_rsi['avg_gain'][i - 1] * (periods - 1)
        df_rsi['avg_gain'][i] = df_rsi['avg_gain'][i] + df_rsi['gain'][i]
        df_rsi['avg_gain'][i] = df_rsi['avg_gain'][i]/periods
        
        df_rsi['avg_loss'][i] = df_rsi['avg_loss'][i - 1] * (periods - 1) 
        df_rsi['avg_loss'][i] = df_rsi['avg_loss'][i] + df_rsi['loss'].abs()[i]
        df_rsi['avg_loss'][i] = df_rsi['avg_loss'][i]/ periods

    # Calculate the RSI
    df_rsi['rs'] = df_rsi['avg_gain'] / df_rsi['avg_loss']
    df_rsi['rsi'] = 100 - (100 / (1 + df_rsi['rs']))

    return df_rsi['rsi']

def rsl(value_series, periods = 15):
    """
    Method to calculate RSI indicator.

    Args:
    value_series (pandas.Series): A pandas Series ordered from oldest to 
        newest.
    periods (int): Periods of the calculation (default 15).
    
    Returns:
        pandas.Series: Calculated RSL indicator.
    """
    
    sma = simple_moving_average(value_series, periods)


    return np.round((value_series/sma)-1, 4)*100