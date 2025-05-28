# IDENTIFY_PAIRS
# User-editable variables
yfinance_stock_price_directory = 'yfinance'
total_correlation_threshold = 0.4
five_day_correlation_threshold = 0.4
price_difference_threshold = 10
price_change_percentage_threshold = 0.08

# PURCHASE_TIMING
'''
DEFINITIONS
Stock1 is always the increases stock
Stock2 is always the trailing stock
'''

'''
We want to look for stocks pairs where:
    Stock 1 has inceased by at least the percent_increase_threshold_high
    Stock 2 has not increased by more than percent_increase_threshold_low
    Stock 2 has not decreased by more than decrease_threshold
'''
percent_increase_threshold_high = 0.05
percent_increase_threshold_low = 0.02
decrease_threshold = -0.02
