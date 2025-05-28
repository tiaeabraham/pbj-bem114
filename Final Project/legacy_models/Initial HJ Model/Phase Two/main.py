from identify_pairs import calculate_correlation_pairs
# from purchase_timing import main

# ([sorted_pairs_total_corr], [sorted_pairs_five_day_corr])
sorted_pairs = calculate_correlation_pairs()

print(f'{sorted_pairs[0]}\n\n{sorted_pairs[1]}')
