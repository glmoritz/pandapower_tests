import csv
from datetime import datetime, timedelta
from collections import defaultdict

input_filename = 'dados-carregamento.csv'
output_filename = 'dados-carregamento-normal.csv'
interval = timedelta(minutes=15)

# Dictionary to accumulate values per timestamp
accumulator = defaultdict(lambda: [])

with open(input_filename, newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile, delimiter=';')

    for row in reader:
        if len(row) < 6:
            continue  # Skip invalid lines

        date_str = row[2]
        start_str = row[3]
        end_str = row[4]

        try:
            start_dt = datetime.strptime(f'{date_str} {start_str}', '%d/%m/%Y %H:%M:%S')
            end_dt = datetime.strptime(f'{date_str} {end_str}', '%d/%m/%Y %H:%M:%S')
        except ValueError:
            continue

        # Round start time to next 15-minute mark
        aligned_start = start_dt.replace(minute=(start_dt.minute // 15) * 15, second=0)
        if aligned_start < start_dt:
            aligned_start += interval

        current = aligned_start
        while current <= end_dt:
            iso_time = current.isoformat()
            # Convert numeric fields from string to float, handling comma decimal
            try:
                values = [float(val.replace(',', '.')) for val in row[6:9]]
            except ValueError:
                values = [0.0] * (len(row) - 5)

            if iso_time in accumulator:
                accumulator[iso_time] = [a + b for a, b in zip(accumulator[iso_time], values)]
            else:
                accumulator[iso_time] = values

            current += interval

# Write output
with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    for timestamp in sorted(accumulator.keys()):
        # Format floats with 2 decimal digits, using comma as decimal separator
        formatted_values = [f'{-1*v:.2f}'.replace('.', '.') for v in accumulator[timestamp]]
        writer.writerow([timestamp] + formatted_values)
