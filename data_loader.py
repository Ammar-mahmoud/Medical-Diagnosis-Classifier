import csv
import random

def encode(value):
    mapping = {
        'yes': 1, 'no': 0,
        'normal': 1, 'abnormal': 0,
        'present': 1, 'notpresent': 0,
        'good': 1, 'poor': 0,
        'ckd': 1, 'notckd': 0
    }
    try:
        return float(value)
    except:
        return mapping.get(value.strip().lower(), 0)

def calculate_column_defaults(data_rows):
    n_cols = len(data_rows[0])
    columns = [[] for _ in range(n_cols)]

    for row in data_rows:
        for i in range(n_cols):
            val = row[i].strip().lower()
            if val != '':
                columns[i].append(val)

    defaults = []
    for col_vals in columns:
        try:
            nums = []
            for v in col_vals:
                nums.append(float(v))
            mean_val = sum(nums) / len(nums)
            defaults.append(str(mean_val))
        except:
            counts = {}
            for val in col_vals:
                counts[val] = counts.get(val, 0) + 1

            most_common = max(counts, key=counts.get)
            defaults.append(most_common)

    return defaults

def normalize_dataset(dataset):
    n_features = len(dataset[0]) - 2  # exclude ID and class
    for col in range(1, n_features + 1):
        col_values = [row[col] for row in dataset]
        min_val = min(col_values)
        max_val = max(col_values)
        if max_val == min_val:
            continue
        for row in dataset:
            row[col] = (row[col] - min_val) / (max_val - min_val)

def load_data(filename, percentage):
    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    headers = data[0][1:]
    raw_rows = [row[1:] for row in data[1:] if len(row[1:]) == len(headers)]

    column_defaults = calculate_column_defaults(raw_rows)

    filled_rows = []
    for row in raw_rows:
        new_row = []
        for i in range(len(row)):
            cell = row[i]
            if cell.strip() != '':
                new_row.append(cell)
            else:
                new_row.append(column_defaults[i])
        filled_rows.append(new_row)

    records = []
    for i in range(len(filled_rows)):
        row = filled_rows[i]
        encoded_row = [encode(cell) for cell in row]
        records.append([str(i + 1)] + encoded_row)

    random.shuffle(records)
    normalize_dataset(records)

    total = int(len(records) * percentage / 100)
    records = records[:total]
    split_index = int(0.75 * total)
    train_set = records[:split_index]
    test_set = records[split_index:]

    return train_set, test_set
