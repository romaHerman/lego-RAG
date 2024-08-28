import json
import csv
from collections import defaultdict

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Load JSON data
with open('sets_reviews.json', 'r') as f:
    data = json.load(f)

# Prepare data for main CSV and reviews CSV
main_data = []
reviews_data = []

for year, sets in data.items():
    for set_data in sets:
        # Extract and remove reviews from set_data
        reviews = set_data.pop('reviews', [])
        
        # Flatten and add main set data
        flat_set = flatten_dict(set_data)
        flat_set['year'] = year
        main_data.append(flat_set)
        
        # Process reviews
        for review in reviews:
            flat_review = flatten_dict(review)
            flat_review['setID'] = set_data['setID']  # Add setID to link reviews to sets
            flat_review['year'] = year
            reviews_data.append(flat_review)

# Function to write data to CSV
def write_to_csv(data, filename):
    if not data:
        print(f"No data to write to {filename}")
        return
    
    keys = set().union(*(d.keys() for d in data))
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(keys))
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"CSV file '{filename}' has been created successfully.")

# Write main data to CSV
write_to_csv(main_data, 'lego_sets_main.csv')

# Write reviews to CSV
write_to_csv(reviews_data, 'lego_sets_reviews.csv')