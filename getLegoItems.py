import requests
import json
import time

def get_years(api_key):
    url = "https://brickset.com/api/v3.asmx/getYears"
    params = {
        "apiKey": api_key,
        "theme": ''
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        years = response.json()  # Assuming the API returns a JSON response
        return years
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def get_sets_for_year(api_key, user_hash, year, set_count):
    url = "https://brickset.com/api/v3.asmx/getSets"
    all_sets = []
    page = 1
    total_fetched = 0
    while total_fetched < set_count:
        params = {
            "apiKey": api_key,
            "userHash": user_hash,
            "params": json.dumps({
                "year": year,
                "pageSize": 500,
                "pageNumber": page,
                "extendedData": True
            })
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('status') != 'success':
                print(f"Failed to retrieve sets for year {year}, page {page}: {data.get('message')}")
                break
            sets = data.get('sets', [])
            all_sets.extend(sets)
            total_fetched += len(sets)
            print(f"Fetched {len(sets)} sets for year {year}, page {page}. Total so far: {total_fetched}/{set_count}")
            if len(sets) < 500:  # If less than 500 sets, we're done for this year
                break
            page += 1
            time.sleep(1)  # To avoid overwhelming the server with requests
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching sets for year {year}, page {page}: {e}")
            break
    return all_sets

def download_sets(api_key, user_hash, years_data):
    sets_by_year = {}
    for year_data in years_data:
        year = year_data['year']
        set_count = year_data['setCount']
        print(f"Processing year {year} with {set_count} sets...")
        sets = get_sets_for_year(api_key, user_hash, year, set_count)
        sets_by_year[year] = sets
        print(f"Finished processing year {year}. Total sets retrieved: {len(sets)}")
    return sets_by_year

def get_reviews_for_set(api_key, set_id):
    url = "https://brickset.com/api/v3.asmx/getReviews"
    params = {
        "apiKey": api_key,
        "setID": set_id
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('status') != 'success':
            print(f"Failed to retrieve reviews for set ID {set_id}: {data.get('message')}")
            return []
        reviews = data.get('reviews', [])
        print(f"Fetched {len(reviews)} reviews for set ID {set_id}")
        return reviews
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching reviews for set ID {set_id}: {e}")
        return []
    
def download_reviews_for_sets(api_key, sets_by_year):
    for year, sets in sets_by_year.items():
        print(f"Processing year {year} with {len(sets)} sets...")
        for i, set_data in enumerate(sets):
            if set_data.get("reviewCount", 0) > 0:
                set_id = set_data['setID']
                print(f"Downloading reviews for set ID {set_id} ({set_data['name']}) with {set_data['reviewCount']} reviews...")
                reviews = get_reviews_for_set(api_key, set_id)
                sets_by_year[year][i]['reviews'] = reviews  # Explicitly update the structure
                time.sleep(0.5)  # To avoid overwhelming the server with requests
            else:
                print(f"Skipping set ID {set_data['setID']} ({set_data['name']}) as it has no reviews.")
    return sets_by_year


def save_to_json(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data successfully saved to {filename}")
    except IOError as e:
        print(f"An error occurred while saving data to {filename}: {e}")

def load_from_json(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Data successfully loaded from {filename}")
        return data
    except IOError as e:
        print(f"An error occurred while loading data from {filename}: {e}")
        return None
    
if __name__ == "__main__":
    api_key = ""
    years = get_years(api_key)
    if years:
        print("Years fetched from API:")
        print(years)
        sets_by_year = download_sets(api_key, '', years['years'])
        save_to_json(sets_by_year, "sets.json")

        sets_with_reviews = download_reviews_for_sets(api_key, load_from_json("sets.json"))
        save_to_json(sets_with_reviews, "sets_reviews.json")

        sets_with_reviews = download_reviews_for_sets(api_key, load_from_json("sets.json"))
        save_to_json(sets_with_reviews, "sets_reviews.json")
    else:
        print("Failed to retrieve years.")

