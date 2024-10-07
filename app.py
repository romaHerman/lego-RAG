import streamlit as st
from openai import OpenAI
import requests
import json
import psycopg2
from elasticsearch import Elasticsearch
import time
import os

from elasticsearch.exceptions import ConnectionError, RequestError

import re
from datetime import datetime
import spacy

from sentence_transformers import SentenceTransformer

from tqdm.auto import tqdm
from elasticsearch.helpers import bulk


# Initialize OpenAI API (Make sure to set your API key in the environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)
#ES index name
index_name = "lego_sets"

# Initialize Elasticsearch
es_client = Elasticsearch('http://localhost:9200')

# Initialize PostgreSQL connection
# conn = psycopg2.connect(
#     dbname="your_database",
#     user="your_username",
#     password="your_password",
#     host="localhost"
# )

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

model_name = 'all-mpnet-base-v2'#multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)

prompt_template = """
You're a lego shop assistant. Answer the QUESTION based on the CONTEXT from Lego sets database.
Use only the facts from the CONTEXT when answering the QUESTION. Return brief description of the set bsaed on the sample data use bricksetURL to provde it's link
If there is no answer in CONTEXT reply: please paraphrase your question 

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

# Define the mapping for the index
mapping = {
    "properties": {
        "name_vector": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine",
        },
        "subtheme_vector": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine",
        },
        "category_vector": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine",
        },
        "theme_vector": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine",
        },
        "tags_vector": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine",
        },
        "description_vector": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine",
        },
        "review_vector": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine",
        },
        "year_vector": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine",
        },
        "setID": {"type": "long"},
        "number": {"type": "keyword"},
        "numberVariant": {"type": "integer"},
        "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "year": {"type": "integer", "null_value": 0},
        "theme": {"type": "keyword",  "null_value": ""},
        "themeGroup": {"type": "keyword",  "null_value": ""},
        "category": {"type": "keyword", "null_value": ""},
        "released": {"type": "boolean"},
        "pieces": {"type": "integer"},
        "minifigs": {"type": "integer", "null_value": 0},
        "image": {
            "properties": {
                "thumbnailURL": {"type": "keyword"},
                "imageURL": {"type": "keyword"}
            }
        },
        "bricksetURL": {"type": "keyword"},
        "collection": {"type": "object"},
        "collections": {
            "properties": {
                "ownedBy": {"type": "integer"},
                "wantedBy": {"type": "integer"}
            }
        },
        "LEGOCom": {
            "properties": {
                "US": {
                    "properties": {
                        "retailPrice": {"type": "float"},
                        "dateFirstAvailable": {"type": "date"},
                        "dateLastAvailable": {"type": "date"}
                    }
                },
                "UK": {
                    "properties": {
                        "retailPrice": {"type": "float"},
                        "dateFirstAvailable": {"type": "date"},
                        "dateLastAvailable": {"type": "date"}
                    }
                },
                "CA": {
                    "properties": {
                        "retailPrice": {"type": "float"},
                        "dateFirstAvailable": {"type": "date"},
                        "dateLastAvailable": {"type": "date"}
                    }
                },
                "DE": {"type": "object"}
            }
        },
        "rating": {"type": "float", "null_value": 0.0},
        "reviewCount": {"type": "integer"},
        "packagingType": {"type": "keyword"},
        "availability": {"type": "keyword"},
        "instructionsCount": {"type": "integer"},
        "additionalImageCount": {"type": "integer"},
        "ageRange": {
            "properties": {
                "min": {"type": "integer"},
                "max": {"type": "integer"}
            }
        },
        "dimensions": {
            "properties": {
                "height": {"type": "float"},
                "width": {"type": "float"},
                "depth": {"type": "float"},
                "weight": {"type": "float"}
            }
        },
        "barcode": {
            "properties": {
                "EAN": {"type": "keyword"},
                "UPC": {"type": "keyword"}
            }
        },
        "extendedData": {
            "properties": {
                "tags": {"type": "keyword"},
                "description": {"type": "text"}
            }
        },
        "lastUpdated": {"type": "date"},
        "reviews": {
            "type": "nested",
            "properties": {
                "author": {"type": "keyword"},
                "datePosted": {"type": "date"},
                "rating": {
                    "properties": {
                        "overall": {"type": "float"},
                        "parts": {"type": "float"},
                        "buildingExperience": {"type": "float"},
                        "playability": {"type": "float"},
                        "valueForMoney": {"type": "float"}
                    }
                },
                "title": {"type": "text"},
                "review": {"type": "text"},
                "HTML": {"type": "boolean"}
            }
        },
    }
}


def extract_search_parameters(query):
    params = {
        'user_query': query,
        'year_from': None,
        'year_to': None,
        'pieces_min': None,
        'pieces_max': None
    }
    
    # Extract years
    year_pattern = r'\b(19\d{2}|20\d{2})\b'
    years = re.findall(year_pattern, query)
    if years:
        year_ints = [int(y) for y in years]
        params['year_from'] = min(year_ints)
        params['year_to'] = max(year_ints)
        query = re.sub(year_pattern, '', query)
    
    # Extract date ranges
    date_patterns = [
        (r'\b(?:from|since|after)\s+(19\d{2}|20\d{2})\b', 'year_from'),
        (r'\b(?:to|until|before)\s+(19\d{2}|20\d{2})\b', 'year_to'),
        (r'\b(?:in|during|around)\s+(19\d{2}|20\d{2})\b', 'year_specific')
    ]
    for pattern, param in date_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            if param == 'year_specific':
                params['year_from'] = year
                params['year_to'] = year
            else:
                params[param] = year
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
    
    # Handle cases where only one year is specified
    current_year = datetime.now().year
    if params['year_from'] and not params['year_to']:
        params['year_to'] = min(params['year_from'] + 5, current_year)
    elif params['year_to'] and not params['year_from']:
        params['year_from'] = max(1949, params['year_to'] - 5)
    
    # Extract piece count
    piece_pattern = r'\b(\d+)(?:\s*-\s*(\d+))?\s*pieces?\b'
    piece_match = re.search(piece_pattern, query, re.IGNORECASE)
    if piece_match:
        params['pieces_min'] = int(piece_match.group(1))
        params['pieces_max'] = int(piece_match.group(2)) if piece_match.group(2) else params['pieces_min']
        query = re.sub(piece_pattern, '', query, flags=re.IGNORECASE)
    
    # Clean up the query
    params['user_query'] = ' '.join(query.split())
    
    return params

def build_elasticsearch_query(params):
    input_embedding = model.encode(params['user_query'])
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": params['user_query'],
                            "fields": ["name", "theme", "subtheme", "category"],
                            "type": "best_fields",
                            "fuzziness": "2",
                            "operator": "or"
                        }
                    },
                    {
                        "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": """
                            double score = 0.0;
                            double vector_score = 0.0;
                            int vector_count = 0;
                            
                            if (doc.containsKey('name_vector') && !doc['name_vector'].empty) {
                                vector_score += cosineSimilarity(params.query_vector, 'name_vector');
                                vector_count++;
                            }
                            if (doc.containsKey('theme_vector') && !doc['theme_vector'].empty) {
                                vector_score += cosineSimilarity(params.query_vector, 'theme_vector');
                                vector_count++;
                            }
                            if (doc.containsKey('subtheme_vector') && !doc['subtheme_vector'].empty) {
                                vector_score += cosineSimilarity(params.query_vector, 'subtheme_vector');
                                vector_count++;
                            }
                            if (doc.containsKey('description_vector') && !doc['description_vector'].empty) {
                                vector_score += cosineSimilarity(params.query_vector, 'description_vector');
                                vector_count++;
                            }

                            if (doc.containsKey('review_vector') && !doc['review_vector'].empty) {
                                vector_score += cosineSimilarity(params.query_vector, 'review_vector');
                                vector_count++;
                            }
                            
                            if (vector_count > 0) {
                                score = vector_score + 1;
                            }

                            return score;
                            """,
                            "params": {
                                "query_vector": input_embedding,
                            }
                    }
                    }
                    }
                ],
                "filter": [
                ]
            },
        },
        "aggs": {
            "themes": {
                "terms": {
                    "field": "theme.keyword",
                    "size": 10
                }
            },
            "years": {
                "date_histogram": {
                    "field": "year",
                    "calendar_interval": "year"
                }
            },
        },
        "size": 15
    }

    # Add year filter if specified
    if params['year_from'] is not None or params['year_to'] is not None:
        year_filter = {"range": {"year": {}}}
        if params['year_from'] is not None:
            year_filter["range"]["year"]["gte"] = params['year_from']
        if params['year_to'] is not None:
            year_filter["range"]["year"]["lte"] = params['year_to']
        query["query"]["bool"]["filter"].append(year_filter)

    # Add piece count filter if specified
    if params['pieces_min'] is not None or params['pieces_max'] is not None:
        piece_filter = {"range": {"pieces": {}}}
        if params['pieces_min'] is not None:
            piece_filter["range"]["pieces"]["gte"] = params['pieces_min']
        if params['pieces_max'] is not None:
            piece_filter["range"]["pieces"]["lte"] = params['pieces_max']
        query["query"]["bool"]["filter"].append(piece_filter)

    return query

def elastic_search_vector(user_input):
    doc = nlp(user_input)
    cleaned_query = []
    for ent in doc:
        if ent.is_stop == False:
            cleaned_query.append(ent.text)
    user_input = ' '.join(cleaned_query)
    params = extract_search_parameters(user_input) 
    es_query = build_elasticsearch_query(params)

    # Execute the search query
    try:
        response = es_client.search(index=index_name, body=es_query)
    except ConnectionError as e:
        print(f"ConnectionError during search: {e}")
        return str(e)
    except RequestError as e:
        print(f"RequestError during search: {e}")
        print("This might be due to an invalid query structure or non-existent fields.")
        return f"RequestError during search: {e}"

    result_docs = [hit['_source'] for hit in response['hits']['hits']]
    return result_docs

def build_prompt(query, search_results):
    context = ""
    #+ ''.join(doc.get("extendedData", "")) + ''.join(doc.get("reviews", "")) 
    for doc in search_results:
        #context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
        context = context + doc["name"] + doc["bricksetURL"] + str(doc["year"]) + doc["theme"] + doc["category"] + f"\n\n"
        if "minifigs" in doc:
            context = context + str(doc["minifigs"])
        if "themeGroup" in doc:
            context = context + doc["themeGroup"]
        if "reviews" in doc:
            reviews_str = "\n\n".join(
                f"Author: {review['author']}\nDate: {review['datePosted']}\nTitle: {review['title']}\nReview: {review['review']}"
                for review in doc["reviews"]
            )
            context = context + reviews_str
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    response = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )
    return response#.choices[0].message.content

def query_openai(prompt):
    start_time = time.time()
    response = llm(prompt)
    end_time = time.time()
    
    answer = response.choices[0].message.content.strip()
    cost = response.usage.total_tokens * 0.00250 / 1000
    duration = end_time - start_time
    
    return answer, cost, duration

def query_ollama(prompt):
    start_time = time.time()
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'llama2',
        'prompt': prompt
    })
    end_time = time.time()
    
    answer = response.json()['response']
    duration = end_time - start_time
    
    return answer, 0, duration  # Ollama is free, so cost is 0

def store_metrics(model, prompt, answer, cost, duration):
    # with conn.cursor() as cur:
    #     cur.execute("""
    #         INSERT INTO metrics (model, prompt, answer, cost, duration)
    #         VALUES (%s, %s, %s, %s, %s)
    #     """, (model, prompt, answer, cost, duration))
    # conn.commit()
    return

def index_document(doc):
    es_client.index(index="rag_index", body=doc)

def search_documents(query):
    result = es_client.search(index="rag_index", body={"query": {"match": {"content": query}}})
    return result['hits']['hits']

# Streamlit UI
st.title("RAG Streamlit App")

# Model selection
model = st.selectbox("Select Model", ["OpenAI", "Ollama"])

# User prompt
user_prompt = st.text_area("Enter your prompt:")

if st.button("Submit"):
    if model == "OpenAI":
        answer, cost, duration = query_openai(user_prompt)
    else:
        answer, cost, duration = query_ollama(user_prompt)
    
    st.write("Answer:", answer)
    st.write(f"Cost: ${cost:.4f}")
    st.write(f"Duration: {duration:.2f} seconds")
    
    # Store metrics
    #store_metrics(model, user_prompt, answer, cost, duration)
    
    # Thumbs up/down feedback
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍"):
            st.write("Thank you for your positive feedback!")
    with col2:
        if st.button("👎"):
            st.write("We're sorry the response wasn't helpful. We'll work on improving!")

def load_from_json(filename):
    try:
        with open(filename, 'r', encoding="utf-8") as f:
            data = json.load(f)
        print(f"Data successfully loaded from {filename}")
        return data
    except IOError as e:
        print(f"An error occurred while loading data from {filename}: {e}")
        return None
# Function to read documents from the backup file
def read_documents(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line.strip())

# Function to prepare documents for bulk indexing
def doc_generator(documents):
    for doc in documents:
        yield {
            "_index": index_name,
            "_source": doc
        }
# Elasticsearch indexing (only done once)
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
    

if not st.session_state.indexed:
    es_client.options(ignore_status=[400,404]).indices.delete(index=index_name)
    es_client.indices.create(index=index_name, body={"mappings": mapping})

    #lego_doc = load_from_json("sets_reviews.json")

    # Flattening the structure
    #flattened_lego_doc = []

    # for year, sets in lego_doc.items():
    #     flattened_lego_doc.extend(sets)

    # Read documents from the backup file
    documents = read_documents('lego_sets_backup.json')

    # Bulk index the documents
    success, failed = bulk(es_client, doc_generator(tqdm(documents, desc="Indexing documents")), stats_only=True)

    print(f"Indexed {success} documents successfully. {failed} documents failed.")

    st.session_state.indexed = True
    st.write("Elasticsearch index created!")

# Note: Grafana setup is not included in this code as it's typically set up separately
# You would need to configure Grafana to connect to your PostgreSQL database
# and create dashboards to visualize the metrics stored in the 'metrics' table.
