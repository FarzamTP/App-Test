import pandas as pd
import numpy as np
import spacy
import re

from sentence_transformers.cross_encoder import CrossEncoder
from rank_bm25 import BM25Okapi
from unidecode import unidecode
from ast import literal_eval
from conf import EMBEDDING_ENDPOINT, EMBEDDING_API_KEY
from openai import AzureOpenAI
import streamlit as st
from stqdm import stqdm

pd.options.mode.chained_assignment = None

embedding_file_path = "data/LL_with_embeddings_text-embedding-3-large.csv"
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", )

embedding_client = AzureOpenAI(
    azure_endpoint=EMBEDDING_ENDPOINT,
    api_key=EMBEDDING_API_KEY,
    api_version="2024-02-01",
)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def read_ll_file():
    stqdm.pandas()
    df = pd.read_excel('data/LL.xlsx')
    return df


@st.cache_data
def load_embeddings(embedding_file_path):
    stqdm.pandas()
    df = pd.read_csv(embedding_file_path)
    df["embedding"] = df.embedding.progress_apply(literal_eval).apply(np.array)
    return df


nlp = spacy.load("en_core_web_lg")
st.session_state.first_page_load = False

# Load data into session state if not already loaded
if 'df_with_embedding' not in st.session_state:
    with st.spinner('Loading AI model...'):
        st.session_state.df_with_embedding = load_embeddings(embedding_file_path)
        st.session_state.ll_df = read_ll_file()
        st.session_state.first_page_load = True
    st.success('AI model loaded!')



def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = unidecode(text)
    text = text.replace('_x000D_', ' ').replace('\n', ' ')
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)


def filter_dataframe(df, include_terms, exclude_terms):
    exclude_terms = [term for term in exclude_terms if term]
    include_patterns = [re.compile(r'\b{}\b'.format(re.escape(term)), re.IGNORECASE) for term in include_terms]

    def matches_include_terms(text):
        return any(pattern.search(text) for pattern in include_patterns)

    include_mask = df['combined'].apply(matches_include_terms)

    if exclude_terms:
        exclude_patterns = [re.compile(r'\b{}\b'.format(re.escape(term)), re.IGNORECASE) for term in exclude_terms]

        def matches_exclude_terms(text):
            return any(pattern.search(text) for pattern in exclude_patterns)

        exclude_mask = df['combined'].apply(matches_exclude_terms)
        filtered_df = df[include_mask & ~exclude_mask]
    else:
        filtered_df = df[include_mask]

    if filtered_df.empty:
        return None

    return filtered_df


def get_embedding(text: str, model="text-embedding-3-large", **kwargs):
    text = text.replace("\n", " ")

    response = embedding_client.embeddings.create(input=[text], model=model, **kwargs)
    return response.data[0].embedding


def normalize_column(df, column_name, method='min-max'):
    if method == 'min-max':
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        df.loc[:, column_name] = (df[column_name] - min_val) / (max_val - min_val)
    elif method == 'z-score':
        mean_val = df[column_name].mean()
        std_val = df[column_name].std()
        df.loc[:, column_name] = (df[column_name] - mean_val) / std_val
    else:
        raise ValueError("Normalization method not supported. Use 'min-max' or 'z-score'.")
    return df


def create_suggestions_list(topic):
    weights = {'cross_score': 0.4, 'similarity': 0.3, 'bm25_score': 0.3}

    df_with_embedding = load_embeddings(embedding_file_path)
    df_copy = df_with_embedding.copy()

    penalty_words = ""

    embedding_model = "text-embedding-3-large"

    include_terms = [element.strip() for element in topic.split(',')]
    exclude_terms = [element.strip() for element in penalty_words.split(',')]

    filtered_df = filter_dataframe(df_copy, include_terms, exclude_terms)
    if filtered_df is None:
        return None

    preprocessed_topic = preprocess_text(topic)

    corpus = filtered_df.combined.to_list()
    sentence_combinations = [[preprocessed_topic, sentence] for sentence in corpus]
    scores = model.predict(sentence_combinations)
    filtered_df.loc[:, 'cross_score'] = scores

    query_embedding = get_embedding(preprocess_text(preprocessed_topic), model=embedding_model)
    filtered_df.loc[:, "similarity"] = filtered_df.embedding.apply(lambda x: cosine_similarity(x, query_embedding))

    filtered_df = normalize_column(filtered_df, 'similarity', method='min-max')
    filtered_df = normalize_column(filtered_df, 'cross_score', method='min-max')

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = preprocessed_topic.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)

    filtered_df.loc[:, 'bm25_score'] = bm25_scores
    filtered_df = normalize_column(filtered_df, 'bm25_score', method='min-max')

    filtered_df["similarity"] = np.ceil(filtered_df["similarity"] * 100)
    filtered_df["cross_score"] = np.ceil(filtered_df['cross_score'] * 100)
    filtered_df["bm25_score"] = np.ceil(filtered_df['bm25_score'] * 100)

    filtered_df.loc[:, "Final Score"] = np.ceil(
        weights['cross_score'] * filtered_df['cross_score'] +
        weights['similarity'] * filtered_df['similarity'] +
        weights['bm25_score'] * filtered_df['bm25_score']
    )
    filtered_df = filtered_df.sort_values(by='Final Score', ascending=False)[['LESSON_LEARNED_ID', 'cross_score', 'similarity', 'bm25_score', 'Final Score']]

    selected_columns = ['LESSON_LEARNED_ID', 'TITLE', 'DESCRIPTION', 'CORRECTIVE_PREVENTIVE_ACTION', 'LINK']
    columns_to_show = selected_columns + ['Final Score']

    merged_df = pd.merge(filtered_df, st.session_state.ll_df[selected_columns], on='LESSON_LEARNED_ID', how='left')[
        columns_to_show]
    merged_df["DESCRIPTION"] = merged_df["DESCRIPTION"].apply(lambda x: x.replace('_x000D_', '\n'))
    merged_df["CORRECTIVE_PREVENTIVE_ACTION"] = merged_df["CORRECTIVE_PREVENTIVE_ACTION"].apply(
        lambda x: x.replace('_x000D_', '\n'))
    return merged_df[columns_to_show]
