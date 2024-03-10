# Task 21 - Capstone Project - NLP Applications - Sentiment Analysis

# This programme performs sentiment analysis on Amazon Reviews dataset. 
# It also compares similarity scores between two randomly chosen reviews
# in the datset.


import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_md")

df = pd.read_csv("D:\Rupa_CoGrammar\Rupa_python_code_files\Task submission Codes\Task 21 - Capstone Project  NLP Applications\Amazon_product_reviews.csv", sep = ',', dtype={1: str, 10: str})


def find_sentiment(sample_rev):
    '''Returns sentiment label and polarity
        score of a sample review.'''

    no_stop_reviews = ""
    sentiment_label = ""

    review_doc = nlp(sample_rev)

    # Remove all tokens of stopwords and punctuations.
    no_stop_reviews_list = [token.text for token in review_doc if not 
                            token.is_stop and not token.is_punct]

    # Create string from the tokens list to work with TextBlob.
    no_stop_reviews = " ".join(no_stop_reviews_list)

    review_text = TextBlob(no_stop_reviews)

    review_pol = review_text.sentiment.polarity
    
    if review_pol > 0.5 and review_pol <= 1:
        sentiment_label = "Very Positive"
    if review_pol > 0 and review_pol <= 0.5:
        sentiment_label = "Positive"
    elif review_pol == 0:
        sentiment_label = "Neutral"
    elif review_pol < 0 and review_pol >= -0.5:
        sentiment_label = "Negative" 
    elif review_pol < -0.5 and review_pol >= -1:
        sentiment_label = "Very Negative" 

    return(sentiment_label, review_pol)


def find_similarity(sample_review_1, sample_review_2):
    '''Returns similarity score and similarity label
    between two sample reviews.'''
    
    sample_review_1_doc = nlp(sample_review_1)
    sample_review_2_doc = nlp(sample_review_2)

    sim_score = sample_review_1_doc.similarity(sample_review_2_doc)
    sim_label = ""

    if sim_score > 0.75 and sim_score < 1:
        sim_label = "Very Similar"
    elif sim_score > 0.5 and sim_score < 0.75:
        sim_label = "Similar"
    elif sim_score > 0.25 and sim_score < 0.5:
        sim_label = "Slightly Similar"
    elif sim_score > 0 and sim_score < 0.25:
        sim_label = "Not Similar"

    return(sim_score, sim_label)


reviews_data = df["reviews.text"]

# Drop all rows with null values.
no_null_data = df.dropna(subset=["reviews.text"])

# Select one row randomly.
one_row_df = no_null_data.take(np.random.permutation(len(no_null_data))[:1])

# Store the review text in variable sample_review.
sample_review = one_row_df.iat[0,16]
print(f"\nRandomly chosen sample review: '{sample_review}'\n")

sentiment_analysis = find_sentiment(sample_review)

print(f"Sentiment Analysis for the above review is {sentiment_analysis[0]} \
with polarity score of {sentiment_analysis[1]}.\n")

# Select two random sample reviews from the cleaned data.
nth_row_df = no_null_data.take(np.random.permutation(len(no_null_data))[:1])
sample_review_n = nth_row_df.iat[0,16]

mth_row_df = no_null_data.take(np.random.permutation(len(no_null_data))[:1])
sample_review_m = mth_row_df.iat[0,16]

review_similarity = find_similarity(sample_review_n, sample_review_m)

print("Comparing similarity of two random sample reviews:\n")
print(f"Sample Review 1: '{sample_review_n}'\n")
print(f"Sample Review 2: '{sample_review_m}'\n")
print(f"The above two reviews are {review_similarity[1]} with similarity \
score of {review_similarity[0]}.\n")
print()
