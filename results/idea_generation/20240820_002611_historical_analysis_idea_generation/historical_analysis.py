import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def scrape_arxiv(query, start_year, end_year):
    base_url = "http://export.arxiv.org/api/query?"
    results = []
    
    for year in range(start_year, end_year + 1):
        query_url = f"{base_url}search_query={query}+AND+submittedDate:[{year}01010000+TO+{year}12312359]&start=0&max_results=1000"
        response = requests.get(query_url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        for entry in soup.find_all("entry"):
            title = entry.title.text
            summary = entry.summary.text
            published = entry.published.text
            results.append({"title": title, "summary": summary, "published": published})
    
    return pd.DataFrame(results)

def analyze_historical_data(df):
    df['year'] = pd.to_datetime(df['published']).dt.year
    topic_counts = df['summary'].str.split(expand=True).stack().value_counts()
    return topic_counts

def get_historical_insights(query, start_year, end_year):
    df = scrape_arxiv(query, start_year, end_year)
    topic_counts = analyze_historical_data(df)
    return topic_counts
