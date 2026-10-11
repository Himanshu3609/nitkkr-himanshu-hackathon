import os
import pandas as pd
import kagglehub
import glob
import re

def clean_tweet(text):
    """Basic preprocessing for tweets."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'#\w+', '', text) # Remove hashtags
    return text.strip()

def download_datasets(data_dir="../data"):
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading Financial News Sentiment dataset...")
    news_path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    
    print("Downloading Tweet Sentiment Impact dataset...")
    tweets_path = kagglehub.dataset_download("thedevastator/tweet-sentiment-s-impact-on-stock-returns")
    
    return news_path, tweets_path

def load_financial_news(news_path, num_samples=10):
    news_file = os.path.join(news_path, "all-data.csv")
    try:
        news_df = pd.read_csv(news_file, encoding='latin-1', header=None, names=['sentiment', 'text'])
        return news_df.sample(num_samples)['text'].tolist()
    except Exception as e:
        print(f"Error loading news data: {e}")
        return ["Markets rally as inflation cools down unexpectedly."] * num_samples

def load_tweets(tweets_path, num_samples=10):
    sample_texts = []
    try:
        csv_files = glob.glob(f"{tweets_path}/*.csv")
        if csv_files:
            tweets_df = pd.read_csv(csv_files[0])
            
            # Find the text column
            text_col = None
            for col in tweets_df.columns:
                if 'text' in col.lower() or 'tweet' in col.lower():
                    text_col = col
                    break
            
            if text_col:
                # Clean tweets and drop empties
                tweets_df['cleaned_text'] = tweets_df[text_col].apply(clean_tweet)
                valid_tweets = tweets_df[tweets_df['cleaned_text'] != '']
                sample_texts = valid_tweets['cleaned_text'].sample(num_samples).tolist()
    except Exception as e:
        print(f"Error loading tweets data: {e}")
        sample_texts = ["Excited about the new product launch today! $AAPL"] * num_samples
        
    return sample_texts

class DataStreamer:
    """Simulates a real-time data streaming pipeline."""
    def __init__(self, news_path, tweets_path):
        self.news = load_financial_news(news_path, num_samples=50)
        self.tweets = load_tweets(tweets_path, num_samples=50)
        
        # Interleave news and tweets to create a unified feed
        self.unified_feed = []
        for n, t in zip(self.news, self.tweets):
            self.unified_feed.append({"source": "News", "text": n})
            self.unified_feed.append({"source": "Twitter", "text": t})
            
        self.current_idx = 0
        
    def get_next(self):
        if self.current_idx < len(self.unified_feed):
            item = self.unified_feed[self.current_idx]
            self.current_idx += 1
            return item
        return None
