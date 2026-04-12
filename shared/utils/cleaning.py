import re
from bs4 import BeautifulSoup
import html
import urllib.parse


def remove_special_characters(text):
    # Remove HTML tags using BeautifulSoup
    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

    except Exception as e:
        print(f"Error decoding text: {e}")

    text = html.unescape(text)
    # Decode the URL-encoded string
    text = urllib.parse.unquote(text)
    # General case to clean Unicode escape sequences
    try:
        text = bytes(text, "utf-8").decode("unicode_escape")
    except Exception as e:
        print(f"Error decoding text: {e}")

    # Remove URLs using regular expressions
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]\[URL:.*?\]', '', text)  # Remove [LINKED_TEXT] and [URL] if any
    text = re.sub(r'\[.*?\]', '', text)  # Remove any leftover [text] placeholders
    emoji_pattern = re.compile("["  
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text) 
    text = re.sub(r'#\w+', '', text)  
    text = re.sub(r'[^\w\s\d.,!?\'"()-;]', '', text) 
    text = re.sub(r'\s+([.,!?;])', r'\1', text)
    text = re.sub(r'([.,!?;])(\S)', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text