import pandas as pd
import numpy as np
import unicodedata
import MeCab
from collections import Counter
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import PyPDF2
import unidic
import gensim
from gensim import corpora
from pprint import pprint
from textblob import TextBlob

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_japanese_text(text):
    # Japanese Regular Expression Patterns
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F\n\r\t\s]+')
    # Extract only Japanese matching the regular expression pattern
    japanese_text = ''.join(japanese_pattern.findall(text))
    return japanese_text

# Extract text from the first PDF file
pdf_path = "data.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

# Extract only Japanese text
japanese_text = extract_japanese_text(pdf_text)

# Combine both Japanese texts and output
combined_japanese_text = japanese_text
print(combined_japanese_text)

# Text that we analyze
text = combined_japanese_text
# Function Settings
def mecab_tokenizer(text):

    replaced_text = unicodedata.normalize("NFKC",text)
    replaced_text = replaced_text.upper()
    replaced_text = re.sub(r'[【】 () （） 『』「」]', '' ,replaced_text) # Removal of【】()「」『』
    replaced_text = re.sub(r'[\[\［］\]]', ' ', replaced_text)   # Removal of ［］
    replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)  # Removal of mention
    replaced_text = re.sub(r'\d+\.*\d*', '', replaced_text) # 数字を0にする
    mecab = MeCab.Tagger()
    parsed_lines = mecab.parse(replaced_text).split("\n")[:-2]

    # Get surface system
    surfaces = [l.split("\t")[0] for l in parsed_lines]
    # Get part of speech
    pos = [l.split("\t")[1].split(",")[0] for l in parsed_lines]
    # Narrow down to nouns, verbs, and adjectives
    target_pos = ["名詞", "動詞", "形容詞"]
    token_list = [t for t , p in zip(surfaces, pos) if p in target_pos]

    # Exclude hiragana-only words
    kana_re = re.compile("^[ぁ-ゖ]+$")
    token_list = [t for t in token_list if not kana_re.match(t)]

    # Join each token with a little space (' ')
    return ' '.join(token_list)

# Function Execution
words = mecab_tokenizer(text)
print(words)
    
# Setting of colors
colormap="Paired"

wordcloud = WordCloud(
    background_color="white",
    width=800,
    height=800,
    font_path = r'C:\Windows\Fonts\YuGothB.ttc',
    colormap = colormap,
    stopwords=["する", "ある", "こと", "ない"],
    max_words=100,
    ).generate(words)

plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("wordcloud_image.png", format="png")
plt.show()


#-----Q4-----#
word_list = words.split()
dictionary = corpora.Dictionary([word_list])
corpus = [dictionary.doc2bow(word_list)]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=10)

pprint(lda_model.print_topics())

#-----Q5-----#
# Emotion analysis using TextBlob
blob = TextBlob(words)
sentiment_score = blob.sentiment.polarity

if sentiment_score > 0:
    print("This is a positive feeling")
elif sentiment_score < 0:
    print("This is a negetive feeling")
else:
    print("This is a neutral feeling")
