import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

text = """Acting is a powerful form of art that allows people to express emotions, tell stories, and connect with others. It is not simply about memorizing dialogues or pretending to be someone else; it is about understanding a character and bringing that character to life. An actor uses facial expressions, body language, voice, and emotions to make a performance believable. Acting also requires confidence, creativity, patience, and the ability to observe people and understand their feelings. Through acting, a person can experience different perspectives and understand situations that may be very different from their own life. It can also help improve communication skills and overcome stage fear. Whether it is performed in a movie, theatre, television show, or on a small stage, acting has the power to make audiences laugh, cry, think, and feel connected to a story. Good actors make their characters feel real and memorable. Acting is also a continuous learning process because every character presents a new challenge. For me, acting is an exciting way to express myself and explore different personalities. It gives me the freedom to be creative, emotional, and confident while entertaining others and sharing meaningful stories."""

print(text)

# Tokenization of words
words = word_tokenize(text.lower())
print(words)

# Remove stopwords
stop_words = set(stopwords.words('english'))

filtered_words = [
    word for word in words
    if word.isalpha() and word not in stop_words
]

print(filtered_words)

# Count word frequency
word_freq = Counter(filtered_words)
print(word_freq)

# Create WordCloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate_from_frequencies(word_freq)

# Display WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()