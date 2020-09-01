# rap_songs
Natural Language Processing and KMeans clustering to group rap songs based on the words used in lyrics.

**genius_songs.py:** 
Retrive list of artists and their songs using the genius.com API and the LyricsGenius wrapper. Will require your own API token.

**rap_genius.ipynb:** 
Jupyter Notebook performing NLP and clustering on rap song lyrics

## Cleaning up text:
  * Convert all charcters to lowercase
  * Remove punctuation
  * Remove stop words
  * Remove numbers
  * Remove words containing only 1 character
  * Lemmatize words (converting words to their root forms)
 
## Basic Text Analysis
<img src="/images/top words.png">

<img src="/images/top artists unique words.png">

## Transforming words into a TDIDF Vector
Doing this essentially converts each unique word in the text into a feature and counting its frequency. We will be excluding words that occur in less than 3% of all songs.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.03, ngram_range =(1,2))
X = vectorizer.fit_transform(df['lyrics_clean']).toarray()
X.shape
```

## Unsupervised learning using KMeans Clustering
It was found that 5 clusters was the optimal number of clusters to group the songs in based on their lyrics. This was determined by calculating the mean silhouette score of the clusters. Although not very high, the mean score was the greatest when the songs were split into 5 clusters. More detail on this methodology can be found in the [Jupyter Notebook](https://github.com/austyngo/rap_songs/blob/master/rap_genius.ipynb).
<img src= "/images/best cluster.png">
           
## Results Analysis
### How the clusters differ between artists.
J. Cole

<img src= "/images/j cole cluster.png">
     
Drake

<img src= "/images/drake clusters.png">
