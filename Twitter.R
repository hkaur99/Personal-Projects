library(twitteR)
library(RCurl)
library(SnowballC)
library(tm)
library(wordcloud)


API_key='z7AbkHu8r9lpg78VBIuWF0XAL'
API_secret_key='dxDeR51kVuL26KcMCHclWBdYKU6QqGaV9JSfcLq9PRgXvvXXbI'
Access_Token='1192171907145256962-7O0qOHyfXaODM88TJ4G591De5nvxDC'
Access_Token_Secret='vH8SIfvZOLrh7eSmgPboT2JxDU0v0DWiYme9D4Yjv5MD2'

#handsaking mode or function
setup_twitter_oauth(API_key, API_secret_key, Access_Token, Access_Token_Secret)

MyTweets = searchTwitter('UEFA', n = 200, lang = 'en' )
MyTweets

TweetDf = do.call('rbind', lapply( MyTweets, as.data.frame))
 
TweetDf$text = sapply(TweetDf$text, function(row) iconv(row, 'latin1', 'ASCII', sub = ''))
Tweets = TweetDf$text
Tweets


corpus = Corpus(VectorSource(Tweets))#always character vector
corpus = tm_map(corpus, removeWords, stopwords('en'))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removeWords, c("rt",'the','have','are', '\\n\n'))
corpus = Corpus(VectorSource(corpus))

TDM = TermDocumentMatrix(corpus)
TDM = as.matrix(TDM)
TDM = sort(rowSums(TDM), decreasing = TRUE)
TDM = data.frame(word = names(TDM), freq = TDM)

head(TDM, n = 10)
wordcloud(words = TDM$word, freq = TDM$freq, min.freq = 1,
          max.words = 200,random.order = FALSE, rot.per = 0.30, 
          colors = brewer.pal(8, 'Dark2'))

barplot(TDM[1:10, ]$freq, las = 2, names.arg = TDM[1:10,]$word,
        col = 'lightblue', main = 'Moat Frequent Word',
        ylab = 'Frequency')