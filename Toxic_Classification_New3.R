#Load Libraries
library(dplyr)
library(ggplot2)
library(wordcloud)
library(tm)
library(stringr)
library(caret)
library(xgboost)
library(Matrix)
library(pROC)
library(glmnet)

#Load data
train<- read.csv("D:/Data Science/Kaggle/Jigsaw-toxic-comment-classification-challenge/Data/New/train.csv")
test<- read.csv("D:/Data Science/Kaggle/Jigsaw-toxic-comment-classification-challenge/Data/New/test.csv")

#Data Exploration

#Variable Identification
nrow(train)
nrow(test)

colnames(train)
colnames(test)



str(train)
str(test)

train$comment_text<-as.character(train$comment_text)
test$comment_text<-as.character(test$comment_text)



#Treating missing values

colSums(train=='')
colSums(is.na(train))

colSums(test=='')
colSums(is.na(test))

#no missing value


#feature Engineering
colSums(train[,3:8]=="1")
barplot(colSums(train[,3:8]=="1"))


# function for wordcloud
comment_func <- function(comm_text){
  # #Create corpus from text 
  docs_toxic<- Corpus(VectorSource(train$comment_text))
  #inspect(docs_toxic)
  
  ##data processing
  docs_toxic <- tm_map(docs_toxic, content_transformer(tolower))
  docs_toxic<-tm_map(docs_toxic, removeNumbers)
  docs_toxic<- tm_map(docs_toxic,removePunctuation)
  docs_toxic<-tm_map(docs_toxic, stripWhitespace)
  docs_toxic <- tm_map(docs_toxic,removeWords, stopwords("english"))
  docs_toxic<- gsub("\n", " ", docs_toxic, perl = T)
  # remove links
  docs_toxic <- gsub("(f|ht)tp(s?)://\\S+", "LINK", docs_toxic, perl = T)
  docs_toxic <- gsub("http\\S+", "LINK", docs_toxic, perl = T)
  docs_toxic <- gsub("xml\\S+", "LINK", docs_toxic, perl = T)
  
  # transform short forms
  docs_toxic <- gsub("'ll", " will", docs_toxic, perl = T)
  docs_toxic <- gsub("i'm", "i am", docs_toxic, perl = T)
  docs_toxic <- gsub("'re", " are", docs_toxic, perl = T)
  docs_toxic <- gsub("'s", " is", docs_toxic, perl = T)
  docs_toxic <- gsub("'ve", " have", docs_toxic, perl = T)
  docs_toxic <- gsub("'d", " would", docs_toxic, perl = T)
  docs_toxic <- gsub("can't", "can not", docs_toxic, perl = T)
  docs_toxic <- gsub("don't", "do not", docs_toxic, perl = T)
  docs_toxic <- gsub("doesn't", "does not", docs_toxic, perl = T)
  docs_toxic <- gsub("isn't", "is not", docs_toxic, perl = T)
  docs_toxic <- gsub("aren't", "are not", docs_toxic, perl = T)
  docs_toxic <- gsub("couldn't", "could not", docs_toxic, perl = T)
  docs_toxic <- gsub("mustn't", "must not", docs_toxic, perl = T)
  docs_toxic <- gsub("didn't", "did not", docs_toxic, perl = T)
  
  
  docs_toxic <- gsub("(?<=\\b\\w)\\s(?=\\w\\b)", "", docs_toxic, perl = T)
  
  # remove "shitdocs_toxic"
  docs_toxic <- gsub("\\b(a|e)w+\\b", "AWWWW", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(y)a+\\b", "YAAAA", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(w)w+\\b", "WWWWW", docs_toxic, perl = T)
  #docs_toxic <- gsub("a?(ha)+\\b", "", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(b+)?((h+)((a|e|i|o|u)+)(h+)?){2,}\\b", "HAHEHI", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(b+)?(((a|e|i|o|u)+)(h+)((a|e|i|o|u)+)?){2,}\\b", "HAHEHI", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(m+)?(u+)?(b+)?(w+)?((a+)|(h+))+\\b", "HAHEHI", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b((e+)(h+))+\\b", "HAHEHI", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b((h+)(e+))+\\b", "HAHEHI", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b((o+)(h+))+\\b", "HAHEHI", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b((h+)(o+))+\\b", "HAHEHI", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b((l+)(a+))+\\b", "LALALA", docs_toxic, perl = T)
  docs_toxic <- gsub("(w+)(o+)(h+)(o+)", "WOHOO", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(d?(u+)(n+)?(h+))\\b", "UUUHHH", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(a+)(r+)(g+)(h+)\\b", "ARGH", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(a+)(w+)(h+)\\b", "AAAWWHH", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(p+)(s+)(h+)\\b", "SHHHHH", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b((s+)(e+)?(h+))+\\b", "SHHHHH", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(s+)(o+)\\b", "", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(h+)(m+)\\b", "HHMM", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b((b+)(l+)(a+)(h+)?)+\\b", "BLABLA", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b((y+)(e+)(a+)(h+)?)+\\b", "YEAH", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b((z+)?(o+)(m+)(f+)?(g+))+\\b", "OMG", docs_toxic, perl = T)
  docs_toxic <- gsub("aa(a+)", "a", docs_toxic, perl = T)
  docs_toxic <- gsub("ee(e+)", "e", docs_toxic, perl = T)
  docs_toxic <- gsub("i(i+)", "i", docs_toxic, perl = T)
  docs_toxic <- gsub("oo(o+)", "o", docs_toxic, perl = T)
  docs_toxic <- gsub("uu(u+)", "u", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(u(u+))\\b", "u", docs_toxic, perl = T)
  docs_toxic <- gsub("y(y+)", "y", docs_toxic, perl = T)
  docs_toxic <- gsub("hh(h+)", "h", docs_toxic, perl = T)
  docs_toxic <- gsub("gg(g+)", "g", docs_toxic, perl = T)
  docs_toxic <- gsub("tt(t+)\\b", "t", docs_toxic, perl = T)
  docs_toxic <- gsub("(tt(t+))", "tt", docs_toxic, perl = T)
  docs_toxic <- gsub("mm(m+)", "m", docs_toxic, perl = T)
  docs_toxic <- gsub("ff(f+)", "f", docs_toxic, perl = T)
  docs_toxic <- gsub("cc(c+)", "c", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(kkk)\\b", "KKK", docs_toxic, perl = T)
  docs_toxic <- gsub("\\b(pkk)\\b", "PKK", docs_toxic, perl = T)
  docs_toxic <- gsub("kk(k+)", "kk", docs_toxic, perl = T)
  docs_toxic <- gsub("fukk", "fuck", docs_toxic, perl = T)
  docs_toxic <- gsub("k(k+)\\b", "k", docs_toxic, perl = T)
  docs_toxic <- gsub("f+u+c+k+\\b", "fuck", docs_toxic, perl = T)
  #gsub("((a+)|(h+)){3,}", "", "ishahahah hanibal geisha")
  docs_toxic <- gsub("((a+)|(h+)){3,}", "HAHEHI", docs_toxic, perl = T)
  
  docs_toxic <- gsub("yeah", "YEAH", docs_toxic, perl = T)
  # remove modified docs_toxic
  #gsub("(?<=\\b\\w)\\s(?=\\w\\b)", "", "f u c k  y o u  a s  u  a r e  a  b i t c h  a s s  n i g g e r", perl = T)
  #gsub("(?<=\\b\\w)\\s(?=\\w\\b)", "", "n i g g e r f a g g o t", perl = T)
  docs_toxic <- gsub("(?<=\\b\\w)\\s(?=\\w\\b)", "", docs_toxic, perl = T)
  
  otherstopwords <- c("put", "far", "bit", "well", "still", "much", "one", "two", "don", "now", "even", 
                      #"article", "articles", "edit", "edits", "page", "pages",
                      #"talk", "editor", "ax", "edu", "subject", "lines", "like", "likes", "line",
                      "uh", "oh", "also", "get", "just", "hi", "hello", "ok", "ja", #"editing", "edited",
                      "dont", "wikipedia", "hey", "however", "id", "yeah", "yo", 
                      #"use", "need", "take", "give", "say", "user", "day", "want", "tell", "even", 
                      #"look", "one", "make", "come", "see", "said", "now",
                      "wiki", 
                      #"know", "talk", "read", "time", "sentence", 
                      "ain't", "wow", #"image", "jpg", "copyright",
                      "wikiproject", #"background color", "align", "px", "pixel",
                      "org", "com", "en", "ip", "ip address", "http", "www", "html", "htm",
                      "wikimedia", "https", "httpimg", "url", "urls", "utc", "uhm",
                      #"i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                      #"you", "your", "yours", "yourself", "yourselves", 
                      "he", "him", "his", "himself", 
                      "she", "her", "hers", "herself", 
                      "it", "its", "itself",    
                      #"they", "them", "their", "theirs", "themselves",
                      #"i'm", "you're", "he's", "i've", "you've", "we've", "we're",
                      #"she's", "it's", "they're", "they've", 
                      #"i'd", "you'd", "he'd", "she'd", "we'd", "they'd", 
                      #"i'll", "you'll", "he'll", "she'll", "we'll", "they'll",
                      "what", "which", "who", "whom", "this", "that", "these", "those",
                      #"am", "can", "will", "not",
                      "is", "was", "were", "have", "has", "had", "having", "wasn't", "weren't", "hasn't",
                      #"are", "cannot", "isn't", "aren't", "doesn't", "don't", "can't", "couldn't", "mustn't", "didn't",    
                      "haven't", "hadn't", "won't", "wouldn't",  
                      "do", "does", "did", "doing", "would", "should", "could",  
                      "be", "been", "being", "ought", "shan't", "shouldn't", "let's", "that's", "who's", "what's", "here's",
                      "there's", "when's", "where's", "why's", "how's", "a", "an", "the", "and", "but", "if",
                      "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
                      "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
                      "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
                      "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
                      "most", "other", "some", "such", "no", "nor", "only", "own", "same", "so", "than",
                      "too", "very")
  #is.element(otherstopwords, stopwords("en"))
  #docs_toxic<- docs_toxic %>%
   # str_replace_all("[^[:alpha:]]", " ") %>%
    #str_replace_all("\\s+", " ") 
  #docs_toxic <- tm_map(docs_toxic,removeWords, otherstopwords)
  #docs_toxic <- removeWords(docs_toxic, otherstopwords)
  
  #Create structured data from text
  dtm_toxic<-DocumentTermMatrix(docs_toxic)
  dtm_toxic = removeSparseTerms(dtm_toxic, 0.99)
  
  #Making word cloud using structured text
  m_toxic<-as.matrix(dtm_toxic)
  v_toxic <- sort(colSums(m_toxic),decreasing=TRUE)
  
  #head(v_toxic)
  
  words_toxic <- names(v_toxic)
  d_toxic <- data.frame(word=words_toxic, freq=v_toxic)
  d_toxic<-d_toxic[order(d_toxic$freq,decreasing = TRUE),]
  wordcloud(d_toxic$word,d_toxic$freq,min.freq=100,max.words = 30, color = c("red"))
  
  table(d_toxic$freq>500)
  return(as.data.frame(m_toxic))
  # return(d_toxic[d_toxic$freq>500,])
  
  # barplot(d_toxic[1:10,2])
  #text(x = d_toxic[1:10,1], y = d_toxic[1:10,2], label = d_toxic[1:10,2], pos = 3, cex = 0.8, col = "red")
}


memory.limit(size=100000)
df<-NULL
df$toxic<-comment_func(train_toxic$comment_text[train_toxic$toxic=="1"])
df$severe_toxic<-comment_func(train_toxic$comment_text[train_toxic$severe_toxic=="1"])
df$obscene<-comment_func(train_toxic$comment_text[train_toxic$obscene=="1"])
df$threat<-comment_func(train_toxic$comment_text[train_toxic$threat=="1"])
df$insult<-comment_func(train_toxic$comment_text[train_toxic$insult=="1"])
df$identity_hate<-comment_func(train_toxic$comment_text[train_toxic$identity_hate=="1"])

#df of words for train dataset
df_train<-NULL
df_train<-comment_func(train$comment_text)

#df of words for test dataset
df_test<-NULL
df_test<-comment_func(test$comment_text)

colnamesSame = intersect(colnames(df_train),colnames(df_test))

df_train = df_train[ , (colnames(df_train) %in% colnamesSame)]
df_test = df_test[ , (colnames(df_test) %in% colnamesSame)]


boost_func2<-function(ctrain,dtest){
  xb <- xgboost(ctrain, 
                silent=1,
                eta = 0.1,
                max_depth = 15, 
                nround=50, 
                subsample = 0.5,
                colsample_bytree = 0.5,
                seed = 1,
                eval_metric = "auc",
                objective = "binary:logistic",
                nthread = 3
  )
  
  
  xb.predict <- predict(xb, newdata = dtest)
  return(xb.predict)
  }

#Toxic Prediction Calculation
submission<-NULL
submission$id<-test$id


data_train<-NULL
data_train<-df_train
data_train$toxic<-(train$toxic)

#library(caTools)
#set.seed(123)

#val<-NULL
#split = sample.split(data_train$toxic, SplitRatio = 2/3)
#training_set = subset(data_train, split == TRUE)
#test_set = subset(data_train, split == FALSE)

#ctrain <- xgb.DMatrix(Matrix(data.matrix(training_set[,!colnames(training_set) %in% c('toxic')])), label = (training_set$toxic))
#dtest <- xgb.DMatrix(Matrix(data.matrix(test_set[,-390])) )

#val_cv<-val_func(ctrain)
#val$toxic<-boost_func2(ctrain,dtest)
#y_pred = ifelse(val$toxic > 0.5, 1, 0)

# Making the Confusion Matrix
#cm = table(test_set[, 390], y_pred)

ctrain <- xgb.DMatrix(Matrix(data.matrix(data_train[,!colnames(data_train) %in% c('toxic')])), label = (data_train$toxic))
dtest <- xgb.DMatrix(Matrix(data.matrix(df_test)) )
submission$toxic<-boost_func2(ctrain,dtest)


#glm.model <-cv.glmnet(data.matrix(data_train[,!colnames(data_train) %in% c('toxic')]), factor(data_train$toxic), alpha = 0, family = "binomial", type.measure = "auc",
 #                     parallel = T, standardize = T, nfolds = 4, nlambda = 50)
#cat(" AUC:", max(glm.model$cvm))
#subm <- predict(glm.model,data.matrix(df_test), type = "response", s = "lambda.min")


  #Severe Toxic Prediction ``````
data_train<-NULL
ctrain<-NULL
data_train<-df_train
data_train$severe_toxic<-(train$severe_toxic)
ctrain <- xgb.DMatrix(Matrix(data.matrix(data_train[,!colnames(data_train) %in% c('severe_toxic')])), label =(data_train$severe_toxic))
submission$severe_toxic<-boost_func2(ctrain,dtest)


#obscene Prediction Calculation
ctrain<-NULL
data_train<-NULL
data_train<-df_train
data_train$obscene<-(train$obscene)
ctrain <- xgb.DMatrix(Matrix(data.matrix(data_train[,!colnames(data_train) %in% c('obscene')])), label = (data_train$obscene))
submission$obscene<-boost_func2(ctrain,dtest)

#threat Prediction Calculation
ctrain<-NULL
data_train<-NULL
data_train<-df_train
data_train$threat<-(train$threat)
ctrain <- xgb.DMatrix(Matrix(data.matrix(data_train[,!colnames(data_train) %in% c('threat')])), label = (data_train$threat))
submission$threat<-boost_func2(ctrain,dtest) 

#insult Prediction Calculation
ctrain<-NULL
data_train<-NULL
data_train<-df_train
data_train$insult<-(train$insult)
ctrain <- xgb.DMatrix(Matrix(data.matrix(data_train[,!colnames(data_train) %in% c('insult')])), label = (data_train$insult))
submission$insult<-boost_func2(ctrain,dtest)


#identity_hate Prediction Calculation
ctrain<-NULL
data_train<-NULL
data_train<-df_train
data_train$identity_hate<-(train$identity_hate)
ctrain <- xgb.DMatrix(Matrix(data.matrix(data_train[,!colnames(data_train) %in% c('identity_hate')])), label = (data_train$identity_hate))
submission$identity_hate<-boost_func2(ctrain,dtest)

sub<-as.data.frame(submission)
colSums(sub[,2:7]>="0.5")
barplot(colSums(sub[,2:7]>="0.5"))


#sub$id<-test$id

#final submission
write.csv(sub, 'D:/Data Science/Kaggle/Jigsaw-toxic-comment-classification-challenge/Data/New/Submission5.csv', row.names = FALSE)

#auc
auc( sub$toxic, train$toxic )  #0.4984



mat<-xgb.importance (feature_names = colnames(sub),model = xb)
xgb.plot.importance (importance_matrix = mat) 
