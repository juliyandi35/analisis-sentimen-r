#Data Cleaning
library(tm)#membersihkan data
library(vroom)#load dataset
library(here)#menyimpan dataset
library(corpus)

#Cleaning data
d<-vroom(here('Data Sentimen.csv'))
ulasan<-d$reviewText
ulasan1<-Corpus(VectorSource(ulasan))
removeURL<-function(x)gsub("http[^[:space:]]*", "",x)
reviewclean<-tm_map(ulasan1,removeURL)
removeNL<-function(y)gsub("\n"," ",y)
reviewclean<-tm_map(reviewclean,removeNL)
replacecomma<-function(y)gsub(",", "",y)
reviewclean<-tm_map(reviewclean,replacecomma)
removetitik2<-function(y)gsub(":", "",y)
reviewclean<-tm_map(reviewclean,removetitik2)
removetitikkoma<-function(y)gsub(";", " ",y)
reviewclean<-tm_map(reviewclean,removetitikkoma)
removetitik3<-function(y)gsub("p...", "",y)
reviewclean<-tm_map(reviewclean,removetitik3)
removeamp<-function(y)gsub("&amp;", "",y)
reviewclean<-tm_map(reviewclean,removeamp)
removeUN<-function(z)gsub("@\\w+", "",z)
reviewclean<-tm_map(reviewclean,removeUN)
remove.all<-function(xy)gsub("[^[:alpha:][:space:]]*", "",xy)
reviewclean<-tm_map(reviewclean,remove.all)
#tanda baca
reviewclean<-tm_map(reviewclean,removePunctuation)
#mengubah huruf kecil
reviewclean<-tm_map(reviewclean,tolower) 
#menghapus slangwords
slang = readLines('Slangwords.csv')
reviewclean<-tm_map(reviewclean,removeWords,slang)
#menghapus stopwords
myStopwords <- readLines('Stopwords.csv')
reviewclean <- tm_map(reviewclean,removeWords,myStopwords)
#stemming kata
library(SnowballC)
reviewclean <- tm_map(reviewclean, stemDocument)
#menghapus kata-kata yang tidak dibutuhkan
gakpenting <- readLines('gakpenting.csv')
reviewclean <- tm_map(reviewclean,removeWords,gakpenting)
#Membuat file baru
dataframe<-data.frame(text=unlist(sapply(reviewclean,`[`)),stringsAsFactors = F)
View(dataframe)
write.csv(dataframe,file='ulasanclean.csv')
#Algoritma
library(e1071)#Untuk Naive Bayes
library(caret)#Untuk klasifikasi data
library(syuzhet)#Untuk membaca fungsi get_nrc

Eduworkdata<-read.csv("ulasanclean.csv",stringsAsFactors = FALSE)
review<-as.character(Eduworkdata$text)#merubah text jadi karakter
s<-get_nrc_sentiment(review)

review_combine<-cbind(Eduworkdata$text,s)#klasifikasi data
par(mar=rep(3,4))
a<-barplot(colSums(s),col=rainbow(10),ylab='count',main='Analisis Sentimen')
brplt<-a

#Wordcloud
#Cleaning data
library(tm)
library(RTextTools)
#Algoritma Naive Bayes
library(e1071)
library(dplyr)
library(caret)
df<-read.csv('ulasanclean.csv',stringsAsFactors = F)
glimpse(df)

#Set seed
set.seed(20)
df<-df[sample(nrow(df)),]
glimpse(df)

corpus<-Corpus(VectorSource(df$text))
corpus
inspect(corpus[1:10])
#Membersihkan data-data yang tidak dibutuhkan
corpus.clean<-corpus%>%
  tm_map(content_transformer(tolower))%>%
  tm_map(removePunctuation)%>%
  tm_map(removeNumbers)%>%
  tm_map(removeWords,stopwords(kind="en"))%>%
  tm_map(stripWhitespace)
dtm<-DocumentTermMatrix(corpus.clean)

inspect(dtm[1:10,1:20])

df.train<-df[1:10,]
dftest<-df[1:10,]

dtm.train<-dtm[1:10,]
dtm.test<-dtm[1:10,]

corpus.clean.train<-corpus.clean[1:10]
corpus.clean.test<-corpus.clean[1:10]

dim(dtm.train)
fivefreq<-findFreqTerms(dtm.train,5)
length(fivefreq)

dtm.train.nb<-DocumentTermMatrix(corpus.clean.train,control=list(dictionary=fivefreq))

dtm.test.nb<-DocumentTermMatrix(corpus.clean.test,control=list(dictionary=fivefreq))

dim(dtm.test.nb)

convert_count<-function(x){
  y<-ifelse(x>0,1,0)
  y<-factor(y,levels = c(0,1),labels = c("no","yes"))
  y
}
trainNB<-apply(dtm.train.nb,2,convert_count)
testNB<-apply(dtm.test.nb,1,convert_count)

nb_model <- naiveBayes(trainNB, df.train$text)
predicted <- predict(nb_model, testNB)
# convert actual test set labels and predicted labels to factors with the same levels
actual_labels <- factor(dftest$text, levels = unique(dftest$text))
predicted_labels <- factor(predicted, levels = levels(actual_labels))

# compute accuracy and error rate
confusionMatrix(predicted_labels,actual_labels[1:2])$overall['Accuracy']
error_rate <- 1 - confusionMatrix(predicted_labels, actual_labels[1:2])$overall['Accuracy']
error_rate
library(wordcloud)
wordcloud(corpus.clean,min.freq=4,max.words=100,random.order=F,colors=brewer.pal(8,"Dark2"))

