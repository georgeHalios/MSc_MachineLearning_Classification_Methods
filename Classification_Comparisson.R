# import relevant libraries
library(tidyverse) 
library(MASS)
library(car)
library(e1071)
library(caret)
library(dplyr)
library(ISLR)
library(repr)
library(corrplot)
library(plotrix)

# import the dataset
US.stock <- read.csv("US-stock.csv")

# remove variables dominated by NA.s
NAs <- US.stock %>% summarise_all(funs(sum(is.na(.))/n()))
NAs <- gather(NAs, key = "variables", value = "percent_missing")
NAs <- subset(NAs,NAs$percent_missing>0.10)

US.stock <- subset(US.stock,select = -c(43,53,76,77,78,82,83,84,85,87,88,90,91,96,97,
                                       99,100,101,103,105,112,114,120,127,128,129,130,
                                       132,143,146,148,149,150,151,152,153,155,159,172,
                                       173,174,175,186,201,202,204,205,207,208,210,211,213,214))

# observer variables
str(US.stock)
summary(US.stock$operatingProfitMargin)

# remove singular variables
US.stock <- subset(US.stock,select = -c(80))

# remove rows with missing value
US.stock <- US.stock[complete.cases(US.stock), ]

#The absolute values of pair-wise correlations are considered. 
#If two variables have a high correlation, the function looks at the mean absolute correlation 
#of each variable and removes the variable with the largest mean absolute correlation.

cor.US.stock <- cor(US.stock[,2:153] , use = "complete.obs")
findCorrelation(cor.US.stock, cutoff = .80, verbose = T,names = T)

# used that function to find the variable number
#match("daysOfSalesOutstanding",names(US.stock))

US.stock <- subset(US.stock,select = -c(5,8,21,25,26,28,29,30,31,32,
                                        35,38,43,46,48,52,55,56,59,60,61,62,
                                        63,66,69,73,78,81,83,94,95,98,99,102,105,106,107,111,
                                        113,117,125,126,133,134,80,135,138,144,146,151,2,9,11,15,
                                        24,33,27,37,17,82,100,75,76,74,93,92,19,89
                                        ,96,101,36,44,90,91,84,141,152))

#Use tree based algorithim r-part to determine the most important variables in the dataset
rpartMod<-train(Class~.-X -Sector ,data = US.stock,method="rpart",na.action = na.exclude)
rpartImp<-varImp(rpartMod)
print(rpartImp, top = 5)
plot(rpartImp, top = 5)

# used that function to find the variable number
# match("Payout.Ratio",names(US.stock))


US.stock <- subset(US.stock,select = c(1,93,92,64,11,49,82,58))

US.stock$Class <- as.factor(US.stock$Class)
US.stock$Sector <- as.factor(US.stock$Sector)

#change name for simplicity 
colnames(US.stock)[7] <- "NIG/share"

# summary data

summary(US.stock)
str(US.stock)

cor1 <- cor(US.stock[,4:8])
corrplot(cor1)

# Exploratory data 

# on sector 

counts <- table(US.stock$Sector)
plot1 <- pie(counts, main="Pie Chart of Sectors")

# Class split 
summary(US.stock$Class)
percentage.of.1s <- 2130/(849+2130)

options(repr.plot.width = 17, repr.plot.height = 10)
ggplot(US.stock, aes(x=Sector,fill=Class))+ geom_bar(position = 'fill')+theme_bw()


# boxpltos on numeric variables

ggplot(US.stock, aes(x=Class, y=EPS, fill=Class)) +geom_violin() +
  geom_boxplot(width=.1, fill="white") + labs(title="EPS") + ylim(-10,10)

ggplot(US.stock, aes(x=Class, y=Payout.Ratio, fill=Class)) +geom_violin() +
  geom_boxplot(width=.1, fill="white") + labs(title="Payout.Ratio") + ylim(-1,1)

ggplot(US.stock, aes(x=Class, y=`NIG/share`, fill=Class)) +geom_violin() +
  geom_boxplot(width=.1, fill="white") + labs(title="NIG/share") + ylim(-1,3)

ggplot(US.stock, aes(x=Class, y=Graham.Number, fill=Class)) + geom_violin() +
  geom_boxplot(width=.1, fill="white") + labs(title="Graham.Number") + ylim(-1,150)

ggplot(US.stock, aes(x=Class, y=PE.ratio, fill=Class)) + geom_violin() +
  geom_boxplot(width=.1, fill="white") + labs(title="PE.ratio") + ylim(0,1)



# Split the data into train and test
set.seed(123)
Index =createDataPartition(US.stock$Class, p = .60, list = FALSE)
train=US.stock[Index,]
test=US.stock[-Index,]

#set controls
control = trainControl(method="repeatedcv", number = 5, repeats = 5)
metric = "Accuracy"

# LDA
set.seed(123)
fit.lda = train(Class~.-X , data=train, method="lda", metric=metric, trControl=control,na.action = na.pass)
pred=predict(fit.lda,test[,-2],type="raw")
table(pred, test[,2])
mean(pred==test[,2])

# KNN
set.seed(123)
fit.knn = train(Class~.-X , data=train, method="knn", metric=metric, trControl=control,na.action = na.pass)
pred=predict(fit.knn,test[,-2],type="raw")
table(pred, test[,2])
mean(pred==test[,2])

# QDA
set.seed(123)
fit.qda = train(Class~.-X , data=train, method="qda", metric=metric, trControl=control,na.action = na.pass)
pred=predict(fit.qda,test[,-2],type="raw")
table(pred, test[,2])
mean(pred==test[,2])

# SVM
set.seed(123)
fit.svm = train(Class~.-X , data=train, method="svmRadial", metric=metric, trControl=control,na.action = na.pass)
pred=predict(fit.svm,test[,-2],type="raw")
table(pred, test[,2])
mean(pred==test[,2])

# RANDOM FOREST
set.seed(123)
fit.rf = train(Class~.-X , data=train, method="rf", metric=metric, trControl=control,na.action = na.pass)
pred=predict(fit.rf,test[,-2],type="raw")
table(pred, test[,2])
mean(pred==test[,2])

# NEURAL NETWORKS
set.seed(123)
fit.nnet = train(Class~.-X , data=train, method="nnet", metric=metric, trControl=control,na.action = na.pass)
pred=predict(fit.nnet,test[,-2],type="raw")
table(pred, test[,2])
mean(pred==test[,2])


