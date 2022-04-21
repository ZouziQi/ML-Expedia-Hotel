
# R script set up and loading in data -------------------------------------
rm(list=ls())

# Load in useful libraries
library(data.table)  # fread function #读取数据就是fread这些函数
library(caret) #拆分数据  
library(dplyr) #  %>% 管道操作符 
library(randomForest) # filled the data #去做数据填充
library(ranger) #package in random forest 
library(Metrics) #mapk
library(mlbench)
#library(rpart)
#library(C50)
#library(class)


# Load in training and test data
# file.choose()
traindata <- fread(".\\ML Team Project\\train.csv") # read the train set
colnames(traindata) 
# filter the is_booking  # 筛选booking数据
table(traindata$is_booking)  #check train set IS_booking=1/0 conditions 
traindata2 <- traindata %>%   #筛选 就是binary variable 就是选择 =1 的值 就是预定的意思！筛选得到筛选后的training dataset 1 booking 0 just click
  filter(is_booking == "1")     
# remove traindata to save memory usage.
rm(traindata)  # remove the traindata to save the memory usage.
#我们删除最原始的training set 来降低我们的memory使用的情况！ 由于train set 有4.07 GB #这边我们就可以删除掉traindata

testdata  <- fread(".\\ML Team Project\\test.csv")  # read the test data 
#导入我们的test dataset  #testing只包含booking的情况，所以这是为什么我们filter train is booking
# testdata是为已经订购的就是booking=1的所有结果
# skimr::skim(testdata)

destinations <- 
  fread(".\\ML Team Project\\destinations.csv") # read the destinations set 
#导入我们的destination dataset 里面有destination的id和所搜索的region的描述
colnames(destinations)
sample_submission <- 
  fread(".\\ML Team Project\\sample_submission.csv") # sample of submission set 
#最后的输出样式 each customer and hotel cluster 酒店集群的样式 没有意义



# Deal with missing data # 数据缺失处理
sapply(traindata2, function(y) sum(((is.na(y))))) # check the NA in the traindata2 set 
#查看training dataset 是否存在缺失 #可以看到orig_destination_distance 有缺失值
sapply(testdata, function(y) sum((is.na(y)))) # check the NA in the testdata set 
#可以看到orig_destination_distance 有缺失值
# 所有的features 都无NA的情况

#using the median of train to fill the NA in train and test 
# reason is to Simulate the actual situation
median_odd_train <- median(traindata2$orig_destination_distance,na.rm = T) #
#Physical distance between a hotel and a customer at the time of search. A null means the distance could not be calculated
#orig_destination_distance 顾客和酒店的距离， 我们使用median查看中位数的距离，这边na.rm=T表示遇到NA就排除缺失值=完成每个计算，同时排除缺失值
median_odd_train #get the median 距离的中位数值

#fill the median value into the train and test 
traindata2$orig_destination_distance[is.na(traindata2$orig_destination_distance)] <- median_odd_train 
# 就是再traindata2中将所有的Orig_destination距离的 空缺值做判断如果是true的话就填充刚刚我们得到的中位数
testdata$orig_destination_distance[is.na(testdata$orig_destination_distance)] <- median_odd_train 
# 就是再testdata中将所有的Orig_destination距离的 空缺值做判断如果是true的话就填充刚刚我们得到的中位数

class(traindata2$srch_ci) #check the class of the search check in feature
# Generate search_ checkIn Month build feature in train and test 
traindata2$checkInMonth <- month(as.Date(traindata2$srch_ci)) 
#srch_ci 是 checkin的date 然后我们转换到date然后在提取month
testdata$checkInMonth <- month(as.Date(testdata$srch_ci))
#srch_ci 是 checkin的date 然后我们转换到date然后在提取month
class(traindata2$srch_ci)

# combine destinations  
all_destinations <- data.frame(             #因为有一些srch_destination_id在destination的dataset中是没有的，所以我们要合并train和test做一个填充
  srch_destination_id = unique(                   
    c(unique(traindata2$srch_destination_id),     #srch_destination_id in train ### ID of the destination where the hotel search was performed
      unique(testdata$srch_destination_id))       #srch_destination_id in test  ###ID of the destination where the hotel search was performed
  )      #get all srch_destination_id  from train and test                                      #上面是我们创建了新的数据集dataframe只包含train/test数据集有的srch_destination_id。
) %>%                                                                                           #第二步与我们已经存在的数据集进行合并，因为我们destination的数据集中也会有相对应的缺失值，所以我们要去做一个数据填充。
  left_join(destinations, by = "srch_destination_id") %>%    # 使用destinations的这个dataset进行left_join, key是srch_destination_id。
  na.roughfix()                                 #na.roughfix就是randomforest提供的填充函数，就是刚刚的中位数进行填充。
#我们这里就是为了所有的srch_destination_id在train和test都出现,并且d1-d149都是完整的有值的。

traindata3 <- traindata2 %>% # add 149 columns latent features in train
  left_join(all_destinations, by = "srch_destination_id")  
colnames(traindata3) #查看数据情况

testdata3 <- testdata %>%    # add 149 columns latent features in test
  left_join(all_destinations, by = "srch_destination_id")  
#colnames(testdata3)


#finish the data process and remove all useless set
rm(destinations)
rm(all_destinations)

#数据拆分 有300万数据 ，提取部分数据 
# Training data too large - Split train to be more manageable for data analysis
set.seed(2016)
splitIndex <- createDataPartition(traindata3$hotel_cluster, 
                                  p = 0.005,    #38million*0.01=30k
                                  list = FALSE,
                                  times = 1)

expediaTrain <- traindata3[splitIndex[, 1], ]  
#splitIndex生成的是一个矩阵matrix是一列的代表的是样本的行号，然后我们就得到了这3万数据的行号，然后放入traindata3中获得数据
#构建模型评估模型的数据已经生成
rm(traindata3) # we can remove the traindata3

# split for train and test
set.seed(2016)
trainIndex <- createDataPartition(expediaTrain$hotel_cluster,  
                                  p = .8,   #split 80% train and 20% test 30k*0.8   30k*0.2
                                  list = FALSE,
                                  times = 1)
expediaTraining <- expediaTrain[trainIndex[, 1], ]  #training 
expediaTesting <- expediaTrain[-trainIndex[, 1], ]  #testing
rm(expediaTrain)


# Fitting random forest model  #我们将在training上训练模型
# skimr::skim(expediaTraining)
colnames(expediaTraining)
length(unique(expediaTraining$hotel_cluster))  #100 hotel_cluster


fit2 <- ranger(x = expediaTraining[,-24],  
               y = expediaTraining$hotel_cluster,  
               classification = T, 
               oob.error = F,  #我们不去计算误差，节省时间
               min.node.size = 500, # number higher will more simple, if lower will case overfit
               probability = T, #求得概率，预测概率最高的前5个，所以我们要有概率
               seed = 42, #随机种子，结果有可重复性
               importance = 'impurity') 

# important x features check 
importance <- importance(fit2)
plot(importance,ylim=c(0,6))
fit2$variable.importance

# modify the x features according to the important 
fit1 <- ranger(x = expediaTraining[, c(7, 9:10, 14:16, 23, 25:174)],  
               y = expediaTraining$hotel_cluster,  
               classification = T, 
               oob.error = F, #我们不去计算误差，节省时间
               min.node.size = 500, 
               probability = T, #求得概率，预测概率最高的前5个，所以我们要有概率
               seed = 42) #随机种子，结果有可重复性

save(fit1, file = "model1.rda") # Save for easy loading later 



# Prediction using test set 
pred <- predict(fit1, expediaTesting, type = "response") #把模型、数据集放进去，然后response就是概率输出，我们这边产生一个list
predprob <- pred$predictions #get the probability         #这时候我们把list中prediction提取出来。
colnames(predprob) <- paste(0:99) #设置新的列名称0-99 正好100个

predprob2 <- predprob %>%  #将我们刚刚得到的predprobability转换为数据框dataframe 
  as.data.frame() %>%    #以下只用于数据框dataframe：
  mutate(id = expediaTesting$user_id) %>% #生成新的变量给它一个id，我们目前是宽数据就是每一行表示一个id对应0-99 100个类的概率
  tidyr::pivot_longer(-id) %>% #从宽数据转换为长数据，就是每一个id会有100行，每一行代表着所属id分类的概率，方便我们去提取每一个id的最大的probability
  group_by(id) %>%
  slice_max(order_by = value, n = 5) # 最高的5个

# mapk5  
actuals <- list() 
predicted <- list() 
for (i in 1:length(unique(predprob2$id))) {
  actuals[[i]] <- expediaTesting$hotel_cluster[expediaTesting$user_id == unique(predprob2$id)[i]]
  predicted[[i]] <- predprob2$name[predprob2$id == unique(predprob2$id)[i]]
}

mapk(5, actuals, predicted) #是 Metrics 是mapk k就是几个

# Generating submission file ----------------------------------------------

load("model1.rda")

# Prediction using test set 
sapply(testdata3, function(y) sum(length(which(is.na(y)))))
testdata3$checkInMonth[is.na(testdata3$checkInMonth)] <-
  median(testdata3$checkInMonth, na.rm = T)
pred.test <- predict(fit1, testdata3, type = "response", num.threads = 3) 
#进行预测，使用模型在使用新数据testdata3，方式type来生成probablity， num.threads就是根据我们自己的电脑的有几个核来跑就像CPU
predprob.test <- pred.test$predictions #test的probability拿出
colnames(predprob.test) <- paste(0:99) #然后给predprob.test 生产新的列名称100个

# Formatting into submission 
submit <- apply(predprob.test, 1, FUN = function(x) { # 在predprob.test上每一行应用一个函数，把每一行里对应的概率最高的5个分类的列名称拿出来
  i1 <- x!=0
  i2 <- order(-x[i1])
  head(colnames(predprob.test)[i1][i2], 5)}) 

submit2 <- list()        #最后我们把它dataframe存成list，每一个list就是一个dataframe对应一个样本
for (i in seq_along(testdata3$id)) {  # 把id拿出来
  submit2[[i]] <-                          # 列的话就是5个最高分类的名称是个向量，5个5个连一起，就是我们每5个提取来对上id的dataframe
    data.frame(id = testdata3$id[i],           
               hotel_cluster = paste(submit[(i*5-4):(i*5)], collapse = " "))
}

# Generating submission file
submission <- bind_rows(submit2)
write.csv(submission, file = "submission.csv", row.names = FALSE)
# write.csv(submission, file = "submission.csv", row.names = FALSE,quote = F)
#生成的csv是100个样本！ cluster  
