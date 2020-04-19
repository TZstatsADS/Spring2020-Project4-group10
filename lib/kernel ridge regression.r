library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)

setwd("D:/CU/GR5243/Spring2020-Project4-spring2020-project4-group10/lib")

data <- read.csv("../data/ml-latest-small/ratings.csv")

test_idx <- sample(1:nrow(data), round(nrow(data)/5, 0))
train_idx <- setdiff(1:nrow(data), test_idx)
data.train <- data[train_idx,]
data.test <- data[test_idx,]

source("../lib/Probalistic_ Matrix_Factorization_w_cv.R")


##to normalized V
normalized<-function(c){
  l<-sqrt(sum(c^2))
  return(c/l)
}

#input:  data(data.train & data.test);V
#output: list of 610 data.frame(rating+fetures)
movie_id<-function(data,V){
  
  #normalized feature matrix for movie
  V_new<-apply(V,2,normalized)
  colnames(V_new)<-colnames(V)
  
  #get fetures and ratings of movie rated for each user 
  user<-unique(data$userId)
  movie_unique<-sort(unique(data$movieId))
  
  movieId_rated<-map(user,function(x) {return(data[which(data$userId == x),]$movieId)})
  movie_features<-map(movieId_rated,function(x) {
    interim<-map_dbl(x,function(x) which(movie_unique == x))
    return(V[,interim])
    })
  
  ratings<-map(user,function(x) {return(data[which(data$userId == x),]$rating)})
  
  return(list(user = user,
              movie = movieId_rated,
              rating = ratings,
              features = movie_features))

}

##linear ridge function
myridge<-function(l,lambda = 0.5,kernel = c("Linear","Gaussian"),data.new = NA,V){
  x<-l$features
  y<-l$rating
  prediction<-c()
  
  if(kernel == "Gaussian"){
    #build the model
    beta_hat<-map2(x,y,function(x,y){
      solve(exp(2*(t(x)%*%(x)-1))+diag(lambda,ncol(x)))%*%y
    })
    type <- "Gaussian"
  }
  
  else{
    beta_hat<-map2(x,y,function(x,y) {
      solve(x%*%t(x)+diag(lambda,nrow(x)))%*%x%*%y
            })
    type <- "Linear"
  }
  return(list(parameter = beta_hat,
              type = type,
              features = x))
}

#prediction
predict.myridge<-function(data.new,m){
  
  if(m$type == "Gaussian"){
    movie_unique<-sort(unique(data$movieId))
    interim_1<-map_dbl(data.new$movie,~which(movie_unique == .x))
    interim_2<-map(interim_1,function(x) V[,x])
    prediction<-unlist(map2(interim_2,data.new$userId, function(x,y){
      exp(2 * (unlist(t(x)) %*% m$features[[y]] -1) ) %*% m$parameter[[y]]
    }))
  }
  
  if(m$type == "linear"){
      movie_unique<-sort(unique(data$movieId))
      interim_1<-map_dbl(data.new$movie,~which(movie_unique == .x))
      interim_2<-map(interim_1,function(x) V[,x])
      prediction<-unlist(map2(interim_2,data.new$userId,function(x,y) t(x)%*%m$parameter[[y]]))
  }
  return(prediction)
}

#################################################################################

output<-gradPMF(D = 5, data, data.train, data.test, sigma = .5,sigma_V = .5, sigma_U = .5, max.iter = 200, lrate = 0.0005, stopping.deriv = 0.01)

V<-output$V

l<-movie_id(data.train,V)

m1<-myridge(l,kernel = "Gaussian",lambda = 0.5)
m2<-myridge(l,kernel = "linear",lambda = 0.5)

###userid:525
###movieid:44191
###Score:3.5
prediction_1<-predict.myridge(data.new = data.test, m = m1)
prediction_2<-predict.myridge(data.new = data.test, m = m2)
sqrt(mean((data.train$rating-prediction)^2))

################################################################################
#tunes parameters
t_lrate <- .0005

t_D <- c(5, 10)
t_sigma_V <- c(.1, .5)
t_sigma_U <- c(.5, 1)
t_lambda <- c(0.1,0.5,1,2)
par<-parameters <- expand.grid(D = t_D, sigma_V = t_sigma_V, sigma_U = t_sigma_U,t_lambda = t_lambda)


#cross validation
