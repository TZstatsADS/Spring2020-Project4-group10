library(dplyr)
library(tidyr)
library(purrr)

##kernel ridge function
P3<-function(data = data.train, V, lambda = 0.5,kernel = c("Linear","Gaussian")){
  
  ##to normalized V
  normalized<-function(c){
    l<-sqrt(sum(c^2))
    return(c/l)
  }
  
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
  
  l<-movie_id(data.train,V)
  
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
predict.P3<-function(data.new, m, V = V, d = data){
  
  if(m$type == "Gaussian"){
    movie_unique<-sort(unique(d$movieId))
    interim_1<-map_dbl(data.new$movieId,~which(movie_unique == .x))
    interim_2<-map(interim_1,function(x) V[,x])
    prediction<-unlist(map2(interim_2,data.new$userId, function(x,y){
      exp(2 * (unlist(t(x)) %*% m$features[[y]] -1) ) %*% m$parameter[[y]]
    }))
  }
  
  if(m$type == "Linear"){
    movie_unique<-sort(unique(d$movieId))
    interim_1<-map_dbl(data.new$movieId,~which(movie_unique == .x))
    interim_2<-map(interim_1,function(x) V[,x])
    prediction<-unlist(map2(interim_2,data.new$userId,function(x,y) 
      t(x)%*%m$parameter[[y]]))
  }
  return(prediction)
}


##cross validation
cv_P3.function <- function(data, dat_train, K, D, sigma_V, sigma_U, lrate = .0005,
                           lambda, kernel = c("Gaussian","Linear")){
  ### Input:
  ### - train data frame
  ### - K: a number stands for K-fold CV
  ### - kernel: Gaussian, Linear
  ### - tuning parameters: D, sigma_V, sigma_U, lambda
  
  n <- dim(dat_train)[1]
  n.fold <- round(n/K, 0)
  set.seed(0)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  train_rmse <- c()
  test_rmse <- c()
  
  for (i in 1:K){
    train.data <- dat_train[s != i,]
    test.data <- dat_train[s == i,]
    
    result <- gradPMF(D = D, data = data, data.train = train.data, data.test = test.data,
                      sigma_V = sigma_V, sigma_U = sigma_U, lrate = lrate)
    
    model <- A2(data = train.data, V = result$V, kernel = kernel,lambda = lambda)
    
    prediction1 <- predict.A2(data.new = train.data, m = model,V = result$V)
    prediction2 <- predict.A2(data.new = test.data, m = model, V = result$V)
    
    train_rmse[i] <-  sqrt(mean((train.data$rating-prediction1)^2))
    test_rmse[i]  <-  sqrt(mean((test.data$rating-prediction2)^2))
    
  }		
  return(list(mean_train_rmse = mean(train_rmse),
              mean_test_rmse = mean(test_rmse),
              sd_train_rmse = sd(train_rmse),
              sd_test_rmse = sd(test_rmse)))
}


