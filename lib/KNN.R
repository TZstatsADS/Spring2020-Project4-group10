source("../lib/Probalistic_ Matrix_Factorization_w_cv.R")
library(dplyr)
library(tidyr)
library(ggplot2)
library(doBy)


#Define a function to calculate RMSE
RMSE <- function(rating, est_rating){
  sqr_err <- function(obs){
    sqr_error <- (obs[3] - est_rating[as.character(obs[1]), as.character(obs[2])])^2
    return(sqr_error)
  }
  return(sqrt(mean(apply(rating, 1, sqr_err))))  
}


expand_r <- function(data, data.train){
  a <- unique(data$userId)
  b <- as.integer(levels(as.factor(data$movieId)))
  c <- expand.grid("movieId" = b, "userId" = a) %>% select(c(userId, movieId))
  new <- left_join(c, data.train, by = c("userId", "movieId")) %>% select(-timestamp)
  new[is.na(new)] <- 0
  return(new)
}

dist.cosine = function(x){
  y =  as.dist( x %*% t(x) / (sqrt(rowSums(x^2) %*% t(rowSums(x^2)))) ) 
  y <- as.matrix(y)
  diag(y) <- 1
  return(y)
}

data <- read.csv("../data/ml-latest-small/ratings.csv")
set.seed(0)
test_idx <- sample(1:nrow(data), round(nrow(data)/5, 0))
train_idx <- setdiff(1:nrow(data), test_idx)
data.train <- data[train_idx,]
data.test <- data[test_idx,]

P2_KNN <- function(data = data, data.train, data.test,K, D=5, sigma_V = 0.5, sigma_U = 1){
  grad <- gradPMF(D, data, data.train, data.test,sigma_V, sigma_U)
  v <- grad$V
  u <- grad$U
  A <- t(u)%*%v
  dist <- dist.cosine(t(v))
  
  d <- matrix(0,9724,9724)
  for (i in 1:nrow(dist)){
    d[i,which.maxn(dist[i,], K)]=1
  }
  r <- expand_r(data, data.train) %>% pivot_wider(names_from = movieId, values_from = rating) %>% 
    select(-userId) %>% as.matrix()
  r_new <- r/r
  C <- matrix(1,nrow=610,9724)
  for (i in 1:nrow(C)){
    C[i,which(r_new[i,]==1)]=0
  }
  f <- (A*C)+r
  R <- (f%*%d)/K
  colnames(R) <- levels(as.factor(data$movieId))
  train_RMSE <- RMSE(data.train,R)
  test_RMSE <- RMSE(data.test,R)
  
  
  return(list(pred=R,train_RMSE=train_RMSE,test_RMSE=test_RMSE))
}
# t3<-Sys.time()
# p <- P2_KNN(data = data, data.train, data.test,K=30, D=5, sigma_V = 0.5, sigma_U = 1)
# t4<-Sys.time()
# t4-t3
# 
# p$train_RMSE
# p$test_RMSE
# rating <- p$pred
# tr <- p$train_RMSE
# te <- p$test_RMSE
# save(rating,file="../output/ratings_A2_P2.RData")
# save(tr,file="../output/train_RMSE_A2_P2.RData")
# save(te,file="../output/test_RMSE_A2_P2.RData")


cv_P2.function <- function(data, dat_train, K, D, sigma_V, sigma_U, k){
  n <- dim(dat_train)[1]
  n.fold <- round(n/K, 0)
  set.seed(0)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  train_rmse <- c()
  test_rmse <- c()
  for (i in 1:K){
    train.data <- dat_train[s != i,]
    test.data <- dat_train[s == i,]
    
    result <- P2_KNN(D=D, data = data, data.train = train.data, data.test = test.data,
                     K=k,sigma_V = sigma_V, sigma_U = sigma_U)
    train_rmse[i] <-  result$train_RMSE
    test_rmse[i] <-   result$test_RMSE
    
  }		
  return(list(mean_train_rmse = mean(train_rmse),
               mean_test_rmse = mean(test_rmse),
               sd_train_rmse = sd(train_rmse),
               sd_test_rmse = sd(test_rmse)))
}


################################################################################
#tunes parameters
# 
# t_D <- c(5, 10)
# t_sigma_V <- c(.1, .5)
# t_sigma_U <- c(.5, 1)
# t_K <- c(20,30)
# par <- expand.grid(D = t_D, sigma_V = t_sigma_V, sigma_U = t_sigma_U,K = t_K)
# dim(par)[1]
# 
# #cross validation
# cv_train_rmse <- c()
# cv_test_rmse <- c()
# 
# for(i in dim(par)[1]){
#   tmp_D <- par$D[i]
#   tmp_sigma_V <- par$sigma_V[i]
#   tmp_sigma_U <- par$sigma_U[i]
#   tmp_k  <- par$K[i]
#   
#   ret <- cv_P2.function(data, data.train, K = 5, D = tmp_D, 
#                         sigma_V = tmp_sigma_V, sigma_U = tmp_sigma_U, k = tmp_k)
#   
#   cat("--------------------------------------- No.",i,"Done --------------------------------------------", "\n")
#   cv_train_rmse[i] <- ret$mean_train_rmse
#   cv_test_rmse[i] <- ret$mean_test_rmse
#   
# }
# 
# cv_P2_train_rmse <- cv_train_rmse
# cv_P2_test_rmse <- cv_test_rmse
# 
# rmse_P2<-cbind(par,cv_P2_train_rmse,cv_P2_test_rmse)
# 
# save(rmse_P2, file = "../output/RMSE_A2_P2.RData")
