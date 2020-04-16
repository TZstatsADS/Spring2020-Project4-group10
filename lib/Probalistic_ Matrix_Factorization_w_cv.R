library(tidyverse)

update_U <- function(r, I, U, V, lambda_U, lrate){
  res <- I * (r - t(U) %*% V)
  grad_U <- (-1) * V %*% t(res) + lambda_U * U
  U <- U - lrate * grad_U
  return(U)
}

update_V <- function(r, I, U, V, lambda_V, lrate){
  res <- I * (r - t(U) %*% V)
  grad_V <- (-1) * U %*% res + lambda_V * V
  V <- V - lrate * grad_V
  return(V)
}

expand_r <- function(data, data.train){
  a <- unique(data$userId)
  b <- as.integer(levels(as.factor(data$movieId)))
  c <- expand.grid("movieId" = b, "userId" = a) %>% select(c(userId, movieId))
  
  new <- left_join(c, data.train, by = c("userId", "movieId")) %>% select(-timestamp)
  new[is.na(new)] <- 0
  
  return(new)
}

RMSE <- function(rating, est_rating){
  sqr_err <- function(obs){
    sqr_error <- (obs[3] - est_rating[as.character(obs[1]), as.character(obs[2])])^2
    return(sqr_error)
  }
  return(sqrt(mean(apply(rating, 1, sqr_err))))  
}

gradPMF <- function(D = 5, data, data.train, data.test, sigma = .5,
                     sigma_V = .5, sigma_U = .5, max.iter = 200, lrate = 0.0005, stopping.deriv = 0.01){
  
  set.seed(0)
  u <- length(unique(data$userId))
  m <- length(unique(data$movieId))
  
  # initialize U(D X #user) and V(D X #movie)
  U <- matrix(rnorm(D * u, 0, sigma_U), ncol = u)
  colnames(U) <- as.character(1:u)
  V <- matrix(runif(D * m, 0, sigma_V), ncol = m)
  colnames(V) <- levels(as.factor(data$movieId))
  
  # expand to get the whole r matrix. (I * R)
  r <- 
    expand_r(data, data.train) %>% 
    pivot_wider(names_from = movieId, values_from = rating) %>% 
    select(-userId) %>% as.matrix()
  
  # get the I matrix
  I <- r
  I[I != 0] <- 1
  I <- as.matrix(I)
  
  # save the train and test RMSE
  # train_RMSE <- c(Inf)
  # test_RMSE <- c(Inf)
  train_RMSE <- c()
  test_RMSE <- c()
  
  lambda_U <- sigma / sigma_U
  lambda_V <- sigma / sigma_V
  
  for(i in 1:max.iter){
    # update U
    U <- update_U(r, I, U, V, lambda_U, lrate)
    # update V
    V <- update_V(r, I, U, V, lambda_V, lrate)
    cat("=>")
    if (i %% 10 == 0){
      cat("epoch:", i, "\t")
      est_rating <- t(U) %*% V
      
      train_RMSE_cur <- RMSE(data.train, est_rating)
      cat("training RMSE:", train_RMSE_cur, "\t")
      train_RMSE <- c(train_RMSE, train_RMSE_cur)
      
      test_RMSE_cur <- RMSE(data.test, est_rating)
      cat("test RMSE:",test_RMSE_cur, "\n")
      test_RMSE <- c(test_RMSE, test_RMSE_cur)
    }
    # early-stopping to prevent overfitting
    # if((tail(test_RMSE,1) > tail(test_RMSE,2)[1])){
    #   cat("Stop to prevent over-fitting", "\n")
    #   break
    # }
  }
  # train_RMSE <- tail(train_RMSE, length(train_RMSE)-1)
  # test_RMSE <- tail(test_RMSE, length(test_RMSE)-1)
  return(list(U = U, V = V, train_RMSE = train_RMSE, test_RMSE = test_RMSE))
}

cv_A2.function <- function(data, dat_train, K, D, sigma_V, sigma_U, lrate){
  ### Input:
  ### - train data frame
  ### - K: a number stands for K-fold CV
  ### - tuning parameters: D, sigma_V, sigma_U, lrate
  
  n <- dim(dat_train)[1]
  n.fold <- round(n/K, 0)
  set.seed(0)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  train_rmse <- matrix(NA, ncol = 20,nrow = K) # this need change
  test_rmse <- matrix(NA, ncol = 20, nrow = K) # this need change
  
  for (i in 1:K){
    train.data <- dat_train[s != i,]
    test.data <- dat_train[s == i,]
    
    result <- gradPMF(D=D, data = data, data.train = train.data, data.test = test.data,
                      sigma_V = sigma_V, sigma_U = sigma_U, lrate = lrate)
      
    train_rmse[i,] <-  result$train_RMSE
    test_rmse[i,] <-   result$test_RMSE
    
  }		
  return(list(mean_train_rmse = apply(train_rmse, 2, mean), mean_test_rmse = apply(test_rmse, 2, mean),
              sd_train_rmse = apply(train_rmse, 2, sd), sd_test_rmse = apply(test_rmse, 2, sd)))
}