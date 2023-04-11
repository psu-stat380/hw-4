library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(stringr)
library(corrplot)
library(car)
library(caret)
library(torch)
library(nnet)
library(broom)


# Question 1 ---
# 1.a Plot the function 
# $f(x, y) = (x - 3)^2 + (y - 4)^2$

f <- \(z) (z^4) + 2 * (z^2) + 3 * z + 4
g <- \(x, y) (x - 3)^2 + (y - 4)^2
h <- \(u, v) torch_sum((u * v)^3)

z <- torch_tensor(2, requires_grad = TRUE)
x <- torch_tensor(3, requires_grad=TRUE)
y <- torch_tensor(4, requires_grad=TRUE)

u <- torch_tensor(
  rep(c(-1, 1), 5), 
  requires_grad=TRUE
)
v <- torch_tensor(
  c(rep(-1, 5), rep(1, 5)),
  requires_grad=TRUE
)


f(z)$backward()
z$grad

g(x, y)$backward()
x$grad
y$grad

h(u, v)$backward()
u$grad
v$grad


f = \(z) (z^4) - 6 * (z^2) - 3 * z + 4
df = \(z) 4* (z^3) - 12 * (z) - 3

if(T){
  lr <- 0.02
  n <- 100
  Z <- rep(-3.5, n)
  for(i in 2:n){
    Z[i] <- Z[i-1] - lr * df(Z[i-1])
  }
  curve(f, -4, 4)
  points(Z, f(Z), col="red", type="b")
}

# Question 2
# Perform logistic regression

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
col_names <- c("id", "diagnosis", paste0("X", 1:30))
df <- read_csv(
    url, col_names, col_types = cols()
    ) %>% 
    select(-id) %>% 
    mutate(y = diagnosis == "M") %>%
    select(-diagnosis)
m <- 30
df$y


#

titanic <- read.csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
rownames(titanic) <- titanic$Name
titanic <- titanic %>% select(-Name)
tmp <- glm(Survived ~ ., titanic, family = binomial())
summary(tmp)

module <- nn_module(
  initialize = function() {
    self$fc1 <- nn_linear(m, 1)
    self$fc2 <- nn_sigmoid()
  },
  forward = function(x) {
    x %>% 
      self$fc1() %>% 
      self$fc2()
  }
)

Loss <- function(x, y, model){
  nn_bce_loss()(model(x), y)
}

penalty <- function(model){
  sum(abs(model$parameters$fc1.weight))
}

fit <- function(f, lambda, n=2000, lr=0.01){
  optimizer <- optim_adam(f$parameters, lr=lr)

  for (i in 1:n){
    loss <- Loss(train_x, train_y, f) + lambda * penalty(f)
    optimizer$zero_grad()
    loss$backward()
    optimizer$step()

    if (i < 10 || i %% 100 == 0){
      cat(sprintf("Step: %d, Loss: %.4f\n", i, loss$item()))
    }
  }
  return(f)
}

if (T){
  set.seed(123)
  f <- module()
  train_index <- sample(1:nrow(df), 0.2 * nrow(df), replace=FALSE)

  train <- df[-train_index, ]
  test <- df[train_index, ]

  train_x <- torch_tensor(
    train[, -(m+1)] %>% as.matrix, dtype=torch_float()
    )
  train_y <- torch_tensor(
    train[[m+1]] %>% as.numeric, dtype=torch_float()
    )
  test_x <- torch_tensor(
    test[, -(m+1)] %>% as.matrix, dtype=torch_float()
    )
  test_y <- torch_tensor(
    test[[m+1]] %>% as.numeric, dtype=torch_float()
    )
}


if (T){
  set.seed(123)
  f <- module()
  train_index <- sample(1:nrow(df), 0.2 * nrow(df), replace=FALSE)

  train <- df[-train_index, ]
  test <- df[train_index, ]
  train <- test <- df

  train_x <- torch_tensor(
    model.matrix(model)[, -1], dtype=torch_float()
    )
  train_y <- torch_tensor(
    train[[m+1]] %>% as.numeric, dtype=torch_float()
    )
  test_x <- torch_tensor(
    model.matrix(model)[, -1], dtype=torch_float()
    )
  test_y <- torch_tensor(
    test[[m+1]] %>% as.numeric, dtype=torch_float()
    )
}

evaluate <- function(f){
  y_pred <- f(test_x)
  y_output <- ifelse(as_array(y_pred) < 0.5, 0, 1)
  mean(y_output != as_array(test_y))
}


f(train_x) %>% as_array()


pred1 <- predict(glm(y ~ ., train, family=binomial()), test, type="response") > 0.5
table(pred1, test$y)

g <- fit(f, 1e-10, 5000, 0.01)
pred2 <- as_array(g(test_x) > 0.5)
table(pred2, test$y)

lambdas <- 2^seq(-20, -2, by=0.5)
errors <- sapply(lambdas, \(x) fit(f, x, 1000, 0.005) %>% evaluate)
plot(log2(lambdas), errors, type="b", col="blue")
abline(v = log2(lambdas[which(errors == min(errors))]), col="red")


# #################
# n <- 10000
# m <- 21
# b <- c(-3, 2, -4)
# s = \(x) (1 + exp(-x))^-1

# x <- list()
# x[[1]] <- 1 + rnorm(n)
# x[[2]] <- 3 + rnorm(n)
# x[[3]] <- -3 + rnorm(n)

# df <- data.frame(matrix(NA, nrow=n, ncol=m))
# df$X1 <- x[[1]]
# df$X2 <- x[[2]]
# df$X3 <- x[[3]]
# df$y <- runif(n) < s(-6 + df[, 1:3] %>% as.matrix %*% b)
# summary(glm(y ~ X1 + X2 + X3, df, family=binomial()))

# for(i in 3:m){
#   df[, i] <- x[[1 + i %% 3]] * rnorm(n)
# }
# df <- as_tibble(df)


# model <- glm(y ~ ., df, family=binomial())
# summary(model)
# # preds <- ifelse(predict(model, data.frame(X1=x1, X2=x2, X3=x3), type="response") <= 0.5, 0, 1)
# # table(preds, y)
# # coef(model)

df <- tibble(titanic) %>% 
mutate_if(\(x) is.character(x), as.factor) %>% 
mutate(y = as.factor(Survived)) %>% 
select(-Survived) %>% 
(\(x) {names(x) <- tolower(names(x)); x})
df

#################
model <- glm(y ~ ., df, family=binomial())
summary(model)


set.seed(pi)
m <- which(colnames(df) == "y") - 1
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats=10)
fit <- train(
  x = model.matrix(model)[, -1],
  y = df[[m+1]] %>% as.factor(),
  method = "glmnet",
  trControl = fitControl, 
  tuneGrid = expand.grid(
    alpha = 1,
    lambda = 2^seq(-20, 0, length.out = 40)
    ),
  family = "binomial"
)
fit$bestTune$lambda
plot(fit$results$lambda %>% log2, fit$results$Accuracy, type="b", col="blue")
abline(v=fit$bestTune$lambda %>% log2, col="red")

# Print the model coefficients
c <- coef(fit$finalModel, fit$bestTune$lambda)
c
# Make predictions on the test set



pred1 <- predict(model, df, type="response") > 0.5
pred2 <- predict(step_model, df, type="response") > 0.5
pred3 <- predict(fit,  model.matrix(model)[, -1], type="raw") == "1"
# Evaluate the model
table(pred1, df$y)
table(pred2, df$y)
table(pred3, df$y)
