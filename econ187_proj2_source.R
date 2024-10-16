# ~~~~~~~ Project 2 ~~~~~~~
# Crime in Communities: Predicting Violent Crime by County
# By Jacob Titcomb
# UCLA Economics 187
# Spring 2024
# Professor: Randall Rojas
# ~~~~~~~~~~~~~~~~~~~~~~~~~

### run upon completion:
# stopCluster(cl)


library(tidyverse)
library(e1071)
library(doParallel)
library(caret)
library(MLmetrics)
library(patchwork)
library(corrplot)
library(viridis)
library(glmnet)
library(Matrix)
library(earth)
library(gam)
library(kernlab)
library(vip)


### ===== DATA IMPORT & PROCESSING ==========

# IMPORTING THE DATA
csv_path <- '/Users/jake/R/Econ 187/Project 2/unscaled_crime.csv'
unscaled_crime_raw <- read_csv(csv_path) %>%
  filter(!is.na(target))

cols_w_na <- names(
  which(
    sapply(unscaled_crime_raw, \(x) sum(is.na(x))) > 100
  )
)

# remove missing values
unscaled_crime <- unscaled_crime_raw %>%
  filter(!is.na(target), !is.na(otherPerCap)) %>% # 2 rows with NAs
  filter(target > 0) %>% # 1 row of 0 violent crimes
  dplyr::select(- all_of(cols_w_na)) %>% # columns with many NAs
  mutate_at("State", factor) %>%
  mutate_at("target", log)

select
# one-hot encode states
unscaled_crime2 <- data.frame(
  target = unscaled_crime$target,
  model.matrix(target ~ ., data = unscaled_crime)[,-1]
)

# save column names
columns <- colnames(unscaled_crime2)
columns.numeric <- columns[!str_detect(string = columns, pattern = '\\A[Ss]tate')]

# create train-test split
set.seed(90024)
N <- nrow(unscaled_crime2)
shuffled_indices <- sample(seq(N))
partition_index <- shuffled_indices[seq(round(0.8 * N))]

train_unscaled <- unscaled_crime2[partition_index,]
test_unscaled <- unscaled_crime2[-partition_index,]


# PCA to determine kept variables, based off of training set
pc_to_reduce <- prcomp(train_unscaled[,-1], center = TRUE, scale. = TRUE)
pc_to_reduce.loadings <- abs(pc_to_reduce$rotation)
pc_to_reduce.importance <- summary(pc_to_reduce)$importance['Cumulative Proportion',]
# pc_to_reduce.importance[70]
# 70 principal components capture over 95% of the variance

# rank original features by contributions (sum of absolute loadings)
loadings_order <- sort(rowSums(pc_to_reduce.loadings[,1:70]), decreasing = TRUE)
loadings_cum_percent <- 100 * cumsum(loadings_order) / sum(loadings_order)
# after 70 features (ordered by decreasing importance), diminishing returns

# find proportional cumulative sum of most important features
prop_loadings_80 <- min(seq_along(loadings_cum_percent)[loadings_cum_percent >= 80])
# 94 features contribute to over 80% of the total absolute loadings

# which features to keep, based off of PCA
keep_features <- names(loadings_cum_percent[1:prop_loadings_80])
keep_features.numeric <- keep_features[keep_features %in% columns.numeric]
keep_features.state <- keep_features[!(keep_features %in% keep_features.numeric)]

# length(keep_features[keep_features %in% columns.numeric])
# length(keep_features[!(keep_features %in% columns.numeric)])
# CONCLUSION: 49 numeric columns, 45 states represented


# scale data
train.df <- train_unscaled %>%
  select(all_of(c('target', keep_features))) %>%
  mutate_at(keep_features.numeric, scale)

test.df <- test_unscaled %>%
  select(all_of(c('target', keep_features))) %>%
  mutate_at(keep_features.numeric, scale)

# as matrices
train.x <- as.matrix(train.df[,-1])
train.y <- train.df[,'target']
test.x <- as.matrix(test.df[,-1])


## kolmogorov-smirnov test outcome:
## OUTCOME NOT NORMALLY DISTRIBUTED, but close
# ks.test(train.df$target, 'pnorm')


### ===== EXPLORATORY DATA ANALYSIS ==========

# correlation plots
corr_all <- cor(bind_rows(train.df, test.df))
corr_target <- corr_all[,"target"]
corr_target_high <- corr_target[abs(corr_target) > 0.5]
corr_target_high.mat <- matrix(t(data.frame(corr_target_high))[,-1], nrow = 1)
rownames(corr_target_high.mat) <- "vCrimePerCapita"
colnames(corr_target_high.mat) <- colnames(t(data.frame(corr_target_high)))[-1]

### put in rmd file
# corrplot(corr_all,
#          method="color", order="hclust",
#          tl.pos = "n",
#          col = mako(300))
# corrplot(corr_target_high.mat,
#          method = 'color', addCoef.col = 'white', cl.pos="r", cl.length = 2,
#          tl.srt = 30, tl.col = "grey30", tl.cex = 0.8,
#          col = mako(60))





target_hist <- ggplot(unscaled_crime2, aes(target)) +
  geom_histogram(binwidth = 0.45, fill = mako(8)[5], color = mako(8)[1]) +
  theme_light() +
  labs(x = "Log-violent crimes per 100,000",
       y = "Frequency",
       title = "Outcome variable histogram")

## --- statistical summaries ---
# summary(unscaled_crime2$target)
# diff(range(unscaled_crime2$target))
# sd(unscaled_crime2$target)

## --- testing for normality ---
# kolmogorov-smirnov test outcome:
# OUTCOME: not normally distributed, but close
# p-value: 0.03069

# ks.test(scale(unscaleddf$target), 'pnorm')


### ===== AUXILIARY FUNCTIONS TO ASSIST ==========

out_of_sample_metrics <- function(pred, obs = test.df$target) {
  predicted <- pred
  observed <- obs
  output <- postResample(pred = predicted, obs = observed)
  return(output)
}

# edit: changed to ACTUAL VS PREDICTED
plot_pred.v.true <- function(pred, model.title, true = test.df$target) {
  df <- cbind(x = as.vector(pred), y = as.vector(true))
  output <- ggplot(data=df, aes(x=x, y=y)) +
    geom_point(color = mako(8)[3], size=2) +
    theme_light() +
    labs(y = "True Values", x="Predicted",
         title = paste("Actual vs Predicted:", model.title))
  
  return(output)
}

plot_glmnet <- function(glmnet_object, title, lambda.1se) {
  lam <- glmnet_object$lambda %>% 
    as.data.frame() %>%
    mutate(penalty = glmnet_object$a0 %>% names()) %>%
    rename(lambda = ".")
  
  results <- glmnet_object$beta %>% 
    as.matrix() %>% 
    as.data.frame() %>%
    rownames_to_column() %>%
    gather(penalty, coefficients, -rowname) %>%
    left_join(lam, by = join_by(penalty)) %>%
    mutate_at("lambda", log)
  
  result_labels <- results %>%
    group_by(rowname) %>%
    filter(lambda == min(lambda)) %>%
    ungroup() %>%
    top_n(10, wt = abs(coefficients)) %>%
    mutate(var = paste0("x", 1:10))
  
  output <- ggplot() +
    geom_line(data = results,
              aes(lambda, coefficients, group = rowname, color = rowname),
              show.legend = FALSE) +
    geom_text(data = result_labels,
              aes(lambda, coefficients, label = var, color = rowname),
              nudge_x = -.3, show.legend = FALSE) +
    scale_color_viridis(option = "mako", discrete = TRUE) +
    theme_light() +
    geom_vline(xintercept = log(lambda.1se), lty = 2, color = "grey30", alpha = 0.7) + 
    labs(y = "Coefficients", x = "Log-Lambda", title = title)
  
  return(output)
}

plot_glmnet.cv <- function(tune_lambda, cv.avg, cvup.avg, cvlo.avg, title, lambda.1se, lambda.min) {
  output <- data.frame(
    log_lambda = log(tune_lambda),
    est = sqrt(cv.avg),
    est.upper = sqrt(cvup.avg),
    est.lower = sqrt(cvlo.avg)
  ) %>%
    ggplot(aes(x=log_lambda, y=est)) +
    geom_errorbar(aes(ymin=est.lower, ymax=est.upper), width=0.1,linewidth=0.2,
                  color=mako(8)[2]) +
    geom_point(color=mako(8)[4], size=1) +
    geom_vline(xintercept = log(lambda.1se), lty = 2, color = "grey30", alpha = 0.7) + 
    geom_vline(xintercept = log(lambda.min), lty = 2, color = "grey50", alpha = 0.7) + 
    labs(title=paste(title, "Cross Validation"), y="RMSE", x="Log-Lambda") +
    theme_light()
  
  return(output)
}


expand_power <- function(df) {
  df.poly2 <- df
  df.poly3 <- df
  
  for (label in keep_features.numeric) {
    column_squared <- paste(label, ".2", sep="")
    column_cubed <- paste(label, ".3", sep="")
    df.poly2 <- df.poly2 %>%
      bind_cols(
        X = scale(df[[label]]^2)
      )
    colnames(df.poly2)[colnames(df.poly2) == "X"] <- column_squared
    
    df.poly3 <- df.poly3 %>%
      bind_cols(
        X = scale(df[[label]]^2),
        Y = scale(df[[label]]^3)
      )
    colnames(df.poly3)[colnames(df.poly3) == "X"] <- column_squared
    colnames(df.poly3)[colnames(df.poly3) == "Y"] <- column_cubed
  }
  output <- list()
  output[[1]] <- df.poly2
  output[[2]] <- df.poly3
  return(output)
}



### ===== TRAIN CONTROL SETUP ==========

# initialize parallel processing
cores <- detectCores()
n_workers <- cores - 3
cl <- makeCluster(n_workers)
registerDoParallel(cl)

# stopCluster(cl)

# train control setup
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     allowParallel = TRUE) # allow parallel processing


### ===== MODEL: OLS ==========
# (RMD file) mod.ols.cv.rmse; mod.ols.performance; mod.ols.vi; mod.ols.pvt_plot

mod.ols <- train(target ~ ., data = train.df, trControl = ctrl,
                 method = "lm")
mod.ols.pred <- predict(mod.ols, newdata = test.df)

mod.ols.cv.rmse <- mod.ols$results[,'RMSE']
mod.ols.cv.rmse_sd <- mod.ols$results[,'RMSESD']
mod.ols.performance <- out_of_sample_metrics(pred = mod.ols.pred,
                                             obs = test.df$target)
mod.ols.vi <- vi(mod.ols)
mod.ols.pvt_plot <- plot_pred.v.true(mod.ols.pred, "OLS")


### ===== NOT INCLUDED: poisson regression ==========

# pois.df <- train.df %>%
#   select(all_of(c('target', keep_features))) %>%
#   # mutate_at(keep_features.numeric, scale) %>%
#   select(-target) %>%
#   bind_cols(target = round(exp(train_unscaled$target)))
#   
# hist(pois.df$target)
# AIC(glm(target ~ ., family="poisson", data=pois.df))

# # very high AIC = 231294.3
# # linear model had AIC = 2761.459
# # thus we do not include poisson model in work


### ===== MODEL: LASSO ==========
# (RMD file) mod.lasso.finaln_vars; mod.lasso.cv.1se; mod.lasso.vi
# (RMD file) mod.lasso.cv.rmse; mod.lasso.performance
# (RMD file) mod.lasso.survival_plot; mod.lasso.cv_plot; mod.lasso.pvt_plot

# repeated cross validation to find error measures
mod.lasso.tune_cvm <- data.frame(
  iter1 = rep(NA, 82),
  iter2 = rep(NA, 82),
  iter3 = rep(NA, 82),
  iter4 = rep(NA, 82),
  iter5 = rep(NA, 82)
)
mod.lasso.tune_cvsd <- mod.lasso.tune_cvm
mod.lasso.tune_cvup <- mod.lasso.tune_cvm
mod.lasso.tune_cvlo <- mod.lasso.tune_cvm

# for loop to repeat 5 times
set.seed(90024)
for(i in 1:5){
  mod.lasso.cv <- cv.glmnet(x = train.x, y = train.y, alpha = 1,
                            nfolds = 10, parallel = TRUE)
  mod.lasso.tune_cvm[[paste('iter', i, sep="")]] <- mod.lasso.cv$cvm
  mod.lasso.tune_cvsd[[paste('iter', i, sep="")]] <- mod.lasso.cv$cvsd
  mod.lasso.tune_cvup[[paste('iter', i, sep="")]] <- mod.lasso.cv$cvup
  mod.lasso.tune_cvlo[[paste('iter', i, sep="")]] <- mod.lasso.cv$cvlo
}
mod.lasso.tune_lambda <- mod.lasso.cv$lambda

# take mean of cross validation values
mod.lasso.cv.avg <- rowMeans(mod.lasso.tune_cvm)
mod.lasso.cvsd.avg <- rowMeans(mod.lasso.tune_cvsd)
mod.lasso.cvup.avg <- rowMeans(mod.lasso.tune_cvup)
mod.lasso.cvlo.avg <- rowMeans(mod.lasso.tune_cvlo)

# minimum RMSE
mod.lasso.cv.min_lambda <- mod.lasso.tune_lambda[which.min(mod.lasso.cv.avg)]

# maximum lambda within 1 se
mod.lasso.cv.1se_index <- min(seq(1, 100)[
  (mod.lasso.cvlo.avg - min(mod.lasso.cv.avg) <= 0)
])
mod.lasso.cv.1se <- mod.lasso.tune_lambda[mod.lasso.cv.1se_index]
# RMSE approximation
mod.lasso.cv.rmse <- sqrt(mod.lasso.cv.avg[mod.lasso.cv.1se_index])
# RMSE standard deviation approximation
mod.lasso.cv.rmse_sd <- sqrt(5) * mod.lasso.cvsd.avg[mod.lasso.cv.1se_index] / (2 * mod.lasso.cv.rmse)

mod.lasso <- glmnet(x = train.x, y = train.y,
                    alpha = 1, lambda = mod.lasso.cv.1se)
# coefficients
mod.lasso.finaln_vars <- mod.lasso$df
mod.lasso.coef <- data.frame(as.matrix(coef(mod.lasso))) %>%
  filter(abs(s0) > 1e-10) %>%
  arrange(desc(abs(s0)))

# for performance evaluation
mod.lasso.no_lambda <- glmnet(x = train.x, y = train.y, alpha = 1)
mod.lasso.pred <- predict(mod.lasso, newx = test.x)
mod.lasso.performance <- out_of_sample_metrics(pred = mod.lasso.pred,
                                               obs = test.df$target)
mod.lasso.vi <- vi(mod.lasso) %>%
  mutate_at("Importance", \(x) 100 * x / max(x))


# plots
mod.lasso.survival_plot <- plot_glmnet(mod.lasso.no_lambda, title = "Lasso Coefficients",
                                       lambda.1se = mod.lasso.cv.1se)

mod.lasso.cv_plot <- plot_glmnet.cv(tune_lambda = mod.lasso.tune_lambda, title = "Lasso",
                                    cv.avg = mod.lasso.cv.avg,
                                    cvup.avg = mod.lasso.cvup.avg,
                                    cvlo.avg = mod.lasso.cvlo.avg,
                                    lambda.1se = mod.lasso.cv.1se,
                                    lambda.min = mod.lasso.cv.min_lambda)
mod.lasso.pvt_plot <- plot_pred.v.true(mod.lasso.pred, "LASSO")


### ===== MODEL: ridge ==========
# (RMD file) mod.ridge.cv.1se; mod.ridge.vi
# (RMD file) mod.ridge.cv.rmse; mod.ridge.performance
# (RMD file) mod.ridge.survival_plot; mod.ridge.cv_plot; mod.ridge.pvt_plot


# repeated cross validation to find error measures
mod.ridge.tune_cvm <- data.frame(
  iter1 = rep(NA, 100),
  iter2 = rep(NA, 100),
  iter3 = rep(NA, 100),
  iter4 = rep(NA, 100),
  iter5 = rep(NA, 100)
)
mod.ridge.tune_cvsd <- mod.ridge.tune_cvm
mod.ridge.tune_cvup <- mod.ridge.tune_cvm
mod.ridge.tune_cvlo <- mod.ridge.tune_cvm

# for loop to repeat 5 times
set.seed(90024)
for(i in 1:5){
  mod.ridge.cv <- cv.glmnet(x = train.x, y = train.y, alpha = 0,
                            nfolds = 10, parallel = TRUE)
  mod.ridge.tune_cvm[[paste('iter', i, sep="")]] <- mod.ridge.cv$cvm
  mod.ridge.tune_cvsd[[paste('iter', i, sep="")]] <- mod.ridge.cv$cvsd
  mod.ridge.tune_cvup[[paste('iter', i, sep="")]] <- mod.ridge.cv$cvup
  mod.ridge.tune_cvlo[[paste('iter', i, sep="")]] <- mod.ridge.cv$cvlo
}
mod.ridge.tune_lambda <- mod.ridge.cv$lambda

# take mean of cross validation values
mod.ridge.cv.avg <- rowMeans(mod.ridge.tune_cvm)
mod.ridge.cvsd.avg <- rowMeans(mod.ridge.tune_cvsd)
mod.ridge.cvup.avg <- rowMeans(mod.ridge.tune_cvup)
mod.ridge.cvlo.avg <- rowMeans(mod.ridge.tune_cvlo)

# minimum RMSE
mod.ridge.cv.min_lambda <- mod.ridge.tune_lambda[which.min(mod.ridge.cv.avg)]

# maximum lambda within 1 se
mod.ridge.cv.1se_index <- min(seq(1, 100)[
  (mod.ridge.cvlo.avg - min(mod.ridge.cv.avg) <= 0)
])
mod.ridge.cv.1se <- mod.ridge.tune_lambda[mod.ridge.cv.1se_index]
# RMSE approximation
mod.ridge.cv.rmse <- sqrt(mod.ridge.cv.avg[mod.ridge.cv.1se_index])
# RMSE standard deviation approximation
mod.ridge.cv.rmse_sd <-  sqrt(5) * mod.ridge.cvsd.avg[mod.ridge.cv.1se_index] / (2 * mod.ridge.cv.rmse)

mod.ridge <- glmnet(x = train.x, y = train.y,
                    alpha = 0, lambda = mod.ridge.cv.1se)
# coefficients
mod.ridge.coef <- data.frame(as.matrix(coef(mod.ridge))) %>%
  arrange(desc(abs(s0)))

# for performance evaluation
mod.ridge.no_lambda <- glmnet(x = train.x, y = train.y, alpha = 0)
mod.ridge.pred <- predict(mod.ridge, newx = test.x)
mod.ridge.performance <- out_of_sample_metrics(pred = mod.ridge.pred,
                                               obs = test.df$target)
mod.ridge.vi <- vi(mod.ridge) %>%
  mutate_at("Importance", \(x) 100 * x / max(x))

# plots
mod.ridge.survival_plot <- plot_glmnet(mod.ridge.no_lambda, title = "Ridge Coefficients",
                                       lambda.1se = mod.ridge.cv.1se)

mod.ridge.cv_plot <- plot_glmnet.cv(tune_lambda = mod.ridge.tune_lambda, title = "Ridge",
                                    cv.avg = mod.ridge.cv.avg,
                                    cvup.avg = mod.ridge.cvup.avg,
                                    cvlo.avg = mod.ridge.cvlo.avg,
                                    lambda.1se = mod.ridge.cv.1se,
                                    lambda.min = mod.ridge.cv.min_lambda)
mod.ridge.pvt_plot <- plot_pred.v.true(mod.ridge.pred, "ridge")



### ===== MODEL: elastic net ==========
# (RMD file) mod.elast.alpha; mod.elast.lambda
# (RMD file) mod.elast.cv.rmse; mod.elast.performance
# (RMD file) mod.elast.vi; mod.elast.pvt_plot; mod.elast.survival_plot

enetGrid <- expand.grid(lambda = seq(0, 0.9, length.out = 150),
                        alpha = seq(0, 0.5, length.out = 150))
mod.elast <- train(target ~ ., data = train.df, trControl = ctrl,
                   method = "glmnet",
                   tuneGrid = enetGrid
)
mod.elast.pred <- predict(mod.elast, newdata = test.df)

mod.elast.alpha <- mod.elast$bestTune[,'alpha']
mod.elast.lambda <- mod.elast$bestTune[,'lambda']
mod.elast.cv.rmse <- mod.elast$results[rownames(mod.elast$bestTune), 'RMSE']
mod.elast.cv.rmse_sd <- mod.elast$results[rownames(mod.elast$bestTune), 'RMSESD']
mod.elast.performance <- out_of_sample_metrics(pred = mod.elast.pred,
                                               obs = test.df$target)
mod.elast.vi <- vi(mod.elast)
mod.elast.pvt_plot <- plot_pred.v.true(mod.elast.pred, "elastic net")
mod.elast.survival_plot <- plot_glmnet(mod.elast$finalModel, title = "Elastic Net Coefficients",
                                       lambda.1se = mod.elast.lambda)



### ===== MODEL: PCR ==========
# (RMD file) mod.pcr.n_comp
# (RMD file) mod.pcr.cv.rmse; mod.pcr.performance
# (RMD file) mod.pcr.vi; mod.pcr.pvt_plot

set.seed(92009)
mod.pcr <- train(target ~ ., data = train.df, trControl = ctrl,
                 method = "pcr",
                 preProcess = c("center","scale"), tuneLength = 90)
mod.pcr.pred <- predict(mod.pcr, newdata = test.df)

# plot(mod.pcr) = RECREATE
mod.pcr.n_comp <- mod.pcr$bestTune[['ncomp']]
mod.pcr.cv.rmse <- mod.pcr$results[rownames(mod.pcr$bestTune), 'RMSE']
mod.pcr.cv.rmse_sd <- mod.pcr$results[rownames(mod.pcr$bestTune), 'RMSESD']
mod.pcr.performance <- out_of_sample_metrics(pred = mod.pcr.pred,
                                             obs = test.df$target)
mod.pcr.vi <- vi(mod.pcr)
mod.pcr.pvt_plot <- plot_pred.v.true(mod.pcr.pred, "PCR")



### ===== MODEL: piece-wise polynomial ==========
# (RMD file) mod.pp2.cv.rmse; mod.pp2.performance; mod.pp2.pvt_plot

# compute higher order polynomials
expand_full.df <- expand_power(bind_rows(train.df, test.df))

# extract polynomial variables
poly_vars <- colnames(expand_full.df[[1]])[
  !(colnames(expand_full.df[[1]]) %in% c("target", keep_features.state))
]

# construct cuts in data
expand_full.poly <- expand_full.df[[1]] %>%
  mutate_at(poly_vars, \(x) cut(x, 3))
train.df.poly2 <- expand_full.poly[1:nrow(train.df),]
test.df.poly2 <- expand_full.poly[(nrow(train.df) + 1):nrow(expand_full.poly),]

# run the regressions
mod.pp2 <- train(target ~ ., data = train.df.poly2, trControl = ctrl, method = "lm")
mod.pp2.pred <- predict(mod.pp2, newdata = test.df.poly2)

mod.pp2.cv.rmse <- mod.pp2$results[,'RMSE']
mod.pp2.cv.rmse_sd <- mod.pp2$results[,'RMSESD']
mod.pp2.performance <- out_of_sample_metrics(pred = mod.pp2.pred,
                                             obs = test.df.poly2$target)
mod.pp2.pvt_plot <- plot_pred.v.true(mod.pp2.pred, "piece-wise polynomial",
                                     true = test.df.poly2$target)


### ===== MODEL: MARS ==========
# (RMD file) mod.mars.nprune; mod.mars.degree
# (RMD file) mod.mars.cv.rmse; mod.mars.performance
# (RMD file) mod.mars.vi; mod.mars.pvt_plot

marsGrid <- expand.grid(
  degree = 1:3, 
  nprune = seq(6, 34, by = 2)
)

mod.mars <- train(target ~ ., data = train.df, trControl = ctrl,
                  method = "earth",
                  tuneGrid = marsGrid)

mod.mars.pred <- predict(mod.mars, newdata = test.df)
mod.mars.vi <- vi(mod.mars)

# plot(mod.mars) = RECREATE
mod.mars.nprune <- mod.mars$bestTune[['nprune']]
mod.mars.degree <- mod.mars$bestTune[['degree']]
mod.mars.cv.rmse <- mod.mars$results[rownames(mod.mars$bestTune), 'RMSE']
mod.mars.cv.rmse_sd <- mod.mars$results[rownames(mod.mars$bestTune), 'RMSESD']
mod.mars.performance <- out_of_sample_metrics(pred = mod.mars.pred,
                                              obs = test.df$target)
mod.mars.pvt_plot <- plot_pred.v.true(mod.mars.pred, "MARS")


### ===== MODEL: GAM ==========
# (RMD file) mod.gam.df; mod.gam.lam_cv
# (RMD file) mod.gam.cv.rmse; mod.gam.performance
# (RMD file) mod.gam.vi; mod.gam.pvt_plot

gamGrid <- expand.grid(df = seq(1.5, 4, length.out = 15))

mod.gam <- train(target ~ ., data = train.df, trControl = ctrl,
                 method = "gamSpline",
                 tuneGrid = gamGrid)

mod.gam.pred <- predict(mod.gam, newdata = test.df)

mod.gam.df <- mod.gam$bestTune[['df']]
mod.gam.cv.rmse <- mod.gam$results[rownames(mod.gam$bestTune), 'RMSE']
mod.gam.cv.rmse_sd <- mod.gam$results[rownames(mod.gam$bestTune), 'RMSESD']
mod.gam.performance <- out_of_sample_metrics(pred = mod.gam.pred,
                                             obs = test.df$target)

mod.gam.vi <- vi(mod.gam)

# grid search plot to find lambda
mod.gam.lam_cv <- ggplot(mod.gam$results, aes(x=df, y=RMSE)) +
  geom_vline(xintercept = mod.gam.df, lty = 2, color = 'grey30') +
  geom_line(color = mako(5)[2]) +
  geom_point(color = mako(5)[2], size = 2, shape=1) + theme_light() +
  theme(panel.grid.minor = element_blank()) +
  labs(x='Degrees of freedom', title='CV to select df')
mod.gam.pvt_plot <- plot_pred.v.true(mod.gam.pred, "GAM")



### ===== MODEL: Gaussian process ==========
# (RMD file) mod.gpr.cv.rmse; mod.gpr.performance
# (RMD file) mod.gpr.pvt_plot

# log approximation if contains 0 or negative
arcsinh <- function(x) log(x + sqrt(x^2 + 1))

gpr.train.df <- train.df
gpr.test.df <- test.df

# transform to make more normal
for (k in keep_features.numeric) {
  if (any(train.df[[k]] <= 0)) {
    transf <- arcsinh(train.df[[k]])
    transf.test <- arcsinh(test.df[[k]])
  } else {
    transf <- log(train.df[[k]])
    transf.test <- log(test.df[[k]])
  }
  skew.untransf <- skewness(train.df[[k]])
  
  if (abs(skewness(transf)) < abs(skew.untransf)) {
    gpr.train.df[[k]] <- transf
    gpr.test.df[[k]] <- transf.test
  }
}

mod.gpr <- train(target ~ ., data = gpr.train.df, trControl = ctrl,
                 method = "gaussprLinear")

mod.gpr.pred <- predict(mod.gpr, newdata = gpr.test.df)
mod.gpr.vi <- vi(mod.gpr)

mod.gpr.cv.rmse <- mod.gpr$results[,'RMSE']
mod.gpr.cv.rmse_sd <- mod.gpr$results[, 'RMSESD'] / 2
mod.gpr.performance <- out_of_sample_metrics(pred = mod.gpr.pred,
                                             obs = gpr.test.df$target)
mod.gpr.pvt_plot <- plot_pred.v.true(mod.gpr.pred, "GPR")



### ===== MODEL: Bayesian ridge ==========
# (RMD file) mod.brid.cv.rmse; mod.brid.performance
# (RMD file) mod.brid.pvt_plot

library(monomvn) # I dislike the MASS package, so I am loading this here

mod.brid <- train(target ~ ., data = train.df, trControl = ctrl,
                  method = "bridge", verb = 0)

mod.brid.pred <- predict(mod.brid, newdata = test.df)
mod.brid.vi <- vi(mod.brid)

mod.brid.cv.rmse <- mod.brid$results[,'RMSE']
mod.brid.cv.rmse_sd <- mod.brid$results[, 'RMSESD']
mod.brid.performance <- out_of_sample_metrics(pred = mod.brid.pred,
                                              obs = test.df$target)
mod.brid.pvt_plot <- plot_pred.v.true(mod.brid.pred, "Bayesian ridge")


### ===== CONCLUSIONS ==========
# (RMD file) cv_results.plot; performance.df

cv_results <- bind_cols(
  OLS = c(mod.ols.cv.rmse, mod.ols.cv.rmse_sd),
  LASSO = c(mod.lasso.cv.rmse, mod.lasso.cv.rmse_sd),
  ridge = c(mod.ridge.cv.rmse, mod.ridge.cv.rmse_sd),
  elastNet = c(mod.elast.cv.rmse, mod.elast.cv.rmse_sd),
  PCR = c(mod.pcr.cv.rmse, mod.pcr.cv.rmse_sd),
  polynomial = c(mod.pp2.cv.rmse, mod.pp2.cv.rmse_sd),
  MARS = c(mod.mars.cv.rmse, mod.mars.cv.rmse_sd),
  GAM = c(mod.gam.cv.rmse, mod.gam.cv.rmse_sd),
  GPR = c(mod.gpr.cv.rmse, mod.gpr.cv.rmse_sd),
  bRidge = c(mod.brid.cv.rmse, mod.brid.cv.rmse_sd)
  ) %>% t() %>% data.frame() %>%
  rename(RMSE = X1, sd = X2) %>%
  rownames_to_column(var = "model")

cv_results.plot <- cv_results %>%
  ggplot(aes(x=model, y=RMSE, group = model)) +
  geom_errorbar(aes(ymin=qnorm(0.025, mean = RMSE, sd = sd),
                    ymax=qnorm(0.975, mean = RMSE, sd = sd)),
                width=0.1, linewidth=1, color=mako(8)[6]) +
  geom_errorbar(aes(ymin=qnorm(0.1, mean = RMSE, sd = sd),
                    ymax=qnorm(0.9, mean = RMSE, sd = sd)),
                width=0.1, linewidth=1.5, color=mako(8)[5]) +
  geom_point(color=mako(8)[3], size=4) +
  labs(title="Cross Validation Performance", y="RMSE", x=NULL) +
  theme_light()


performance.df <- bind_cols(
  measures = c("RMSE", "R squared", "MAE"),
  OLS = mod.ols.performance,
  LASSO = mod.lasso.performance,
  ridge = mod.ridge.performance,
  'elastic net' = mod.elast.performance,
  PCR = mod.pcr.performance,
  'p-w polynomial' = mod.pp2.performance,
  MARS = mod.mars.performance,
  GAM = mod.gam.performance,
  GPR = mod.gpr.performance,
  'Bayesian ridge' = mod.brid.performance) %>%
  column_to_rownames("measures") %>% t() %>% data.frame() %>%
  arrange(RMSE)


stopCluster(cl)






