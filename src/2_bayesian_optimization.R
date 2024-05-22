# Define the objective function to be optimized
# Initialize a global variable to store problematic parameters
problematic_params <- list()

# Modify the objective function to skip errors and log problematic parameter sets
objective_function <- function(max_depth, min_child_weight, subsample, colsample_bytree, gamma, alpha, lambda) {
  params <- list(
    booster = "gbtree",
    eta = 0.1,
    max_depth = as.integer(max_depth),
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    gamma = gamma,
    eval_metric = "rmse",
    objective = "reg:squarederror",
    lambda = lambda,
    alpha = alpha,
    nthread = 8  # Adjust based on your setup
  )
  
  # Try-catch block to handle errors during model training
  result <- tryCatch({
    cv_results <- xgb.cv(
      params = params,
      data = dtrain_highland,  # Ensure dtrain is correctly defined
      nrounds = 100,
      nfold = 5,
      showsd = TRUE,
      stratified = FALSE,
      print_every_n = 10,
      early_stopping_rounds = 10,
      maximize = FALSE
    )
    score <- -min(cv_results$evaluation_log$test_rmse_mean)
    if(is.finite(score)) {
      return(list(Score = score, Pred = min(cv_results$evaluation_log$test_rmse_mean)))
    } else {
      stop("Non-finite value encountered.")  # Trigger error handling
    }
  }, error = function(e) {
    # Log problematic parameters and the error
    cat("Error with parameters:", toString(params), "Error message:", e$message, "\n")
    # Append the problematic parameters and error message to the global list
    problematic_params <<- c(problematic_params, list(list(params = params, error = e$message)))
    return(list(Score = -Inf, Pred = Inf))  # Return a high penalty for error cases
  })
  
  return(result)
}


# Define the bounds of the hyperparameters
bounds <- list(
  max_depth = c(3L, 10L),
  min_child_weight = c(1, 8),
  subsample = c(0.5, 0.8),
  colsample_bytree = c(0.5, 0.8),
  gamma = c(0.3, 0.7),
  alpha = c(5, 25),
  lambda = c(5, 25)
)

# Run Bayesian Optimization
## dtrain from 7.XGBOOST.R
bayes_opt_result <- BayesianOptimization(
  FUN = objective_function,
  bounds = bounds,
  init_points = 5,  # Number of randomly chosen points to sample the target function before fitting the Gaussian process
  n_iter = 200,      # Number of iterations to perform
  acq = "ucb",       # Acquisition function type: expected improvement, ei to ucb
  kappa = 2.5,           # Higher kappa for more exploration
  verbose = TRUE
)

# Print the best parameters and the corresponding RMSE
print(bayes_opt_result$Best_Par)
