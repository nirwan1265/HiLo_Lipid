library(dplyr)
# Read the files
info <- read.csv("data/Supplementary_File_1.csv")
hilo_lipids <- read.csv("data/hilo_lipids.csv")

# Check the data
head(info)
hilo_lipids[1:5,1:10]

# Find the unique IDs
unique(info$ID)
unique(hilo_lipids$genotype)

# Check whether they intersect
intersect(unique(info$ID),unique(hilo_lipids$genotype))

# Separate highland and lowland
highland <- hilo_lipids[which(hilo_lipids$elevation == "High"), ]
lowland <- hilo_lipids[which(hilo_lipids$elevation == "Low"), ]

# Join the long and lat data to the HiLo
highland <- dplyr::left_join(highland,info, by = c("genotype"="ID"), unmatched = "drop")
lowland <- dplyr::left_join(lowland,info, by = c("genotype"="ID"), unmatched = "drop")

# Drop rows with NA in LOng ana Lat
highland <- highland %>% drop_na(Long)
lowland <- lowland %>% drop_na(Long)

# Create a SpatialPointsDataFrame from your data
sp::coordinates(highland) <- ~Long+Lat
sp::coordinates(lowland) <- ~Long+Lat

#Setting CRS and Extracting Values for All .tif Files
proj4string(highland) <- CRS(SRS_string = "EPSG:4326")
proj4string(lowland) <- CRS(SRS_string = "EPSG:4326")

# Define the directory containing your .tif files
dir_path <- "/Users/nirwantandukar/Library/Mobile Documents/com~apple~CloudDocs/Github/Phosphorus_prediction/data/phosphorus_prediction/predictors"

# List all .tif files
tif_files <- list.files(dir_path, pattern = "\\.tif$", full.names = TRUE)

# Function to extract values from a single .tif file
extract_values <- function(tif_path, data_points) {
  raster_layer <- raster::raster(tif_path)
  extracted_values <- raster::extract(raster_layer, data_points)
  return(extracted_values)
}

# Iterate over .tif files and bind the results to the phos
for (tif_path in tif_files) {
  values <- extract_values(tif_path, highland)
  col_name <- gsub("^.+/(.+).tif$", "\\1", tif_path) # Create a column name based on the .tif file name
  highland[[col_name]] <- values
}

for (tif_path in tif_files) {
  values <- extract_values(tif_path, lowland)
  col_name <- gsub("^.+/(.+).tif$", "\\1", tif_path) # Create a column name based on the .tif file name
  lowland[[col_name]] <- values
}


highland <- highland@data
lowland <- lowland@data


# predictors
predictors_highland <- highland %>% dplyr::select(AM_herbs_roots_colonized:`wv1500_5-15cm_mean_1000`)
predictors_lowland <- lowland %>% dplyr::select(AM_herbs_roots_colonized:`wv1500_5-15cm_mean_1000`)


# Getting PC's and LPC's
PC_highland <- highland[grepl("^PC",names(highland))]
LPC_highland <- highland[grepl("LPC",names(highland))]

PC_lowland <- lowland[grepl("^PC",names(lowland))]
LPC_lowland <- lowland[grepl("LPC",names(lowland))]

# Summing up all the PC's and LPC's
PC_sum_highland <- rowSums(PC_highland, na.rm = T)
LPC_sum_highland <- rowSums(LPC_highland, na.rm = T)

PC_sum_lowland <- rowSums(PC_lowland, na.rm = T)
LPC_sum_lowland <- rowSums(LPC_lowland, na.rm = T)


# Taking Ratios
PC_LPC_sum_highland <- PC_sum_highland/LPC_sum_highland
PC_LPC_sum_lowland <- PC_sum_lowland/LPC_sum_lowland


# Getting the training data
# PC/LPC is the response variable
y_highland <- PC_LPC_sum_highland
X_highland <- predictors_highland

y_lowland <- PC_LPC_sum_lowland
X_lowland <- predictors_lowland


train_data_highland <- X_highland
train_y_highland <- y_highland

train_data_lowland <- X_lowland
train_y_lowland <- y_lowland


# Factorizing
train_data_highland <- train_data_highland %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric)

# Convert data to matrix, as xgboost doesn't accept data frames
set.seed(1)
dtrain_highland <- xgb.DMatrix(data = as.matrix(train_data_highland), label = train_y_highland, missing = NA)
dtrain_lowland <- xgb.DMatrix(data = as.matrix(train_data_lowland), label = train_y_lowland, missing = NA)

# Read the functions in file 2_bayesian_optimization
# Run Bayesian Optimization
# Remember to change dtrain to dtrain_highland or dtrain_lowland in the objective_function
bayes_opt_result <- BayesianOptimization(
  FUN = objective_function,
  bounds = bounds,
  init_points = 5,  # Number of randomly chosen points to sample the target function before fitting the Gaussian process
  n_iter = 75,      # Number of iterations to perform
  acq = "ucb",       # Acquisition function type: expected improvement, ei to ucb
  kappa = 2.5,           # Higher kappa for more exploration
  verbose = TRUE
)


# Print the best parameters and the corresponding RMSE
print(bayes_opt_result$Best_Par)

#For Highland
#> print(bayes_opt_result$Best_Par)
#max_depth min_child_weight        subsample colsample_bytree            gamma            alpha 
#5.000000         3.095288         0.652949         0.500000         0.578890         5.075037 
#lambda 
#15.000000 


# Getting those parameters:
# Parameters
params <- list(
  booster = "gbtree",
  eta = 0.1,
  max_depth = 5,
  min_child_weight = 3.095288,
  subsample = 0.652949,
  colsample_bytree = 0.500000,
  eval_metric = "rmse",
  objective = "reg:squarederror",
  lambda = 15.000000,  # L2 regularization
  alpha = 5.075037,    # L1 regularization
  gamma = 0.578890,        # Minimum loss reduction required to make a further partition
  nthread = 10,
  num_parallel_tree = 1  # Use more than 1 for boosted random forests
)


### FOR CV and nrounds:
# Perform 10-fold cross-validation
cv.nfold <- 10
cv.nrounds <- 1000
set.seed(123) # For reproducibility
cv_results <- xgb.cv(
  params = params,
  data = dtrain_highland,
  nrounds = cv.nrounds,
  nfold = cv.nfold,
  showsd = TRUE,
  stratified = TRUE,
  print_every_n = 10,
  early_stopping_rounds = 10,
  maximize = FALSE
)

# Review the cross-validation results
print(cv_results)

# BEST nrounds
# highland = 14


## Running the model:
xgb_model <- xgb.train(params = params, data = dtrain_highland, nrounds = 14)

## importance matrix
importance_matrix <- xgb.importance(feature_names = colnames(train_data_highland), model = xgb_model)
print(importance_matrix)


# Convert data to matrix, as xgboost doesn't accept data frames
set.seed(1)
dtest_maize <- xgb.DMatrix(data = as.matrix(test_data_maize), label = test_y_maize, missing = NA)

