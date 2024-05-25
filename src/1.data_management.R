library(dplyr)
# Read the files
info <- read.csv("data/Supplementary_File_1.csv")
hilo_lipids <- read.csv("data/hilo_lipids.csv")


# Check the data
head(info)
hilo_lipids[1:10,1:10]

# Just getting the genotypes that are grown in high and low
hilo_lipids <- hilo_lipids %>%
  group_by(genotype) %>%
  filter(n() > 1) %>%
  ungroup()

# Separate highland and lowland
highland <- hilo_lipids[which(hilo_lipids$elevation == "High"), ]
lowland <- hilo_lipids[which(hilo_lipids$elevation == "Low"), ]

# Separate high and low land that have grown in high and low
highland_high <- highland[which(highland$field == "Highland"), ]
highland_low <- highland[which(highland$field == "Lowland"), ]

lowland_high <- lowland[which(lowland$field == "Highland"), ]
lowland_low <- lowland[which(lowland$field == "Lowland"), ]


# Join all the highlands and lowlands
highland_all <- rbind(highland_high,lowland_high)
lowland_all <- rbind(highland_low,lowland_low)

# Add the long and lat 
highland_all <- dplyr::left_join(highland_all,info, by =c("genotype" = "ID"))
lowland_all <- dplyr::left_join(lowland_all,info, by =c("genotype" = "ID"))


# Drop rows with NA in LOng ana Lat
highland_all <- highland_all %>% drop_na(Long)
lowland_all <- lowland_all %>% drop_na(Long)

# Create a SpatialPointsDataFrame from your data
sp::coordinates(highland_all) <- ~Long+Lat
sp::coordinates(lowland_all) <- ~Long+Lat

#Setting CRS and Extracting Values for All .tif Files
proj4string(highland_all) <- CRS(SRS_string = "EPSG:4326")
proj4string(lowland_all) <- CRS(SRS_string = "EPSG:4326")

# Define the directory containing your .tif files
dir_path <- "/Users/nirwantandukar/Documents/Research/data/HiLo_lipids/predictors"

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
  values <- extract_values(tif_path, highland_all)
  col_name <- gsub("^.+/(.+).tif$", "\\1", tif_path) # Create a column name based on the .tif file name
  highland_all[[col_name]] <- values
}

for (tif_path in tif_files) {
  values <- extract_values(tif_path, lowland_all)
  col_name <- gsub("^.+/(.+).tif$", "\\1", tif_path) # Create a column name based on the .tif file name
  lowland_all[[col_name]] <- values
}


highland_all <- highland_all@data
lowland_all <- lowland_all@data


# predictors
predictors_highland <- highland_all %>% dplyr::select(c(elevation,AM_herbs_roots_colonized:`wc2.1_30s_wind_12`))
predictors_lowland <- lowland_all %>% dplyr::select(c(elevation,AM_herbs_roots_colonized:`wc2.1_30s_wind_12`))

# Getting PC's and LPC's
PC_highland <- highland_all[grepl("^PC",names(highland_all))]
LPC_highland <- highland_all[grepl("LPC",names(highland_all))]

PC_lowland <- lowland_all[grepl("^PC",names(lowland_all))]
LPC_lowland <- lowland_all[grepl("LPC",names(lowland_all))]

# Summing up all the PC's and LPC's
PC_sum_highland <- rowSums(PC_highland, na.rm = T)
LPC_sum_highland <- rowSums(LPC_highland, na.rm = T)

PC_sum_lowland <- rowSums(PC_lowland, na.rm = T)
LPC_sum_lowland <- rowSums(LPC_lowland, na.rm = T)


# Taking Ratios
PC_LPC_sum_highland <- PC_sum_highland/LPC_sum_highland
PC_LPC_sum_lowland <- PC_sum_lowland/LPC_sum_lowland

PC_LPC_16_32_highland <- PC_highland$PC_32_0/LPC_highland$LPC_16_0
PC_LPC_16_32_lowland <- PC_lowland$PC_32_0/LPC_lowland$LPC_16_0

# Getting the training data
# PC/LPC is the response variable
y_highland <- log10(PC_LPC_sum_highland)
X_highland <- predictors_highland

y_highland <- log10(PC_LPC_16_32_highland)
y_lowland <- log10(PC_LPC_16_32_lowland)


y_lowland <- log10(PC_LPC_sum_lowland)
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
  n_iter = 100,      # Number of iterations to perform
  acq = "ucb",       # Acquisition function type: expected improvement, ei to ucb
  kappa = 2.5,           # Higher kappa for more exploration
  verbose = TRUE
)


# Print the best parameters and the corresponding RMSE
#For Highland
print(bayes_opt_result$Best_Par)
#max_depth min_child_weight        subsample colsample_bytree            gamma            alpha           lambda 
#9.0000000        7.4680538        0.7747523        0.5301410        0.4837789        7.5664646        7.3439911 

# Getting those parameters:
# Parameters
params <- list(
  booster = "gbtree",
  eta = 0.1,
  max_depth = 9.0000000,
  min_child_weight = 7.4680538,
  subsample = 0.7747523,
  colsample_bytree = 0.5301410,
  eval_metric = "rmse",
  objective = "reg:squarederror",
  lambda = 7.3439911,  # L2 regularization
  alpha = 7.5664646,    # L1 regularization
  gamma = 0.4837789,        # Minimum loss reduction required to make a further partition
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
# highland = 6


## Running the model:
set.seed(123)
xgb_model <- xgb.train(params = params, data = dtrain_lowland, nrounds = 1000)

## importance matrix
importance_matrix <- xgb.importance(feature_names = colnames(train_data_highland), model = xgb_model)
print(importance_matrix)


# Prediction on test set Maize
preds_lowland <- predict(xgb_model, dtrain_highland)
hist(preds_lowland)
cor(preds_lowland,y_highland)



