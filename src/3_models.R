library(tidyverse)
library(lme4)
library(caret)
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


head(hilo_lipids)

# Add the long and lat 
hilo_lipids_all <- dplyr::left_join(hilo_lipids,info, by =c("genotype" = "ID"))

# Drop rows with NA in LOng ana Lat
hilo_lipids_all <- hilo_lipids_all %>% drop_na(Long)

# Create a SpatialPointsDataFrame from your data
sp::coordinates(hilo_lipids_all) <- ~Long+Lat

#Setting CRS and Extracting Values for All .tif Files
proj4string(hilo_lipids_all) <- CRS(SRS_string = "EPSG:4326")

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

for (tif_path in tif_files) {
  values <- extract_values(tif_path, hilo_lipids_all)
  col_name <- gsub("^.+/(.+).tif$", "\\1", tif_path) # Create a column name based on the .tif file name
  hilo_lipids_all[[col_name]] <- values
}

hilo_lipids_all <- hilo_lipids_all@data

str(hilo_lipids_all)

# predictors
predictors_hilo_lipids_all <- hilo_lipids_all %>% dplyr::select(c(genotype,field,AM_herbs_roots_colonized:`wc2.1_30s_wind_12`))

# Getting PC's and LPC's
LPC_hilo_lipids_all <- hilo_lipids_all[grepl("LPC",names(hilo_lipids_all))]
PC_hilo_lipids_all <- hilo_lipids_all[grepl("PC",names(hilo_lipids_all))]


# Summing up all the PC's and LPC's
LPC_sum_hilo_lipids_all <- rowSums(LPC_hilo_lipids_all, na.rm = T)
PC_sum_hilo_lipids_all <- rowSums(PC_hilo_lipids_all, na.rm = T)


# Taking Ratios
PC_LPC_sum_hilo_lipids_all <- log10(PC_sum_hilo_lipids_all/LPC_sum_hilo_lipids_all)

head(PC_LPC_sum_hilo_lipids_all)
head(predictors_hilo_lipids_all)


# Ensure predictors_hilo_lipids_all is a data frame
predictors_hilo_lipids_all <- as.data.frame(predictors_hilo_lipids_all)
data <- predictors_hilo_lipids_all
data$lipid_concentration <- PC_LPC_sum_hilo_lipids_all


# Split the data into training and testing sets
set.seed(568)
train_index <- createDataPartition(data$lipid_concentration, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Remove columns with any NA values
train_data <- train_data %>% select_if(~ !any(is.na(.)))
test_data <- test_data %>% select_if(~ !any(is.na(.)))

# Ensure test data has the same columns as train data
common_cols <- intersect(colnames(train_data), colnames(test_data))
train_data <- train_data %>% dplyr::select(all_of(common_cols))
test_data <- test_data %>% dplyr::select(all_of(common_cols))

# Separate predictors and response
train_predictors <- train_data %>% dplyr::select(-lipid_concentration)
test_predictors <- test_data %>% dplyr::select(-lipid_concentration)
train_response <- train_data$lipid_concentration
test_response <- test_data$lipid_concentration

# Perform PCA separately on training and test predictors
pca_train <- PCA(train_predictors[,-c(1,2)], scale.unit = TRUE, ncp = 5, graph = FALSE)
pca_test <- PCA(test_predictors[,-c(1,2)], scale.unit = TRUE, ncp = 5, graph = FALSE)



# Extract the top 5 principal components for both training and test sets
train_pca <- as.data.frame(pca_train$ind$coord[, 1:5])
test_pca <- as.data.frame(pca_test$ind$coord[, 1:5])

# Add response variable and genotype back to the PCA-transformed data
train_pca$lipid_concentration <- train_response
train_pca$genotype <- train_data$genotype

test_pca$lipid_concentration <- as.numeric(test_response)
test_pca$genotype <- test_data$genotype


# Fit a linear model using the top 5 principal components
lm_model <- lm(lipid_concentration ~ Dim.1 + Dim.2 + Dim.3 + Dim.4 + Dim.5, data = train_pca)

# Summarize the model
summary(lm_model)

# Predict on the test set
test_pca$predicted_lm <- predict(lm_model, newdata = test_pca)

# Evaluate the model
mse_lm <- mean((test_pca$lipid_concentration - test_pca$predicted_lm)^2)
cat("Mean Squared Error (LM): ", mse_lm, "\n")

cor(test_pca$predicted_lm, test_data$lipid_concentration)

# Fit a generalized linear model using the top 5 principal components
glm_model <- glm(lipid_concentration ~ Dim.1 + Dim.2 + Dim.3 + Dim.4 + Dim.5, data = train_pca, family = gaussian())

# Summarize the model
summary(glm_model)

# Predict on the test set
test_pca$predicted_glm <- predict(glm_model, newdata = test_pca)

# Evaluate the model
mse_glm <- mean((test_pca$lipid_concentration - test_pca$predicted_glm)^2)
cat("Mean Squared Error (GLM): ", mse_glm, "\n")


cor(test_pca$predicted_glm, test_data$lipid_concentration)




# Prepare data for XGBoost
train_matrix <- as.matrix(train_pca %>% dplyr::select(-lipid_concentration, -genotype))
test_matrix <- as.matrix(test_pca %>% dplyr::select(-lipid_concentration, -genotype))
train_label <- train_pca$lipid_concentration
test_label <- test_pca$lipid_concentration

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# Train XGBoost model
params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 6)
xgb_model <- xgboost(params = params, data = dtrain, nrounds = 100)

# Predict on the test set
test_pca$predicted_xgb <- predict(xgb_model, newdata = dtest)

# Evaluate the model
mse_xgb <- mean((test_pca$lipid_concentration - test_pca$predicted_xgb)^2)
cor_xgb <- cor(test_pca$predicted_xgb, test_pca$lipid_concentration)
cat("Mean Squared Error (XGBoost): ", mse_xgb, "\n")
cat("Correlation (XGBoost): ", cor_xgb, "\n")




### using all
# Ensure predictors_hilo_lipids_all is a data frame
predictors_hilo_lipids_all <- as.data.frame(predictors_hilo_lipids_all)
data <- predictors_hilo_lipids_all
data$lipid_concentration <- PC_LPC_sum_hilo_lipids_all

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data$lipid_concentration, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]


# Remove columns with any NA values
train_data <- train_data %>% select_if(~ !any(is.na(.)))
test_data <- test_data %>% select_if(~ !any(is.na(.)))

# Ensure test data has the same columns as train data
common_cols <- intersect(colnames(train_data), colnames(test_data))
train_data <- train_data %>% dplyr::select(all_of(common_cols))
test_data <- test_data %>% dplyr::select(all_of(common_cols))


# Separate predictors and response
train_predictors <- train_data %>% dplyr::select(-lipid_concentration)
test_predictors <- test_data %>% dplyr::select(-lipid_concentration)
train_response <- train_data$lipid_concentration
test_response <- test_data$lipid_concentration


dtrain <- xgb.DMatrix(data = as.matrix(train_predictors[,-c(1,2)]), label = as.vector(train_response), missing = NA)
dtest <- xgb.DMatrix(data = as.matrix(test_predictors[,-c(1,2)]), label = test_response,missing = NA)

# Train XGBoost model

# Define cross-validation parameters
params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 6)
cv_folds <- 5  # Number of cross-validation folds
cv_results <- xgb.cv(params = params, data = ddata, nrounds = 100, nfold = cv_folds, metrics = "rmse", verbose = 0)

# Print cross-validation results
print(cv_results)

params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 6)
xgb_model <- xgboost(params = params, data = dtrain, nrounds = 100, verbose = 0)

# Predict on the test set
test_data$predicted_xgb <- predict(xgb_model, newdata = dtest)

# Evaluate the model
mse_xgb <- mean((test_data$lipid_concentration - test_data$predicted_xgb)^2)
cor_xgb <- cor(test_data$predicted_xgb, test_data$lipid_concentration)
cat("Mean Squared Error (XGBoost): ", mse_xgb, "\n")
cat("Correlation (XGBoost): ", cor_xgb, "\n")











# Define cross-validation parameters
params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 6)
cv_folds <- 5  # Number of cross-validation folds
cv_results <- xgb.cv(params = params, data = ddata, nrounds = 100, nfold = cv_folds, metrics = "rmse", verbose = 0)

# Print cross-validation results
print(cv_results)

# Extract the best iteration
best_nrounds <- cv_results$best_iteration

# Train the final model with the best number of rounds
xgb_model <- xgboost(params = params, data = ddata, nrounds = best_nrounds, verbose = 0)

# Cross-validated predictions
predictions <- predict(xgb_model, newdata = ddata)

# Evaluate the model using cross-validated predictions
mse_cv <- mean((response - predictions)^2)
cor_cv <- cor(predictions, response)
cat("Mean Squared Error (Cross-Validated): ", mse_cv, "\n")
cat("Correlation (Cross-Validated): ", cor_cv, "\n")