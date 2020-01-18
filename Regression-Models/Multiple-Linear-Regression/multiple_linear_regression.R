# Multiple Linear Regression

# Data Preprocessing Template
# Importing the dataset
dataset = read.csv('50_Startups.csv')
# dataset = dataset[, 2:3]

# Data Preprocessing

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the sataset into the Training set and Test Set
# installed.packages('caTools')

library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# # Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])
regressor = lm(formula = Profit ~ .,
               data = training_set)
# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

#Independent variables (added with +) Which variable is significant?
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
# same as line 27 :)
# 1st Run of Backward Elimination -->
summary(regressor)

# 4th Backward, remove Marketing.Spend
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
# Final Backward
summary(regressor)
