# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling ---> Not required for this machine
# training_set = scale(training_set
# test_set = scale(test_set)

# fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)
# visualizing the Training set results
# install.packages('ggplot2') --> if required

library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = 'red') + 
  geom_line (aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
  color = 'blue') + 
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# visualizing the test set results

library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'green') + 
  geom_line (aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
             color = 'blue') + 
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary')