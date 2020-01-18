# Polynomial Regression

# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Lines 10-20 not required for tutorial at this time. Dataset does not need splitting nor scaling.
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
#library(caTools)
#set.seed(123)
#split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Step 2
# Fitting Linear Regression to dataset (to see a difference)
lin_reg = lm(formula = Salary ~ .,
             data = dataset)
# Fitting Polynomial Regression to dataset (more appropriate for the dataset)
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
# Visualizing the Linear Regression
# install, if needed // install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Truth of Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualizing the Polynomial Regression
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Truth of Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')
# Step 4
# Predicting a new result with the Linear Regression (@Level 6.5)
y_pred = predict(lin_reg, data.frame(Level = 6.5))
# Predicting a new result with the Polynomial Regression (=$158+ vs. $330K+ in linear)
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))