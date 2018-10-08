# Question 1
# Read csv file

data.df <- read.csv("train.csv", header = TRUE)


# Question 2

# Number of Rows
data.df.n_rows <- nrow(data.df)
# Number of Columns
data.df.n_cols  <- ncol(data.df)


# Question 3
data.df.subset <-subset(data.df , select = c("PassengerId", "Age", "Fare", "Embarked"))

# Question 4

# Make all the empty blanks to NA in the table
data.df.subset[data.df.subset==""]  <- NA 

  # part a
# Replace NA in Age by median of Age
data.df.subset$Age[is.na(data.df.subset$Age)] <-median(data.df.subset$Age, na.rm=TRUE)

  # part b
# Function for Mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Mode of Embarked
data.df.subset$Embarked[is.na(data.df.subset$Embarked)] <-Mode(data.df.subset$Embarked)

# part c
# Replace the NA of Fare with median
#There are no missing values in the data.df.subset


# Question 5
  #part a
hist(data.df.subset$Age, main="Histogram for Age", xlab = "Age", ylab = "Frequency")
  #part b
plot(data.df.subset$Age, data.df.subset$Fare, main="Scatterplot between Age and Fare", xlab=" Age ", ylab="Fare ", pch=19)

# Question 6
# Mean of Age
age_mean = mean(data.df.subset$Age)
# Standard Deviation of Age
std_dev = sd(data.df.subset$Age)

# Anamoly Detection
anomalous_indices <- c(data.df.subset[which ((data.df.subset$Age > (age_mean + std_dev)) | (data.df.subset$Age < (age_mean - std_dev))), "PassengerId"])


#Question 7
# Subsetting of data
data.df.subset.v2 <-data.df.subset[which ((data.df.subset$Age >= 25) & (data.df.subset$Age <= 80)), c("Age","Fare","Embarked")]


#Question 8
#Max fare calculation
maxFare <-max(data.df.subset.v2$Fare)
# Min fare calculation
minFare <-min(data.df.subset.v2$Fare)
# Calculation of Fare_Rescaled
data.df.subset.v2$Fare_Rescaled <-((data.df.subset.v2$Fare - minFare)/(maxFare-minFare))*100

