
## Installation

# install latest miniconda
# install latest RTools.exe
# install latest R version
install.packages("devtools")
library(devtools)
install.packages("keras")
library(keras)

# pip install keras

# Load the data from the website, any dataset you want (dataset_....())
# 10000 makes it manageable, 10000 most frequently used words
imdb <- dataset_imdb(num_words = 10000)

# Split the raw data into suitable groupings
# x = Input (pixel values train/test), y = Output (number categories train/test)
train_data <- imdb$train$x
train_labels <- imdb$train$y
test_data <- imdb$test$x
test_labels <- imdb$test$y

# train_labels and test_labels are lists of 0s and 1s, 
# where 0 stands for negative and 1 stands for positive
str(train_data[[1]])
train_labels[[1]]

#Because you're restricting yourself to the top 10,000 most frequent words, 
#no word index will exceed 10,000
max(sapply(train_data, max))


###############################################################################
# decode one of these reviews back to English words
# Named list mapping words to an integer index.
word_index <- dataset_imdb_word_index()  
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index


# Decodes the review. Note that the indices are offset by 3 because 0, 1, and 
# 2 are reserved indices for "padding," "start of sequence," and "unknown."
# change train data for different review
decoded_review <- sapply(train_data[[56]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "---"
})
cat(decoded_review)
################################################################################

################################################################
# PREPARING THE DATA

vectorize_sequences <- function(sequences, dimension = 10000) {
# Creates an all-zero matrix of shape (length(sequences), dimension)
results <- matrix(0, nrow = length(sequences), ncol = dimension) 
for (i in 1:length(sequences))
  # Sets specific indices of results[i] to 1s
  results[i, sequences[[i]]] <- 1 
results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# sample
str(x_train[1,])

# convert your labels from integer to numeric
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

###############################################################
# BUILDING THE NETWORK

model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

  
##############################################################
# LOSS FUNCTION AND OPTIMISER, 3 OPTIONs

# 1
# Binary CrossEntropy loss function
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# 2
# configure the parameters of your optimizer 
# can be done by passing an optimizer instance as the optimizer argument:
model %>% compile(
  optimizer = optimizer_rmsprop(lr=0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# 3
# Custom loss and metrics functions can be provided by 
# passing function objects as the loss and/or metrics arguments
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
  loss = loss_binary_crossentropy,
  metrics = metric_binary_accuracy
)

############################################################
# VALIDATING YOUR APPROACH

# Moniter during training accuracy of model on data it hasnt seen before
# Creating a validation set by setting apart 10 000 samples from original training data
val_indices <- 1:10000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

# Model for # epochs, in mini batches of 512 samples
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Monitoring of loss and accuracy on the 10 000 samples that were set apart
# You do so by passing the validation data as the validation_data argument
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 40,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

history

# Note that the call to 'fit()' returns a history object. 
# The history object has a 'plot()' method that enables us to visualize the 
# training and validation metrics by epoch
##################################################################################

# training loss decreases with every epoch
# training accuracy increases with every epoch

# That's what you would expect when running a gradient-descent optimization - 
# the quantity you're trying to minimize should be less with every iteration

# But that isn't the case for the validation loss and accuracy: 
# they seem to peak at the fourth epoch

# A model that performs better on the training data isn't necessarily a model that 
# will do better on data it has never seen before
# ie/ OVERFITTING

# to prevent overfitting, you could stop training after three epochs. 
# In general, you can use a range of techniques to mitigate overfitting


##################################################################################
# train a new network from scratch for four epochs and then 
# evaluate it on the test data
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)

results
# Loss and Accuracy

##############################################################
# GENERATING PREDICTIONS

# You can generate the likelihood of reviews being positive 
# by using the predict method
model %>% predict(x_test[1:100,])
