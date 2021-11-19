from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):

  def __init__(self, dim_hidden, perceptrons_out):
    """
    dim_hidden: dimensions of hidden layers (hardcoded as dense layers)
                1st arg: n_layers
                2nd arg: n_perceptrons per layers
    perceptrons_out: n of perceptrons in output layer
    """
    super(MyModel, self).__init__()
    n_layers, n_perceptrons = dim_hidden
    self.hidden = [Dense(n_perceptrons, activation=tf.sigmoid)
                          for _ in range(n_layers)]
    self.out = Dense(perceptrons_out, activation=tf.nn.softmax)

    # for visualization of training
    self.test_losses = []
    self.test_accuracies = []
    self.training_losses = []

  @tf.function
  def call(self, x):
    """
    forward propagating the inputs through the network
    """
    for layer in self.hidden:
          x = layer(x)
    x = self.out(x)
    return x       


  def training_loop(self, train_dataset, test_dataset, num_epochs,learning_rate ):
    
    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()

    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer = tf.keras.optimizers.Adam(learning_rate)


    #testing once before we begin
    test_loss, test_accuracy = self.test( test_dataset, cross_entropy_loss)
    self.test_losses.append(test_loss)
    self.test_accuracies.append(test_accuracy)
   
    #check how model performs on train data once before we begin
    train_loss, _ = self.test( train_dataset, cross_entropy_loss)
    self.training_losses.append(train_loss)
    
    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {self.test_accuracies[-1]}')

        #training (and checking in with training)
        epoch_loss_agg = []
        for input,target in train_dataset:
            train_loss = self.train_step( input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)

        #track training loss
        self.training_losses.append(tf.reduce_mean(epoch_loss_agg))

        #testing, so we can track accuracy and test loss
        
        test_loss, test_accuracy = self.test( test_dataset, cross_entropy_loss)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_accuracy)


  def train_step(self, input, target, loss_function, optimizer):
    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
      prediction = self(input)
      loss = loss_function(target, prediction)
      gradients = tape.gradient(loss, self.trainable_variables)
    optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    
    return loss

  def test(self, test_data, loss_function):
    # test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:

      prediction = self(input)
      sample_test_loss = loss_function(target, prediction)
      sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
      sample_test_accuracy = np.mean(sample_test_accuracy)
      test_loss_aggregator.append(sample_test_loss.numpy())
      test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy
