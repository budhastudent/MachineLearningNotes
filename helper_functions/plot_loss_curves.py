
# Plot the validation and training curves separately
def plot_loss_curves(model_fit_out):
  """
  Returns separe loss curves for training & validation metrics
  """
  loss = model_fit_out.history["loss"]
  val_loss = model_fit_out.history["val_loss"]
  accuracy = model_fit_out.history["accuracy"]
  val_accuracy = model_fit_out.history["val_accuracy"]
  epochs = range(len(model_fit_out.history["loss"]))  # how many epochs did we run for?

  # plot loss
  plt.plot(epochs, loss, label="Training loss")
  plt.plot(epochs, val_loss, label="Validation loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  # plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label="Training accuracy")
  plt.plot(epochs, val_accuracy, label="Validation Accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()
  plt.show()
