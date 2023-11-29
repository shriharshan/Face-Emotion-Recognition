# Deep Learning Face Emotion Recognition: Code Explanation

<h2>Directory Setup</h2>

```
train_dir = "train"
test_dir = "test"
```
<h4>Specify the directories where training and testing image data is located.</h4>

<h2>Data Preprocessing</h2>

```
dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.2)
```
<h4>Create an image data generator for preprocessing. Images are rescaled to [0, 1], horizontally flipped, and divided into training and validation subsets.</h4>

<h2>Validation Data Generator</h2>

```
validation_set = dataGenerator.flow_from_directory(train_dir, batch_size=64, target_size=(48, 48), shuffle=True, color_mode='grayscale', class_mode='categorical', subset='validation')
```
<h4>Create a data generator for validation images with similar preprocessing as training dat</h4>

<h2>Testing Data Generator</h2>

```
testDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_data = testDataGenerator.flow_from_directory(test_dir, batch_size=64, target_size=(48, 48), shuffle=True, color_mode='grayscale', class_mode='categorical')
```
<h4>Prepare a data generator for testing images, applying rescaling and horizontal flipping.</h4>

<h2>Model Architecture</h2>

```
def create_model():
    weight_decay = 1e-4
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (4, 4), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=(48, 48, 1)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (4, 4), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(128, (4, 4), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(128, (4, 4), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (4, 4), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="linear"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    
    return model
```
<h4> This architecture outlines how the neural network processes and learns from the input images.<br><br></h4>

```
def create_model():
    weight_decay = 1e-4
    model = tf.keras.models.Sequential()

```
<h4>

   -  We start by defining a regularization parameter weight_decay to control the amount of regularization applied to the model. Regularization helps prevent overfitting by discouraging the model from becoming overly complex.<br><br>
   -  We then create an instance of tf.keras.models.Sequential(), which represents a linear stack of layers. This is the foundational structure of the neural network.<br><br>
</h4>

```
    model.add(tf.keras.layers.Conv2D(64, (4, 4), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), input_shape=(48, 48, 1)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
```
  </h4>

  - Here, we're adding the first convolutional layer (Conv2D). This layer detects different features in the input images using small filters (also called kernels). The 64 indicates the number of filters, and (4, 4) specifies the size of each filter.

  - padding='same' ensures that the output feature maps have the same dimensions as the input, and kernel_regularizer=tf.keras.regularizers.l2(weight_decay) applies L2 regularization to the layer's weights.

  - input_shape=(48, 48, 1) defines the input shape of the images, which are resized to 48x48 pixels and have only one channel (grayscale).

  - After the convolutional layer, we apply the ReLU activation function (Activation('relu')) to introduce non-linearity.

  - The BatchNormalization layer normalizes the output of the previous layer, helping stabilize and speed up the training process.</h4>

  ```
      model.add(tf.keras.layers.Conv2D(64, (4, 4), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
  ```
<h4>

  - This section adds another convolutional layer, followed by ReLU activation and batch normalization, as explained earlier.

  - The MaxPool2D layer performs pooling by taking the maximum value within a specified window (pool_size=(2, 2) in this case), reducing the spatial dimensions of the feature maps.

  - Dropout(0.2) introduces dropout regularization by randomly setting a fraction of input units to zero during training. This prevents the model from relying too heavily on any specific feature.

  - This pattern of adding convolutional layers, activation functions, batch normalization, pooling, and dropout repeats in the subsequent sections of the create_model function. The architecture gradually transforms the image data, extracting essential features and patterns, and reduces the spatial dimensions.</h4>

```
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="linear"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    
    return model
```
<h4>

- Flatten() reshapes the 2D feature maps into a 1D vector, preparing them for fully connected layers.

- Dense(128, activation="linear") is a fully connected (dense) layer with 128 neurons and a linear activation function. This layer captures complex combinations of features.

- Activation('relu') applies ReLU activation to the output of the dense layer, adding non-linearity.

- The final Dense layer with 7 neurons and a softmax activation function performs the classification. It converts the model's outputs into probability scores for each class.
</h4>

<h2>Model Compilation</h2>

```
model = create_model()
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0003), metrics=['accuracy'])
```
<h4>
Instantiate the model and compile it for training, specifying the loss function, optimizer, and evaluation metric.
</h4>

<h2>Training Callbacks</h2>

```
checkpointer = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=1, restore_best_weights=True, mode="max", patience=10),
                tf.keras.callbacks.ModelCheckpoint(filepath='final_model_weights.hdf5', monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")]
```

<h4>Set up callbacks to control training, including early stopping and model checkpointing.</h4>

<h2>Training Process</h2>

```
steps_per_epoch = training_data.n // training_data.batch_size
validation_steps = validation_set.n // validation_set.batch_size
history = model.fit(x=training_data, validation_data=validation_set, epochs=100, callbacks=[checkpointer], steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
```

<h4>Initiate the model training process, utilizing the training and validation data generators, along with the defined callbacks.</h4>

<h2>Testing Performance</h2>

```
print(f"Test accuracy = {model.evaluate(test_data, batch_size=test_data.batch_size, steps=test_data.n // test_data.batch_size)[1]*100}%")
```
<h4>Evaluate the trained model's performance on the test data and print the test accuracy.</h4>