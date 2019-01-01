from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input,Dense,BatchNormalization,Reshape,UpSampling2D,Conv2D,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_train = (X_train - 127.5) / 127.5

print(X_train.shape)

generator = Sequential([
        Dense(128*7*7, input_shape=(100,), activation='relu'),
        BatchNormalization(),
        Reshape((7,7,128)),
        UpSampling2D(),
        Conv2D(64, (5, 5),padding='same',activation='relu'),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(1, (5, 5),padding='same',activation='tanh')
    ])
print(generator.summary())

discriminator = Sequential([
        Conv2D(64, (5, 5), input_shape=(28,28,1),strides=(2,2), padding='same', activation='relu'),
        Dropout(0.3),
        Conv2D(128, (5, 5), strides=(2,2),padding='same', activation='relu'),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

print(discriminator.summary())

generator.compile(loss='binary_crossentropy', optimizer=Adam())
discriminator.compile(loss='binary_crossentropy', optimizer=Adam())
discriminator.trainable = False
ganInput = Input(shape=(100,))

x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=Adam())
gan.summary()

def train(epoch=10, batch_size=128):
    batch_count = X_train.shape[0] // batch_size
    
    for i in range(epoch):
        for j in tqdm(range(batch_count)):
            # Input for the generator
            noise_input = np.random.rand(batch_size, 100)
            
            # getting random images from X_train of size=batch_size 
            # these are the real images that will be fed to the discriminator
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
            
            # these are the predicted images from the generator
            predictions = generator.predict(noise_input, batch_size=batch_size)
            
            # the discriminator takes in the real images and the generated images
            X = np.concatenate([predictions, image_batch])
            
            # labels for the discriminator
            y_discriminator = [0]*batch_size + [1]*batch_size
            
            # Let's train the discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_discriminator)
            
            # Let's train the generator
            noise_input = np.random.rand(batch_size, 100)
            y_generator = [1]*batch_size
            discriminator.trainable = False
            gan.train_on_batch(noise_input, y_generator)
            
train(10,128)            
            
def plot_output():
    try_input = np.random.rand(100, 100)
    preds = generator.predict(try_input)

    plt.figure(figsize=(10,10))
    for i in range(preds.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(preds[i, :, :, 0], cmap='gray')
        plt.axis('off')
    
    # tight_layout minimizes the overlap between 2 sub-plots
    plt.tight_layout()
    
    
plot_output()