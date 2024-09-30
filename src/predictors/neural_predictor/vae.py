from numpy import concatenate, expand_dims, zeros, linspace, array, arange, round
from tensorflow import shape, reduce_sum, GradientTape, reduce_mean, square, exp
from tensorflow.random import normal
from keras import Input, Model
from keras.layers import Layer, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from keras.metrics import Mean
from keras.losses import binary_crossentropy
from keras.datasets import fashion_mnist
from keras.optimizers import Adam

from matplotlib.pyplot import figure, xticks, yticks, xlabel, ylabel, show, imshow, scatter, colorbar

def plot_latent_space(vae, n=10, figsize=5):
    # display a n*n 2D manifold of images
    img_size = 28
    scale = 0.5
    latent_space = zeros((img_size * n, img_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of images classes in the latent space
    grid_x = linspace(-scale, scale, n)
    grid_y = linspace(-scale, scale, n)[::-1]
 
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            sample = array([[xi, yi]])
            x_decoded = vae.decoder.predict(sample, verbose=0)
            images = x_decoded[0].reshape(img_size, img_size)
            latent_space[
                i * img_size : (i + 1) * img_size,
                j * img_size : (j + 1) * img_size,
            ] = images
 
    figure(figsize=(figsize, figsize))
    start_range = img_size // 2
    end_range = n * img_size + start_range
    pixel_range = arange(start_range, end_range, img_size)
    sample_range_x = round(grid_x, 1)
    sample_range_y = round(grid_y, 1)
    xticks(pixel_range, sample_range_x)
    yticks(pixel_range, sample_range_y)
    xlabel("z[0]")
    ylabel("z[1]")
    imshow(latent_space, cmap="Greys_r")
    show()

def plot_label_clusters(encoder, decoder, data, test_lab):

    labels = {0    :"T-shirt / top",
    1:    "Trouser",
    2:    "Pullover",
    3:    "Dress",
    4:    "Coat",
    5:    "Sandal",
    6:    "Shirt",
    7:    "Sneaker",
    8:    "Bag",
    9:    "Ankle boot"}
    z_mean, _, _ = encoder.predict(data)
    figure(figsize =(12, 10))
    sc = scatter(z_mean[:, 0], z_mean[:, 1], c = test_lab)
    cbar = colorbar(sc, ticks = range(10))
    cbar.ax.set_yticklabels([labels.get(i) for i in range(10)])
    xlabel("z[0]")
    ylabel("z[1]")
    show()


class Sampling(Layer):
    """Uses (mean, log_var) to sample z, the vector encoding a digit."""
 
    def call(self, inputs):
        mean, log_var = inputs
        batch = shape(mean)[0]
        dim = shape(mean)[1]
        epsilon = normal(shape=(batch, dim))
        return mean + exp(0.5 * log_var) * epsilon

latent_dim = 2
 
encoder_inputs = Input(shape=(28, 28, 1))
x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = Flatten()(x)
x = Dense(16, activation="relu")(x)
mean = Dense(latent_dim, name="mean")(x)
log_var = Dense(latent_dim, name="log_var")(x)
z = Sampling()([mean, log_var])
encoder = Model(encoder_inputs, [mean, log_var, z], name="encoder")
print(encoder.summary())


latent_inputs = Input(shape=(latent_dim,))
x = Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")
print(decoder.summary())


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = Mean(name="kl_loss")
 
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
 
    def train_step(self, data):
        with GradientTape() as tape:
            mean,log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = reduce_mean(
                reduce_sum(
                    binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + log_var - square(mean) - exp(log_var))
            kl_loss = reduce_mean(reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
   
(x_train, y_train), (x_test, _) = fashion_mnist.load_data()
fashion_mnist_data = concatenate([x_train, x_test], axis=0)
fashion_mnist_data = expand_dims(fashion_mnist_data, -1).astype("float32") / 255
 
vae = VAE(encoder, decoder)
vae.compile(optimizer=Adam())
vae.fit(fashion_mnist_data, epochs=10, batch_size=128)

plot_latent_space(vae)
 
x_train = expand_dims(x_train, -1).astype("float32") / 255
plot_label_clusters(encoder, decoder, x_train, y_train)