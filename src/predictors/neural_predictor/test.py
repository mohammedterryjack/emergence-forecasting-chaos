from numpy import concatenate, expand_dims
from keras.datasets import fashion_mnist
from keras.optimizers import Adam

from vae import VAE, encoder, decoder
from visualisation_utils import plot_latent_space, plot_label_clusters

(x_train, y_train), (x_test, _) = fashion_mnist.load_data()
fashion_mnist_data = concatenate([x_train, x_test], axis=0)
fashion_mnist_data = expand_dims(fashion_mnist_data, -1).astype("float32") / 255
 
vae = VAE(encoder, decoder)
vae.compile(optimizer=Adam())
vae.fit(fashion_mnist_data, epochs=10, batch_size=128)

plot_latent_space(vae)
 
x_train = expand_dims(x_train, -1).astype("float32") / 255
plot_label_clusters(encoder, decoder, x_train, y_train)