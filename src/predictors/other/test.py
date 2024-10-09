from numpy import concatenate, expand_dims
from keras.datasets import fashion_mnist

from vae import VAE
from visualisation_utils import plot_latent_space, plot_label_clusters

model = VAE(input_shape=(28, 28, 1), hidden_latent_dimension=2)


(x_train, y_train), (x_test, _) = fashion_mnist.load_data()

fashion_mnist_data = expand_dims(concatenate([x_train, x_test], axis=0), -1).astype("float32") / 255
model.fit(fashion_mnist_data, epochs=1, batch_size=128)

plot_latent_space(predictor=lambda sample:model.decoder.predict(sample, verbose=0))
 
x_train_ = expand_dims(x_train, -1).astype("float32") / 255
#print(x_train.shape) #(60000, 28, 28)
#print(x_train_.shape) #(60000, 28, 28, 1)
h_embedded, _, _ = model.encoder.predict(x_train_)

plot_label_clusters(
    inputs_embedded=h_embedded,
    outputs=y_train, 
    output_labels=[
        "T-shirt / top","Trouser", "Pullover","Dress","Coat","Sandal","Shirt", "Sneaker","Bag","Ankle boot"
    ]
)