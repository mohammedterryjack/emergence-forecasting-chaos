
from tensorflow import shape, reduce_sum, GradientTape, reduce_mean, square, exp
from tensorflow.random import normal
from keras import Input, Model
from keras.layers import Layer, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from keras.metrics import Mean
from keras.losses import binary_crossentropy


class Sampling(Layer):
    def call(self, mean, log_var):
        """sample the hidden vector encoding of an input"""
        batch = shape(mean)[0]
        dimension = shape(mean)[1]
        epsilon = normal(shape=(batch, dimension))
        return mean + exp(0.5 * log_var) * epsilon

#TODO: continue refactoring from here...
class VAE(Model):
    def __init__(self) -> None:
        super().__init__()

        latent_dim = 2 
        encoder_inputs = Input(shape=(28, 28, 1))
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
        x = Flatten()(x)
        x = Dense(16, activation="relu")(x)
        mean = Dense(latent_dim, name="mean")(x)
        log_var = Dense(latent_dim, name="log_var")(x)
        z = Sampling()(mean, log_var)
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
    
   
