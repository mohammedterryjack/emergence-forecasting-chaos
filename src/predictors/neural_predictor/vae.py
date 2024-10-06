
from tensorflow import shape, reduce_sum, GradientTape, reduce_mean, square, exp, Tensor
from tensorflow.random import normal
from keras import Input, Model
from keras.layers import Layer, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from keras.metrics import Mean
from keras.losses import binary_crossentropy
from keras.optimizers import Adam


class Sampling(Layer):
    def call(self, mean:Dense, log_var:Dense):
        """sample the hidden vector encoding of an input"""
        batch = shape(mean)[0]
        dimension = shape(mean)[1]
        print(batch, dimension)
        epsilon = normal(shape=(batch, dimension))
        return mean + exp(0.5 * log_var) * epsilon

class VAE(Model):
    def __init__(
        self, 
        input_shape:tuple[int,int,int], 
        hidden_latent_dimension:int,
    ) -> None:
        
        super().__init__()
        
        self.encoder = self.create_encoder(input_shape=input_shape, hidden_latent_dimension=hidden_latent_dimension)
        self.decoder = self.create_decoder(hidden_latent_dimension=hidden_latent_dimension)
        
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

        self.compile(optimizer=Adam())

        print(self.encoder.summary())
        print(self.decoder.summary())

    @property
    def metrics(self) -> list[Mean]:
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, train_data) -> dict[str,float]:
        with GradientTape() as tape:
            mean,log_var, hidden_latent_layer = self.encoder(train_data)
            train_data_reconstructed = self.decoder(hidden_latent_layer)
            reconstruction_loss = self.calculate_reconstruction_loss(
                y=train_data,
                y_hat=train_data_reconstructed
            ) 
            kl_loss = self.calculate_kl_loss(
                mean=mean,
                log_var=log_var
            )
            total_loss=reconstruction_loss+kl_loss
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

    @staticmethod
    def create_decoder(hidden_latent_dimension:int) -> Model:
        latent_inputs = Input(shape=(hidden_latent_dimension,))
        layer1 = Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        layer2 = Reshape((7, 7, 64))(layer1)
        layer3 = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(layer2)
        layer4 = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(layer3)
        decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(layer4)
        return Model(latent_inputs, decoder_outputs, name="decoder")

    @staticmethod
    def create_encoder(input_shape:tuple[int,int,int], hidden_latent_dimension:int) -> Model:
        #TODO: (make input 1D vector, make output 1D vector)
        encoder_inputs = Input(shape=input_shape) 
        layer1 = Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        layer2 = Conv2D(128, 3, activation="relu", strides=2, padding="same")(layer1)
        layer3 = Flatten()(layer2)
        layer4 = Dense(16, activation="relu")(layer3)
        mean = Dense(hidden_latent_dimension, name="mean")(layer4)
        log_var = Dense(hidden_latent_dimension, name="log_var")(layer4)
        layer_latent_hidden = Sampling()(mean=mean, log_var=log_var)
        return Model(encoder_inputs, [mean, log_var, layer_latent_hidden], name="encoder")

    @staticmethod
    def calculate_reconstruction_loss(y:Tensor, y_hat:Tensor) -> Tensor:
        return reduce_mean(
            reduce_sum(
                binary_crossentropy(y, y_hat),
                axis=(1, 2),
            )
        )        

    @staticmethod
    def calculate_kl_loss(mean:Dense, log_var:Dense) -> Tensor:
        kl_loss_ = -0.5 * (1 + log_var - square(mean) - exp(log_var))
        return reduce_mean(reduce_sum(kl_loss_, axis=1)) 
