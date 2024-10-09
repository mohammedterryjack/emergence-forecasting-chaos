
from tensorflow import shape, reduce_sum, GradientTape, reduce_mean, square, exp, Tensor, print, round
from tensorflow.random import normal
from keras import Input, Model
from keras.layers import Layer, Flatten, Dense
from keras.metrics import Mean
from keras.losses import binary_crossentropy
from keras.optimizers import Adam


class Sampling(Layer):
    def call(self, mean:Dense, log_var:Dense):
        """sample the hidden vector encoding of an input"""
        batch = shape(mean)[0]
        dimension = shape(mean)[1]
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
        x, y = train_data
        with GradientTape() as tape:
            mean,log_var, hidden_latent_layer = self.encoder(x)
            y_reconstructed = self.decoder(hidden_latent_layer)
            reconstruction_loss = reduce_mean(binary_crossentropy(y, y_reconstructed)) 
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
        layer1 = Dense(256, activation="relu")(latent_inputs)
        layer2 = Dense(128, activation="relu")(layer1)
        layer3 = Dense(64, activation="relu")(layer2)
        decoder_outputs = Dense(9, activation="sigmoid")(layer3)
        return Model(latent_inputs, decoder_outputs, name="decoder")

    @staticmethod
    def create_encoder(input_shape:tuple[int,int,int], hidden_latent_dimension:int) -> Model:
        encoder_inputs = Input(shape=input_shape) 
        layer1 = Flatten()(encoder_inputs)
        layer2 = Dense(64, activation="relu")(layer1)
        layer3 = Dense(128, activation="relu")(layer2)
        layer4 = Dense(256, activation="relu")(layer3)
        mean = Dense(hidden_latent_dimension, name="mean")(layer4)
        log_var = Dense(hidden_latent_dimension, name="log_var")(layer4)
        layer_latent_hidden = Sampling()(mean=mean, log_var=log_var)
        return Model(encoder_inputs, [mean, log_var, layer_latent_hidden], name="encoder")


    @staticmethod
    def calculate_kl_loss(mean:Dense, log_var:Dense) -> Tensor:
        kl_loss_ = -0.5 * (1 + log_var - square(mean) - exp(log_var))
        return reduce_mean(reduce_sum(kl_loss_, axis=1)) 
