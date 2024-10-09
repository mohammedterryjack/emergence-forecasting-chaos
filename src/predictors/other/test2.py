from numpy import array
from matplotlib.pyplot import imshow, show 


x = array([
    [0,1,0,0,1,0,0,1,0],
    [0,1,1,0,1,0,0,0,0],
    [0,1,0,1,1,0,1,0,1],            
    [1,0,1,0,1,1,0,0,0],
])
y = array([
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0,0,0],            
])

imshow(y)
show()

from tensorflow import round 
from vae import VAE

model = VAE(input_shape=(9, 1), hidden_latent_dimension=10)
model.fit(x, y, epochs=50, batch_size=1)

_,_, z = model.encoder(x)
y_hat = model.decoder(z)

print(y_hat)

imshow(y_hat)
show()

y_hat_ = round(y_hat)
imshow(y_hat_)
show()
