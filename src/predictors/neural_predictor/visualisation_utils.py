from numpy import zeros, linspace, array, arange, round
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
