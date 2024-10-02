from numpy import zeros, linspace, array, arange, round, ndarray
from matplotlib.pyplot import figure, xticks, yticks, xlabel, ylabel, show, imshow, scatter, colorbar

def plot_latent_space(predictor:callable, n:int=10, image_size:int = 28, scale:float = 0.5) -> None:
    """display a n*n 2D manifold of images"""
    latent_space = zeros((image_size * n, image_size * n))
    grid_x = linspace(-scale, scale, n)
    grid_y = linspace(-scale, scale, n)[::-1] 
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            sample = array([[xi, yi]])
            x_decoded = predictor(sample) 
            images = x_decoded[0].reshape(image_size, image_size)
            latent_space[
                i * image_size : (i + 1) * image_size,
                j * image_size : (j + 1) * image_size,
            ] = images
 
    figure(figsize=(5, 5))
    start_range = image_size // 2
    end_range = n * image_size + start_range
    pixel_range = arange(start_range, end_range, image_size)
    sample_range_x = round(grid_x, 1)
    sample_range_y = round(grid_y, 1)
    xticks(pixel_range, sample_range_x)
    yticks(pixel_range, sample_range_y)
    xlabel("z[0]")
    ylabel("z[1]")
    imshow(latent_space, cmap="Greys_r")
    show()

def plot_label_clusters(inputs_embedded:list[ndarray], outputs:list[int], output_labels:list[str]) -> None:
    figure(figsize =(12, 10))
    sc = scatter(inputs_embedded[:, 0], inputs_embedded[:, 1], c = outputs)
    cbar = colorbar(sc, ticks = range(len(output_labels)))
    cbar.ax.set_yticklabels(output_labels)
    xlabel("z[0]")
    ylabel("z[1]")
    show()
