from numpy import ndarray
from matplotlib.pyplot import subplots, show, tight_layout, legend

def plot_spacetime_diagrams(predicted:ndarray, target:ndarray) -> None:
    _, axes = subplots(1, 2)
    axes[0].set_title('Expected')
    axes[0].imshow(target)
    axes[1].set_title('Predicted')
    axes[1].imshow(predicted)

    tight_layout()
    show()


def plot_spacetime_diagrams_binarised(source:ndarray, target:ndarray, predicted:ndarray, batch_size:int, binary_threshold:int) -> None:
    _, axes = subplots(batch_size, 4)
    if batch_size==1:
        axes[0].imshow(source[0], cmap='gray')
        axes[0].set_title(f'Source')
        axes[0].axis('off') 

        axes[1].imshow(target[0], cmap='gray')
        axes[1].set_title(f'Target')
        axes[1].axis('off') 

        axes[2].imshow(predicted[0] > binary_threshold, cmap='gray')
        axes[2].set_title(f'Predicted (Binarised)')
        axes[2].axis('off') 

        axes[3].imshow(predicted[0], cmap='gray')
        axes[3].set_title(f'Predicted (Real)')
        axes[3].axis('off') 
    else:
        for b in range(batch_size):

            axes[b, 0].imshow(source[b], cmap='gray')
            axes[b, 0].set_title(f'Source {b+1}')
            axes[b, 0].axis('off') 

            axes[b, 1].imshow(target[b], cmap='gray')
            axes[b, 1].set_title(f'Target {b+1}')
            axes[b, 1].axis('off') 

            axes[b, 2].imshow(predicted[b] > binary_threshold, cmap='gray')
            axes[b, 2].set_title(f'Predicted (Binarised) {b+1}')
            axes[b, 2].axis('off') 

            axes[b, 3].imshow(predicted[b], cmap='gray')
            axes[b, 3].set_title(f'Predicted (Real) {b+1}')
            axes[b, 3].axis('off') 

    tight_layout()
    show()

def plot_trajectories(target:ndarray, predicted:ndarray, batch_size:int) -> None:
    _, axes = subplots(batch_size, 1)
    if batch_size==1:
        axes.plot(target[0], label='expected', color='b')
        axes.plot(predicted[0], label='predicted', linestyle=':', color='g')
    else:
        for b in range(batch_size):
            axes[b].set_title(f'Batch {b+1}')
            axes[b].plot(target[b], label='expected', color='b')
            axes[b].plot(predicted[b], label='predicted', linestyle=':', color='g')

    legend()
    tight_layout()
    show()