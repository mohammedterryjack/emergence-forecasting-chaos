from numpy import ndarray
from matplotlib.pyplot import subplots, show, tight_layout, legend

def plot_spacetime_diagrams(predicted:ndarray, target:ndarray, batch_size:int) -> None:
    _, axes = subplots(batch_size, 2)
    if batch_size==1:
        axes[0].set_title('Expected')
        axes[0].imshow(target[0])
        axes[1].set_title('Predicted')
        axes[1].imshow(predicted[0])
    else:
        for b in range(batch_size):
            axes[b,0].set_title('Expected')
            axes[b,0].imshow(target[b])
            axes[b,1].set_title('Predicted')
            axes[b,1].imshow(predicted[b])

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