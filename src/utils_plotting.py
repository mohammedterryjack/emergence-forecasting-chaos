from matplotlib.pyplot import subplots, show, tight_layout

from utils_projection import projector 

def plot_spacetime_diagrams(predicted:list[list[int]], target:list[list[int]], batch_size:int) -> None:
    _, axes = subplots(batch_size, 2)
    if batch_size==1:
        axes[0].set_title('Expected')
        axes[0].imshow(target[0], cmap='gray')
        axes[1].set_title('Predicted')
        axes[1].imshow(predicted[0], cmap='gray')
    else:
        for b in range(batch_size):
            axes[b,0].set_title('Expected')
            axes[b,0].imshow(target[b], cmap='gray')
            axes[b,1].set_title('Predicted')
            axes[b,1].imshow(predicted[b], cmap='gray')

    tight_layout()
    show()


def plot_trajectories(target:list[float], predicted:list[float], batch_size:int) -> None:
    _, axes = subplots(batch_size, 1)
    if batch_size==1:
        axes.plot(target[0], label='expected', color='b')
        axes.plot(predicted[0], label='predicted', linestyle=':', color='g')
        axes.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    else:
        for b in range(batch_size):
            axes[b].set_title(f'Batch {b+1}')
            axes[b].plot(target[b], label='expected', color='b')
            axes[b].plot(predicted[b], label='predicted', linestyle=':', color='g')
            axes[b].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    tight_layout()
    show()

def plot_results(target:list[list[int]], predicted:list[list[int]], batch_size:int, lattice_width:int) -> None:
    #TODO: combine both above in one graph
    #- highlight the parts on the spacetime where they differ using new color overlay
    #- project in here


    plot_trajectories(
        target=[
            [
                projector(
                    embedding=vector,
                    lattice_width=lattice_width
                ) for vector in target[b]
            ] for b in range(batch_size)
        ],
        predicted=[
            [
                projector(
                    embedding=vector,
                    lattice_width=lattice_width
                ) for vector in predicted[b]
            ] for b in range(batch_size)
        ],
        batch_size=batch_size
    )

    plot_spacetime_diagrams(
        target=target, 
        predicted=predicted,
        batch_size=batch_size
    )