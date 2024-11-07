from numpy import array
from matplotlib.pyplot import subplots, show, subplots_adjust

from utils.projection import projector
from utils.evaluation import errors

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
    show()

def plot_results(target:list[list[int]], predicted:list[list[int]], timesteps:list[int], batch_size:int, lattice_width:int) -> None:

    target_projected=[
        [
            projector(
                embedding=vector,
                lattice_width=lattice_width,
            ) for vector in target[b]
        ] for b in range(batch_size)
    ]
    predicted_projected=[
        [
            projector(
                embedding=vector,
                lattice_width=lattice_width,
            ) for vector in predicted[b]
        ] for b in range(batch_size)
    ]
    scores = errors(
        target=target,
        predicted=predicted,
        batch_size=batch_size,
        lattice_width=lattice_width
    )    

    _, axes = subplots(batch_size, 4, sharey=True)
    if batch_size==1:
        axes[0].set_title('Expected')
        axes[0].imshow(target[0], cmap='gray')
        axes[1].set_title('Predicted')
        axes[1].imshow(predicted[0], cmap='gray')
        axes[2].plot(target_projected[0],timesteps, label='expected', color='b')
        axes[2].plot(predicted_projected[0],timesteps, label='predicted', linestyle=':', color='g')
        axes[2].invert_yaxis()
        axes[2].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        axes[3].plot(scores['distance'][0],timesteps, label='cosine distance', color='orange')
        axes[3].plot(scores['mae'][0],timesteps, label='MAE', color='r')
        axes[3].plot(scores['rmse'][0],timesteps, label='RMSE', color='gray')
        axes[3].invert_yaxis()
        axes[3].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    else:
        for b in range(batch_size):
            axes[b, 0].imshow(target[b], cmap='gray')
            axes[b, 1].imshow(predicted[b], cmap='gray')
            axes[b, 2].plot(target_projected[b],timesteps, label='expected', color='b')
            axes[b, 2].plot(predicted_projected[b], timesteps, label='predicted', linestyle=':', color='g')
            axes[b, 2].invert_yaxis()
            axes[b, 3].plot(scores['distance'][b],timesteps, label='cosine distance', color='orange')
            axes[b, 3].plot(scores['mae'][b],timesteps, label='MAE', color='r')
            axes[b, 3].plot(scores['rmse'][b],timesteps, label='RMSE', color='gray')
            axes[b, 3].invert_yaxis()

        #sharey
        lowers2,uppers2 = [],[]
        lowers3,uppers3 = [],[]
        for b in range(batch_size):
            lower2,upper2 = axes[b, 2].get_xlim()
            lower3,upper3 = axes[b, 3].get_xlim()
            lowers2.append(lower2)
            uppers2.append(upper2)
            lowers3.append(lower3)
            uppers3.append(upper3)
        lowest2 = min(lowers2)
        uppest2 = max(uppers2)
        lowest3 = min(lowers3)
        uppest3 = max(uppers3)

        for b in range(batch_size):
            axes[b, 2].set_xlim(lowest2,uppest2)
            axes[b, 3].set_xlim(lowest3,uppest3)

        axes[-1, 2].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        axes[-1, 3].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    show()