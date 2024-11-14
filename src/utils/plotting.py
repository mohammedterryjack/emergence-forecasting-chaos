from matplotlib.pyplot import subplots, show
from numpy import array

from utils.projection import projector
from utils.evaluation import errors, time_steps_for_good_forecast

def plot_results_with_emergence(
    real:list[list[int]],
    predicted:list[list[int]],
    lattice_width:int,
    emergence_spacetime_filter:callable
) -> None:
    real = array(real)
    predicted = array(predicted)
    emergent_real = array(emergence_spacetime_filter(real))
    emergent_predicted = array(emergence_spacetime_filter(predicted))
    projected_real=[
        projector(
            embedding=vector,
            lattice_width=lattice_width,
        ) for vector in real
    ]
    projected_predicted=[
        projector(
            embedding=vector,
            lattice_width=lattice_width,
        ) for vector in predicted
    ]
    projected_emergent_real=[
        projector(
            embedding=vector,
            lattice_width=lattice_width,
        ) for vector in emergent_real
    ]
    projected_emergent_predicted=[
        projector(
            embedding=vector,
            lattice_width=lattice_width,
        ) for vector in emergent_predicted
    ]

    scores = errors(
        target=[real],
        predicted=[predicted],
        batch_size=1,
        lattice_width=lattice_width
    )    
    scores_emergent = errors(
        target=[emergent_real],
        predicted=[emergent_predicted],
        batch_size=1,
        lattice_width=lattice_width
    )    

    _, axes = subplots(5, 2, sharex=True)

    axes[0,0].set_title('Spacetime')
    axes[0,0].imshow(real.T, cmap='gray', aspect='auto')
    axes[1,0].imshow(predicted.T, cmap='gray', aspect='auto')
    axes[2,0].imshow(real.T, cmap='summer', alpha=0.5, aspect='auto')
    axes[2,0].imshow(predicted.T, cmap='winter', alpha=0.5, aspect='auto')
    axes[3,0].plot(projected_real, label='expected', color='b')
    axes[3,0].plot(projected_predicted, label='predicted', linestyle=':', color='g')
    axes[4,0].plot(scores['distance'][0], label='cosine distance', color='orange')
    axes[4,0].plot(scores['mae'][0], label='MAE', color='r')
    axes[4,0].plot(scores['rmse'][0], label='RMSE', color='gray')
    axes[4,0].plot(scores['bce'][0], label='BCE', color='pink')

    axes[0,1].set_title('Emergent Properties')
    axes[0,1].imshow(emergent_real.T, cmap='gray', aspect='auto')
    axes[1,1].imshow(emergent_predicted.T, cmap='gray', aspect='auto')
    axes[2,1].imshow(emergent_real.T, cmap='summer', alpha=0.5, aspect='auto')
    axes[2,1].imshow(emergent_predicted.T, cmap='winter', alpha=0.5, aspect='auto')
    axes[3,1].plot(projected_emergent_real, label='expected', color='b')
    axes[3,1].plot(projected_emergent_predicted, label='predicted', linestyle=':', color='g')
    axes[3,1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=1)
    axes[4,1].plot(scores_emergent['distance'][0], label='cosine distance', color='orange')
    axes[4,1].plot(scores_emergent['mae'][0], label='MAE', color='r')
    axes[4,1].plot(scores_emergent['rmse'][0], label='RMSE', color='gray')
    axes[4,1].plot(scores_emergent['bce'][0], label='BCE', color='pink')
    axes[4,1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    show()

def plot_results(
    target:list[list[int]], 
    predicted:list[list[int]], 
    timesteps:list[int], 
    batch_size:int, 
    lattice_width:int
) -> None:

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
    
    ts = time_steps_for_good_forecast(
        errors=scores['mae'],
        threshold=0.5,
        batch_size=batch_size
    )
    print(ts)
    
    _, axes = subplots(batch_size, 5, sharey=True)
    if batch_size==1:
        axes[0].set_title('Expected')
        axes[0].imshow(target[0], cmap='gray')
        axes[1].set_title('Predicted')
        axes[1].imshow(predicted[0], cmap='gray')
        axes[2].imshow(target[0], cmap='summer', alpha=0.5)
        axes[2].imshow(predicted[0], cmap='winter', alpha=0.5)
        axes[3].plot(target_projected[0],timesteps, label='expected', color='b')
        axes[3].plot(predicted_projected[0],timesteps, label='predicted', linestyle=':', color='g')
        axes[3].invert_yaxis()
        axes[3].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        axes[4].plot(scores['distance'][0],timesteps, label='cosine distance', color='orange')
        axes[4].plot(scores['mae'][0],timesteps, label='MAE', color='r')
        axes[4].plot(scores['rmse'][0],timesteps, label='RMSE', color='gray')
        axes[4].plot(scores['bce'][0],timesteps, label='BCE', color='pink')
        axes[4].invert_yaxis()
        axes[4].legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    else:
        for b in range(batch_size):
            axes[b, 0].imshow(target[b], cmap='gray')
            axes[b, 1].imshow(predicted[b], cmap='gray')
            axes[b, 2].imshow(target[b], cmap='summer', alpha=0.5)
            axes[b, 2].imshow(predicted[b], cmap='winter', alpha=0.5)
            axes[b, 3].plot(target_projected[b],timesteps, label='expected', color='b')
            axes[b, 3].plot(predicted_projected[b], timesteps, label='predicted', linestyle=':', color='g')
            axes[b, 3].invert_yaxis()
            axes[b, 4].plot(scores['distance'][b],timesteps, label='cosine distance', color='orange')
            axes[b, 4].plot(scores['mae'][b],timesteps, label='MAE', color='r')
            axes[b, 4].plot(scores['rmse'][b],timesteps, label='RMSE', color='gray')
            axes[b, 4].plot(scores['bce'][b],timesteps, label='BCE', color='pink')
            axes[b, 4].invert_yaxis()

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
