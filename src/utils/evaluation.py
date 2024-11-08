from numpy import array, clip, mean, log, ndarray, where

from utils.projection import cosine_similarity

def time_steps_for_good_forecast(
    errors:list[list[float]], 
    threshold:float, 
    batch_size:int
) -> list[int]:
    results = []
    for b in range(batch_size):
        indices = where(errors[b] > threshold)[0]
        timesteps = indices[0] if indices.size > 0 else None
        results.append(timesteps)
    return results

def bce(target:ndarray, predicted:ndarray, lattice_width:int, epsilon:float = 1e-15) -> ndarray:    
    target_probabilities = target/lattice_width
    predicted_probabilities = predicted/lattice_width
    predicted_probabilities = clip(predicted_probabilities, epsilon, 1 - epsilon)    
    return -mean(target_probabilities * log(predicted_probabilities) + (1 - target_probabilities) * log(1 - predicted_probabilities))

def errors(target:list[list[int]], predicted:list[list[int]], batch_size:int, lattice_width:int) -> dict[str,list[list[float]]]:
    cosine_distances=[
        [
            1.-cosine_similarity(a=t,b=p)
            for t,p in zip(target[b],predicted[b])
        ] for b in range(batch_size)
    ]
    mean_absolute_errors = array(
        [
            [
                sum(abs(p-t))/lattice_width for p,t in zip(predicted[b],target[b])
            ]
            for b in range(batch_size)
        ]
    )
    root_mean_squared_errors = array(
        [
            [
                (sum((p-t)**2)/lattice_width)**0.5 for p,t in zip(predicted[b],target[b])
            ]
            for b in range(batch_size)
        ]
    )

    binary_cross_entropy = array(
        [
            [
                bce(target=t,predicted=p, lattice_width=lattice_width) for p,t in zip(predicted[b], target[b])
            ]
            for b in range(batch_size)
        ]
    )

    return dict(
        distance=cosine_distances,
        mae=mean_absolute_errors,
        rmse=root_mean_squared_errors,
        bce=binary_cross_entropy
    )

