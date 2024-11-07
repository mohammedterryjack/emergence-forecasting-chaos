from numpy import array 

from utils.projection import cosine_similarity

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
    return dict(
        distance=cosine_distances,
        mae=mean_absolute_errors,
        rmse=root_mean_squared_errors
    )
