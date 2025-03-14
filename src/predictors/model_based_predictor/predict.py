from numpy import ndarray, array, append
from torch import no_grad, tensor

from predictors.model_based_predictor.transformer import Transformer

def predict_next(model:Transformer, source:ndarray, target:ndarray, return_distribution:bool=False) -> ndarray:
    """
    Given a sequence of integers as a numpy array
    the model will predict the next in the sequece
    If return_distribution is False
    the output is given as an integer like the input
    otherwise the raw distribution over the target vocab is returned"""
    predictions_tensor = model(src=tensor(source), tgt=tensor(target))
    predictions = predictions_tensor.detach().numpy()
    return predictions if return_distribution else predictions.argmax(-1)



def predict_n(
    model:Transformer, 
    source:ndarray,
    target:ndarray,
    batch_size:int,
    forecast_horizon:int,
) -> ndarray:
    """autoregressively predict next n steps in sequence"""
    model.eval()
    with no_grad():            
        for iteration in range(forecast_horizon):
            predicted_next_indexes_tgt = predict_next(
                model=model,
                source=source,
                target=target,
                return_distribution=False
            )[:,-1:]
            
            target = array([
                append(target[b], predicted_next_indexes_tgt[b])
                for b in range(batch_size)
            ])

    return target


def predict_n_encoded(
    model:Transformer, 
    source:ndarray,
    target:ndarray,
    batch_size:int,
    forecast_horizon:int,
    vector_to_index:callable
) -> ndarray:
    """autoregressively predict next n steps in sequence"""
    model.eval()
    with no_grad():            
        for iteration in range(forecast_horizon):
            predicted_next_vectors = predict_next(
                model=model,
                source=source,
                target=target,
                return_distribution=True
            )[:,-1:]

            predicted_next_indexes_tgt = array([
                [
                    vector_to_index(vector) for vector in predicted_next_vectors[b]
                ]
                for b in range(batch_size)
            ])

            target = array([
                append(target[b], predicted_next_indexes_tgt[b])
                for b in range(batch_size)
            ])

    return target