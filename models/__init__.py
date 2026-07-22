from .mamba import MambaRiskPredictor
from .rnn import GRURiskPredictor, LSTMRiskPredictor, RNNRiskPredictor
from .transformer import TransformerRiskPredictor


MODEL_REGISTRY = {
    "rnn": RNNRiskPredictor,
    "gru": GRURiskPredictor,
    "lstm": LSTMRiskPredictor,
    "transformer": TransformerRiskPredictor,
    "mamba": MambaRiskPredictor,
}


def build_model(
    name: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
):
    try:
        model_class = MODEL_REGISTRY[name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    return model_class(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
