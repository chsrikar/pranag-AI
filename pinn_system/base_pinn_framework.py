"""
Module: base_pinn_framework.py
Purpose: Reusable DeepXDE-based wrapper that all physics models inherit from.

Hard-Constraint Design (Professor's Requirement)
------------------------------------------------
The output layer uses a **sigmoid** activation, which guarantees:
    0 < output < 1  for ALL inputs  — always, structurally.

This is a *hard* constraint baked into the architecture:
  - The model cannot explore invalid space (no clamping needed post-hoc).
  - The constraint is automatically respected during gradient descent.
  - It works perfectly with Min-Max normalised targets in [0, 1].

Hidden layers keep 'tanh' to preserve expressive power and gradient flow.
"""
import deepxde as dde
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BasePINNFramework:
    def __init__(self):
        self.model = None
        self.net = None
        self.geom = None
        self.timedomain = None

    def define_domain(self, geometry, time_range=None):
        """
        Define spatial and optionally temporal domain.
        """
        self.geom = geometry
        self.timedomain = time_range
        
    def build_network(
        self,
        inputs: int,
        outputs: int,
        hidden_layers: int = 4,
        neurons: int = 128,
        activation: str = "tanh",
        output_activation: str = "sigmoid",
    ):
        """
        Builds the Fully Connected Network (MLP) with a separate output activation.

        Parameters
        ----------
        activation        : Activation for ALL hidden layers (default: 'tanh').
        output_activation : Activation for the output layer (default: 'sigmoid').
                            'sigmoid' enforces 0 < output < 1 as a HARD constraint —
                            the model is structurally forbidden from leaving [0,1].
                            Set to None to use the same activation as hidden layers.
        """
        layer_size = [inputs] + [neurons] * hidden_layers + [outputs]

        if output_activation is not None and output_activation != activation:
            # DeepXDE FNN accepts a list of per-layer activations.
            # Length must equal number of weight layers = hidden_layers + 1.
            activations = [activation] * hidden_layers + [output_activation]
            self.net = dde.nn.FNN(layer_size, activations, "Glorot normal")
            logging.info(
                f"Network built: {layer_size}  "
                f"hidden={activation}  output={output_activation} (HARD bound)"
            )
        else:
            # Uniform activation across all layers (legacy / fallback)
            self.net = dde.nn.FNN(layer_size, activation, "Glorot normal")
            logging.info(f"Network built: {layer_size}  activation={activation}")

        return self.net

    def compile(self, optimizer="adam", learning_rate=1e-3, loss_weights=None):
        """
        Compiles the DeepXDE model.
        """
        if self.model is None:
            raise ValueError("Model is not built yet. Ensure subclass sets self.model = dde.Model(...)")
        
        kwargs = {}
        if loss_weights is not None:
            kwargs["loss_weights"] = loss_weights
            
        self.model.compile(optimizer, lr=learning_rate, **kwargs)
        logging.info(f"Model compiled with optimizer={optimizer}, lr={learning_rate}, loss_weights={loss_weights}")

    def train(self, iterations=1000):
        """
        Trains the PINN model.
        """
        if self.model is None:
            raise ValueError("Model not compiled/built.")
        
        logging.info(f"Training started for {iterations} iterations...")
        losshistory, train_state = self.model.train(iterations=iterations)
        return losshistory, train_state

    def predict(self, X):
        """
        Generates predictions.
        """
        if self.model is None:
            raise ValueError("Model not compiled/built.")
        return self.model.predict(X)
