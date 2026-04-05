"""
Module: base_pinn_framework.py
Purpose: Reusable DeepXDE-based wrapper that all physics models inherit from.
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
        
    def build_network(self, inputs, outputs, hidden_layers=4, neurons=128, activation="tanh"):
        """
        Builds the Fully Connected Network (MLP).
        """
        layer_size = [inputs] + [neurons] * hidden_layers + [outputs]
        self.net = dde.nn.FNN(layer_size, activation, "Glorot normal")
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
