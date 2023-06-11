"""
Statically-linked deep learning library
Copyright (C) 2020 Dušan Erdeljan, Nedeljko Vignjević

This file is part of neural-network

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
"""

from pynn.util.dll import DLLUtil
from pynn.layers.dense import Dense
from pynn.type.output import Output
from pynn import optimizers, losses, regularizers, weightinitializers
from pynn.validation import validate_compile, validate_fit
from pynn.state import State
import numpy as np
import ctypes as C
import pickle
import os


class NeuralNetwork(object):

    def __init__(self, layers=None):
        self._lib = DLLUtil.load_dll()
        self._layers = None
        self._compiled = False
        self._state = State()
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Dense):
        if not self._layers:
            if layer.inputs <= 0:
                raise Exception("Invalid layer inputs.")
            self._layers = [layer]
        elif self._layers[-1].neurons != layer.inputs and layer.inputs != 0:
            raise Exception("Invalid layer inputs.")
        else:
            layer.inputs = self._layers[-1].neurons
            self._layers.append(layer)
        self._state.add_layer(layer)
        self._compiled = False

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int = 1):
        if not self._compiled:
            raise Exception("Model is not compiled.")
        validate_fit(epochs, batch_size)
        for x, y in zip(x_train, y_train):
            self._lib.add_training_sample(np.asarray(x, dtype=np.double),
                                          np.asarray(y if np.isscalar(y) else [y], dtype=np.double))
        self._lib.train(C.c_uint(epochs), C.c_uint(batch_size))

    def predict(self, inputs: np.array) -> Output:
        return self._lib.eval(np.asarray(inputs, dtype=np.double))

    def _update_state(self, optimizer, loss, initializer, regularizer):
        self._state.update(optimizer, loss, initializer, regularizer, self._layers[0].inputs, self._layers[-1].neurons)

    def compile(self, optimizer='sgd', loss='mean_squared_error', initializer='random', regularizer='none'):
        if not self._layers:
            raise Exception("No layers specified.")
        validate_compile(optimizer, loss, initializer, regularizer)
        for layer in self._layers:
            self._lib.add(layer)
        self._update_state(optimizer, loss, initializer, regularizer)
        if isinstance(optimizer, str):
            self._lib.compile(optimizers.optimizers[optimizer], losses.losses[loss],
                              weightinitializers.weight_initializers[initializer],
                              regularizers.regularizers[regularizer])
        else:
            self._lib.compile_optimizer(C.byref(optimizer), optimizers.optimizers[optimizer.__class__.__name__.lower()],
                                        losses.losses[loss],
                                        weightinitializers.weight_initializers[initializer],
                                        regularizers.regul