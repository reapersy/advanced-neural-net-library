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
          