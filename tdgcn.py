# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd

import utils
from DTW_matrix import dtw_adj

tf.compat.v1.disable_v2_behavior()

from tensorflow.contrib.rnn import RNNCell
# RNNCell = tf.nn.rnn_cell.BasicRNNCell
# RNNCell=tf.compat.v1.nn.rnn_cell.RNNCell
from utils import calculate_laplacian


class tdgcnCell(RNNCell):
    """Temporal-DTW Graph Convolutional Network（T-DGCN） """

    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, adj, num_nodes, dataname, input_size=None,
                 act=tf.nn.tanh, reuse=None):
        super(tdgcnCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self._adj = []

        # Description: Read the corresponding geological feature file according to the dataset name,
        #               and call the DTW function to calculate the geological similarity matrix based on it.
        # Input: dataset name
        # Output: geological similarity matrix, which is transformed by np.mat()
        sml = np.mat(dtw_adj(dataname))
        self._sml = sml

        # Description: Compute the Laplace operator
        # input: the aggregated matrix with geological and spatial characteristics
        # Output: Append the computed result to self.adj
        adj_lapl = calculate_laplacian(adj)
        self._adj = adj_lapl

    @property
    def state_size(self):
        return self._nodes * self._units

    @property
    def output_size(self):
        return self._units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "tdgcn"):
            with tf.variable_scope("gates"):
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope, dtw_input=self._sml))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope, dtw_input=self._sml))
            new_h = u * state + (1 - u) * c
        return new_h, new_h

    def _gc(self, inputs, state, output_size, dtw_input, bias=0.0, scope=None):
        """
        卷积层内部处理
        """
        ## inputs:(-1,num_nodes)
        inputs = tf.expand_dims(inputs, 2)
        ## state:(batch,num_node,gru_units)
        state = tf.reshape(state, (-1, self._nodes, self._units))
        ## concat
        x_s = tf.concat([inputs, state], axis=2)
        input_size = x_s.get_shape()[2].value
        ## (num_node,input_size,-1)
        x0 = tf.transpose(x_s, perm=[1, 2, 0])
        x0 = tf.reshape(x0, shape=[self._nodes, -1])
        scope = tf.get_variable_scope()
        # Description: Each time this function is called, a Hadamard aggregation is performed,
        #               which serves to dynamically correct the matrix in the model
        self._adj = utils.hadamard_polymerization(self._adj, dtw_input)
        with tf.variable_scope(scope):
            x1 = tf.sparse_tensor_dense_matmul(self._adj, x0)
            x = tf.reshape(x1, shape=[self._nodes, input_size, -1])
            x = tf.transpose(x, perm=[2, 0, 1])
            x = tf.reshape(x, shape=[-1, input_size])
            weights = tf.get_variable(
                'weights', [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)
            biases = tf.get_variable(
                "biases", [output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32))
            x = tf.nn.bias_add(x, biases)
            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x
