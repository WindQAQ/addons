# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements npairs loss."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
@tf.function
def npairs_loss(y_true, y_pred):
    """Computes the npairs loss between `y_true` and `y_pred`.

    Npairs loss expects paired data where a pair is composed of samples from the
    same labels and each pairs in the minibatch have different labels. The loss
    has two components. The first component is the L2 regularizer on the
    embedding vectors. The second component is the sum of cross entropy loss
    which takes each row of the pair-wise similarity matrix as logits and
    the remapped one-hot labels as labels.

    The pair-wise similarity matrix `y_pred` between two embedding matrics
    `a` and `b` with shape [batch_size, hidden_size] can be computed
    as follows:

    ```python
    # y_pred = a * b^T
    y_pred = tf.linalg.matmul(a, b, transpose_a=False, transpose_b=True)
    ```

    See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

    Args:
      y_true: 1-D integer `Tensor` with shape [batch_size] of labels.
      y_pred: 2-D float `Tensor` with shape [batch_size, batch_size]
        of the pair-wise similarity matrix between two embedding matrices.

    Returns:
      npairs_loss: 1-D float `Tensor` with shape [batch_size].
    """
    # Expand [batch_size] label tensor to a [batch_size, 1] label tensor.
    y_true = tf.expand_dims(y_true, axis=-1)

    y_true = tf.dtypes.cast(
        tf.math.equal(y_true, tf.transpose(y_true)), y_pred.dtypes)
    y_true /= tf.math.reduce_sum(y_true, 1, keepdims=True)

    # Compute softmax loss.
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(y_true), logits=y_pred)


@keras_utils.register_keras_custom_object
class NpairsLoss(keras_utils.LossFunctionWrapper):

    def __init__(self,
                 refuction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="npairs_loss")
        super(NpairsLoss, self).__init__(
            npairs_loss,
            reduction=reduction,
            name=name)
