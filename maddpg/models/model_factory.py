"""Implements a model factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from models import graph_net

MODEL_MAP = {
    'gcn_max': functools.partial(graph_net.GraphNet, pool_type='max')
}


def get_model_fn(name):
  assert name in MODEL_MAP
  return MODEL_MAP[name]
