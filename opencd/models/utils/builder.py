# Copyright (c) Open-CD. All rights reserved.
import warnings

from opencd.registry import MODELS

ITERACTION_LAYERS = MODELS

def build_interaction_layer(cfg):
    """Build backbone."""
    warnings.warn('``build_interaction_layer`` would be deprecated soon, please use '
                  '``opencd.registry.MODELS.build()`` ')
    return ITERACTION_LAYERS.build(cfg)
