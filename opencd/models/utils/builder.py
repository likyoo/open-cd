from mmcv.utils import Registry, build_from_cfg

ITERACTION_LAYERS = Registry('interaction layer')

def build_interaction_layer(cfg, default_args=None):
    """Builder for Interaction layer."""
    return build_from_cfg(cfg, ITERACTION_LAYERS, default_args)
