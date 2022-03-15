from omegaconf import OmegaConf
from mmcv import Config


def try_get_key(cfg, *keys, default=None):
    """# modified from detectron2 to also support mmcv Config
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    from detectron2.config import CfgNode

    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    elif isinstance(cfg, Config):
        cfg = OmegaConf.create(cfg._cfg_dict.to_dict())
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    for k in keys:
        # OmegaConf.select(default=) is supported only after omegaconf2.1,
        # but some internal users still rely on 2.0
        parts = k.split(".")
        # https://github.com/omry/omegaconf/issues/674
        for p in parts:
            if p not in cfg:
                break
            cfg = OmegaConf.select(cfg, p)
        else:
            return cfg
    return default
