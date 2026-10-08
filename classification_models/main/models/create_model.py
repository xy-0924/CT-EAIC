import torch
from timm.models.factory import parse_model_name
from timm.models.registry import is_model, is_model_in_modules, model_entrypoint
from timm.models.hub import load_model_config_from_hf
from timm.models.layers import set_layer_config

def create_model_custom(
    model_name,
    pretrained=False,
    checkpoint_path='',
    scriptable=None,
    exportable=None,
    no_jit=None,
    **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    source_name, model_name = parse_model_name(model_name)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
            " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if source_name in ('hf_hub', 'hf-hub'):
        # For model names specified in the form `hf_hub:path/architecture_name#revision`,
        # load model weights + default_cfg from Hugging Face hub.
        hf_default_cfg, model_name = load_model_config_from_hf(model_name)
        kwargs['external_default_cfg'] = hf_default_cfg  # FIXME revamp default_cfg interface someday

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(pretrained=pretrained, **kwargs)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        model_dict = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    skipped_keys.append(f"{k}: ckpt={tuple(v.shape)} model={tuple(model_dict[k].shape)}")
            else:
                skipped_keys.append(f"{k}: key not in model")

        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded {len(filtered_state_dict)}/{len(model_dict)} keys from checkpoint '{checkpoint_path}'")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} keys due to mismatch:")
            for sk in skipped_keys[:10]:
                print(f"  - {sk}")
            if len(skipped_keys) > 10:
                print(f"  ... and {len(skipped_keys) - 10} more")

    return model
