import os
import yaml


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if "$import" not in data:
        return data
    import_info = data.pop("$import")
    ref = import_info["ref"].split("#")[0]
    base_path = (
        ref
        if os.path.isabs(ref)
        else os.path.join(os.path.dirname(os.path.abspath(yaml_path)), ref)
    )
    base = load_yaml(base_path)
    return deep_merge(base, data)


def deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result
