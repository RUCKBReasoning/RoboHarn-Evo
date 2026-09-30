import os
import yaml

# This is a default path for localizing all the benchmark related files
libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runtime")
)
config_file = os.path.join(libero_config_path, "config.yaml")


def get_path_dict(root_location=os.path.dirname(os.path.abspath(__file__))):
    benchmark_root_path = root_location

    # This is a default path for localizing all the default bddl files
    bddl_files_default_path = os.path.join(benchmark_root_path, "./bddl_files")

    # This is a default path for localizing all the default bddl files
    init_states_default_path = os.path.join(benchmark_root_path, "./init_files")

    # This is a default path for localizing all the default datasets
    dataset_default_path = os.path.join(benchmark_root_path, "../datasets")

    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": bddl_files_default_path,
        "init_states": init_states_default_path,
        "datasets": dataset_default_path,
    }


def get_libero_path(key):
    if not os.path.isfile(config_file):
        config = get_path_dict()
        assert key in config, f"Unknown LIBERO path key: {key}"
        return config[key]
    with open(config_file, "r") as f:
        config = dict(yaml.safe_load(f.read()))
    assert key in config, f"Key {key} not found in config file {config_file}"
    return config[key]


def set_libero_path(custom_location=os.path.dirname(os.path.abspath(__file__))):
    new_config = get_path_dict(custom_location)
    os.makedirs(libero_config_path, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.safe_dump(new_config, f)
