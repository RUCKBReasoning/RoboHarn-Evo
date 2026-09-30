import os
import yaml

# This is a default path for localizing all the benchmark related files
libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runtime")
)
config_file = os.path.join(libero_config_path, "config.yaml")

_PATH_ENVIRONMENT = {
    "bddl_files": "LIBERO_PRO_BDDL_ROOT",
    "init_states": "LIBERO_PRO_INIT_STATES_ROOT",
    "datasets": "LIBERO_PRO_DATASETS_ROOT",
    "assets": "LIBERO_PRO_ASSETS_ROOT",
}


def get_default_path_dict(custom_location=None):
    if custom_location is None:
        benchmark_root_path = os.path.dirname(os.path.abspath(__file__))
    else:
        benchmark_root_path = custom_location

    # This is a default path for localizing all the default bddl files
    bddl_files_default_path = os.path.join(benchmark_root_path, "./bddl_files")

    # This is a default path for localizing all the default bddl files
    init_states_default_path = os.path.join(benchmark_root_path, "./init_files")

    # This is a default path for localizing all the default datasets
    dataset_default_path = os.path.join(benchmark_root_path, "../datasets")

    # This is a default path for localizing all the default assets
    assets_default_path = os.path.join(benchmark_root_path, "./assets")

    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": bddl_files_default_path,
        "init_states": init_states_default_path,
        "datasets": dataset_default_path,
        "assets": assets_default_path,
    }


def get_libero_path(query_key):
    override_name = _PATH_ENVIRONMENT.get(query_key)
    if override_name and os.environ.get(override_name, "").strip():
        return os.path.abspath(os.path.expanduser(os.environ[override_name]))
    if not os.path.isfile(config_file):
        config = get_default_path_dict()
        assert query_key in config, f"Unknown LIBERO path key: {query_key}"
        return config[query_key]
    with open(config_file, "r") as f:
        config = dict(yaml.safe_load(f.read()))

    assert (
        query_key in config
    ), f"Key {query_key} not found in config file {config_file}. You need to modify it. Available keys are: {config.keys()}"
    return config[query_key]


def set_libero_default_path(custom_location=os.path.dirname(os.path.abspath(__file__))):
    print(
        "[Warning] You are changing the default path for Libero config. This will affect all the paths in the config file."
    )
    new_config = get_default_path_dict(custom_location)
    os.makedirs(libero_config_path, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.safe_dump(new_config, f)
