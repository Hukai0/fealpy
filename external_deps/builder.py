#!/usr/bin/env python3
import os
import sys
import platform
import json
import importlib
import traceback

# add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def build_ext():
    """Build the extension modules for the current platform.
    Returns a list of extension descriptors (ext_modules) or raises on fatal error.
    """
    ext_modules = []
    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")

    # Load json file with dependencies
    with open(config_path, "r", encoding="utf-8") as f:
        deps = json.load(f)

    # deps should be an iterable of dicts
    if not isinstance(deps, (list, tuple)):
        raise RuntimeError("config.json should contain a list of dependency descriptors")

    for i, dep in enumerate(deps):
        try:
            if not isinstance(dep, dict):
                print(f"[builder] skipping entry #{i}: not a dict -> {repr(dep)}")
                continue

            # safe-get keys
            dep_env = dep.get("env")      # may be None
            dep_name = dep.get("name")    # module name expected

            # If dep_env provided and that environment variable is set -> skip or do conditional build
            # Original logic seemed to _run_ build when env var is set; keep that semantics but safe-check.
            if dep_env:
                if not os.getenv(dep_env):
                    print(f"[builder] skipping {dep_name or '<unknown>'}: env var {dep_env} not set")
                    continue
            else:
                # If no env key, we proceed (or you may want to skip — adjust as needed)
                # print(f"[builder] no env key for {dep_name or '<unknown>'}, proceeding")
                pass

            if not dep_name:
                print(f"[builder] entry #{i} missing 'name' key, skipping -> {dep}")
                continue

            # import module and call its build function (if present)
            try:
                module = importlib.import_module(dep_name)
            except Exception as e:
                print(f"[builder] failed to import module '{dep_name}': {e}")
                traceback.print_exc()
                continue

            build_fn = getattr(module, "build", None)
            if not callable(build_fn):
                print(f"[builder] module '{dep_name}' has no callable 'build(dep, ext_modules)'; skipping")
                continue

            # call module.build(dep, ext_modules)
            try:
                build_fn(dep, ext_modules)
            except Exception as e:
                print(f"[builder] exception while building '{dep_name}': {e}")
                traceback.print_exc()
                continue

        except Exception as e:
            print(f"[builder] unexpected error processing entry #{i}: {e}")
            traceback.print_exc()
            # continue processing other deps rather than aborting
            continue

    return ext_modules


# For manual testing
if __name__ == "__main__":
    modules = build_ext()
    print("Built ext modules:", modules)
