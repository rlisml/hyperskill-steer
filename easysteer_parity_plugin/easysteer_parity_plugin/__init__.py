"""easysteer-parity-plugin — registers the `steerling_adapter` steering
algorithm with the EasySteer fork via the standard vLLM general-plugins
entry-point mechanism (no fork file and no EasySteer-package file is
modified). The entry point is loaded by vLLM in every engine/worker process,
so the algorithm registry is populated wherever `create_algorithm` runs.
"""

__all__ = ["register"]


def register():
    from . import steerling_adapter  # noqa: F401  (registers on import)
