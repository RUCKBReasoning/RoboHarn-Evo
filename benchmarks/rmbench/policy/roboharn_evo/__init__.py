"""RMBench policy facade backed by the independent top-level :mod:`tcm` runtime.

The copied donor Agent and model trees remain available as compatibility and
reference code, but formal deployment is exported only through this facade.
"""

from .deploy_policy import eval, get_model, reset_model

__all__ = ["eval", "get_model", "reset_model"]
