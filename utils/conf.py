import dataclasses
from typing import Any

@dataclasses
class WeightModuleConfig:
    name: str
    params: dict[str, Any]
