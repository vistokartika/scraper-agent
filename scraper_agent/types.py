from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ExtractResult:
    data: Any
    raw: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
