from hashlib import md5
from typing import Any, Mapping


def hash_dict(x: Mapping[Any, Any]) -> str: return md5(str(sorted(x.items())).encode("utf-8")).hexdigest()
