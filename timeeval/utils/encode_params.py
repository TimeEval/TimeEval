import json
from pathlib import Path
from typing import Dict, Any, TextIO, Union

from numpyencoder import NumpyEncoder


def dumps_params(params: Dict[str, Any]) -> str:
    return json.dumps(params, cls=NumpyEncoder)


def dump_params(params: Dict[str, Any], fh: Union[str, Path, TextIO]) -> None:
    if isinstance(fh, str):
        fh = Path(fh)
    if isinstance(fh, Path):
        fh = fh.open("w", encoding="utf-8")
    json.dump(params, fh, cls=NumpyEncoder)
    fh.close()
