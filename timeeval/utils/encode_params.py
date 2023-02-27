import json
from pathlib import Path
from typing import TextIO, Union

from numpyencoder import NumpyEncoder

from timeeval.params import Params


def dumps_params(params: Params) -> str:
    return json.dumps(params.to_dict(), cls=NumpyEncoder)


def dump_params(params: Params, fh: Union[str, Path, TextIO]) -> None:
    if isinstance(fh, str):
        fh = Path(fh)
    if isinstance(fh, Path):
        fh = fh.open("w", encoding="utf-8")
    json.dump(params.to_dict(), fh, cls=NumpyEncoder)
    fh.close()
