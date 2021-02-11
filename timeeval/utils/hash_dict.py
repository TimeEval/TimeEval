from hashlib import md5


def hash_dict(x: dict) -> str:
    return str(md5(str(sorted(x.items()))))
