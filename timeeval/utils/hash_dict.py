def hash_dict(x: dict) -> str:
    return str(hash(str(sorted(x.items()))))
