from hashlib import md5


def hash_dict(x: dict) -> str: return md5(str(sorted(x.items())).encode("utf-8")).hexdigest()
