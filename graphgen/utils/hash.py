from hashlib import md5


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def compute_content_hash(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def compute_dict_hash(d: dict, prefix: str = ""):
    items = tuple(sorted(d.items()))
    return prefix + md5(str(items).encode()).hexdigest()
