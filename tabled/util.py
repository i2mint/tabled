"""
Utils
"""


def identity(x):
    return x


def split_keys(d):
    """Returns a dictionary where keys that had spaces were split into multiple keys

    Meant to be a convenience function for the user to use when they want to define a
    mapping where several keys map to the same value.

    >>> split_keys({'apple': 1, 'banana carrot': 2})
    {'apple': 1, 'banana': 2, 'carrot': 2}
    """
    return {split_k: v for k, v in d.items() for split_k in k.split()}
