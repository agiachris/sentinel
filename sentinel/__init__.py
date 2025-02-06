import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections
    import collections.abc

    collections.__dict__["Mapping"] = collections.abc.Mapping
    collections.__dict__["Sequence"] = collections.abc.Sequence
    collections.__dict__["MutableMapping"] = collections.abc.MutableMapping
    collections.__dict__["MutableSequence"] = collections.abc.MutableSequence
