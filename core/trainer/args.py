import argparse

from .utils import str2bool


class _MyArgParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flag_for_overwrite = False

    def add_argument_overwrite(self, *args, **kwargs):
        dest = self._get_optional_kwargs(*args, **kwargs)["dest"]
        for action in self._actions:
            if action.dest == dest:
                for k, v in kwargs.items():
                    if hasattr(action, k):
                        setattr(action, k, v)
                return action
        return super().add_argument(*args, **kwargs)

    def add_argument_general(self, *args, **kwargs):
        fn = self.add_argument_overwrite if self.flag_for_overwrite else super().add_argument
        return fn(*args, **kwargs)


_global_parser = _MyArgParser()

class _ArgValues(object):
    """Global container and accessor for flags and their values."""

    def __init__(self):
        self.__dict__["__flags"] = {}
        self.__dict__["__actions"] = {}
        self.__dict__["__parsed"] = False

    def _parse_flags(self, args=None):
        result = _global_parser.parse_args(args=args)
        for flag_name, val in vars(result).items():
            self.__dict__["__flags"][flag_name] = val
        self.__dict__["__parsed"] = True

    def get_dict(self):
        if not self.__dict__["__parsed"]:
            self._parse_flags()
        return self.__dict__["__flags"]

    def set_dict(self, newdict, overwrite=False):
        if not self.__dict__["__parsed"]:
            self._parse_flags()
        for k in newdict:
            self.__dict__["__flags"][k] = newdict[k]

    def __getattr__(self, name):
        """Retrieves the 'value' attribute of the flag --name."""
        if not self.__dict__["__parsed"]:
            self._parse_flags()
        if name not in self.__dict__["__flags"]:
            raise AttributeError(name)
        return self.__dict__["__flags"][name]

    def __setattr__(self, name, value):
        """Sets the 'value' attribute of the flag --name."""
        if not self.__dict__["__parsed"]:
            self._parse_flags()
        self.__dict__["__flags"][name] = value

    def Enable_OverWrite(self):
        _global_parser.flag_for_overwrite = True

    def Disable_OverWrite(self):
        _global_parser.flag_for_overwrite = False

    def DEFINE_argument(self, *args, default=None, rep=False, **kwargs):
        if rep is True:
            kwargs["nargs"] = "+"
        _global_parser.add_argument_general(*args, default=default, **kwargs)

    def DEFINE_boolean(self, *args, default=None, docstring=None, **kwargs):
        docstring = "" if docstring is None else docstring
        _global_parser.add_argument_general(
            *args, help=docstring, default=default, type=str2bool, **kwargs
        )