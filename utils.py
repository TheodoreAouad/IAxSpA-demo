from typing import Any
import logging
import yaml
import re

import numpy as np


def ceil_(x: float) -> int:
    return int(np.ceil(x))


def log_console(to_print: Any = '', *args, level: str = 'info', logger: logging.Logger = None, **kwargs) -> None:
    if logger is None:
        print(to_print, *args, **kwargs)
    else:
        to_print = '{}'.format(to_print)
        for st in args:
            to_print = to_print + ' {} '.format(st)
        getattr(logger, level.lower())(to_print)


def load_yaml(path,):

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(path, 'r') as f:
        try:
            content = yaml.load(f, Loader=loader)
        except yaml.YAMLError as exc:
            print(exc)
    return content
