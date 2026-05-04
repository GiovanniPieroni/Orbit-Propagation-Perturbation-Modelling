from inspect import getsource, getfile, getframeinfo
from os.path import abspath, dirname, relpath
import sys

def getsourcefunc(func):
    fpath = abspath(getfile(func))
    caller = abspath(dirname(sys._getframe(1).f_code.co_filename))
    return [relpath(fpath, caller)], getsource(func)
