# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from . import option_list

option_list['mxnet'] = {
    'enabled': True,
    'executable': 1,
}
'''

def find_executable(path=None):
    """
    Finds th on the given path and returns it if found
    If path is None, searches through PATH
    """
    if path is None:
        dirnames = os.environ['PATH'].split(os.pathsep)
        suffixes = ['th']
    else:
        dirnames = [path]
        # fuzzy search
        suffixes = ['th',
                    os.path.join('bin', 'th'),
                    os.path.join('install', 'bin', 'th')]

    for dirname in dirnames:
        dirname = dirname.strip('"')
        for suffix in suffixes:
            path = os.path.join(dirname, suffix)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None


if 'MXNET_ROOT' in os.environ:
    executable = find_executable(os.environ['MXNET_ROOT'])
    if executable is None:
        pass
#        raise ValueError('Mxnet executable not found at "%s" (MXNET_ROOT)'
#                        % os.environ['MXNET_ROOT'])
elif 'MXNET_HOME' in os.environ:
#    executable = find_executable(os.environ['MXNET_HOME'])
 #   if executable is None:
  #      raise ValueError('Mxnet executable not found at "%s" (MXNET_HOME)'
  #                       % os.environ['MXNET_HOME'])
    pass

else:
    executable = find_executable()


if executable is None:
    option_list['mxnet'] = {
        'enabled': True,
        'executable': executable,
    }
else:
    option_list['mxnet'] = {
        'enabled': True,
        'executable': executable,
    }
'''