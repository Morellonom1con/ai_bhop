""" small timer function """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time


def timing(function):
    """ lazy method to time my function """

    def wrap(*args, **kw):
        """ wrap the function to time it"""
        time1 = time.time()
        ret = function(*args, **kw)
        time2 = time.time()
        print("----- %s took %0.3f s" % (function.func_name, (time2 - time1)))
        return ret

    return wrap
