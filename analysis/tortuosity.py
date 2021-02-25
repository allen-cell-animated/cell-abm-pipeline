#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Provide a means of quantifying straightness of a path """


import numpy as np


def tortuosity(trace):
    """Provide the tortuosity of the 2D path passed in

    Parameters
    ----------
    trace: np.array
        shape of n,2 for n observations

    Returns
    -------
    tortuosity: float
        as defined by https://en.wikipedia.org/wiki/Tortuosity
    """
    assert trace.shape[1] == 2, "Trace needs to be shape (n,2)"
    along = np.linalg.norm(trace[:-1] - trace[1:], axis=1).sum()
    end_to_end = np.linalg.norm(trace[0] - trace[-1])
    return along / end_to_end


def test_tortuosity():
    """Make sure outputs are as expected"""
    assert tortuosity(np.array(((0, 0), (1, 0), (2, 0)))) == 1.0
    path = np.array(((1, 1), (1, 2), (2, 2)))
    np.testing.assert_approx_equal(tortuosity(path), np.sqrt(2), 3)
