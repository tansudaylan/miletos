import os
import pickle


def retr_pathcache(gdat):
    """Return the path to the cached workflow output bundle."""

    path = gdat.pathdatatarg + 'dict_miletos_output.pickle'

    return path


def read_cached_output(gdat):
    """Return cached workflow output when a completed run is available."""

    path = retr_pathcache(gdat)
    if gdat.boolwritover or not os.path.exists(path):
        return None

    if gdat.typeverb > 0:
        print('Reading from %s...' % path)
    with open(path, 'rb') as objthand:
        gdat.dictmileoutp = pickle.load(objthand)

    return gdat.dictmileoutp