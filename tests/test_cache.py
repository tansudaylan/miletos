import pickle

from miletos.cache import read_cached_output, retr_pathcache


class DummyGdat:
    pass


def test_read_cached_output_reads_existing_bundle(tmp_path):
    gdat = DummyGdat()
    gdat.pathdatatarg = str(tmp_path) + '/'
    gdat.boolwritover = False
    gdat.typeverb = 0

    payload = {'status': 'cached', 'value': 3.14}
    with open(retr_pathcache(gdat), 'wb') as objthand:
        pickle.dump(payload, objthand)

    result = read_cached_output(gdat)

    assert result == payload
    assert gdat.dictmileoutp == payload


def test_read_cached_output_skips_when_overwriting(tmp_path):
    gdat = DummyGdat()
    gdat.pathdatatarg = str(tmp_path) + '/'
    gdat.boolwritover = True
    gdat.typeverb = 0

    assert read_cached_output(gdat) is None