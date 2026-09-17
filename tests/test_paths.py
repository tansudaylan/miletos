import miletos
from miletos.paths import chec_path_input, setp_alle_path, setp_base_paths, setp_feature_paths, setp_mast_path, setp_target_paths


class DummyGdat:
    pass


def test_retr_tsecpathlocl_reads_cached_sector_table(monkeypatch, tmp_path):
    monkeypatch.setenv('TESS_DATA_PATH', str(tmp_path))

    pathdir = tmp_path / 'data' / 'lcur' / 'tsec'
    pathdir.mkdir(parents=True)
    pathfile = pathdir / 'tsec_spoc_0000000123456789.csv'
    pathfile.write_text('5,/tmp/sector05.fits\n12,/tmp/sector12.fits\n', encoding='utf-8')

    listtsec, listpath = miletos.retr_tsecpathlocl(123456789, typeverb=0)

    assert listtsec.tolist() == [5, 12]
    assert listpath == ['/tmp/sector05.fits', '/tmp/sector12.fits']


def test_setp_target_paths_and_input_check():
    gdat = DummyGdat()
    gdat.pathtarg = '/tmp/run/'
    gdat.pathbase = None
    gdat.pathdatatarg = None
    gdat.pathvisutarg = None
    gdat.strgcnfg = 'Demo'
    gdat.booldiag = False

    assert chec_path_input(gdat)

    setp_target_paths(gdat)

    assert gdat.pathtargcnfg == '/tmp/run/Demo/'
    assert gdat.pathdatatarg == '/tmp/run/Demo/data/'
    assert gdat.pathvisutarg == '/tmp/run/Demo/visuals/'


def test_path_input_check_precedes_default_base_path(monkeypatch, tmp_path):
    monkeypatch.setenv('MILETOS_DATA_PATH', str(tmp_path / 'miletos'))
    monkeypatch.setenv('LYGOS_DATA_PATH', str(tmp_path / 'lygos'))

    for values in (
        (None, None, None, None),
        ('/tmp/run/', None, None, None),
        (None, None, '/tmp/data/', '/tmp/visuals/'),
    ):
        gdat = DummyGdat()
        gdat.pathtarg, gdat.pathbase, gdat.pathdatatarg, gdat.pathvisutarg = values
        assert chec_path_input(gdat)

    gdat = DummyGdat()
    gdat.pathtarg = '/tmp/run/'
    gdat.pathbase = None
    gdat.pathdatatarg = None
    gdat.pathvisutarg = None
    setp_base_paths(gdat)
    assert not chec_path_input(gdat)


def test_setp_mast_path(monkeypatch, tmp_path):
    gdat = DummyGdat()
    monkeypatch.setenv('MAST_DATA_PATH', str(tmp_path / 'mast-cache'))

    setp_mast_path(gdat)

    assert gdat.pathdatamast.endswith('/')
    assert 'mast-cache' in gdat.pathdatamast


def test_setp_feature_paths():
    gdat = DummyGdat()
    gdat.pathvisutarg = '/tmp/run/Demo/visuals/'
    gdat.liststrgpdfn = ['prio']

    setp_feature_paths(gdat)

    assert gdat.pathvisufeat == '/tmp/run/Demo/visuals/feat/'
    assert gdat.pathvisufeatplanprio == '/tmp/run/Demo/visuals/feat/prio/featplan/'
    assert gdat.pathvisufeatsystprio == '/tmp/run/Demo/visuals/feat/prio/featsyst/'
    assert gdat.pathvisudataplanprio == '/tmp/run/Demo/visuals/feat/prio/dataplan/'


def test_setp_alle_path(tmp_path):
    gdat = DummyGdat()
    gdat.pathallebase = str(tmp_path) + '/'
    gdat.pathalle = {}

    setp_alle_path(gdat, 'fitt')

    assert gdat.pathalle['fitt'].endswith('/allesfit_fitt/')
    assert (tmp_path / 'allesfit_fitt').is_dir()