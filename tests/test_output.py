import pandas as pd

from miletos.output import retr_listnamecols, write_cluster_output_csv, write_target_output_csv


class DummyGdat:
    pass


def test_write_target_output_csv_writes_scalar_outputs(tmp_path):
    gdat = DummyGdat()
    gdat.pathdatatarg = str(tmp_path) + '/'
    gdat.dictmileoutp = {
        'strgtarg': 'TOI-123',
        'timeexec': 12.5,
        'boolflag': True,
        'listskip': [1, 2, 3],
    }

    write_target_output_csv(gdat, typeverb=0)

    text = (tmp_path / 'miletos_output.csv').read_text(encoding='utf-8')

    assert 'strgtarg, TOI-123' in text
    assert 'timeexec, 12.5' in text
    assert 'boolflag, 1' in text
    assert 'listskip' not in text


def test_write_cluster_output_csv_creates_header_and_row(tmp_path):
    gdat = DummyGdat()
    gdat.pathdataclus = str(tmp_path) + '/'
    gdat.strgtarg = 'TOI-456'
    gdat.dictmileoutp = {
        'strgtarg': 'TOI-456',
        'timeexec': 2.5,
        'lygo_pathsaverflx_demo': '/tmp/skip',
        'lygo_strgtitlcntpplot_demo': 'skip',
        'boolflag': False,
    }

    write_cluster_output_csv(gdat, typeverb=0)

    fram = pd.read_csv(tmp_path / 'miletos_cluster_output.csv')

    assert fram.columns.tolist() == ['strgtarg', 'timeexec', 'boolflag']
    assert fram.iloc[0]['strgtarg'] == 'TOI-456'
    assert fram.iloc[0]['timeexec'] == 2.5
    assert fram.iloc[0]['boolflag'] == 0


def test_retr_listnamecols_filters_transient_plot_keys():
    listnamecols = retr_listnamecols({
        'strgtarg': 'TOI-1',
        'lygo_pathsaverflx_demo': 'skip',
        'lygo_strgtitlcntpplot_demo': 'skip',
        'timeexec': 1.0,
    })

    assert listnamecols == ['strgtarg', 'timeexec']