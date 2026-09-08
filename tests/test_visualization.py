import os

from miletos.visualization import plot_binned_rms, plot_work_tser


class DummyGdat:
    pass


def test_plot_work_tser_forwards_standard_arguments():
    gdat = DummyGdat()
    gdat.pathvisutarg = '/tmp/visu/'
    gdat.timeoffs = 2457000.0
    gdat.boolwritover = False
    gdat.boolbrekmodl = True

    calls = {}

    def fake_plot_tser(pathvisu, **kwargs):
        calls['pathvisu'] = pathvisu
        calls['kwargs'] = kwargs
        return 'plot.png'

    pathplot = plot_work_tser(
        fake_plot_tser,
        gdat,
        timedata=[1, 2],
        tserdata=[3, 4],
        strgextn='Demo',
        strgtitl='Title',
        dictmodl={'model': 1},
        lablyaxi='Residual relative flux',
        booldiag=True,
    )

    assert pathplot == 'plot.png'
    assert calls['pathvisu'] == '/tmp/visu/'
    assert calls['kwargs']['timedata'] == [1, 2]
    assert calls['kwargs']['tserdata'] == [3, 4]
    assert calls['kwargs']['timeoffs'] == 2457000.0
    assert calls['kwargs']['strgextn'] == 'Demo'
    assert calls['kwargs']['strgtitl'] == 'Title'
    assert calls['kwargs']['boolwritover'] is False
    assert calls['kwargs']['boolbrekmodl'] is True
    assert calls['kwargs']['dictmodl'] == {'model': 1}
    assert calls['kwargs']['lablyaxi'] == 'Residual relative flux'
    assert calls['kwargs']['booldiag'] is True


def test_plot_binned_rms_writes_plot(tmp_path):
    gdat = DummyGdat()
    gdat.figrsizeydob = (4, 3)
    gdat.cadetimeplot = 0.5
    gdat.typeverb = 0

    path = str(tmp_path / 'stdvrebn.png')
    delt = [0.5, 1.0, 2.0]
    stdvresi = [1e-6, 2e-6, 3e-6]

    pathout = plot_binned_rms(gdat, path, delt, stdvresi)

    assert pathout == path
    assert os.path.exists(path)