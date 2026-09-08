import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from miletos.report import retr_pathpage, setp_dvrp_output


class DummyGdat:
    pass


def _write_demo_png(path):
    data = np.arange(9).reshape((3, 3))
    plt.figure(figsize=(1, 1))
    plt.imshow(data)
    plt.axis('off')
    plt.savefig(path, dpi=50)
    plt.close()


def test_setp_dvrp_output_builds_summary_pages(tmp_path):
    inpt_path = tmp_path / 'input.png'
    _write_demo_png(inpt_path)

    gdat = DummyGdat()
    gdat.pathvisutarg = str(tmp_path) + '/'
    gdat.strgtarg = 'DemoTarget'
    gdat.indxpage = np.array([0])
    gdat.listdictdvrp = [[{'path': str(inpt_path), 'limt': [0.1, 0.1, 0.8, 0.8]}]]
    gdat.typeverb = 0
    gdat.dictmileoutp = {}

    listpathdvrp = setp_dvrp_output(gdat)

    assert listpathdvrp == [retr_pathpage(gdat, 0)]
    assert os.path.exists(listpathdvrp[0])
    assert gdat.dictmileoutp['listpathdvrp'] == listpathdvrp