import os

import miletos
import numpy as np


def main(pathvisu=None):
    # generate synthetic data
    ## number of time samples
    numbtime = 10000
    ## time axis
    time = np.linspace(0., 30., numbtime)
    ## number of sinusoidal components
    numbcomp = 10

    ## light curve
    lcur = np.zeros(numbtime)
    ## standard deviation of the light curve
    stdvlcur = np.full(numbtime, 0.5)
    for k in range(numbcomp):
        ## amplitude of the component
        ampl = np.random.rand()
        ## period of the component
        peri = 10. * np.random.rand()
        ## add the component to the light curve
        lcur += ampl * np.sin(2. * np.pi * time / peri)
    ## add noise
    lcur += np.random.randn(numbtime) * stdvlcur

    if pathvisu is None:
        pathvisu = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(pathvisu, exist_ok=True)

    # plot the initial light curve
    miletos.plot_tser(
        timedata=time,
        tserdata=lcur,
        pathvisu=pathvisu,
        strgextn='raw',
    )

    # list of time scales for detrending [days]
    listtimescalbdtr = [0.5, 0.7, 1., 3., 10., 30.]
    paths = []
    for timescalbdtr in listtimescalbdtr:
        # detrend the light curve
        lcurbdtr = miletos.bdtr_tser(time, lcur, stdvlcur, timescalbdtr=timescalbdtr)[0]

        # plot the detrended light curve
        path = miletos.plot_tser(
            timedata=time,
            tserdata=lcurbdtr,
            strgtitl='Detrended at %.3g days' % timescalbdtr,
            pathvisu=pathvisu,
            strgextn='detrended_%.3g_days' % timescalbdtr,
        )
        paths.append(path)

    return paths


if __name__ == '__main__':
    main()


def test_detrend_demo_writes_file(tmp_path):
    paths = main(pathvisu=str(tmp_path))

    assert paths
    assert all(os.path.exists(path) for path in paths)
    assert os.path.exists(os.path.join(str(tmp_path), 'TimeSeries_raw.png'))

