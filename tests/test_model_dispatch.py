from types import SimpleNamespace

import numpy as np
import pytest

import miletos.main as miletos_main


def test_compact_object_dispatch_passes_stellar_mass_and_radius_separately(monkeypatch, tmp_path):
    captured = {}

    class DispatchReached(Exception):
        pass

    def capture_dispatch(*args, **kwargs):
        captured.update(kwargs)
        raise DispatchReached

    model = SimpleNamespace(
        boolmodlpsys=True,
        indxcomp=[0],
        listnamecompmodl=['Transit'],
        numbcomp=1,
        typemodl='CompactObjectStellarCompanion',
        typemodllmdkener='linr',
        typemodlenerfitt='full',
    )
    data = SimpleNamespace(
        booldiag=False,
        boolmakeanimefestrue=False,
        boolmakeplotefestrue=True,
        fitt=model,
        indxdatatser=[0],
        indxinst=[[0]],
        liststrginst=[['test']],
        numbener=[1],
        numbenermodl=1,
        pathvisutarg=f'{tmp_path}/',
    )
    parameters = {
        'cosicom0': 0.1,
        'epocmtracom0': 0.0,
        'masscom0': 5.0,
        'massstar': 1.4,
        'pericom0': 2.0,
        'radistar': 0.8,
        'rratcom0': 0.01,
        'rsmacom0': 0.1,
    }
    monkeypatch.setattr(miletos_main.ephesos, 'eval_modl', capture_dispatch)

    with pytest.raises(DispatchReached):
        miletos_main.retr_dictmodl_mile(
            data, [[np.array([0.0, 1.0])]], parameters, 'fitt'
        )

    assert captured['massstar'] == parameters['massstar']
    assert captured['radistar'] == parameters['radistar']