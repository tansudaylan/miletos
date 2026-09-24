import numpy as np
from astropy.io import fits

from miletos.main import read_tesskplr_file


def test_read_tesskplr_file_reads_target_pixel_metadata(tmp_path):
    primary = fits.PrimaryHDU()
    primary.header["SECTOR"] = 7
    primary.header["CAMERA"] = 3
    primary.header["CCD"] = 2
    cadence_table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TIME", format="D", array=np.array([1.0, 2.0, np.nan])),
            fits.Column(name="QUALITY", format="J", array=np.array([0, 1, 0])),
        ]
    )
    path = tmp_path / "public_target_tp.fits"
    fits.HDUList([primary, cadence_table]).writeto(path)

    hdus, good_indices, sector, camera, ccd = read_tesskplr_file(str(path))

    try:
        np.testing.assert_array_equal(good_indices, [0])
        assert (sector, camera, ccd) == (7, 3, 2)
    finally:
        hdus.close()