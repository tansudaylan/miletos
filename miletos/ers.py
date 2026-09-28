"""Reproduce published JWST Early Release Science analyses."""

from dataclasses import dataclass
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import visualization
from .paths import get_data_path, get_repository_path
from .pipeline import run_observational_pipeline


WASP39_ARCHIVE_URL = (
    "https://zenodo.org/records/7185300/files/G395H_paper_data.zip?download=1"
)
WASP39_ARCHIVE_NAME = "G395H_paper_data.zip"
WASP39_PRODUCT_ROOT = "G395H_paper_data"
WASP39_PRODUCTS = {
    "nrs1": (
        "2_LIGHT_CURVES/"
        "raw-white-light-curve-W39-G395H-NRS1-custom-Alam-Fig1a.xc"
    ),
    "nrs2": (
        "2_LIGHT_CURVES/"
        "raw-white-light-curve-W39-G395H-NRS2-custom-Alam-Fig1a.xc"
    ),
    "spectroscopic_light_curves": (
        "2_LIGHT_CURVES/"
        "fitted-binned-light-curve-W39-G395H-10pix-custom-Grant-Fig1bc.nc"
    ),
    "spectrum": (
        "3_TRANSMISSION_SPECTRA/"
        "transit-spectrum-W39b-G395H-10pix_weighted-average.nc"
    ),
    "model": "4_THEORY/fitted-model-spectrum-WASP-39b-ATMO-eq-Goyal-Fig3.nc",
}


@dataclass(frozen=True)
class WASP39ERSResult:
    """Published NIRSpec G395H observations and their ATMO model comparison."""

    wavelength_microns: np.ndarray
    bin_half_width_microns: np.ndarray
    transit_depth_ppm: np.ndarray
    transit_depth_uncertainty_ppm: np.ndarray
    model_transit_depth_ppm: np.ndarray
    model_residual_ppm: np.ndarray
    nrs1_time_bjd: np.ndarray
    nrs1_flux: np.ndarray
    nrs1_flux_uncertainty: np.ndarray
    nrs2_time_bjd: np.ndarray
    nrs2_flux: np.ndarray
    nrs2_flux_uncertainty: np.ndarray
    spectroscopic_time_bjd: np.ndarray
    spectroscopic_wavelength_microns: np.ndarray
    raw_spectroscopic_flux: np.ndarray
    raw_spectroscopic_flux_uncertainty: np.ndarray
    systematic_model: np.ndarray
    corrected_spectroscopic_flux: np.ndarray
    corrected_spectroscopic_flux_uncertainty: np.ndarray
    transit_model: np.ndarray
    spectroscopic_residual_flux: np.ndarray
    detector_shift_x_pixels: np.ndarray
    detector_shift_y_pixels: np.ndarray
    data_precision_ppm: np.ndarray
    photon_precision_ppm: np.ndarray
    reduced_chi_squared: float


def get_wasp39_ers_data_path() -> Path:
    """Return the ignored runtime directory for the Alderson et al. products."""

    return get_data_path() / "WASP-39" / "ERS_G395H_Alderson2023"


def download_wasp39_ers_products(refresh: bool = False) -> dict[str, Path]:
    """Download and extract the public products used by the reproduction."""

    data_path = get_wasp39_ers_data_path()
    archive_path = data_path / WASP39_ARCHIVE_NAME
    if refresh or not archive_path.is_file():
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"Reading from {WASP39_ARCHIVE_URL}...")
        print(f"Writing to {archive_path}...")
        with urlopen(WASP39_ARCHIVE_URL) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)

    product_paths = {
        name: data_path / relative_path for name, relative_path in WASP39_PRODUCTS.items()
    }
    missing_products = [name for name, path in product_paths.items() if not path.is_file()]
    if missing_products:
        print(f"Reading from {archive_path}...")
        with zipfile.ZipFile(archive_path) as archive:
            for name in missing_products:
                member = f"{WASP39_PRODUCT_ROOT}/{WASP39_PRODUCTS[name]}"
                output_path = product_paths[name]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Writing to {output_path}...")
                with archive.open(member) as source, output_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
    return product_paths


def _read_hdf5_arrays(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    print(f"Reading from {path}...")
    with h5py.File(path) as product:
        return {name: product[name][...] for name in names}


def analyze_wasp39_ers_g395h(refresh_data: bool = False) -> WASP39ERSResult:
    """Compare the published weighted G395H spectrum with its best-fit ATMO model."""

    paths = download_wasp39_ers_products(refresh=refresh_data)
    spectrum = _read_hdf5_arrays(
        paths["spectrum"],
        ("central_wavelength", "bin_half_width", "transit_depth", "transit_depth_error"),
    )
    model = _read_hdf5_arrays(paths["model"], ("wavelength", "transit_depth"))
    light_curves = {}
    for detector in ("nrs1", "nrs2"):
        light_curves[detector] = _read_hdf5_arrays(
            paths[detector], ("time_flux", "raw_flux", "raw_flux_error")
        )
    spectroscopic = _read_hdf5_arrays(
        paths["spectroscopic_light_curves"],
        (
            "time_flux",
            "central_wavelength",
            "raw_flux",
            "raw_flux_error",
            "systematic_model",
            "corrected_flux",
            "corrected_flux_error",
            "light_curve_model",
            "residuals",
            "shift_x",
            "shift_y",
        ),
    )

    if not np.allclose(spectrum["central_wavelength"], model["wavelength"], atol=1e-8):
        raise ValueError("The published spectrum and ATMO model wavelength bins differ.")
    expected_corrected_flux = (
        spectroscopic["raw_flux"] - spectroscopic["systematic_model"] + 1.0
    )
    if not np.allclose(spectroscopic["corrected_flux"], expected_corrected_flux):
        raise ValueError("The published systematics correction is internally inconsistent.")
    expected_residuals = (
        spectroscopic["raw_flux"] - spectroscopic["light_curve_model"]
    )
    if not np.allclose(spectroscopic["residuals"], expected_residuals):
        raise ValueError("The published light-curve residuals are internally inconsistent.")

    transit_depth_ppm = 1e6 * spectrum["transit_depth"]  # [ppm]
    uncertainty_ppm = 1e6 * spectrum["transit_depth_error"]  # [ppm]
    model_depth_ppm = 1e6 * model["transit_depth"]  # [ppm]
    residual_ppm = transit_depth_ppm - model_depth_ppm  # [ppm]
    degrees_of_freedom = transit_depth_ppm.size - 1
    reduced_chi_squared = float(
        np.sum(np.square(residual_ppm / uncertainty_ppm)) / degrees_of_freedom
    )
    transit_model = (
        spectroscopic["light_curve_model"] - spectroscopic["systematic_model"] + 1.0
    )
    data_precision_ppm = 1e6 * np.std(spectroscopic["residuals"], axis=1)  # [ppm]
    photon_precision_ppm = 1e6 * np.median(
        spectroscopic["raw_flux_error"] / spectroscopic["raw_flux"], axis=1
    )  # [ppm]
    photon_precision_ppm = gaussian_filter1d(photon_precision_ppm, 3, mode="nearest")

    return WASP39ERSResult(
        wavelength_microns=spectrum["central_wavelength"],
        bin_half_width_microns=spectrum["bin_half_width"],
        transit_depth_ppm=transit_depth_ppm,
        transit_depth_uncertainty_ppm=uncertainty_ppm,
        model_transit_depth_ppm=model_depth_ppm,
        model_residual_ppm=residual_ppm,
        nrs1_time_bjd=light_curves["nrs1"]["time_flux"],
        nrs1_flux=light_curves["nrs1"]["raw_flux"],
        nrs1_flux_uncertainty=light_curves["nrs1"]["raw_flux_error"],
        nrs2_time_bjd=light_curves["nrs2"]["time_flux"],
        nrs2_flux=light_curves["nrs2"]["raw_flux"],
        nrs2_flux_uncertainty=light_curves["nrs2"]["raw_flux_error"],
        spectroscopic_time_bjd=spectroscopic["time_flux"],
        spectroscopic_wavelength_microns=spectroscopic["central_wavelength"],
        raw_spectroscopic_flux=spectroscopic["raw_flux"],
        raw_spectroscopic_flux_uncertainty=spectroscopic["raw_flux_error"],
        systematic_model=spectroscopic["systematic_model"],
        corrected_spectroscopic_flux=spectroscopic["corrected_flux"],
        corrected_spectroscopic_flux_uncertainty=spectroscopic["corrected_flux_error"],
        transit_model=transit_model,
        spectroscopic_residual_flux=spectroscopic["residuals"],
        detector_shift_x_pixels=spectroscopic["shift_x"],
        detector_shift_y_pixels=spectroscopic["shift_y"],
        data_precision_ppm=data_precision_ppm,
        photon_precision_ppm=photon_precision_ppm,
        reduced_chi_squared=reduced_chi_squared,
    )


def run_wasp39_ers_g395h_reproduction(
    typefileplot: str = "png", refresh_data: bool = False
) -> WASP39ERSResult:
    """Download, analyze, and plot the public Alderson et al. G395H products."""

    output_path = get_repository_path() / "examples" / "WASP-39b" / "visuals"
    return run_observational_pipeline(
        analyzer=analyze_wasp39_ers_g395h,
        analysis_kwargs={'refresh_data': refresh_data},
        output_path=output_path,
        typefileplot=typefileplot,
        plot_products=(
            ('wasp39_ers_g395h_white_light', visualization.plot_wasp39_ers_white_light_curves),
            ('wasp39_ers_g395h_detector_motion', visualization.plot_wasp39_ers_detector_motion),
            (
                'wasp39_ers_g395h_spectroscopic_detrending',
                visualization.plot_wasp39_ers_spectroscopic_detrending,
            ),
            (
                'wasp39_ers_g395h_corrected_light_curve_map',
                visualization.plot_wasp39_ers_corrected_light_curve_map,
            ),
            (
                'wasp39_ers_g395h_light_curve_precision',
                visualization.plot_wasp39_ers_light_curve_precision,
            ),
            (
                'wasp39_ers_g395h_transmission_spectrum',
                visualization.plot_wasp39_ers_transmission_spectrum,
            ),
        ),
    )