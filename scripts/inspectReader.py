"""Phase 1 inspection: load the simulator data and cross-check the CSV export.

Run under a debugger to inspect ``pulses``, ``referencePulse`` and ``metadata``
directly, or run headless to print shapes/dtypes and the CSV cross-check::

    uv run python scripts/inspectReader.py

This is a throwaway inspection script, not part of the gate. It verifies the
Phase 1 reader: the on-disk array shapes/dtypes and that pulse-0 of look 0 in
``rx_sig.npy`` matches the human-readable ``s_rx_pulse0_look_000.csv`` export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import simValidation as sv

DATA_DIR = Path(
    r"C:\Users\yJoonSio\OneDrive - DSO\trihedral"
    r"\Trihedrals_SAR_ADC\raw_pulse_data_repackaged"
)
# The CSV exports sit one level above the repackaged .npy directory.
CSV_PATH = DATA_DIR.parent / "s_rx_pulse0_look_000.csv"


def main() -> int:
    pulses, referencePulse, metadata = sv.readPulses(DATA_DIR)

    print("=== arrays ===")
    print(f"pulses               {pulses.shape} {pulses.dtype}")
    print(f"referencePulse       {referencePulse.shape} {referencePulse.dtype}")
    print(
        "sensorPositions_ENU_m"
        f" {metadata['sensorPositions_ENU_m'].shape} {metadata['sensorPositions_ENU_m'].dtype}"
    )
    print(
        f"pulseAzimuth_deg     {metadata['pulseAzimuth_deg'].shape} {metadata['pulseAzimuth_deg'].dtype}"
    )
    print(
        f"lookAzimuth_deg      {metadata['lookAzimuth_deg'].shape} {metadata['lookAzimuth_deg'].dtype}"
    )
    print(
        f"lookPositions_ENU_m  {metadata['lookPositions_ENU_m'].shape} {metadata['lookPositions_ENU_m'].dtype}"
    )

    print("\n=== scalars ===")
    for key in (
        "radarCenterFrequency_Hz",
        "radarBandwidth_Hz",
        "rxSamplingFreq_Hz",
        "chirpDuration_s",
        "pulseDuration_s",
        "chirpRate_HzPerSec",
        "wavePropogationSpeed_mPerSec",
        "radarWavelength_m",
        "slantRange_m",
        "groundRange_m",
        "radarAltitude_m",
        "grazingAngle_deg",
        "azimuthCenter_deg",
        "azimuthHalfSpan_deg",
        "lookSpacing_deg",
        "pulseToPulseAzStep_deg",
        "nLooks",
        "pulsesPerLook",
        "totalPulses",
        "nRangeSamples",
    ):
        print(f"{key:<28} {metadata[key]}")

    print("\n=== CSV cross-check (look 0, pulse 0) ===")
    csv = np.genfromtxt(CSV_PATH, delimiter=",", skip_header=1)
    expected = csv[:, 2] + 1j * csv[:, 3]
    actual = np.asarray(pulses[0, 0, :])
    maxAbsErr = float(np.max(np.abs(actual - expected)))
    matched = bool(np.allclose(actual, expected))
    print(f"expected {expected.shape} {expected.dtype}")
    print(f"actual   {actual.shape} {actual.dtype}")
    print(f"maxAbsErr = {maxAbsErr:.3e}  matched = {matched}")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
