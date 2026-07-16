"""Read the simulator's raw pulses, reference chirp, and geometry metadata.

The on-disk layout (under ``.../raw_pulse_data_repackaged``) is::

    rx_sig.npy             (nLooks, pulsesPerLook, nRangeSamples) complex128
    lfm_ref.npy            (nRangeSamples,)                       complex128
    sensor_positions.npy   (nLooks, pulsesPerLook, 3)             float64
    pulse_azimuth_deg.npy  (nLooks, pulsesPerLook)                float64
    look_azimuth_deg.npy   (nLooks,)                              float64
    look_positions.npy     (nLooks, 3)                            float64

with ``Meta_parameters.txt`` one directory up and ``geometry_note.txt`` alongside
the ``.npy`` files. ``rx_sig`` (~6.9 GB) is memory-mapped so the full cube is
never resident; the small geometry arrays are loaded eagerly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from simValidation._parseMeta import parseMeta

__all__ = ["readPulses"]

# ``.npy`` files inside the repackaged data directory.
_RX_SIG_FILE = "rx_sig.npy"
_LFM_REF_FILE = "lfm_ref.npy"
_SENSOR_POS_FILE = "sensor_positions.npy"
_PULSE_AZ_FILE = "pulse_azimuth_deg.npy"
_LOOK_AZ_FILE = "look_azimuth_deg.npy"
_LOOK_POS_FILE = "look_positions.npy"
# Text sidecars: Meta_parameters.txt lives one level above the repackaged dir,
# geometry_note.txt lives alongside the .npy files.
_META_PARAMS_FILE = "Meta_parameters.txt"
_GEOMETRY_NOTE_FILE = "geometry_note.txt"


def readPulses(
    dataDirPath: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read raw pulses, the LFM reference, and geometry metadata from disk.

    Args:
        dataDirPath: Path to the simulator's repackaged data directory
            (``.../raw_pulse_data_repackaged``) containing the ``.npy`` files.

    Returns:
        A tuple ``(pulses, referencePulse, metadata)`` where ``pulses`` is the
        memory-mapped ``rx_sig`` cube with shape ``(nLooks, pulsesPerLook,
        nRangeSamples)`` complex128, ``referencePulse`` is the ``lfm_ref`` chirp
        with shape ``(nRangeSamples,)`` complex128, and ``metadata`` bundles the
        per-pulse geometry arrays (``sensorPositions_ENU_m``, ``pulseAzimuth_deg``,
        ``lookAzimuth_deg``, ``lookPositions_ENU_m``) and the parsed scalar
        parameters from the text sidecars.
    """
    dataDir = Path(dataDirPath)
    pulses = np.load(dataDir / _RX_SIG_FILE, mmap_mode="r")
    referencePulse = np.load(dataDir / _LFM_REF_FILE)
    sensorPositions_ENU_m = np.load(dataDir / _SENSOR_POS_FILE)
    pulseAzimuth_deg = np.load(dataDir / _PULSE_AZ_FILE)
    lookAzimuth_deg = np.load(dataDir / _LOOK_AZ_FILE)
    lookPositions_ENU_m = np.load(dataDir / _LOOK_POS_FILE)

    metadata = parseMeta(dataDir.parent / _META_PARAMS_FILE, dataDir / _GEOMETRY_NOTE_FILE)
    metadata["sensorPositions_ENU_m"] = sensorPositions_ENU_m
    metadata["pulseAzimuth_deg"] = pulseAzimuth_deg
    metadata["lookAzimuth_deg"] = lookAzimuth_deg
    metadata["lookPositions_ENU_m"] = lookPositions_ENU_m
    return pulses, referencePulse, metadata
