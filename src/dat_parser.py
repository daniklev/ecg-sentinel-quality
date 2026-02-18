"""
DAT file parser - Python port of CommwellSentinelAnalysis.ECG1()
from Stream-Analysis-Service.

Parses binary .dat files from Sentinel ECG devices (2-lead, 125 Hz).
Supports protocol 0xAA (legacy) and 0xAC (new with battery status).
"""

import struct
import numpy as np
from typing import Tuple, Optional


SENTINEL_BAD_SAMPLE = 0x8000
MAX_LOST_PACKETS = 300


def _extract_packet_number_aa(data: bytes, offset: int) -> int:
    """Extract packet number from 0xAA protocol."""
    return data[offset + 3] * 256 + data[offset + 4]


def _extract_packet_number_ac(data: bytes, offset: int) -> int:
    """Extract packet number from 0xAC protocol (4-byte)."""
    pn = (data[offset + 1] & 0x3) * 256
    pn = (pn + data[offset + 2]) * 256
    pn = (pn + data[offset + 3]) * 256
    pn = pn + data[offset + 4]
    return pn


def _extract_samples(data: bytes, offset: int) -> list:
    """
    Extract 4 pairs of (lead1, lead2) samples from a 21-byte packet.
    Each sample is a signed 16-bit integer (little-endian, swapped bytes).
    """
    samples = []
    for k in range(4):
        # Lead 1: bytes at offset+6+4k and offset+5+4k (swapped)
        b_lo = data[offset + k * 4 + 6]
        b_hi = data[offset + k * 4 + 5]
        lead1 = struct.unpack('<h', bytes([b_lo, b_hi]))[0]

        # Lead 2: bytes at offset+8+4k and offset+7+4k (swapped)
        b_lo = data[offset + k * 4 + 8]
        b_hi = data[offset + k * 4 + 7]
        lead2 = struct.unpack('<h', bytes([b_lo, b_hi]))[0]

        samples.append((lead1, lead2))
    return samples


def parse_dat_file(file_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a single .dat file into two lead arrays at native 125 Hz.

    Returns:
        (lead1, lead2) - numpy arrays of int16 ECG samples
    """
    if len(file_data) < 1000:
        raise ValueError(f"File too small ({len(file_data)} bytes), minimum 1000")

    lead1_samples = []
    lead2_samples = []
    i = 0
    packet_check = False
    last_packet = 0
    lost_count = 0

    while i < len(file_data) - 20:
        byte_val = file_data[i]

        if byte_val == 0xAA:
            pn = _extract_packet_number_aa(file_data, i)
            if packet_check:
                if pn != (last_packet + 1) & 0xFFFF:
                    # Fill lost packets
                    expected = (last_packet + 1) & 0xFFFF
                    while expected != pn:
                        lead1_samples.append(SENTINEL_BAD_SAMPLE)
                        lead2_samples.append(SENTINEL_BAD_SAMPLE)
                        lost_count += 1
                        if lost_count > MAX_LOST_PACKETS:
                            raise ValueError(f"Too many lost packets ({lost_count})")
                        expected = (expected + 1) & 0xFFFF
            else:
                packet_check = True
            last_packet = pn

            for s1, s2 in _extract_samples(file_data, i):
                lead1_samples.append(s1)
                lead2_samples.append(s2)
            i += 21

        elif byte_val == 0xAC and (file_data[i + 1] & 0xF8) == 0xA0:
            # Battery status packet - skip
            i += 21

        elif byte_val == 0xAC and (file_data[i + 1] & 0xF8) != 0xA0:
            pn = _extract_packet_number_ac(file_data, i)
            if packet_check:
                if pn != last_packet + 1:
                    expected = last_packet + 1
                    while expected != pn:
                        lead1_samples.append(SENTINEL_BAD_SAMPLE)
                        lead2_samples.append(SENTINEL_BAD_SAMPLE)
                        lost_count += 1
                        if lost_count > MAX_LOST_PACKETS:
                            raise ValueError(f"Too many lost packets ({lost_count})")
                        expected += 1
            else:
                packet_check = True
            last_packet = pn

            for s1, s2 in _extract_samples(file_data, i):
                lead1_samples.append(s1)
                lead2_samples.append(s2)
            i += 21

        elif byte_val == 0xAB:
            # Event marker - skip
            i += 21
        else:
            i += 1

    return np.array(lead1_samples, dtype=np.float64), np.array(lead2_samples, dtype=np.float64)


def get_packet_range(file_data: bytes) -> Tuple[int, int]:
    """
    Extract first and last packet numbers from a .dat file.
    Used to check file consecutiveness.
    """
    first = last = 0
    found_first = False
    i = 0

    while i < len(file_data) - 5:
        byte_val = file_data[i]
        pn = None

        if byte_val == 0xAA:
            pn = _extract_packet_number_aa(file_data, i)
            i += 20
        elif byte_val == 0xAC and (file_data[i + 1] & 0xF8) != 0xA0:
            pn = _extract_packet_number_ac(file_data, i)
            i += 20
        else:
            i += 1
            continue

        if pn is not None:
            if not found_first:
                first = pn
                found_first = True
            last = pn

    return first, last


def are_files_consecutive(files_data: list) -> bool:
    """
    Check if a list of .dat file byte arrays are consecutive
    (packet numbers follow sequentially).
    """
    if len(files_data) < 2:
        return True

    prev_last = None
    for fdata in files_data:
        first_pn, last_pn = get_packet_range(fdata)
        if prev_last is not None:
            if first_pn != prev_last + 1:
                return False
        prev_last = last_pn
    return True
