"""Helpers for discovering and selecting audio input devices."""

from __future__ import annotations

import sounddevice as sd


def list_input_devices() -> list[tuple[int, str, int, int]]:
    """Return `(index, name, max_input_channels, default_samplerate)` rows."""
    devices = sd.query_devices()
    rows: list[tuple[int, str, int, int]] = []
    for index, device in enumerate(devices):
        rows.append(
            (
                index,
                device.get("name", ""),
                device.get("max_input_channels", 0),
                int(device.get("default_samplerate", 0) or 0),
            )
        )
    return rows


def print_input_devices(chosen_index: int | None = None) -> None:
    """Pretty-print available inputs and highlight the selected one."""
    print("\nAvailable audio devices:")
    print(f"{'Idx':>3}  {'InCh':>4}  {'DefSR':>6}  Name")
    for index, name, input_channels, sample_rate in list_input_devices():
        marker = "*" if chosen_index == index else " "
        print(f"{index:>3}{marker}  {input_channels:>4}  {sample_rate:>6}  {name}")
    print("('*' marks the selected input device)\n")


def pick_input_device() -> tuple[int, str]:
    """
    Choose an input device using the original fallback order:
    1. `sd.default.device[0]`
    2. a microphone-like device name
    3. the first device with input channels
    """
    devices = sd.query_devices()
    candidate = None

    try:
        default_input = sd.default.device[0]
    except Exception:
        default_input = None

    if isinstance(default_input, int) and 0 <= default_input < len(devices):
        if devices[default_input].get("max_input_channels", 0) > 0:
            candidate = default_input

    if candidate is None:
        keywords = ("microphone", "mic", "built-in", "external", "usb")
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) <= 0:
                continue
            name = (device.get("name") or "").lower()
            if any(keyword in name for keyword in keywords):
                candidate = index
                break

    if candidate is None:
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                candidate = index
                break

    if candidate is None:
        raise RuntimeError("No input device with capture channels found.")

    return candidate, devices[candidate].get("name", "")


def pick_sample_rate(device_index: int) -> int | None:
    """Choose the device default, or fall back to 48000/44100 if needed."""
    info = sd.query_devices(device_index)
    sample_rate = float(info.get("default_samplerate", 0) or 0)
    if sample_rate > 0:
        return int(sample_rate)

    for candidate in (48000, 44100):
        try:
            sd.check_input_settings(device=device_index, samplerate=candidate, channels=1)
            return candidate
        except Exception:
            continue

    return None
