"""Quality preset system for video export."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class QualityPreset:
    """
    Quality preset configuration for video export.

    Attributes:
        width: Video width in pixels
        height: Video height in pixels
        crf: Constant Rate Factor for H.264 encoding (lower = better quality, larger file)
        name: Human-readable name for the preset
    """
    width: int
    height: int
    crf: int
    name: str


# Available quality presets mapped to GUI strings
QUALITY_PRESETS: Dict[str, QualityPreset] = {
    "720 p": QualityPreset(720, 1280, 28, "720p HD"),
    "1080 p": QualityPreset(1080, 1920, 25, "1080p Full HD"),
    "1440 p": QualityPreset(1440, 2560, 23, "1440p 2K"),
    "2160 p (4K)": QualityPreset(2160, 3840, 21, "2160p 4K")
}


def get_preset_for_quality(quality_str: str) -> QualityPreset:
    """
    Get quality preset from GUI quality string.

    Args:
        quality_str: Quality string from GUI (e.g., "1080 p", "720 p")

    Returns:
        QualityPreset object for the requested quality
        Falls back to 1080p if quality string is not recognized
    """
    return QUALITY_PRESETS.get(quality_str, QUALITY_PRESETS["1080 p"])


def cap_resolution_to_source(
    source_width: int,
    source_height: int,
    target_preset: QualityPreset
) -> tuple[QualityPreset, bool]:
    """
    Cap target resolution to source resolution (no upsampling).

    This function ensures that the output resolution never exceeds the source
    resolution. If the user selects a quality higher than the source supports,
    the function returns the highest quality preset that doesn't exceed the source.

    Args:
        source_width: Width of source video in pixels
        source_height: Height of source video in pixels
        target_preset: User's desired quality preset

    Returns:
        Tuple of (final_preset, was_capped)
        - final_preset: The quality preset to use (may be capped)
        - was_capped: True if the preset was capped, False if using target as-is

    Examples:
        >>> source = 1080x1920 (vertical), target = 1440p
        >>> Returns (1080p preset, True) - capped to source resolution

        >>> source = 1080x1920 (vertical), target = 720p
        >>> Returns (720p preset, False) - downsampling is OK
    """
    # For vertical videos, the width is the limiting dimension
    # (narrower dimension in vertical orientation)
    source_limit = min(source_width, source_height)

    # If target doesn't exceed source, use it as-is
    if target_preset.width <= source_limit:
        return (target_preset, False)

    # Target exceeds source - need to cap
    # Find the highest preset that fits within source resolution
    sorted_presets = sorted(
        QUALITY_PRESETS.items(),
        key=lambda x: x[1].width,
        reverse=True  # Start from highest quality
    )

    for quality_str, preset in sorted_presets:
        if preset.width <= source_limit:
            return (preset, True)

    # Edge case: source is lower than all presets
    # Return the lowest preset (720p)
    return (QUALITY_PRESETS["720 p"], True)


def get_capped_preset_for_source(
    source_width: int,
    source_height: int,
    quality_str: str
) -> tuple[QualityPreset, bool]:
    """
    Convenience function to get capped preset in one call.

    Args:
        source_width: Width of source video in pixels
        source_height: Height of source video in pixels
        quality_str: Quality string from GUI (e.g., "1080 p")

    Returns:
        Tuple of (final_preset, was_capped)

    Example:
        >>> preset, capped = get_capped_preset_for_source(720, 1280, "1080 p")
        >>> preset.name
        '720p HD'
        >>> capped
        True
    """
    target_preset = get_preset_for_quality(quality_str)
    return cap_resolution_to_source(source_width, source_height, target_preset)
