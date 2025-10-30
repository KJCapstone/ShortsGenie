"""Video editing utilities for cutting, cropping, and transforming videos."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import ffmpeg


@dataclass
class CropRegion:
    """Defines a rectangular crop region."""
    x: int  # Left position
    y: int  # Top position
    width: int
    height: int


@dataclass
class PanKeyframe:
    """Defines a crop position at a specific time."""
    time: float  # Seconds from start
    x: int
    y: int


@dataclass
class EditSegment:
    """Defines a single video segment to extract and process."""
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    crop_region: Optional[CropRegion] = None  # Static crop
    pan_keyframes: Optional[List[PanKeyframe]] = None  # Dynamic crop (overrides crop_region)
    crop_width: Optional[int] = None  # Crop width for pan_keyframes (if not specified, uses output dimensions)
    crop_height: Optional[int] = None  # Crop height for pan_keyframes (if not specified, uses output dimensions)

    @property
    def duration(self) -> float:
        """Calculate segment duration."""
        return self.end_time - self.start_time


@dataclass
class VideoEditConfig:
    """Configuration for video editing operations."""
    input_path: str
    output_path: str
    segments: List[EditSegment]
    output_width: int = 1080
    output_height: int = 1920
    output_fps: Optional[int] = None  # None = preserve original
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 23  # Quality (lower = better, 18-28 recommended)


class VideoEditor:
    """Handles video editing operations using FFmpeg."""

    def __init__(self):
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify FFmpeg is installed."""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg: "
                "https://ffmpeg.org/download.html"
            )

    def cut_segment(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        copy_streams: bool = False
    ) -> None:
        """
        Cut a video segment without re-encoding (fast).

        Args:
            input_path: Input video path
            output_path: Output video path
            start_time: Start time in seconds
            end_time: End time in seconds
            copy_streams: If True, copy streams without re-encoding (faster but less precise)
        """
        duration = end_time - start_time

        if copy_streams:
            # Fast method (stream copy) - may be imprecise at cut points
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', input_path,
                '-t', str(duration),
                '-c', 'copy',
                output_path
            ]
        else:
            # Slower method (re-encode) - precise cuts
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                output_path
            ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else e.stdout
            raise RuntimeError(f"FFmpeg error: {error_msg}") from e

    def crop_and_scale(
        self,
        input_path: str,
        output_path: str,
        crop: CropRegion,
        output_width: int,
        output_height: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> None:
        """
        Crop and scale a video to target dimensions.

        Args:
            input_path: Input video path
            output_path: Output video path
            crop: Crop region
            output_width: Output video width
            output_height: Output video height
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds
        """
        # Build filter chain
        filters = f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y},"
        filters += f"scale={output_width}:{output_height}"

        cmd = ['ffmpeg', '-y', '-i', input_path]

        # Add time range if specified
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
        if end_time is not None and start_time is not None:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])

        cmd.extend([
            '-vf', filters,
            '-c:v', 'libx264',
            '-crf', '23',
            '-c:a', 'aac',
            output_path
        ])

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else e.stdout
            raise RuntimeError(f"FFmpeg error: {error_msg}") from e

    def dynamic_crop(
        self,
        input_path: str,
        output_path: str,
        width: int,
        height: int,
        keyframes: List[PanKeyframe],
        output_width: int,
        output_height: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> None:
        """
        Apply dynamic cropping with interpolated movement between keyframes.

        Args:
            input_path: Input video path
            output_path: Output video path
            width: Crop width
            height: Crop height
            keyframes: List of position keyframes (time, x, y)
            output_width: Output video width
            output_height: Output video height
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds
        """
        # Sort keyframes by time
        keyframes = sorted(keyframes, key=lambda k: k.time)

        # Build interpolation expressions for x and y
        x_expr = self._build_interpolation_expr(
            [(kf.time, kf.x) for kf in keyframes]
        )
        y_expr = self._build_interpolation_expr(
            [(kf.time, kf.y) for kf in keyframes]
        )

        # Build filter chain
        # Escape commas in expressions for FFmpeg filter syntax
        x_expr_escaped = x_expr.replace(',', '\\,')
        y_expr_escaped = y_expr.replace(',', '\\,')

        filters = f"crop={width}:{height}:{x_expr_escaped}:{y_expr_escaped},"
        filters += f"scale={output_width}:{output_height}"

        cmd = ['ffmpeg', '-y', '-i', input_path]

        # Add time range if specified
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
        if end_time is not None and start_time is not None:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])

        cmd.extend([
            '-vf', filters,
            '-c:v', 'libx264',
            '-crf', '23',
            '-c:a', 'aac',
            output_path
        ])

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else e.stdout
            raise RuntimeError(f"FFmpeg error: {error_msg}") from e

    def _build_interpolation_expr(
        self,
        keyframes: List[Tuple[float, int]]
    ) -> str:
        """
        Build FFmpeg expression for linear interpolation between keyframes.

        Args:
            keyframes: List of (time, value) tuples

        Returns:
            FFmpeg expression string
        """
        if len(keyframes) == 0:
            return "0"
        if len(keyframes) == 1:
            return str(keyframes[0][1])

        # Build nested if expressions for piecewise linear interpolation
        expr_parts = []
        for i in range(len(keyframes) - 1):
            t1, v1 = keyframes[i]
            t2, v2 = keyframes[i + 1]

            # Linear interpolation formula: v1 + (v2-v1) * (t-t1)/(t2-t1)
            slope = (v2 - v1) / (t2 - t1)
            interpolation = f"{v1}+{slope}*(t-{t1})"

            if i == 0:
                # First segment
                expr_parts.append(f"if(lt(t,{t1}),{v1},if(lt(t,{t2}),{interpolation}")
            elif i == len(keyframes) - 2:
                # Last segment
                expr_parts.append(f",if(lt(t,{t2}),{interpolation},{v2})")
            else:
                # Middle segments
                expr_parts.append(f",if(lt(t,{t2}),{interpolation}")

        # Close all if statements
        # We have len(keyframes) - 1 segments, each opening an if except the last which closes one
        # Total ifs to close: (len(keyframes) - 1) + 1 for the outer if from first segment
        expr = "".join(expr_parts) + ")" * (len(keyframes) - 1)
        return expr

    def process_edit_config(self, config: VideoEditConfig) -> None:
        """
        Process a complete video edit configuration.

        This method handles multiple segments and combines them if needed.
        For single segment, directly outputs to config.output_path.
        For multiple segments, creates temporary files and concatenates them.

        Args:
            config: Video editing configuration
        """
        if len(config.segments) == 0:
            raise ValueError("No segments specified")

        if len(config.segments) == 1:
            # Single segment - process directly
            self._process_single_segment(config, config.segments[0], config.output_path)
        else:
            # Multiple segments - process and concatenate
            temp_dir = Path(config.output_path).parent / "temp_segments"
            temp_dir.mkdir(exist_ok=True)

            temp_files = []
            try:
                # Process each segment
                for i, segment in enumerate(config.segments):
                    temp_file = temp_dir / f"segment_{i:03d}.mp4"
                    self._process_single_segment(config, segment, str(temp_file))
                    temp_files.append(str(temp_file))

                # Concatenate segments
                self._concatenate_videos(temp_files, config.output_path)

            finally:
                # Cleanup temp files
                for temp_file in temp_files:
                    Path(temp_file).unlink(missing_ok=True)
                temp_dir.rmdir()

    def _process_single_segment(
        self,
        config: VideoEditConfig,
        segment: EditSegment,
        output_path: str
    ) -> None:
        """Process a single video segment."""
        if segment.pan_keyframes:
            # Dynamic crop with panning
            # Use specified crop dimensions or fall back to output dimensions
            crop_width = segment.crop_width if segment.crop_width else config.output_width
            crop_height = segment.crop_height if segment.crop_height else config.output_height

            self.dynamic_crop(
                input_path=config.input_path,
                output_path=output_path,
                width=crop_width,
                height=crop_height,
                keyframes=segment.pan_keyframes,
                output_width=config.output_width,
                output_height=config.output_height,
                start_time=segment.start_time,
                end_time=segment.end_time
            )
        elif segment.crop_region:
            # Static crop
            self.crop_and_scale(
                input_path=config.input_path,
                output_path=output_path,
                crop=segment.crop_region,
                output_width=config.output_width,
                output_height=config.output_height,
                start_time=segment.start_time,
                end_time=segment.end_time
            )
        else:
            # Just cut the segment (no crop)
            self.cut_segment(
                input_path=config.input_path,
                output_path=output_path,
                start_time=segment.start_time,
                end_time=segment.end_time,
                copy_streams=False
            )

    def _concatenate_videos(
        self,
        input_paths: List[str],
        output_path: str
    ) -> None:
        """Concatenate multiple video files."""
        # Create concat demuxer file list
        concat_file = Path(output_path).parent / "concat_list.txt"

        try:
            with open(concat_file, 'w') as f:
                for path in input_paths:
                    # FFmpeg concat demuxer requires absolute paths
                    abs_path = Path(path).resolve()
                    f.write(f"file '{abs_path}'\n")

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                output_path
            ]

            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else e.stdout
                raise RuntimeError(f"FFmpeg error: {error_msg}") from e

        finally:
            concat_file.unlink(missing_ok=True)


# Convenience function
def edit_video(config: VideoEditConfig) -> None:
    """
    Convenience function to edit a video.

    Args:
        config: Video editing configuration

    Example:
        ```python
        config = VideoEditConfig(
            input_path="input.mp4",
            output_path="output.mp4",
            segments=[
                EditSegment(
                    start_time=10.0,
                    end_time=20.0,
                    crop_region=CropRegion(x=100, y=50, width=1080, height=1920)
                )
            ]
        )
        edit_video(config)
        ```
    """
    editor = VideoEditor()
    editor.process_edit_config(config)
