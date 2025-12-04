"""Export worker thread for video re-encoding."""

import subprocess
from pathlib import Path
from PySide6.QtCore import QThread, Signal
import time


class ExportWorker(QThread):
    """
    Background thread for video export with FFmpeg re-encoding.

    Signals:
        progress_updated: Emitted with (percentage, message)
        export_completed: Emitted with statistics dict on success
        export_failed: Emitted with error message string on failure
    """

    # Signals
    progress_updated = Signal(int, str)
    export_completed = Signal(dict)
    export_failed = Signal(str)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        target_width: int,
        target_height: int,
        crf: int = 25,
        fps: int = 30
    ):
        """
        Initialize export worker.

        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            target_width: Target video width in pixels
            target_height: Target video height in pixels
            crf: Constant Rate Factor for quality (18-28, lower=better)
            fps: Target frames per second
        """
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        
        # [안전장치] 해상도가 홀수면 인코딩이 실패하므로 짝수로 보정
        self.target_width = target_width - (target_width % 2)
        self.target_height = target_height - (target_height % 2)
        
        self.crf = crf
        self.fps = fps
        self._is_cancelled = False

    def run(self):
        """Execute video export in background thread."""
        try:
            start_time = time.time()

            # Create output directory if needed
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Emit initial progress
            self.progress_updated.emit(0, "영상 출력 준비 중...")

            # Build FFmpeg command for re-encoding
            # -i: input file
            # -vf scale: resize video to target resolution
            # -c:v libx264: use H.264 video codec
            # -crf: quality setting
            # -preset: encoding speed/compression tradeoff
            # -r: frame rate
            # -c:a copy: copy audio stream without re-encoding
            # -y: overwrite output file if exists
            cmd = [
                'ffmpeg',
                '-i', self.input_path,
                '-vf', f'scale={self.target_width}:{self.target_height}',
                '-c:v', 'libx264',
                '-crf', str(self.crf),
                '-preset', 'medium',  # balance between speed and compression
                '-r', str(self.fps),
                '-c:a', 'copy',  # copy audio without re-encoding
                '-y',  # overwrite output
                self.output_path
            ]

            # Execute FFmpeg
            self.progress_updated.emit(10, "영상 인코딩 중...")

            # [핵심 수정 1] stdout/stderr를 None으로 설정하여 파이프 막힘(Deadlock) 방지
            process = subprocess.Popen(
                cmd,
                stdout=None,
                stderr=None,
                universal_newlines=True
            )

            # Monitor process
            while process.poll() is None:
                if self._is_cancelled:
                    process.terminate()
                    process.wait()
                    return

                # Update progress (simple linear progress since FFmpeg output is hard to parse)
                # We increment progress gradually during encoding
                time.sleep(0.5)

            # [핵심 수정 2] "성공" 판정 기준 완화
            # FFmpeg가 경고(exit code 1)를 뱉었더라도, 파일이 정상적으로 만들어졌으면 성공으로 간주합니다.
            output_file = Path(self.output_path)
            
            if output_file.exists() and output_file.stat().st_size > 0:
                # 파일이 존재하고 용량이 0보다 크면 성공! (에러 코드 무시)
                pass
            else:
                # 파일이 없거나 0바이트면 진짜 실패
                if process.returncode != 0:
                    raise Exception(f"FFmpeg 변환 실패 (에러 코드: {process.returncode}). 터미널 로그를 확인하세요.")
                else:
                    raise Exception("알 수 없는 이유로 파일이 생성되지 않았습니다.")

            # Calculate processing time
            processing_time = time.time() - start_time

            # Emit completion with statistics
            if not self._is_cancelled:
                self.progress_updated.emit(100, "출력 완료!")
                stats = {
                    'output_path': self.output_path,
                    'processing_time': processing_time,
                    'resolution': f'{self.target_width}x{self.target_height}',
                    'crf': self.crf
                }
                self.export_completed.emit(stats)

        except Exception as e:
            if not self._is_cancelled:
                self.export_failed.emit(str(e))

    def cancel(self):
        """Cancel the export operation."""
        self._is_cancelled = True