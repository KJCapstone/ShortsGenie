"""Main window with page navigation."""

from typing import Dict
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QMessageBox, QProgressDialog, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QFont
from .main_page import MainPage
from .progress_page import ProgressPage
from .highlight_selector import SelectPage
from .preview_page import VideoPreviewPage
from .output_page import OutputSettingsPage
from src.pipeline.export_worker import ExportWorker
from src.utils.quality_presets import get_capped_preset_for_source
from src.utils.video_utils import get_video_resolution

# Constrants
WINDOW_TITLE = "Shorts Genie"
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600
WINDOW_INIT_WIDTH = 800
WINDOW_INIT_HEIGHT = 600

class MainWindow(QMainWindow):
    """Main application window managing page navigation."""
    OPTION_MAP = {

        "⚽ 골 모음 영상": "골",
        "⚡ 경기 주요 영상": "경기",
        "🎵 밈 영상": "밈"     

    }
    
    def __init__(self):
        """Initialize the main window and set up UI components."""
        super().__init__()
        self.selected_video_info = None # 선택된 영상 정보 저장

        # Cache for back navigation
        self.cached_highlights = None
        self.cached_file_path = None
        self.cached_option = None

        self._setup_ui()
        
    def _setup_ui(self):
        """Initialize the user interface."""

        # Configure window properties
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_INIT_WIDTH, WINDOW_INIT_HEIGHT)
        
        # Create central stacked widget for page management
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Initialize pages
        self.main_page = MainPage()
        self.progress_page = ProgressPage()
        self.select_page = SelectPage()
        self.preview_page = VideoPreviewPage()
        self.output_page = OutputSettingsPage()
        
        # Add pages to stack (index order matters - first is default)
        self.stacked_widget.addWidget(self.main_page)     # index 0
        self.stacked_widget.addWidget(self.progress_page)    # index 1
        self.stacked_widget.addWidget(self.select_page)   # index 2
        self.stacked_widget.addWidget(self.preview_page)  # index 3
        self.stacked_widget.addWidget(self.output_page)   # index 4

        # Connect signals to slots
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect signals from child pages to appropriate slots."""
        # Forward navigation
        self.main_page.edit_requested.connect(self.show_progress_page)
        self.progress_page.processing_completed.connect(self.show_select_page)
        self.select_page.video_preview_requested.connect(self.show_preview_page)
        self.preview_page.output_settings_requested.connect(self.show_output_page)

        # Export handler
        self.output_page.export_requested.connect(self.handle_export_request)

        # Back navigation
        self.progress_page.back_requested.connect(self.handle_progress_page_back)
        self.select_page.back_requested.connect(self.handle_select_page_back)
        self.preview_page.back_requested.connect(self.handle_preview_page_back)
        self.output_page.back_requested.connect(self.handle_output_page_back)

    @Slot(str, str)
    def show_progress_page(self, file_path: str, option: str) -> None:
        """
        Switch to editing page and pass data to it.

        Args:
            file_path: Path to the selected video file
            option: Selected editing option/condition
        """
        # Cache for potential back navigation
        self.cached_file_path = file_path
        self.cached_option = option

        selected_option = self.OPTION_MAP.get(option)

        self.progress_page.set_data(file_path, selected_option)
        self.stacked_widget.setCurrentIndex(1)

    @Slot(list)
    def show_select_page(self, highlights) -> None:
        """Show the select page."""
        # Cache highlights for potential back navigation
        self.cached_highlights = highlights

        self.select_page.set_highlights(highlights)
        self.stacked_widget.setCurrentIndex(2)

    @Slot(list, int)
    def show_preview_page(self, highlights, selected_index):
        """
        Show preview page with highlights.
        
        Args:
            highlights: List of highlight dictionaries
            selected_index: Index of initially selected video
        """
        self.preview_page.load_highlights(highlights, selected_index)
        self.stacked_widget.setCurrentIndex(3)

    @Slot(dict)
    def show_output_page(self, video_info: Dict) -> None:
        """
        Show output page with selected video info.
        
        Args:
            video_info: Dictionary containing selected video information
        """
        # 선택된 영상 정보 저장
        self.selected_video_info = video_info
        self.output_page.video_info = video_info

        # 페이지 전환
        self.stacked_widget.setCurrentIndex(4)

    # Back navigation handlers

    @Slot()
    def handle_progress_page_back(self) -> None:
        """Handle back navigation from progress page."""
        # Cancel processing if active
        if hasattr(self.progress_page, 'worker') and self.progress_page.worker:
            if self.progress_page.worker.isRunning():
                # Show confirmation dialog
                reply = QMessageBox.question(
                    self,
                    "처리 취소",
                    "영상 분석이 진행 중입니다.\n취소하고 돌아가시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

                # Cancel worker
                self.progress_page.worker.cancel()

        # Clear cached data
        self.cached_file_path = None
        self.cached_option = None
        self.cached_highlights = None

        # Navigate back
        self.stacked_widget.setCurrentIndex(0)

    @Slot()
    def handle_select_page_back(self) -> None:
        """Handle back navigation from select page."""
        # User wants to select different video - clear cache
        self.cached_highlights = None
        self.cached_file_path = None
        self.cached_option = None

        # Navigate back to main page
        self.stacked_widget.setCurrentIndex(0)

    @Slot()
    def handle_preview_page_back(self) -> None:
        """Handle back navigation from preview page."""
        # Stop video playback before navigating
        if hasattr(self.preview_page, 'media_player'):
            self.preview_page.media_player.stop()

        # Stop preview videos in left panel
        if hasattr(self.preview_page, 'video_items'):
            for item in self.preview_page.video_items:
                if hasattr(item, 'media_player'):
                    item.media_player.stop()

        # Restore select page with cached highlights
        if self.cached_highlights:
            self.select_page.set_highlights(self.cached_highlights)

        # Navigate back to select page
        self.stacked_widget.setCurrentIndex(2)

    @Slot()
    def handle_output_page_back(self) -> None:
        """Handle back navigation from output page."""
        # Navigate back to preview page
        # Video info is already loaded in preview page
        self.stacked_widget.setCurrentIndex(3)

    @Slot(str, str, str)
    def handle_export_request(self, save_path: str, filename: str, quality: str) -> None:
        """
        Handle export request from output page.

        Args:
            save_path: Directory to save output file
            filename: Output filename (without extension)
            quality: Quality string (e.g., "1080 p")
        """
        if not self.selected_video_info:
            QMessageBox.warning(
                self,
                "오류",
                "선택된 영상이 없습니다."
            )
            return

        # Get input video path
        input_path = self.selected_video_info.get('video_path')
        if not input_path or not Path(input_path).exists():
            QMessageBox.warning(
                self,
                "오류",
                "입력 영상 파일을 찾을 수 없습니다."
            )
            return

        # Construct output path
        output_filename = f"{filename}.mp4"
        output_path = str(Path(save_path) / output_filename)

        # Check if file exists
        if Path(output_path).exists():
            reply = QMessageBox.question(
                self,
                "파일 덮어쓰기",
                f"'{output_filename}'이(가) 이미 존재합니다.\n덮어쓰시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        try:
            # Detect source resolution
            source_width, source_height = get_video_resolution(input_path)

            # Get target quality preset and cap to source resolution
            final_preset, was_capped = get_capped_preset_for_source(
                source_width,
                source_height,
                quality
            )

            # Notify user if quality was capped
            if was_capped:
                QMessageBox.information(
                    self,
                    "화질 조정",
                    f"원본 영상 해상도({source_width}x{source_height})보다 높은 화질은 "
                    f"선택할 수 없습니다.\n{final_preset.name}로 출력됩니다."
                )

            # Create and start export worker
            self.export_worker = ExportWorker(
                input_path=input_path,
                output_path=output_path,
                target_width=final_preset.width,
                target_height=final_preset.height,
                crf=final_preset.crf
            )

            # Connect worker signals
            self.export_worker.progress_updated.connect(self._on_export_progress)
            self.export_worker.export_completed.connect(self._on_export_completed)
            self.export_worker.export_failed.connect(self._on_export_failed)

            # Show progress dialog
            self._show_export_progress_dialog()

            # Start export
            self.export_worker.start()

        except Exception as e:
            QMessageBox.critical(
                self,
                "오류",
                f"출력을 시작할 수 없습니다:\n{str(e)}"
            )

    def _show_export_progress_dialog(self) -> None:
        """Show export progress dialog."""
        self.export_progress_dialog = QProgressDialog(
            "영상 출력 중...",
            "취소",
            0, 100,
            self
        )
        self.export_progress_dialog.setWindowTitle("출력 진행 중")
        self.export_progress_dialog.setWindowModality(Qt.WindowModal)
        self.export_progress_dialog.setMinimumDuration(0)
        self.export_progress_dialog.setValue(0)

        # Handle cancel button
        self.export_progress_dialog.canceled.connect(self._on_export_cancelled)
        self.export_progress_dialog.show()

    @Slot(int, str)
    def _on_export_progress(self, percentage: int, message: str) -> None:
        """Update export progress dialog."""
        if hasattr(self, 'export_progress_dialog'):
            self.export_progress_dialog.setValue(percentage)
            self.export_progress_dialog.setLabelText(message)

    @Slot(dict)
    def _on_export_completed(self, stats: dict) -> None:
        """Handle successful export completion."""
        if hasattr(self, 'export_progress_dialog'):
            self.export_progress_dialog.close()

        output_path = stats.get('output_path', '알 수 없음')
        processing_time = stats.get('processing_time', 0)

        # Show custom completion dialog with "메인 페이지로" button
        self._show_export_completion_dialog(output_path, processing_time)

    @Slot(str)
    def _on_export_failed(self, error: str) -> None:
        """Handle export failure."""
        if hasattr(self, 'export_progress_dialog'):
            self.export_progress_dialog.close()

        QMessageBox.critical(
            self,
            "출력 실패",
            f"영상 출력 중 오류가 발생했습니다:\n{error}"
        )

    @Slot()
    def _on_export_cancelled(self) -> None:
        """Handle export cancellation."""
        if hasattr(self, 'export_worker') and self.export_worker.isRunning():
            self.export_worker.cancel()
            self.export_worker.wait()  # Wait for thread to finish

            QMessageBox.information(
                self,
                "출력 취소",
                "영상 출력이 취소되었습니다."
            )

    def _show_export_completion_dialog(self, output_path: str, processing_time: float) -> None:
        """
        Show custom export completion dialog with "메인 페이지로" button.

        Args:
            output_path: Path to exported video file
            processing_time: Processing time in seconds
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("출력 완료")
        dialog.setFixedSize(400, 200)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Success icon and title
        title_label = QLabel("✅ 출력 완료!")
        title_label.setFont(QFont("맑은 고딕", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # File path
        path_label = QLabel(f"파일: {output_path}")
        path_label.setWordWrap(True)
        path_label.setStyleSheet("color: #666;")
        layout.addWidget(path_label)

        # Processing time
        time_label = QLabel(f"처리 시간: {processing_time:.1f}초")
        time_label.setStyleSheet("color: #666;")
        layout.addWidget(time_label)

        layout.addStretch()

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        # "메인 페이지로" button
        main_page_button = QPushButton("메인 페이지로")
        main_page_button.setMinimumHeight(40)
        main_page_button.setCursor(Qt.PointingHandCursor)
        main_page_button.setStyleSheet("""
            QPushButton {
                background-color: #7B68BE;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14pt;
                font-weight: bold;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #6A57AD;
            }
        """)
        main_page_button.clicked.connect(lambda: self._go_to_main_page_from_dialog(dialog))

        # "닫기" button
        close_button = QPushButton("닫기")
        close_button.setMinimumHeight(40)
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                color: #333;
                border: none;
                border-radius: 8px;
                font-size: 14pt;
                font-weight: bold;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #D0D0D0;
            }
        """)
        close_button.clicked.connect(dialog.accept)

        button_layout.addWidget(main_page_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        dialog.exec()

    def _go_to_main_page_from_dialog(self, dialog: QDialog) -> None:
        """
        Navigate to main page and close dialog.

        Args:
            dialog: The dialog to close
        """
        dialog.accept()
        self.stacked_widget.setCurrentIndex(0)  # Navigate to MainPage