from multiprocessing.pool import AsyncResult
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import multiprocessing
import queue
from typing import Iterable, Callable, Optional, TypedDict

from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QLabel, QGraphicsOpacityEffect, QWidget, QHBoxLayout, QFrame
from PyQt6.QtCore import (
    QTimer, Qt, QThread, pyqtSignal, pyqtSlot, QPropertyAnimation, QObject, QEvent, QUrl, QFileInfo,
    QTemporaryDir, QMutex
    )
from PyQt6.QtGui import QFont

import vlc

from . import AlbumReader
from . import FFMPEGObject
from . import utils

logger = utils.get_logger('SlideShow.GUIPyQt6FFMPEG')


class VLCMediaPlayer(QObject):
    mediaStatusChanged = pyqtSignal(int)
    MediaStatus = SimpleNamespace()
    MediaStatus.EndOfMedia = 6
    MediaStatus.OtherStatus = 0

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent=parent)
        self._instance = vlc.Instance('--verbose -1')
        self._player = self._instance.media_player_new()
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._is_playing = False
        # self._timer.timeout.connect(self.update_mediaStatusChanged)
        self.event_manager = self._player.event_manager()
        self.event_manager.event_attach(
            vlc.EventType.MediaPlayerEndReached,
            lambda e: self.mediaStatusChanged.emit(self.MediaStatus.EndOfMedia)
            )

    def setVideoOutput(self, video_widget):
        if sys.platform.startswith('linux'):  # for Linux using the X Server
            self._player.set_xwindow(self.video_widget.winId())
        elif sys.platform == "win32":  # for Windows
            self._player.set_hwnd(self.video_widget.winId())
        elif sys.platform == "darwin":  # for MacOS
            self._player.set_nsobject(int(video_widget.winId()))

    def setSource(self, media):
        self.media = self._instance.media_new(media.url())
        self._player.set_media(self.media)
        # self._player.play()
        self._player.pause()

    def play(self):
        self._player.play()
        self._timer.start()
        self._is_playing = True

    def setPlaybackRate(self, v):
        self._player.set_rate(v)

    def update_mediaStatusChanged(self):
        is_playing = self._player.is_playing()
        if is_playing != self._is_playing:
            self._is_playing = is_playing
            self.mediaStatusChanged.emit(self.MediaStatus.OtherStatus if is_playing else self.MediaStatus.EndOfMedia)


def process_meta(command):
    return {'tag': 'meta', 'content': command}


def _ffmpeg_read(
    queue: queue.Queue[AsyncResult],
    iterable: Iterable[ET.Element],
    delay: float,
    size: tuple[int, int],
    dpath: str,
    segment_time: int = 10,
    log_level: Optional[str] = None
) -> None:
    # with multiprocessing.Pool(None, limit_cpu) as pool:
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        ffmpeg_object = FFMPEGObject.FFMPEGObjectLive(delay, size, log_level=log_level)
        index = 0
        skipped = 0
        for media in iterable:
            if media.tag == 'meta':
                queue.put_nowait(pool.apply_async(process_meta, (media.get('command'),)))
                skipped += 1
                continue
            ffmpeg_object.add_stream(media)

            if ffmpeg_object.total_duration > segment_time:
                if skipped > 0:
                    qput = queue.put_nowait
                    skipped -= 1
                else:
                    qput = queue.put
                    fname = f'{dpath}/{index:0>4}.ts'
                qput(pool.apply_async(ffmpeg_object.compile_call(fname)))
                ffmpeg_object.reset()
                index += 1
        if ffmpeg_object.streams:
            if skipped > 0:
                qput = queue.put_nowait
                skipped -= 1
            else:
                qput = queue.put
            qput(pool.apply_async(ffmpeg_object.compile_call(fname)))
        pool.close()
        pool.join()
        queue.join()


class FStat(TypedDict):
    path: Path
    size: int


class CleanUpThread(QThread):
    max_size: int = 10
    flist: list[FStat] = []
    lock: QMutex = QMutex()

    def run(self) -> None:
        flist = self.flist
        self.lock.lock()
        total_size = 0
        it = reversed(self.flist)
        try:
            next(it)
        except StopIteration:
            self.lock.unlock()
            return
        i = 0
        try:
            for fstat in it:
                path = fstat['path']
                size = fstat['size']
                if size < 0:
                    size = path.stat().st_size
                    fstat['size'] = size
                total_size += size
                if total_size > self.max_size:
                    i = flist.index(fstat)
                    break
            for fstat in flist[:i]:
                fstat['path'].unlink()
            flist[:i] = []
        except FileNotFoundError:
            pass
        self.lock.unlock()


class ReadMediaThread(QThread):
    def __init__(
        self,
        q: queue.Queue[AsyncResult],
        media_objects: Iterable[ET.Element],
        delay: float,
        size: tuple[int, int],
        dpath: str,
        log_level: Optional[str] = None,
        parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self.queue: queue.Queue[AsyncResult] = q
        self.media_objects: Iterable[ET.Element] = media_objects
        self.delay: float = delay
        self.size: tuple[int, int] = size
        self.dpath: str = dpath
        self.log_level = log_level

    def run(self) -> None:
        _ffmpeg_read(self.queue, self.media_objects, self.delay, self.size, self.dpath, log_level=self.log_level)


class StdInfo(QLabel):
    def __init__(
        self,
        text_func: Callable | str,
        font: QFont = QFont('Arial', 36),
        align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        animated: bool = True,
        parent: Optional[QObject] = None,
        **kwds
    ):
        super().__init__(parent, **kwds)

        self.setStyleSheet('QLabel {color: gray;}')
        self.setStyleSheet('QLabel {background-color: rgba(0, 0, 0, 50);}')
        self.setFont(font)
        self.animated: bool = animated
        self.text_func: Callable | str = text_func
        self.align: Qt.AlignmentFlag = align
        self.display: Callable = self.display_func if callable(text_func) else self.display_str
        effect = QGraphicsOpacityEffect(self)
        self.anim: QPropertyAnimation = QPropertyAnimation(effect, b'opacity')
        self.setGraphicsEffect(effect)
        self.anim.setDuration(2000)
        self.anim.setStartValue(1.)
        self.anim.setEndValue(0.)
        # self.anim.setEasingCurve(QEasingCurve.OutQuad)
        self.anim.finished.connect(self.hide)
        self.parent().resized.connect(self.set_position)

    @pyqtSlot()
    def set_position(self, pos=None) -> None:
        if pos is None:
            coords = self.parent().geometry().getRect()
            if self.align & Qt.AlignmentFlag.AlignLeft:
                x = 0
            if self.align & Qt.AlignmentFlag.AlignRight:
                x = coords[2] - self.width()
            if self.align & Qt.AlignmentFlag.AlignTop:
                y = 0
            if self.align & Qt.AlignmentFlag.AlignBottom:
                y = coords[3] - self.height()
            if self.align & Qt.AlignmentFlag.AlignHCenter:
                x = (coords[2] - self.width()) // 2
            if self.align & Qt.AlignmentFlag.AlignVCenter:
                y = (coords[3] - self.height()) // 2
            pos = (x, y)
        self.move(*pos)

    def display_func(self, *x) -> None:
        self.setText(self.text_func(*x))
        self.adjustSize()
        self.set_position()
        if self.animated:
            self.show()
            self.anim.stop()
            self.anim.start()

    def display_str(self) -> None:
        self.setText(self.text_func)
        self.adjustSize()
        self.set_position()
        if self.animated:
            self.show()
            # self.anim.updateCurrentValue(1.)
            self.anim.stop()
            self.anim.start()


class App(QWidget):
    _changed_playspeed = pyqtSignal()
    _play_now = pyqtSignal(str)
    _keyPressed = pyqtSignal(QEvent)

    def __init__(
        self,
        media_files: Iterable[str],
        delay: int = 1000,
        rate: float = 1.0,
        qsize: int = 10,
        parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self.status: bool = True
        self.delay: int = delay
        self.rate: float = rate
        self.album: AlbumReader.AlbumReader = AlbumReader.AlbumReader(*media_files)
        self.set_size()
        self.maxqsize: int = 10
        self.queue: queue.Queue[AsyncResult[dict[str, str]]] = queue.Queue(qsize)
        self._tempd = QTemporaryDir()
        self.destroyed.connect(self._tempd.remove)
        self._played: list[str] = []
        self.media_thread: ReadMediaThread = ReadMediaThread(
            self.queue,
            iter(self.album),
            self.delay,
            self.media_size,
            self._tempd.path(),
            parent=self,
            log_level=utils.LOG_LEVEL
        )
        self.destroyed.connect(self.media_thread.terminate)

        self._keyPressed.connect(self.on_key)

        # layout
        self.Layout = QHBoxLayout(self)
        self.Layout.setSizeConstraint(self.Layout.SizeConstraint.SetNoConstraint)
        self.Layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.Layout)

        # video
        self.MediaPlayer = VLCMediaPlayer(self)

        self.VideoWidget = QFrame(self)

        self.Layout.addWidget(self.VideoWidget)
        self.MediaPlayer.setVideoOutput(self.VideoWidget)
        self.MediaPlayer.setPlaybackRate(self.rate)
        self.MediaPlayer.mediaStatusChanged.connect(self.end_of_media)

        # info
        self.info_fps = StdInfo(
            lambda: f'fps: {1000. / self.delay * self.rate:.1f}',
            align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            parent=self
        )
        self._changed_playspeed.connect(self.info_fps.display)

        self.info_playspeed: StdInfo = StdInfo(
            lambda: f'x {self.rate:.0%}',
            align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            parent=self)
        self._changed_playspeed.connect(self.info_playspeed.display)
        self._changed_playspeed.emit()

        self.debug_info: StdInfo = StdInfo(
            lambda x: f'path: {Path(x).relative_to(".") if Path(x).is_relative_to(".") else x:s}',
            align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            animated=False,
            font=QFont('Arial', 24),
            parent=self)
        
        self.end_info: Optional[StdInfo] = None

        self._play_now[str].connect(self.debug_info.display)
        self._play_now[str].connect(self.clean_up)
        self.debug_info.hide()
        self.setVisible(True)
        for meta in self.album.preprocess_meta():
            print(f'App.{meta.get("command")}')
            exec(f'self.{meta.get("command")}')

    def clean_up(self, path) -> None:
        CleanUpThread.flist.append({'path': Path(path), 'size': -1})
        t = CleanUpThread(self)
        t.start()
        logger.debug(f'Cleaning up {path}')
        self.destroyed.connect(t.terminate)

    def show_slides(self) -> None:
        t = self.media_thread
        t.start()
        self.play_next(True)

    def keyPressEvent(self, e) -> None:
        super().keyPressEvent(e)
        self._keyPressed.emit(e)

    def play_pause(self, status: Optional[bool] = None) -> None:
        if status is None:
            status = not self.status
        self.status = status
        if self.status:
            if self.rate > 0:
                self.MediaPlayer.setPlaybackRate(self.rate)
        else:
            self.MediaPlayer.setPlaybackRate(0)

    def on_key(self, e) -> None:
        match e.key():
            case Qt.Key.Key_Escape:  # Esc: Exit
                self.close()
            case Qt.Key.Key_Space:  # Space: stop/start
                self.play_pause()
            case Qt.Key.Key_S:
                raise NotImplementedError
                from . import Mp4Movie
                app = Mp4Movie.App(self.image_files, self.delay, rate=self.rate, aspect='4X3')
                app.show_slides(None)
                # self.close()
            case 93:  # ]: -10%
                self.change_playspeed(self.rate + 0.1)
            case 91:  # [: +10%
                self.change_playspeed(self.rate - 0.1)
            case Qt.Key.Key_D:
                self.toggle_debug()
            case Qt.Key.Key_Return:
                if e.modifiers() == Qt.KeyboardModifier.ControlModifier:  # Ctrl + Enter: Toggle Full Screen
                    self.toggle

    def toggle(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def toggle_debug(self) -> None:
        debug_is_visible = self.debug_info.isVisible()
        self.debug_info.setVisible(not debug_is_visible)
        if self.end_info is not None:
            self.end_info.setVisible(not debug_is_visible)

    def end_of_media(self, status) -> None:
        if status == self.MediaPlayer.MediaStatus.EndOfMedia:
            self.play_next()

    def play_next(self, first: bool = False) -> None:
        if first or self.queue.unfinished_tasks > 0:
            entry: FFMPEGObject.VideoMeta = self.queue.get().get()
            self.queue.task_done()
        else:
            self.play_pause(False)
            end_info = StdInfo(
                'End',
                QFont('Arial', 108),
                align=Qt.AlignmentFlag.AlignCenter,
                parent=self
            )
            end_info.display()
            return
        content = entry['content']
        if entry['tag'] == 'meta':
            exec(f'self.{content}')
            self.play_next()
            return

        self._play_now.emit(content)
        url = QUrl.fromLocalFile(QFileInfo(content).absoluteFilePath())
        self.MediaPlayer.setSource(url)
        self.MediaPlayer.play()

    def change_playspeed(self, speed):
        if speed < 0:
            speed = 0
        self.rate = speed
        self.MediaPlayer.setPlaybackRate(self.rate)
        self._changed_playspeed.emit()

    def set_size(
        self, width: Optional[int] = None, height: Optional[int] = None, aspect: str = '16X9'
    ) -> None:
        if (width, height) == (None, None):
            match aspect:
                case '4X3':
                    width, height = 1024, 768
                case '16X9':
                    width, height = 1280, 720
                case '9X16':
                    width, height = 720, 1280
                case _:
                    width = 1280

        if None in (width, height):
            aspect = tuple(map(int, aspect.split('X', 1)))
            aspect_ratio = aspect[0] / aspect[1]
            if width is not None:
                height = width * aspect_ratio
            else:
                width = height / aspect_ratio
        self.media_size: tuple[int, int] = (width, height)

    resized = pyqtSignal()

    def resizeEvent(self, event) -> None:
        ev = super().resizeEvent(event)
        self.resized.emit()
        # logger.info('size: {}'.format(self.size()))
        # logger.info('frame size: {}'.format(self.frameSize()))
        return ev
