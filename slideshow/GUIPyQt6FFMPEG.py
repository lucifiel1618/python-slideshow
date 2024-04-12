from fractions import Fraction
from multiprocessing.pool import AsyncResult
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import multiprocessing
import queue
from typing import Iterable, Callable, Literal, NoReturn, Optional, Self, TypedDict, cast, Sequence

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


class Application:
    def __init__(self, qapp: Optional[QApplication] = None):
        if qapp is None:
            qapp = QApplication([])
        self.qapp = qapp
        self.ret_code: int = 0

    def exec(self) -> None:
        self.ret_code = self.qapp.exec()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback) -> NoReturn:
        sys.exit(self.ret_code)

    def quit(self):
        self.qapp.quit()


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
            vlc.EventType.MediaPlayerEndReached,  # type: ignore
            lambda e: self.mediaStatusChanged.emit(self.MediaStatus.EndOfMedia)
        )

    def setVideoOutput(self, video_widget) -> None:
        if sys.platform.startswith('linux'):  # for Linux using the X Server
            self._player.set_xwindow(self.video_widget.winId())  # type: ignore
        elif sys.platform == "win32":  # for Windows
            self._player.set_hwnd(self.video_widget.winId())
        elif sys.platform == "darwin":  # for MacOS
            self._player.set_nsobject(int(video_widget.winId()))

    def setSource(self, media) -> None:
        self.media = self._instance.media_new(media.url())
        self._player.set_media(self.media)
        # self._player.play()
        self._player.pause()

    def play(self) -> None:
        self._player.play()
        self._timer.start()
        self._is_playing = True

    def setPlaybackRate(self, v) -> None:
        self._player.set_rate(v)

    def update_mediaStatusChanged(self) -> None:
        is_playing = self._player.is_playing()
        if is_playing != self._is_playing:
            self._is_playing = is_playing
            self.mediaStatusChanged.emit(self.MediaStatus.OtherStatus if is_playing else self.MediaStatus.EndOfMedia)


def process_meta(command: str) -> FFMPEGObject.VideoMeta:
    return {'tag': 'meta', 'content': command}


def _ffmpeg_read(
    queue: queue.Queue[AsyncResult[FFMPEGObject.VideoMeta]],
    iterable: Iterable[ET.Element],
    delay: float,
    size: tuple[int, int],
    dpath: str,
    segment_time: int = 10,
    loglevel: Optional[FFMPEGObject.LogLevel] = None
) -> None:
    # with multiprocessing.Pool(None, limit_cpu) as pool:
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        logger.debug('Submmiting jobs to media processing queue...')
        ffmpeg_object = FFMPEGObject.FFMPEGObjectLive(delay, size, loglevel=loglevel)
        index = 0
        skipped = 0
        logger.debug('Iterating over files...')
        for media in iterable:
            logger.detail(f'Processing {media}...')
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
                logger.detail(f'Queuing `{fname}`')
                qput(pool.apply_async(ffmpeg_object.compile_call(fname)))  # type: ignore
                ffmpeg_object.reset()
                index += 1
            logger.detail(f'End processing {media}...')
        if ffmpeg_object.streams:
            if skipped > 0:
                qput = queue.put_nowait
                skipped -= 1
            else:
                qput = queue.put
            qput(pool.apply_async(ffmpeg_object.compile_call(fname)))  # type: ignore
        pool.close()
        logger.debug('Submmiting jobs to media processing queue finished. Waiting for jobs to end...')
        pool.join()
        queue.join()
        logger.debug('All media processing jobs finished.')


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
        q: queue.Queue[AsyncResult[FFMPEGObject.VideoMeta]],
        media_objects: Iterable[ET.Element],
        delay: float,
        size: tuple[int, int],
        dpath: str,
        loglevel: Optional[FFMPEGObject.LogLevel] = None,
        parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self.queue: queue.Queue[AsyncResult[FFMPEGObject.VideoMeta]] = q
        self.media_objects = media_objects
        self.delay = delay
        self.size = size
        self.dpath = dpath
        self.loglevel = loglevel

    def run(self) -> None:
        logger.debug('Initializing video thread...')
        _ffmpeg_read(self.queue, self.media_objects, self.delay, self.size, self.dpath, loglevel=self.loglevel)
        logger.debug('Terminating video thread...')


class Resizable(QWidget):
    resized = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setStyleSheet("background-color:black;")

    def resizeEvent(self, event) -> None:
        ev = super().resizeEvent(event)
        self.resized.emit()
        return ev


class StdInfo(QLabel):
    def __init__(
        self,
        text_func: Callable[[], str] | str,
        font: QFont = QFont('Arial', 36),
        align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        animated: bool = True,
        parent: Optional[Resizable] = None,
        **kwds
    ):
        super().__init__(parent, **kwds)
        self.setStyleSheet('QLabel {color: gray;}')
        self.setStyleSheet('QLabel {background-color: rgba(0, 0, 0, 50);}')
        self.setFont(font)
        self.animated = animated
        self.text_func = text_func
        self.align = align
        self.display = self.display_func if callable(text_func) else self.display_str
        effect = QGraphicsOpacityEffect(self)
        self.anim: QPropertyAnimation = QPropertyAnimation(effect, b'opacity')  # type: ignore
        self.setGraphicsEffect(effect)
        self.anim.setDuration(2000)
        self.anim.setStartValue(1.)
        self.anim.setEndValue(0.)
        # self.anim.setEasingCurve(QEasingCurve.OutQuad)
        self.anim.finished.connect(self.hide)
        self.parent().resized.connect(self.set_position)  # type: ignore

    @pyqtSlot()
    def set_position(self, pos: Optional[tuple[int, int]] = None) -> None:
        if pos is None:
            coords: tuple[int, ...] = self.parent().geometry().getRect()  # type: ignore
            if self.align & Qt.AlignmentFlag.AlignLeft:
                x = 0
            elif self.align & Qt.AlignmentFlag.AlignRight:
                x = coords[2] - self.width()
            elif self.align & Qt.AlignmentFlag.AlignHCenter:
                x = (coords[2] - self.width()) // 2
            else:
                raise ValueError('Inappropriate AlignmentFlag')

            if self.align & Qt.AlignmentFlag.AlignTop:
                y = 0
            elif self.align & Qt.AlignmentFlag.AlignBottom:
                y = coords[3] - self.height()
            elif self.align & Qt.AlignmentFlag.AlignVCenter:
                y = (coords[3] - self.height()) // 2
            else:
                raise ValueError('Inappropriate AlignmentFlag')
            pos = (x, y)
        self.move(*pos)

    def display_func(self, *x) -> None:
        text_func = cast(Callable, self.text_func)
        self.setText(text_func(*x))
        self.adjustSize()
        self.set_position()
        if self.animated:
            self.show()
            self.anim.stop()
            self.anim.start()

    def display_str(self) -> None:
        text = cast(str, self.text_func)
        self.setText(text)
        self.adjustSize()
        self.set_position()
        if self.animated:
            self.show()
            # self.anim.updateCurrentValue(1.)
            self.anim.stop()
            self.anim.start()


class App(Resizable):
    _changed_playspeed = pyqtSignal()
    _play_now = pyqtSignal(str)
    _keyPressed = pyqtSignal(QEvent)

    def __init__(
        self,
        media_files: Iterable[str],
        delay: int = 1000,
        rate: float = 1.0,
        qsize: int = 10,
        chapters: Optional[Sequence[str]] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.status: bool = True
        self.delay: int = delay
        self.rate: float = rate
        self.album: AlbumReader.AlbumReader = AlbumReader.AlbumReader(*media_files, chapters=chapters)
        self.set_size()
        self.queue: queue.Queue[AsyncResult[FFMPEGObject.VideoMeta]] = queue.Queue(qsize)
        self._tempd = QTemporaryDir()
        self.destroyed.connect(self._tempd.remove)  # type: ignore
        self._played: list[str] = []
        self.media_thread: ReadMediaThread = ReadMediaThread(
            self.queue,
            iter(self.album),
            self.delay,
            self.media_size,
            self._tempd.path(),
            parent=self,
            loglevel=utils.FFMPEG_LOGLEVEL
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
            lambda x: f'path: {Path(x).relative_to(".") if Path(x).is_relative_to(".") else x:s}',   # type: ignore
            align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            animated=False,
            font=QFont('Arial', 24),
            parent=self)

        self.loading_info: StdInfo = StdInfo(
            'Loading...',
            QFont('Arial', 108),
            align=Qt.AlignmentFlag.AlignCenter,
            animated=False,
            parent=self
        )

        self.end_info: StdInfo = StdInfo(
            'End',
            QFont('Arial', 108),
            align=Qt.AlignmentFlag.AlignCenter,
            parent=self
        )

        self._play_now[str].connect(self.debug_info.display)
        self._play_now[str].connect(self.clean_up)
        self.loading_info.display()
        self.end_info.hide()
        self.debug_info.hide()
        self.setVisible(True)
        for meta in self.album.preprocess_meta():
            cmd = meta.get('command')
            logger.info(f'Executing App.{cmd}')
            exec(f'self.{cmd}')

    def clean_up(self, path) -> None:
        CleanUpThread.flist.append({'path': Path(path), 'size': -1})
        t = CleanUpThread(self)
        t.start()
        logger.debug(f'Cleaning up {path}')
        self.destroyed.connect(t.terminate)

    def show_slides(self) -> None:
        t = self.media_thread
        logger.info('Start playing...')
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

    def end_of_media(self, status) -> None:
        if status == self.MediaPlayer.MediaStatus.EndOfMedia:
            self.play_next()

    def play_next(self, first: bool = False) -> None:
        if first:
            logger.debug('Start pulling files from the media queue...')
        logger.detail('Accessing media queue for the next file...')
        if first or self.queue.unfinished_tasks > 0:
            logger.detail('Waiting for the media processing to finish...')
            try:
                timeout = 60
                entry: FFMPEGObject.VideoMeta = self.queue.get(timeout=timeout).get(timeout=timeout)
            except queue.Empty as e:
                e.args = (f'{timeout}s Timeout. Media queue is empty when it shouldn\'t be.', )
                logger.error(e)
                raise e
            logger.detail(f'Next file found `{entry}`...')
            self.queue.task_done()
        else:
            logger.debug('Empty queue. Exiting gracefully.')
            self.play_pause(False)
            self.end_info.display()
            return
        content = entry['content']
        if entry['tag'] == 'meta':
            exec(f'self.{content}')
            self.play_next()
            return

        self._play_now.emit(content)
        url = QUrl.fromLocalFile(QFileInfo(content).absoluteFilePath())
        self.MediaPlayer.setSource(url)
        if first:
            self.loading_info.hide()
        self.MediaPlayer.play()

    def change_playspeed(self, speed) -> None:
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
            aspect_ratio = Fraction(*aspect.split('X', 1))
            if width is not None:
                height = int(width * aspect_ratio)
            else:
                width = int(cast(int, height) / aspect_ratio)
        self.media_size: tuple[int, int] = (cast(int, width), cast(int, height))

    def get_ffmpeg_loglevel(self) -> Optional[Literal[FFMPEGObject.LogLevel]]:
        return self.media_thread.loglevel

    def set_ffmpeg_loglevel(self, value: Optional[FFMPEGObject.LogLevel] = None) -> None:
        self.media_thread.loglevel = value
