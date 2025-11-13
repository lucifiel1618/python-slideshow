import bisect
from fractions import Fraction
from multiprocessing.pool import AsyncResult
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
import multiprocessing
import queue
from typing import Iterable, Callable, Literal, NoReturn, Optional, Self, TypedDict, cast, Sequence

from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QLabel, QGraphicsOpacityEffect, QWidget, QHBoxLayout, QFrame
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QPropertyAnimation, QObject, QEvent, QUrl, QFileInfo,
    QTemporaryDir, QMutex, QTimer
)
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QFont

from . import AlbumReader
from . import FFMPEGObject
from . import utils
utils.initialize_vlc4()
import vlc  # noqa: E402
USE_VLC = True

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
    playTimeSignal = pyqtSignal(float)

    class MediaStatus:
        EndOfMedia = 6
        Paused = 4
        Playing = 3
        OtherStatus = 0
        MediaPlayerEndReached = 7

        @classmethod
        def get_state(cls, vlc_state):
            match vlc_state:
                case vlc.State.Ended:
                    state = cls.EndOfMedia
                case vlc.State.Paused:
                    state = cls.Paused
                case vlc.State.Playing:
                    state = cls.Playing
                case _:
                    state = cls.OtherStatus
            return state

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent=parent)
        self._instance = vlc.Instance('--verbose -1')
        self._player = self._instance.media_player_new()  # pyright: ignore[reportOptionalMemberAccess]
        self._is_playing = False
        self._show_subtitle = False
        self.events: dict[str, int] = {}
        # self._timer.timeout.connect(self.update_mediaStatusChanged)
        self.event_manager = self._player.event_manager()
        self.events['end_of_media'] = self.event_manager.event_attach(
            vlc.EventType.MediaPlayerEndReached,  # type: ignore
            lambda e: self.mediaStatusChanged.emit(self.MediaStatus.EndOfMedia)
        )
        # self.mediaStatusChanged.connect(self._showSubtitle)
        self.playTime = 0
        self._tick_interval = 50
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.playTimetick)
        self._timer.setInterval(self._tick_interval)

    def playTimetick(self):
        if self._is_playing:
            self.playTime += self._tick_interval
            self.playTimeSignal.emit(self.playTime / 1000)

    def setVideoOutput(self, video_widget) -> None:
        if sys.platform.startswith('linux'):  # for Linux using the X Server
            self._player.set_xwindow(self.video_widget.winId())  # type: ignore
        elif sys.platform == "win32":  # for Windows
            self._player.set_hwnd(self.video_widget.winId())
        elif sys.platform == "darwin":  # for MacOS
            self._player.set_nsobject(int(video_widget.winId()))

    def setSource(self, media: QUrl) -> None:
        self.media = self._instance.media_new(media.url())  # pyright: ignore[reportOptionalMemberAccess]
        self._player.set_media(self.media)
        self.pause()
        self.playTime = 0

    def play(self) -> None:
        self._player.play()
        self._timer.start()
        self._is_playing = True

    def pause(self) -> None:
        self._player.pause()
        self._timer.stop()
        self._is_playing = False

    def _showSubtitle(self, status: int):
        self.showSubtitle(visible=None)

    def showSubtitle(self, visible: Optional[bool] = None):
        if visible is None:
            visible = self._show_subtitle
        else:
            self._show_subtitle = visible
        track_num = 0 if visible else -1
        self._player.video_set_spu(track_num)

    def setPlaybackRate(self, v: float) -> None:
        if v == 0.:
            self.pause()
        else:
            self._player.set_rate(v)
            self.play()

    # def update_mediaStatusChanged(self) -> None:
    #     is_playing = self._player.is_playing()
    #     if is_playing == self._is_playing:
    #         return
    #     self._is_playing = is_playing
    #     self.mediaStatusChanged.emit(self.MediaStatus.get_state(self._player.get_state()))

    def attachTimer(self):
        def _f(ev):
            self.playTime = ev.u.new_time
        self.events['time_changed'] = self.event_manager.event_attach(vlc.EventType.MediaPlayerTimeChanged, _f)
        self._timer.start()

    def deattachTimer(self):
        if 'time_changed' in self.events:
            self.event_manager.event_detach(vlc.EventType.MediaPlayerTimeChanged)
            del self.events['time_changed']
        self._timer.stop()


def process_meta(command: str) -> FFMPEGObject.VideoMeta:
    return {'tag': 'meta', 'content': command}


def _ffmpeg_read(
    queue: queue.Queue[AsyncResult[FFMPEGObject.VideoMeta]],
    iterable: Iterable[ET.Element],
    delay: float,
    size: tuple[int, int],
    dpath: str,
    segment_time: int = 10,
    loglevel: Optional[FFMPEGObject.LogLevel | bool] = None
) -> None:
    # with multiprocessing.Pool(None, limit_cpu) as pool:
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        logger.debug('Submmiting jobs to media processing queue...')
        if isinstance(loglevel, str):
            loglevel = loglevel.lower()  # pyright: ignore[reportAssignmentType]
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
                qput(
                    pool.apply_async(
                        ffmpeg_object.compile_call(
                            fname,  # pyright: ignore[reportArgumentType]
                            meta={'timeline': list(ffmpeg_object.subtitler.timeline())}
                        )
                    )
                )  # type: ignore
                ffmpeg_object.reset()
                index += 1
            logger.detail(f'End processing {media}...')
        if ffmpeg_object.streams:
            if skipped > 0:
                qput = queue.put_nowait
                skipped -= 1
            else:
                qput = queue.put
                fname = f'{dpath}/{index:0>4}.ts'
            qput(
                pool.apply_async(
                    ffmpeg_object.compile_call(
                        fname,  # pyright: ignore[reportArgumentType]
                        meta={'timeline': list(ffmpeg_object.subtitler.timeline())}
                    )
                )
            )  # type: ignore
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
        loglevel: Optional[FFMPEGObject.LogLevel | bool] = None,
        parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self.queue: queue.Queue[AsyncResult[FFMPEGObject.VideoMeta]] = q
        self.media_objects = media_objects
        self.delay = delay
        self.size = size
        self.dpath = dpath
        self.loglevel: Optional[FFMPEGObject.LogLevel | bool] = loglevel

    def run(self) -> None:
        logger.debug('Initializing video thread...')
        _ffmpeg_read(self.queue, self.media_objects, self.delay, self.size, self.dpath, loglevel=self.loglevel)
        logger.debug('Terminating video thread...')


class Resizable(QWidget):
    resized = pyqtSignal(tuple)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setStyleSheet("background-color:black;")

    def resizeEvent(self, event) -> None:
        ev = super().resizeEvent(event)
        coords = self.geometry().getRect()
        self.resized.emit(coords)
        return ev


class StdInfo(QLabel):
    def __init__(
        self,
        text_func: Callable[[*tuple[str, ...]], str] | Callable[[], str] | str,
        font: QFont = QFont('Arial', 36),
        align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        animated: bool = True,
        parent: Optional[Resizable] = None,
        connect_resized: bool = True,
        **kwds
    ):
        super().__init__(parent=parent, **kwds)
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
        if connect_resized and isinstance(parent, Resizable):
            parent.resized.connect(self.set_position)  # type: ignore

    @pyqtSlot(tuple)
    def set_position(
        self,
        coords: tuple[int, int, int, int],
        pos: Optional[tuple[int, int]] = None
    ) -> None:
        if pos is None:
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

    def _ensure_main_thread(self, func, *args):
        if QThread.currentThread() != self.thread():
            QTimer.singleShot(0, lambda: func(*args))
            return False
        return True

    def display_func(self, *x: str) -> None:
        if not self._ensure_main_thread(self.display_func, *x):
            return
        text_func = cast(Callable[[*tuple[str, ...]], str], self.text_func)
        text = text_func(*x)
        self._display_text(text)

    def display_str(self) -> None:
        text = cast(str, self.text_func)
        self._display_text(text)

    def _display_text(self, text: str) -> None:
        self.setText(text)
        self.adjustSize()
        if self.animated:
            self.show()
            self.anim.stop()
            self.anim.start()


class App(Resizable):
    _changed_playspeed = pyqtSignal()
    _play_now_signal = pyqtSignal(str)
    _play_now_source_signal = pyqtSignal(str)
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
        self.play_now = ['', '']
        self.set_size()

        # layout
        self.Layout = QHBoxLayout(self)
        self.Layout.setSizeConstraint(self.Layout.SizeConstraint.SetNoConstraint)
        self.Layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.Layout)

        # video
        if USE_VLC:
            self.MediaPlayer = VLCMediaPlayer(self)
            self.VideoWidget = QFrame(self)
            self.MediaPlayer.playTimeSignal.connect(self.play_now_source_by_time)
        else:
            self.MediaPlayer = QMediaPlayer(self)
            self.VideoWidget = QVideoWidget(self)

        self.Layout.addWidget(self.VideoWidget)
        self.MediaPlayer.setVideoOutput(self.VideoWidget)
        self.MediaPlayer.setPlaybackRate(self.rate)
        self.MediaPlayer.mediaStatusChanged.connect(self.end_of_media)

        self.loading_info: StdInfo = StdInfo(
            'Loading...',
            QFont('Arial', 108),
            align=Qt.AlignmentFlag.AlignCenter,
            animated=False,
            parent=self
        )
        self.setVisible(True)

        self.loading_info.display()

        self.queue: queue.Queue[AsyncResult[FFMPEGObject.VideoMeta]] = queue.Queue(qsize)
        self._tempd = QTemporaryDir()
        # self.destroyed.connect(self._tempd.remove)  # type: ignore
        self.album: AlbumReader.AlbumReader = AlbumReader.AlbumReader(*media_files, chapters=chapters)
        self.media_thread: ReadMediaThread = ReadMediaThread(
            self.queue,
            iter(self.album),
            self.delay,
            self.media_size,
            self._tempd.path(),
            parent=self,
            loglevel=utils.FFMPEG_LOGLEVEL
        )
        # self.destroyed.connect(self.media_thread.terminate)

        self._keyPressed.connect(self.on_key)

        self.timeline: list[tuple[float, str]] = []

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
            self.debug_info_text_func,
            align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            animated=False,
            font=QFont('Arial', 24),
            parent=self,
            connect_resized=False
        )

        self.debug_info.setTextFormat(Qt.TextFormat.RichText)
        self.debug_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.debug_info.setOpenExternalLinks(False)

        def reveal_localurl_in_file_manager(url: str):
            return utils.reveal_in_file_manager(Path(QUrl(url).toLocalFile()))
        self.debug_info.linkActivated.connect(
            reveal_localurl_in_file_manager
        )

        self.end_info: StdInfo = StdInfo(
            'End',
            QFont('Arial', 108),
            align=Qt.AlignmentFlag.AlignCenter,
            parent=self
        )

        self._play_now_signal[str].connect(lambda s: self.play_now.__setitem__(0, s))
        self._play_now_signal[str].connect(self.clean_up)
        self._play_now_source_signal[str].connect(lambda s: self.play_now.__setitem__(1, s))
        self.end_info.hide()
        self.debug_info.hide()

        for meta in self.album.preprocess_meta():
            cmd = meta.get('command')
            logger.info(f'Executing App.{cmd}')
            exec(f'self.{cmd}')

    def clean_up(self, path) -> None:
        CleanUpThread.flist.append({'path': Path(path), 'size': -1})
        t = CleanUpThread(self)
        t.start()
        logger.debug(f'Cleaning up {path}')
        # self.destroyed.connect(t.terminate)

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
                # Ctrl + Enter: Toggle Full Screen
                if e.modifiers() == Qt.KeyboardModifier.ControlModifier:
                    self.toggle()

    def toggle(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def play_now_source_by_time(self, time: float):
        i = bisect.bisect_right(self.timeline, time, key=lambda en: en[0]) - 1
        source = self.timeline[i][1]
        if self.play_now[1] != source:
            self._play_now_source_signal.emit(source)
            self.play_now[1] = source
            if self.debug_info.isVisible():
                self.debug_info.display(*self.play_now)

    def toggle_debug(self) -> None:
        debug_is_visible = not self.debug_info.isVisible()
        self.debug_info.setVisible(debug_is_visible)
        if isinstance(self.MediaPlayer, VLCMediaPlayer):
            if debug_is_visible:
                self.MediaPlayer.attachTimer()
            else:
                self.MediaPlayer.deattachTimer()
        # self.MediaPlayer.showSubtitle(debug_is_visible)  # pyright: ignore[reportAttributeAccessIssue]

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
                logger.detail(f'Next file found `{entry}`...')
            except queue.Empty as e:
                e.args = (f'{timeout}s Timeout. Media queue is empty when it shouldn\'t be.', )
                logger.error(e)
                raise e
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

        self._play_now_signal.emit(content)
        self.debug_info.display(*self.play_now)
        url = QUrl.fromLocalFile(QFileInfo(content).absoluteFilePath())
        self.MediaPlayer.setSource(url)
        self.timeline = cast(dict[str, list[tuple[float, str]]], entry.get('meta', {'timeline': []}))['timeline']
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

    def get_ffmpeg_loglevel(self) -> Optional[Literal[FFMPEGObject.LogLevel] | bool]:
        return self.media_thread.loglevel

    def set_ffmpeg_loglevel(self, value: Optional[FFMPEGObject.LogLevel | bool] = None) -> None:
        self.media_thread.loglevel = value

    @staticmethod
    def debug_info_text_func(fpath: str, sourcepath: str, *args: str) -> str:
        lines = []
        fp = Path(fpath)
        if fp.is_relative_to('.'):
            fp = fp.relative_to('.')
        furl = QUrl.fromLocalFile(str(fp.absolute())).toString()
        lines.append(f'path: <a href="{furl}">{fp}</a>')
        if sourcepath:
            sp = Path(sourcepath)
            if sp.is_relative_to("."):
                sp = sp.relative_to(".")
            surl = QUrl.fromLocalFile(str(sp.absolute())).toString()
            lines.append(f'source: <a href="{surl}">{sp}</a>')
        lines.extend(args)
        l = '<br>'.join(lines)
        return l
