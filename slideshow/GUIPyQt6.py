import sys
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QStackedWidget, QLabel, QGraphicsOpacityEffect, QWidget, QHBoxLayout, QFrame
from PyQt6.QtCore import QTimer, Qt, QUrl, QThread, QFileInfo, pyqtSignal, pyqtSlot, QPropertyAnimation, QSize, QObject, QEvent
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtGui import QPixmap, QMovie, QFont
from PyQt6.QtMultimediaWidgets import QVideoWidget


from PIL import Image
from . import AlbumReader
from .utils import get_logger


logger = get_logger('slideshow.UI.PyQt6')

try:
    # Python2
    from Queue import PriorityQueue, Empty
except ImportError:
    # Python3
    from queue import PriorityQueue, Empty

try:
    import vlc
    __VLC__ = True
except ImportError:
    __VLC__ = False

class QMovieT(QMovie):
    '''
    NOT YET COMPLETELY IMPLEMENTED
    '''
    mediaStatusChanged = pyqtSignal(bool)

    def __init__(self, *args, **kwds):
        self.repeat = kwds.pop('repeat', 1)
        super(self.__class__, self).__init__(*args, **kwds)
        kwds['repeat'] = self.repeat
        self.times = 0

    def gif_frame_changed(self, frame):
        if (frame + 1) >= self.frameCount():
            self.times += 1
            if self.times >= self.repeat:
                self.stop()
                self.mediaStatusChanged.emit(False)


class VLCMediaPlayer(QObject):
    mediaStatusChanged = pyqtSignal(bool)
    MediaStatus = SimpleNamespace()
    MediaStatus.EndOfMedia = 6
    MediaStatus.OtherStatus = 0

    def __init__(self, parent = None):
        super().__init__(parent)
        self._instance = vlc.Instance()
        self._player = self._instance.media_player_new()
        self.parent = parent
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._is_playing = False
        self._timer.timeout.connect(self.update_mediaStatusChanged)

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
    # def __getattr__(self, attr):
    #     if not hasattr(self, attr):
    #         print (attr)
    #     return lambda x: None

    def play(self):
        self._player.play()
        self._timer.start()
        self.MediaStatus.EndOfMedia = False
        self._is_playing = True

    def setPlaybackRate(self, v):
        self._player.set_rate(v)

    def update_mediaStatusChanged(self):
        is_playing = self._player.is_playing()
        if is_playing != self._is_playing:
            self._is_playing = is_playing
            self.mediaStatusChanged.emit(self.MediaStatus.OtherStatus if is_playing else self.MediaStatus.EndOfMedia)


class StdInfo(QLabel):
    def __init__(self, text_func, align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, animated = True, parent = None, font = None, **kwds):
        QLabel.__init__(self, parent, **kwds)
        if font is None:
            font = QFont('Arial', 36)
        elif not isinstance(font, QFont):
            font = QFont(*font)
        self.setStyleSheet("QLabel {color: gray;}")
        self.setStyleSheet("QLabel {background-color: rgba(0, 0, 0, 50);}");
        self.setFont(font)
        self.animated = animated
        self.text_func = text_func
        self.align = align
        if callable(text_func):
            self.display = self.display_func
        else:
            self.setText(self.text_func)
            self.adjustSize()
            self.display = self.display_str
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        self.anim = QPropertyAnimation(effect, b"opacity")
        self.anim.setDuration(2000)
        self.anim.setStartValue(1.)
        self.anim.setEndValue(0.)
        # self.anim.setEasingCurve(QEasingCurve.OutQuad)
        self.anim.finished.connect(self.hide)
        self.parent().resized.connect(self.set_position)
    @pyqtSlot()
    def set_position(self, pos = None):
        if pos == None:
            coords = self.parent().geometry().getRect()
            if self.align & Qt.AlignmentFlag.AlignLeft:
                x = 0
            if self.align & Qt.AlignmentFlag.AlignRight:
                x = coords[2] - self.width()
            if self.align & Qt.AlignmentFlag.AlignTop:
                y = 0
            if self.align & Qt.AlignmentFlag.AlignBottom:
                y = coords[3] - self.height()
            pos = (x, y)
        self.move(*pos)

    def display_func(self, *x):
        self.setText(self.text_func(*x))
        self.adjustSize()
        self.set_position()
        if self.animated:
            self.show()
            self.anim.stop()
            self.anim.start()

    def display_str(self):
        if self.animated:
            self.show()
            # self.anim.updateCurrentValue(1.)
            self.anim.stop()
            self.anim.start()


class ReadMediaThread(QThread):
    def __init__(self, fname, queue, priority, size = QSize(800, 600), parent = None, repeat = 1):
        QThread.__init__(self, parent = parent)
        self.fname = fname
        self.queue = queue
        self.priority = priority
        self.size = size
        self.parent = parent
        self.repeat = repeat if repeat is not None else 1
        self.finished.connect(self.deleteLater)
    # def __del__(self):
    #     self.quit()
    #     self.wait()
    @staticmethod
    def readMedia(fname, size = 0, parent = None, repeat = 1):
        mime = AlbumReader.AlbumReader._get_media_type(fname)
        if mime == 'video':
            media = QUrl.fromLocalFile(QFileInfo(fname).absoluteFilePath())
        elif mime == 'image':
            media = QPixmap(fname).scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        elif mime == 'animation':
            _media = Image.open(fname)
            media = QMovieT(fname, parent = parent, repeat = repeat)#, parent = parent())
            media.setScaledSize(QSize(*_media.size))
            media.setCacheMode(QMovie.CacheMode.CacheAll)
        return {'path': fname, 'object': media}
    def run(self):
        media = self.readMedia(self.fname, self.size, self.parent, self.repeat)
        self.queue.put((self.priority, media))

class App(QWidget):
    _changed_playseed = pyqtSignal()
    _play_now = pyqtSignal(str)
    _keyPressed = pyqtSignal(QEvent)
    def __init__(self, image_files, delay, rate = 1.0, qsize = 100, parent = None):
        super().__init__(parent)
        # self.widget = QWidget(self)
        # self.setCentralWidget(self.widget)
        self.status = True
        self.delay = delay
        self.rate = rate
        self.image_files = image_files
        self.pictures = AlbumReader.AlbumReader(*image_files)
        self.maxqsize = qsize
        self.timer = QTimer(self)
        self.timer.setInterval(self.delay)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.show_slides)
        self._timeout = self.delay / 10
        self._result = PriorityQueue(self.maxqsize)
        self.min_size = QSize(1280, 720)
        self._next = 0
        self._queued = 0
        self._gif_replay_times = [0, 1]
        self.tempd = None

        self._keyPressed.connect(self.on_key)

        # layout
        self.Layout = QHBoxLayout()
        self.Layout.setSizeConstraint(self.Layout.SizeConstraint.SetNoConstraint)
        self.Layout.setContentsMargins(0, 0, 0, 0)
        self.OutputWidget = QStackedWidget(self)
        self.Layout.addWidget(self.OutputWidget)
        self.setLayout(self.Layout)

        # slideshow
        self.SlideShowWidget = QLabel(self.OutputWidget)
        self.SlideShowWidget.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.OutputWidget.addWidget(self.SlideShowWidget)

        # video
        self.MediaPlayer = VLCMediaPlayer(self) if __VLC__ else QMediaPlayer(self)
        # self.VideoWidget = QVideoWidget(self.OutputWidget)
        self.VideoWidget = QFrame(self.OutputWidget) if __VLC__ else QVideoWidget(self.OutputWidget)
        self.MediaPlayer.setVideoOutput(self.VideoWidget)
        self.MediaPlayer.setPlaybackRate(self.rate)
        self.MediaPlayer.mediaStatusChanged.connect(self.end_of_media)
        self.OutputWidget.addWidget(self.VideoWidget)

        # audio
        self.MusicPlayer = VLCMediaPlayer(self) if __VLC__ else QMediaPlayer(self)
        self.bgm_rate = 1.
        self.threading(int(self.maxqsize/10))

        # info
        self.info_fps = StdInfo(lambda: "fps: {:.1f}".format(1000. / self.delay * self.rate),
                                align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                                parent = self)
        self._changed_playseed.connect(self.info_fps.display)

        self.info_playspeed = StdInfo(lambda: "x {:.0%}".format(self.rate),
                                align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                                parent = self)
        self._changed_playseed.connect(self.info_playspeed.display)
        self._changed_playseed.emit()

        self.debug_info = StdInfo(lambda x: "path: {}".format(Path(x).relative_to('.') if Path(x).is_relative_to('.') else x),
                                align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                                animated = False,
                                font = ('Arial', 24),
                                parent = self)
        self._play_now[str].connect(self.debug_info.display)
        self.debug_info.hide()

        self.show()
    @property
    def bgm_rate(self):
        return self._bgm_rate
    @bgm_rate.setter
    def bgm_rate(self, playspeed):
        self._bgm_rate = playspeed
        self.MusicPlayer.setPlaybackRate(self.bgm_rate)

    def threading(self, n = 1):
        i = 0
        while i < n:
            try:
                media = next(self.pictures)
            except StopIteration:
                self._queued = -1
                return 1
            if media.tag == 'meta':
                exec('self.' + media.get('command'))
            elif media.tag == 'bgm':
                self.MusicPlayer.setSource(ReadMediaThread.readMedia(media.get('path')))
                self.MusicPlayer.play()
            else:
                size = self.size()
                if size.height() < self.min_size.height():
                    size = self.min_size
                # assert media.get('path') is not None, ' '.join(('>>>', media.text))
                thread = ReadMediaThread(media.get('path'), self._result, self._queued, size, repeat = media.get('repeat'), parent = self)
                self._queued += 1
                thread.start()
                i += 1

    def show_slides(self):
        try:
            i, media = self._result.get(True, 0.5)
            path = media['path']
            img_object = media['object']
            # if isinstance(img_object, QMediaContent):
            #     self.MediaPlayer.setMedia(img_object)
        except Empty:
            self.timer.stop()
            return
        while i != self._next:
            self._result.put((i, media), False)
            i, media = self._result.get()
            path = media['path']
            img_object = media['object']
        if self.debug_info.isVisible():
            self._play_now.emit(path)
        if isinstance(img_object, QUrl):
            # If a movie
            self.OutputWidget.setCurrentWidget(self.VideoWidget)
            self.MediaPlayer.setSource(img_object)
            self.MediaPlayer.play()
        elif isinstance(img_object, QMovie):
            # If a gif
            size = img_object.scaledSize()
            img_object = QMovieT(img_object.fileName(), repeat = img_object.repeat)
            self.set_gif_replay_times(img_object.repeat)
            img_object.setCacheMode(QMovie.CacheMode.CacheAll)
            self._gif = img_object
            img_object.frameChanged.connect(self.gif_frame_changed)
            self.SlideShowWidget.setMovie(img_object)
            size.scale(self.SlideShowWidget.size(), Qt.AspectRatioMode.KeepAspectRatio)
            img_object.setScaledSize(size)
            img_object.setSpeed(int(self.rate*100*0.3))
            self.OutputWidget.setCurrentWidget(self.SlideShowWidget)
            # self.change_playspeed(self.rate)
            img_object.start()
        else:
            # If a picture
            # print(img_object.size())
            self.SlideShowWidget.setPixmap(img_object.scaled(self.SlideShowWidget.size(), Qt.AspectRatioMode.KeepAspectRatio))
            self.timer.start(int(self.delay / self.rate))
            self.OutputWidget.setCurrentWidget(self.SlideShowWidget)
        self._next += 1
        self.threading(self.maxqsize - self._result.qsize())

    def keyPressEvent(self, e):
        super().keyPressEvent(e)
        self._keyPressed.emit(e)

    def on_key(self, e):
        key = e.key()
        if key == Qt.Key.Key_Escape: # Esc: Exit
            self.close()
            return
        if key == Qt.Key.Key_Space: # Space: stop/start
            self.status = not self.status
            if self.status:
                if self.rate > 0:
                    self.timer.start()
            else:
                self.timer.stop()
            return
        if key == Qt.Key.Key_S:
            import Mp4Movie
            app = Mp4Movie.App(self.image_files, self.delay, rate = self.rate, aspect = '4X3')
            app.show_slides(None)
            # self.close()
            return
        if key == 93: # ]: -10%
            self.change_playspeed(self.rate + 0.1)
            return
        if key == 91: # [: +10%
            self.change_playspeed(self.rate - 0.1)
            return
        if key == Qt.Key.Key_D:
            self.toggle_debug()
        if e.modifiers() == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Return: # Ctrl + Enter: Toggle Full Screen
            self.toggle()

    def toggle(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def toggle_debug(self):
        self.debug_info.setVisible(not self.debug_info.isVisible())

    def end_of_media(self, status):
        if status == self.MediaPlayer.MediaStatus.EndOfMedia:
            self.timer.start(0)

    def change_playspeed(self, speed):
        if speed < 0:
            speed = 0
        self.rate = speed
        if self.rate == 0:
            self.timer.stop()
        elif self.status and not self.timer.isActive() and not hasattr(self, '_gif'):
            self.timer.start()
        if hasattr(self, '_gif'):
            self._gif.setSpeed(int(speed*100*0.3))
        self.MediaPlayer.setPlaybackRate(self.rate)
        self._changed_playseed.emit()

    def gif_frame_changed(self, frame):
        print(self._gif.frameCount())
        if (frame + 1) == self._gif.frameCount():
            self._gif_replay_times[0] += 1
            if self._gif_replay_times[0] == self._gif_replay_times[1]:
                self._gif.stop()
                self.timer.start(0)
                self._gif_replay_times[0] = 0
                del self._gif

    def set_gif_replay_times(self, n):
        self._gif_replay_times[1] = n

    def set_size(self, width = None, height = None, aspect = '4X3'):
        return

    resized = pyqtSignal()
    def resizeEvent(self, event):
        ev = super().resizeEvent(event)
        self.resized.emit()
        # logger.info('size: {}'.format(self.size()))
        # logger.info('frame size: {}'.format(self.frameSize()))
        return ev