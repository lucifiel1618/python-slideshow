from tkinter import Label, Entry, Button, Tk, LEFT, RIGHT, INSERT, X

import os
import shutil
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET
from typing import Iterable, Sequence

import ffmpeg

from . import AlbumReader
from . import FFMPEGObject

from typing import Optional

root = Tk()


def _ffmpeg_read(
    output: Path,
    iterable: Iterable[ET.Element],
    delay: float,
    rate: float,
    size: tuple[int, int],
    fps: Optional[int] = 30,
    parent: Optional[object] = None,
    max_streams: int = 50, #100,  # 200,
    dryrun: bool = False
) -> None:
    ffmpeg_object = FFMPEGObject.FFMPEGObjectOutput(delay, size, fps, rate)
    ffmpeg_object.dryrun = dryrun
    with tempfile.TemporaryDirectory() as tempd:
        tempd_path = Path(tempd)
        temp_output = tempd_path / output.name
        subfiles = []

        for media in iterable:
            if media.tag == 'meta':
                if parent is not None:
                    exec(f'parent.{media.get("command")}', {'parent': parent})
                continue
            ffmpeg_object.add_stream(media)
            if len(ffmpeg_object.streams) >= max_streams:
                subfile = temp_output.with_suffix(f'.part{len(subfiles)+1}{temp_output.suffix}')
                ffmpeg_object.run(subfile)
                subfiles.append(str(subfile))
                ffmpeg_object.reset()

        if ffmpeg_object.streams:
            if subfiles:
                subfile = temp_output.with_suffix(f'.part{len(subfiles)+1}{temp_output.suffix}')
                ffmpeg_object.run(subfile)
                subfiles.append(str(subfile))
            else:
                ffmpeg_object.run(output, temp_output)
                return

        subfiles_list = tempd_path / 'inputs.txt'
        subfiles_list.write_text('\n'.join(f"file '{f}'" for f in subfiles))
        outstream = (
            ffmpeg.input(str(subfiles_list), format='concat', safe=0)
                  .output(str(temp_output), codec='copy')
        )
        if dryrun:
            ffmpeg_object.print_stream_cmd(outstream)
            return
        outstream.run()
        shutil.move(temp_output, output)


class App(object):
    def __init__(
        self,
        image_files: Sequence[str],
        delay: int,
        rate: float = 1.,
        width: Optional[int] = None,
        height: Optional[int] = None,
        aspect: str = '16X9',
        chapters: Optional[Sequence[str]] = None,
        qsize: None = None  # Preserved argument
    ):
        self.set_size(width, height, aspect)
        # _gcd = gcd(width, height)
        self.album: AlbumReader.AlbumReader = AlbumReader.AlbumReader(*image_files, repeat=False, chapters=chapters)
        self.rate: float = rate
        self.delay: int = delay
        self.fps: int = 30  # None
        self.dryrun = False
        for meta in self.album.preprocess_meta():
            print(f'App.{meta.get("command")}')
            exec(f'self.{meta.get("command")}')

    def set_size(self, width: Optional[int] = None, height: Optional[int] = None, aspect: str = '16X9'):
        if (width, height) == (None, None):
            match aspect:
                case '4X3':
                    width, height = 1024, 768
                case '3X4':
                    width, height = 768, 1024
                case '16X9':
                    width, height = 1280, 720
                case '9X16':
                    width, height = 720, 1280
                case '16X10':
                    width, height = 1152, 720
                case '10X16':
                    width, height = 720, 1152
                case _:
                    width = 1280

        if width is not None and height is not None:
            pass
        else:
            aspect = tuple(int(l) for l in aspect.split('X', 1))
            aspect_ratio = aspect[0] / aspect[1]
            if width is not None:
                height = width * aspect_ratio
            else:
                width = height / aspect_ratio
        self.size: tuple[int, int] = (width, height)

    def show_slides(self, output: Optional[Path], dryrun: Optional[bool] = None):
        if output is None:
            self.inputbox()
            output = Path(self._output)
        if dryrun is None:
            dryrun = self.dryrun
        _ffmpeg_read(output, iter(self.album), self.delay, self.rate, self.size, self.fps, dryrun=dryrun)

    def change_playspeed(self, rate: float):
        self.rate = rate

    def showFullScreen(self):
        pass

    def inputbox(self):
        text = Label(root)
        text['text'] = 'Output:'
        text.pack(side=LEFT)
        field = Entry(root)
        field.pack(side=LEFT, fill=X, expand=1)
        execute = Button(root)
        execute['text'] = 'Execute'
        execute.pack(side=RIGHT)
        field.insert(INSERT, os.getcwd() + os.sep)

        def command():
            self._output = field.get()
            root.destroy()
        execute['command'] = command

        root.mainloop()
