import copy
import os
from xml.etree import ElementTree as ET
from itertools import cycle, chain, product
from pathlib import Path
from zipfile import ZipFile
import tempfile
import mimetypes
import unicodedata
from typing import Callable, Iterable, Iterator, Literal, Sequence, Optional, overload
from . import Config
from .utils import get_logger, AspectEstimator

logger = get_logger('slideshow.AlbumReader')
# logger.setLevel(40)

mimetypes.add_type('text/album', '.album')
mimetypes.add_type('video/ogg', '.ogv')  # platform dependent, not directly available on linux
mimetypes.add_type('video/x-ms-wmv', '.wmv')  # platform dependent, not directly available on linux
mimetypes.add_type('image/apng', '.apng')

ALBUM_FILE = 'pictures.album'
CONFIG_FILE = 'pictures.yml'


class FileReader:
    def __init__(self, archive: Optional[Path | str] = None):
        self.archive = archive
        if archive is not None:
            self._tempdir = tempfile.TemporaryDirectory()
            self._archiveobj = ZipFile(self.archive, 'r')
            self.read = self._read_from_archive
        else:
            self._tempdir = None
            self._archiveobj = None
            self.read = self._read_from_flist

    def _read_from_archive(self, f) -> str:
        return self._archiveobj.extract(
            unicodedata.normalize('NFC', str(Path(f).relative_to('.'))), self._tempdir.name
        )

    def _read_from_flist(self, f: str) -> str:
        return f

    def close(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
        if self._archiveobj is not None:
            self._archiveobj.close()


class AlbumReader:
    def __init__(
        self,
        *iterable: str,
        repeat: bool = True,
        chapters: Optional[Sequence[str]] = None
    ):
        self._iterator: Iterator[ET.Element] = self._generator(iterable, repeat=repeat, chapters=chapters)
        self.repeat_video: int = 3

    def __iter__(self) -> Iterator[ET.Element]:
        return self._iterator

    def preprocess_meta(self) -> list[ET.Element]:
        results: list[ET.Element] = []
        try:
            for e in self._iterator:
                if e.tag != 'meta':
                    break
                results.append(e)
            self._iterator = chain.from_iterable([(e,), self._iterator])
        except StopIteration:
            pass
        return results

    @staticmethod
    def _get_media_type(url: str) -> str:
        mimetype_submimetype = mimetypes.guess_type(url)[0]
        if mimetype_submimetype is not None:
            mimetype, submimetype = mimetype_submimetype.split('/')
            if submimetype in ('gif', 'apng'):
                return 'animation'
            elif (mimetype, submimetype) == ('text', 'album'):
                return 'album'
            else:
                return mimetype
        return 'UNKNOWN'

    @overload
    @staticmethod
    def _read_image(
        arg: str,
        ret_fname: Literal[False] = False,
        tag: str = 'item',
        media_type: str = 'image',
        **stat: str
    ) -> ET.Element:
        ...

    @overload
    @staticmethod
    def _read_image(
        arg: str,
        ret_fname: Literal[True] = True,
        tag: str = 'item',
        media_type: str = 'image',
        **stat: str
    ) -> str:
        ...

    @staticmethod
    def _read_image(
        arg: str,
        ret_fname: bool = False,
        tag: str = 'item',
        media_type: str = 'image',
        **stat: str
    ) -> ET.Element | str:
        logger.detail(f'Reading media file: `{arg}`')
        if ret_fname:
            return arg
        return ET.Element(tag, path=arg, media_type=media_type, attrib=stat)

    @overload
    @staticmethod
    def _read_dir(
        arg: str,
        skip_albumfile: bool = False,
        ret_fname: Literal[False] = False,
        softlink_albumfile: bool = False,
        chapters: Optional[Sequence[str]] = None
    ) -> Iterator[ET.Element]:
        ...

    @overload
    @staticmethod
    def _read_dir(
        arg: str,
        skip_albumfile: bool = False,
        ret_fname: Literal[True] = True,
        softlink_albumfile: bool = False,
        chapters: Optional[Sequence[str]] = None
    ) -> Iterator[str]:
        ...

    @staticmethod
    def _read_dir(
        arg: str,
        skip_albumfile: bool = False,
        ret_fname: bool = False,
        softlink_albumfile: bool = False,
        chapters: Optional[Sequence[str]] = None
    ) -> Iterator[ET.Element | str]:
        logger.info('Start loading files in directory...')
        for root, _, files in os.walk(arg):
            index = os.path.join(root, 'pictures.album')
            album_exist = os.path.exists(index)
            if album_exist and not skip_albumfile:
                logger.info('album file exists in directory. Reading from album...')
                yield from AlbumReader._read_xml(index, chapters=chapters)
                return
            elif album_exist and ret_fname and softlink_albumfile:
                yield root
                return
            else:
                files.sort()
                # subdirs.sort()
                # for d in subdirs:
                #     for item in AlbumReader._read_dir(d, skip_albumfile = skip_albumfile, ret_fname = ret_fname):
                #         yield item
                for f in (os.path.join(root, f) for f in files):
                    if (media_type := AlbumReader._get_media_type(f)) in ('image', 'animation', 'video'):
                        yield AlbumReader._read_image(f, ret_fname=ret_fname, media_type=media_type)
        logger.info('End loading files in directory...')

    @staticmethod
    def _read_xml(
        arg: str, repeat: bool = False, chapters: Optional[Sequence[str]] = None
    ) -> Iterator[ET.Element]:
        logger.info('Start loading files in album...')
        directory = os.path.dirname(arg)
        t = ET.parse(arg)
        album = t.getroot()
        head = album.find('head')
        body = album.find('body')
        for part in chain((head,), cycle((body,)) if repeat else (body,)):
            for item in part:
                tag = item.tag
                logger.debug(f'Reading <Element {tag}>...')
                if tag == 'meta':
                    yield item
                elif tag in ('chapter', 'bgm'):
                    id = item.get('id')
                    logger.info(f'Start reading {tag} ["{id or ""}"]...')
                    if tag == 'chapter':
                        # print(chapters)
                        if (chapters != [None] and chapters is not None) and id not in chapters:
                            logger.debug(f'Chapter("{id}") not in {chapters}! SKIP!')
                            continue
                    source = item.get('source', False)
                    for l in AlbumReader._get_tag(
                        id,
                        source or album,
                        directory=directory if not source else '',
                        default=item,
                        tag=tag
                    ):
                        if tag == 'bgm':
                            l.tag = 'bgm'
                        yield l
                    logger.info('End reading {} ["{}"]...'.format(tag, id if id else ''))
        logger.info('End loading files in album...')

    @staticmethod
    def _get_album(album: ET.Element | str, directory: str = '') -> ET.Element:
        album, _ = AlbumReader._get_album_and_directory(album, directory)
        return album

    @staticmethod
    def _get_album_and_directory(
        album: ET.Element | str | Path,
        directory: str = ''
    ) -> tuple[ET.Element, str]:
        if isinstance(album, (str, Path)):
            if os.path.isdir(album):
                directory = album
                albumfile = os.path.join(directory, ALBUM_FILE)
            else:
                directory = directory or os.path.dirname(album)
                albumfile = album
            t = ET.parse(albumfile)
            album = t.getroot()
        return album, directory

    @staticmethod
    def _get_items_of_tag(
        id: Optional[str],
        album: ET.Element,
        directory: str = '',
        tag: str = 'chapter',
        root: str = '.'
    ) -> list[ET.Element] | ET.Element | None:
        '''
        If id is None, return all elements with the tag.
        Otherwise, return the element with the id along with the tag.
        '''
        album = AlbumReader._get_album(album, directory)
        if id is not None:
            pat = f'{root}/{tag}[@id="{id}"]'
            chapters = album.find(pat)
            assert chapters is not None
        else:
            pat = f'{root}/{tag}'
            chapters = album.findall(pat)
        return chapters

    @staticmethod
    def _get_tag(
        id: str,
        album: str | ET.Element,
        directory: str = '',
        default: str = '',
        tag: str = 'chapter'
    ) -> Iterable[ET.Element]:
        album, directory = AlbumReader._get_album_and_directory(album, directory)
        chapter = AlbumReader._get_items_of_tag(id, album, directory, tag) if id else default
        yield from AlbumReader._read_chapter(chapter, directory, album)

    @staticmethod
    def _read_chapter(
        chapter: ET.Element, directory: str = '', album: Optional[ET.Element] = None
    ) -> Iterator[ET.Element]:
        for el in chapter:
            if el.tag == 'f':
                logger.info('Reading files from a file block inside ["{}"]'.format(chapter.get('id', '')))
                yield from AlbumReader._read_text(el.text, directory, archive=el.get('archive', None))
            elif el.tag == 'chapter':
                logger.info('Reading files from another predefined chapter')
                yield from AlbumReader._get_tag(el.get('id'), el.get('source', album), directory)
            elif el.tag == 'overlay':
                logger.detail('Reading an overlay object ["{}"]'.format(chapter.get('id', '')))
                yield from AlbumReader._read_overlay(el, directory, album)
            else:
                yield el

    @staticmethod
    def _read_overlay(
        el: ET.Element, directory: str = '', album: Optional[ET.Element] = None
    ) -> Iterator[ET.Element]:
        last_layers = tuple(AlbumReader._read_text(el.text, directory, archive=el.get('archive', None)))
        layers_stack = product(*(
            AlbumReader._read_text(el_sub.text, directory, archive=el.get('archive', None), attribs=el_sub.attrib)
            for el_sub in el if el_sub.tag == 'layer'
        ))
        # print(list((_el_sub, _el_sub.tag, _el_sub.text) for _el_sub in el if _el_sub.tag == 'layer'))
        try:
            layers = next(layers_stack)
            tag = el.tag
            attribs = el.attrib
            for layers in chain((layers,), layers_stack):
                _el = ET.Element(tag, attrib=attribs)
                _el.extend(layers)
                _el.extend(last_layers)
                # print(*((el_sub, el_sub.tag, el_sub.get('path', None)) for el_sub in _el))
                yield _el
        except StopIteration:
            el.extend(last_layers)
            yield el

    @staticmethod
    def _read_text(
        arg: str, directory: str, archive: Optional[Path | str] = None, attribs: dict | None = None
    ) -> Iterator[ET.Element]:
        file_reader = FileReader(archive)
        for l in map(str.strip, arg.splitlines()):
            if l:
                if not l.startswith('#'):
                    l = l.split('#', 1)[0].strip()
                    if not os.path.isabs(l):
                        assert directory != ''
                        l = os.path.join(directory, l)
                    if os.path.isdir(l):
                        l = file_reader.read(l)
                        yield from AlbumReader._read_dir(l, ret_fname=False)
                    else:
                        stat: dict[str, str | None] = {} if attribs is None else attribs.copy()
                        if l.endswith(']'):
                            path, _stat = l.rsplit(' [', 1)
                            stat.update(eval(f'dict({_stat[:-1]})'))
                        else:
                            path = l
                        media_type = AlbumReader._get_media_type(path)
                        path = file_reader.read(path)
                        yield AlbumReader._read_image(path, media_type=media_type, **stat)
        file_reader.close()

    @staticmethod
    def _generator(
        iterable: Iterable[str], repeat: bool = True, chapters: Optional[Sequence[str]] = None
    ) -> Iterator[ET.Element]:
        # print(chapters)
        for arg in iterable:
            mime = AlbumReader._get_media_type(arg)
            if mime in ('animation', 'video', 'image'):
                yield AlbumReader._read_image(arg)
            elif mime == 'album':
                yield from AlbumReader._read_xml(arg, repeat=repeat, chapters=chapters)
            elif os.path.isdir(arg):
                yield from AlbumReader._read_dir(arg, chapters=chapters)
            else:
                raise IOError(f'TYPE UNKNOWN: {arg}')

    def next(self) -> ET.Element:
        return next(self._iterator)

    def __next__(self) -> ET.Element:
        return self.next()

    @staticmethod
    def make_template(
        path: Path | str,
        reg: dict[str, Sequence[str]] = {},
        empty: bool = False,
        do_open_editor: bool = True,
        output: Path | Optional[str] = None,
        repeat_video: int = 1,
        sorter: Optional[str | Callable] = None,
        maker: Optional[Path | str] = None,
        maker_options: dict = {},
        aspect: Optional[tuple[int, int]] = None
    ) -> None:
        t = """<album>
<head>
<meta command="showFullScreen()"/>
<meta command="change_playspeed(5.)"/>
<meta command="set_size(aspect='{aspect}')"/>
</head>
<body>
{body}
</body>
{chapters}
</album>"""
        ch = '''<chapter{id}>
<f>
{files}
</f>
</chapter>'''
        albumf = Path(output if output is not None else path)
        if albumf.is_dir():
            albumf /= ALBUM_FILE

        if albumf.exists():
            if do_open_editor:
                logger.info(
                    'Album file "{}" already exists. Opening Album file...'.format(albumf.relative_to('.'))
                )
                AlbumReader.open_editor(albumf)
                return
            raise OSError('Album file "{}" already exists.'.format(albumf.relative_to('.')))

        logger.info('Start building album file "{}"'.format(albumf.relative_to('.')))
        d = {'@body': []}

        config = Config.ConfigReader.from_args(reg)

        if maker is not None:
            if maker is True:
                maker = Path(albumf.parent) / CONFIG_FILE
            config = config.combined(AlbumReader._read_config(maker, **maker_options))

        for m in config.header.meta:
            exec(m)

        aspect_estimator = AspectEstimator(0.1, default_aspect=aspect)

        def callback(f: str) -> str:
            aspect_estimator.add_sample_aspect(f)
            mime = AlbumReader._get_media_type(f)
            pat = '{}'
            if mime in ('video', 'audio', 'animation'):
                if repeat_video is not None:
                    pat = f'{{}} [repeat={repeat_video}]'
            return pat.format(f)

        dataset = AlbumReader._read_dir(str(path), True, ret_fname=True, softlink_albumfile=True) if not empty else []
        d = config.process(dataset=map(os.path.relpath, dataset), callback=callback)

        wd = {
            'body': '\n'.join(
                [ch.format(id='', files='\n'.join(d.pop('@body')))] + [f'<chapter id="{dk}"/>' for dk in d]
            ),
            'chapters': '\n'.join(
                ch.format(id=f' id="{dk}"', files='\n'.join(dv)) for dk, dv in d.items()
            ),
            'aspect': aspect_estimator.get_aspect_as_str()
        }
        # logger.debug(wd)
        with albumf.open('w') as f:
            f.write(t.format(**wd))
        if do_open_editor:
            AlbumReader.open_editor(albumf)

    @staticmethod
    def _write_config(config_file: Path | str, pattern: Optional[str | Path]) -> None:
        Config.ConfigWriter.create().write(config_file, pattern)

    @staticmethod
    def _read_config(
        config_file: Path | str, pattern: Optional[str | Path] = None, do_open_editor: bool = True
    ) -> Config.ConfigReader:
        config_file = Path(config_file)
        if not config_file.exists():
            logger.info('Configuration file not exists. Creating now...')
            if pattern is None:
                d = config_file.parent.resolve()
                logger.info(f'No search pattern is given. Resolved from parent folder "{d}"')
                pattern = d
            AlbumReader._write_config(config_file, pattern)
        if do_open_editor:
            AlbumReader.open_editor(config_file.name, True)
        return Config.ConfigReader.read(config_file)

    @staticmethod
    def open_editor(path: str | Path, wait: bool = False) -> None:
        editor = os.getenv('EDITOR', '')
        if editor:
            if editor.startswith('subl'):
                os.system(f'subl{" -w" if wait else ""} "{path}"')
            else:
                os.system(f'{editor} "{path}"')
        else:
            import webbrowser
            webbrowser.open(str(path))


def iomap(
    f: list[str],
    chapters: Optional[list[str]],
    outputs: list[str] | Literal['pipe'],
    for_each: bool,
    output_pattern: Optional[str] = None
) -> dict[Path, list[str] | None]:
    if not f:
        f = ['.']
    if chapters or for_each:
        assert len(f) == 1
    IN = Path(f[0]).resolve().expanduser()
    if not outputs:
        assert output_pattern is not None
    if chapters is None:
        chapters = []
    if for_each:
        assert not chapters
        if IN.is_dir():
            album = IN / ALBUM_FILE
        else:
            assert IN.suffix == '.album'
            album = IN
        for chapter in AlbumReader._get_items_of_tag( None, album, root='body'):
            chapter_id = chapter.get('id', None)
            if chapter_id is not None:
                chapters.append(chapter_id)

    output_map: dict[Path | str, list[str] | None] = {}

    if outputs == 'pipe:':
        output_map['pipe:'] = chapters if chapters else None
    else:
        outfmt = {}
        if IN.is_dir():
            outfmt['IN_DIR'] = IN
            outfmt['IN'] = IN
            outfmt['IN_STEM'] = IN.parts[-1]
        else:
            outfmt['IN_DIR'] = IN.parent
            outfmt['IN'] = IN.with_suffix('')
            outfmt['IN_STEM'] = IN.stem
        if not outputs:
            outputs[:] = (output_pattern for _ in chapters)
        elif len(outputs) > 1:
            assert len(outputs) == len(chapters)
        if len(outputs) == 1:
            p = Path(outputs[0].format(**outfmt)).expanduser().resolve()
            output_map[p] = chapters if chapters else None
        else:
            for output, chapter in zip(outputs, chapters):
                outfmt['CHAPTER_ID'] = chapter
                p = Path(output.format(**outfmt)).expanduser().resolve()
                output_map[p] = [chapter]
    return output_map