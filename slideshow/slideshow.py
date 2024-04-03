#!/opt/homebrew/bin/python3
from __future__ import unicode_literals
from pathlib import Path
import argparse
import sys
from . import utils

PLAY = 'qt+ffmpeg'
logger = utils.get_logger('slideshow')


def main():
    parser = argparse.ArgumentParser(description='Python Slideshow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument(
        '-d', '--delay', type=int, default=1000,
        help='Delay time in milisecond. For example 1000 mean 1 fps'
    )
    parser.add_argument('-r', '--rate', type=float, default=5., help='Playback speed')
    parser.add_argument('--bgm-rate', type=float, default=2., help='BGM Playback speed')
    parser.add_argument('-s', '--qsize', type=int, default=100, help='Queue size')
    parser.add_argument('-C', '--chapter', default=[], action='append', help='Only read the selected chapters')
    parser.add_argument('--log-level', default='DEBUG', choices=['DETAIL', 'DEBUG', 'INFO', 'WARNING', 'ERROR'], help='The log level')
    parser.add_argument('--no-color-log', action='store_true', help='Whether not to use colored loggers. Python packages `coloredlogs` and `humanfriendly` required')
    parser.add_argument(
        '--ffmpeg-loglevel', nargs='?', const=True, default=False,
        help='Specify the log level for FFmpeg. If provided without an argument, defaults to True. If not provided, defaults to False.'
    )

    parent_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[parser])
    backend_parsers = parent_parser.add_subparsers(title='Backend', dest='backend', help='More help under subparsers')
    parser.add_argument('f', nargs='*', type=str, help='Input file name.')
    for backend in [
        'play',
        'qt', 'qt+vlc', 'qt+ffmpeg',
        'tk',
        'mp4',
        'stream',
        'template'
    ]:
        backend_parser = backend_parsers.add_parser(
            backend, parents=[parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        if backend == 'mp4':
            backend_parser.add_argument('--for-each', action='store_true',
                                        help='If activated, output a video file per chapter')
            backend_parser.add_argument('--for-each-output-pattern', type=str,
                                        default='{chapter_id}.mp4',
                                        help='Output file name pattern in `for each` syntax.')
            backend_parser.add_argument('-o', '--output', default=[], action='append', help='Output file names')
            backend_parser.add_argument('-a', '--aspect', default='16X9', help='Aspect ratio')
            backend_parser.add_argument('--dpi', type=int, default=300, help='DPI')
            backend_parser.add_argument('--force', '-F', action='store_true',
                                        help='Force rerun if the output file exists already.')
            backend_parser.add_argument('--m3u', action='store_true', help='Recreate .m3u play list')
            backend_parser.add_argument('--dryrun', action='store_true', help='Print out the command line to be executed')
        if backend == 'stream':
            backend_parser.add_argument('--for-each', action='store_true',
                                        help='If activated, output a video file per chapter')
            backend_parser.add_argument('--for-each-output-pattern', type=str,
                                        default='{chapter_id}.mp4',
                                        help='Output file name pattern in `for each` syntax.')
            backend_parser.add_argument('-o', '--output', default=[], action='append', help='Output file names')
            backend_parser.add_argument('-a', '--aspect', default='16X9', help='Aspect ratio')
            backend_parser.add_argument('--force', '-F', action='store_true',
                                        help='Force rerun if the output file exists already.')
        if backend == 'template':
            backend_parser.add_argument('-o', '--output', default=[], action='append', help='Output file name')
            backend_parser.add_argument('--empty', action='store_true',
                                        help='Files will not be imported if `empty` is set.')
            backend_parser.add_argument('-R', '--reg', nargs='*', default=[], help='<key>:<reg_pattern> pairs')
            backend_parser.add_argument('-S', '--sorter', nargs='?', const='SSIM', choices=['SSIM', 'pixelwise'], help='sorter')
            backend_parser.add_argument('--maker', default=None, const=True, nargs='?',
                                        help='choose a template maker for album file generation or \
                                        activate to auto-generate a template maker')
            backend_parser.add_argument('--maker-options', default=[], action='append',
                                        help='template maker options. e.g. pattern:<title>, pattern:id=:<id>.')

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ModuleNotFoundError:
        pass
    try:
        from IPython.core.ultratb import AutoFormattedTB
        sys.excepthook = AutoFormattedTB()
    except ModuleNotFoundError:
        logger.info('IPython is not installed. Colored Traceback will not be populated.')
    args = parent_parser.parse_args()
    utils.LOG_LEVEL = args.log_level
    utils.COLOR_LOG = not args.no_color_log

    if not args.f:
        args.f = ['.']

    if args.backend == 'play':
        args.backend = PLAY

    backends = args.backend.split('+', 1)

    if len(backends) == 1:
        gui_backend, player_backend = backends[0], backends[0]
    else:
        gui_backend, player_backend = backends

    if len(args.chapter):
        PER_CHAPTER = True
        # assert len(args.chapter) == len(args.output)
        chapters = args.chapter
    else:
        chapters = [None]
        PER_CHAPTER = False

    if gui_backend in ('qt', 'qt6'):
        if player_backend == 'ffmpeg':
            from . import GUIPyQt6FFMPEG as GUIPyQt
        else:
            from . import GUIPyQt6 as GUIPyQt
            GUIPyQt.__VLC__ = (player_backend == 'vlc')
        with GUIPyQt.Application() as a:
            app = GUIPyQt.App(
                args.f, args.delay, rate=args.rate, qsize=args.qsize, chapters=chapters
            )
            app.set_ffmpeg_loglevel(args.ffmpeg_loglevel)
            app.destroyed.connect(a.quit)
            app.bgm_rate = args.bgm_rate
            app.setGeometry(20, 20, 1280, 720)
            logger.info('Initializing GUI...')
            app.show()
            logger.info('Start showing slides...')
            app.show_slides()
            a.exec()
    elif gui_backend == 'stream':
        from .StreamFFMPEG import App
        assert len(args.f) == 1

        if args.for_each:
            chapters.clear()
            PER_CHAPTER = True
            from . import AlbumReader
            for chapter in AlbumReader.AlbumReader._get_items_of_tag(
                None, AlbumReader.ALBUM_FILE, root='body'
            ):
                chapter_id = chapter.get('id', None)
                if chapter_id is not None:
                    chapters.append(chapter_id)

        IN = Path(args.f[0]).resolve()
        output_formatter = {}
        if IN.is_dir():
            output_formatter['IN_DIR'] = IN
            output_formatter['IN'] = IN
            output_formatter['IN_STEM'] = IN.parts[-1]
        else:
            output_formatter['IN_DIR'] = IN.parent
            output_formatter['IN'] = IN.with_suffix('')
            output_formatter['IN_STEM'] = IN.stem
        print(f'{output_formatter=}')
        print(f'{args.output=}')

        outputs = [o if not o.is_dir() else o / (Path(o.name).with_suffix('.ts'))
                   for o in (Path(o.format(**output_formatter)).expanduser().resolve() for o in args.output or ['.'])]

        assert len(outputs) == 1
        app = App(args.f, args.delay, rate=args.rate, aspect=args.aspect.upper(), chapters=chapters)
        app.show_slides(outputs[0])
    elif gui_backend == 'mp4':
        # from Mp4Movie import App
        from .VideoFFMPEG import App
        assert len(args.f) == 1

        if args.for_each:
            chapters.clear()
            PER_CHAPTER = True
            from . import AlbumReader
            for chapter in AlbumReader.AlbumReader._get_items_of_tag(
                None, AlbumReader.ALBUM_FILE, root='body'
            ):
                chapter_id = chapter.get('id', None)
                if chapter_id is not None:
                    chapters.append(chapter_id)
                    args.output.append(args.for_each_output_pattern.format(chapter_id=chapter_id))

        outputs = [o if not o.is_dir() else o / (Path(o.name).with_suffix('.mp4'))
                   for o in (Path(o.format(IN=o)).expanduser().resolve() for o in args.output or ['.'])]
        if PER_CHAPTER:
            maxl_chapters = max(len(c) for c in chapters)
            maxl_outpus = max(len(str(o)) for o in outputs)
            en_fmt = f'{{:<{maxl_chapters}}}:{{:<{maxl_outpus}}}'
        for output, chapter in zip(outputs, chapters):
            if PER_CHAPTER:
                logger.debug(en_fmt.format(chapter, str(output)))

            if output.exists() and not args.force and not args.dryrun:
                logger.info(f'File `{output}` already exists. Skip!')
                continue
            logger.info(f'Start creating file `{output}`...')
            app = App(args.f, args.delay, rate=args.rate, aspect=args.aspect.upper(),
                      qsize=args.qsize, chapters=[chapter])
            app.dryrun = args.dryrun
            app.show_slides(output)
        if args.m3u:
            assert len(args.f) == 1
            d = Path(args.f[0]).resolve()
            assert d.is_dir()
            f = d / (d.name + '.m3u')
            logger.info(f'Creating .m3u file `{f}`')
            with f.open('w') as of:
                of.writelines([f'{output.relative_to(d)}\n' for output in outputs[:-1]] + [str(outputs[-1].relative_to(d))])
    elif gui_backend == 'template':
        from . import AlbumReader
        assert len(args.f) == 1
        d = args.f[0]
        assert Path(d).is_dir()
        reg = {}
        for rkey, rval in (r.split(':', 1) for r in args.reg):
            reg.setdefault(rkey, []).append(rval)
        AlbumReader.AlbumReader.make_template(
            d,
            reg=reg,
            empty=args.empty,
            output=args.output or None,
            sorter=args.sorter,
            maker=args.maker,
            maker_options={k: v for k, v in (opt.split(':', 1)for opt in args.maker_options)}
        )


if __name__ == '__main__':
    main()
