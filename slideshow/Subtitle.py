import abc
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, NotRequired, Optional, TypedDict, override

# =====================
# ENUM CLASSES
# =====================


class ASSAlignment(Enum):
    BOTTOM_LEFT = 1
    BOTTOM_CENTER = 2
    BOTTOM_RIGHT = 3
    MIDDLE_LEFT = 4
    MIDDLE_CENTER = 5
    MIDDLE_RIGHT = 6
    TOP_LEFT = 7
    TOP_CENTER = 8
    TOP_RIGHT = 9


class ASSColor(Enum):
    WHITE = "&H00FFFFFF"
    BLACK = "&H00000000"
    RED = "&H000000FF"
    GREEN = "&H0000FF00"
    BLUE = "&H00FF0000"
    YELLOW = "&H0000FFFF"
    CYAN = "&H00FFFF00"
    MAGENTA = "&H00FF00FF"
    TRANSPARENT = "&HFF000000"  # fully transparent

# =====================
# ASSSubtitle CLASS
# =====================


class SubtitleEntry(TypedDict):
    start_time: float
    end_time: float
    text: str
    idx: int
    alignment: NotRequired[Any]
    pos: NotRequired[Any]
    primary_color: NotRequired[Any]


class Subtitle(abc.ABC):
    SUFFIX = '.nullsub'

    def __init__(
        self,
        w: int = 1920,
        h: int = 1080,
        font_style: str = "Arial",
        font_size: int = 48,
        primary_color: Any = None,
        outline_color: Any = None,
        back_color: Any = None,
        alignment: Any = None
    ):
        self.w = w
        self.h = h
        self.font_style = font_style
        self.font_size = font_size
        self.primary_color = primary_color
        self.outline_color = outline_color
        self.back_color = back_color
        self.alignment = alignment
        self.entries: list[SubtitleEntry] = []
        self.nentries = 0
        self.export_path: Optional[Path] = None

    def get_export_path(self, filepath: Optional[Path] = None):
        if filepath is None:
            if self.export_path is not None:
                filepath = self.export_path
            else:
                raise ValueError("No export path specified. Provide a filepath or set self.export_path first.")
        else:
            self.export_path = filepath
        return filepath

    def clear(self):
        """Clear all subtitle entries"""
        self.entries.clear()
        self.nentries = 0
        # if self.export_path is not None:
        #     self.export_path.unlink(missing_ok=True)

    @abc.abstractmethod
    def _export(self, filepath: Path):
        ...

    @classmethod
    @abc.abstractmethod
    def get_event(cls, entry: SubtitleEntry) -> list[str]:
        ...

    def export(self, filepath: Optional[Path] = None):
        self._export(self.get_export_path(filepath))

    def add_entry(
        self,
        start_time: float,
        end_time: float,
        text: str,
        alignment: Any = None,
        pos: Optional[tuple[int, int]] = None,
        primary_color: Any = None
    ):
        self.entries.append(
            {
                'start_time': start_time,
                'end_time': end_time,
                'text': text,
                'idx': self.nentries + 1,
                'alignment': alignment,
                'pos': pos,
                'primary_color': primary_color
            }
        )
        self.nentries += 1

    def get_event_lines(self) -> Iterable[str]:
        for e in self.entries:
            yield from self.get_event(e)

    def timeline(self) -> Iterator[tuple[float, str]]:
        for e in self.entries:
            yield e['start_time'], e['text']

    def injected[T](self, ffmpeg_stream: T) -> T:
        return ffmpeg_stream.filter("subtitles", str(self.export_path))  # pyright: ignore[reportAttributeAccessIssue]


class NullSubtitle(Subtitle):
    def __init__(
        self,
        w: int = 1920,
        h: int = 1080,
        font_style: str = "Arial",
        font_size: int = 48,
        primary_color: Any = None,
        outline_color: Any = None,
        back_color: Any = None,
        alignment: Any = None
    ):
        self.entries = []

    @override
    def export(self, filepath: Optional[Path] = None):
        ...

    def add_entry(
        self,
        start_time: float,
        end_time: float,
        text: str,
        alignment: Any = None,
        pos: Optional[tuple[int, int]] = None,
        primary_color: Any = None
    ):
        ...

    def _export(self, filepath: Path):
        ...

    @override
    def clear(self):
        ...

    @override
    @classmethod
    def get_event(cls, entry: SubtitleEntry) -> list[str]:
        return []

    @override
    def injected[T](self, ffmpeg_stream: T) -> T:
        return ffmpeg_stream


class InternalSubtitle(Subtitle):
    @classmethod
    def get_event(cls, entry: SubtitleEntry) -> list[str]:
        return [entry['text']]

    def _export(self, filepath: Path):
        ...

    @override
    def injected[T](self, ffmpeg_stream: T) -> T:
        return ffmpeg_stream


class ASSSubtitle(Subtitle):
    SUFFIX = '.ass'

    def __init__(
        self,
        w: int = 1920,
        h: int = 1080,
        font_style: str = "Arial",
        font_size: int = 48,
        primary_color: ASSColor = ASSColor.WHITE,
        outline_color: ASSColor = ASSColor.BLACK,
        back_color: ASSColor = ASSColor.TRANSPARENT,
        alignment: ASSAlignment = ASSAlignment.BOTTOM_CENTER
    ):
        super().__init__(w, h, font_style, font_size, primary_color, outline_color, back_color, alignment)

    @staticmethod
    def format_ass_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds - int(seconds)) * 100)
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

    @classmethod
    def get_event(cls, entry: SubtitleEntry) -> list[str]:
        text = entry['text']
        override = ""
        if pos := entry.get('pos', None):
            override += f"\\pos({int(pos[0])},{int(pos[1])})"
        if primary_color := entry.get('primary_color', None):
            override += f"\\c{primary_color.value}"
        if alignment := entry.get('alignment', None):
            override += f"\\an{alignment.value}"  # <-- use alignment override

        if override:
            text = f"{{{override}}}{text}"
        return [
            f"Dialogue: 0,{cls.format_ass_time(entry['start_time'])},"
            f"{cls.format_ass_time(entry['end_time'])},"
            f"Default,,0,0,0,,{text}\n"
        ]

    def _export(self, filepath: Path):
        """Export subtitles to ASS file"""
        lines = [
            "[Script Info]\n",
            "ScriptType: v4.00+\n",
            "Collisions: Normal\n",
            f"PlayResX: {self.w}\n",
            f"PlayResY: {self.h}\n",
            "\n[V4+ Styles]\n",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n",
            f"Style: Default,{self.font_style},{self.font_size},{self.primary_color.value},&H000000FF,"
            f"{self.outline_color.value},{self.back_color.value},"
            "0,0,0,0,100,100,0,0,1,2,2,"
            f"{self.alignment.value},10,10,10,1\n",
            "\n[Events]\n",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        ]

        lines.extend(self.get_event_lines())

        with filepath.open('w', encoding='utf-8-sig') as f:
            f.writelines(lines)


class SRTSubtitle(Subtitle):
    SUFFIX = '.srt'

    def __init__(
        self,
        w: int = 1920,
        h: int = 1080,
        font_style: str = "Arial",
        font_size: int = 48,
        primary_color: Any = None,
        outline_color: Any = None,
        back_color: Any = None,
        alignment: Any = None
    ):
        super().__init__(w, h, font_style, font_size, primary_color, outline_color, back_color, alignment)

    @staticmethod
    def format_srt_time(seconds: float) -> str:
        """Format time as HH:MM:SS,mmm for SRT format"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @classmethod
    def get_event(cls, entry: SubtitleEntry) -> list[str]:
        return [
            f"{entry['idx']}\n",
            f"{cls.format_srt_time(entry['start_time'])} --> {cls.format_srt_time(entry['end_time'])}\n",
            f"{entry['text']}\n",
            "\n"
        ]

    def _export(self, filepath: Path):
        with filepath.open('w', encoding='utf-8') as f:
            f.writelines(self.get_event_lines())
