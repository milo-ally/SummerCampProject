#!/usr/bin/env python3
"""
YouTube video downloader CLI.

Examples:
  python youtube_downloader.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  python youtube_downloader.py --search "lofi hip hop"
  python youtube_downloader.py --search "python tutorial" --search-count 3
  python youtube_downloader.py URL -o downloads --quality 1080
  python youtube_downloader.py URL --audio-only --audio-format mp3
  python youtube_downloader.py URL --list-formats

This script uses yt-dlp for the actual extraction/downloading work.
Install it with:
  python -m pip install -U yt-dlp

For best video/audio merging support, install ffmpeg and keep it on PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


APP_NAME = "youtube_downloader.py"
DEFAULT_TEMPLATE = "%(title).200B [%(id)s].%(ext)s"


class DownloaderError(RuntimeError):
    """Raised for user-facing downloader errors."""


def import_ytdlp():
    try:
        import yt_dlp  # type: ignore

        return yt_dlp
    except ImportError as exc:
        raise DownloaderError(
            "缺少依赖 yt-dlp。\n"
            "请先运行：python -m pip install -U yt-dlp\n"
            "如果需要自动安装，也可以运行本脚本加参数：--install-deps"
        ) from exc


def install_dependencies() -> None:
    print("Installing/upgrading yt-dlp ...")
    cmd = [sys.executable, "-m", "pip", "install", "-U", "yt-dlp"]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        raise DownloaderError(f"依赖安装失败，退出码：{exc.returncode}") from exc


def ensure_output_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DownloaderError(f"无法创建输出目录：{path}") from exc
    if not path.is_dir():
        raise DownloaderError(f"输出路径不是目录：{path}")
    return path


def parse_quality(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "best": "best",
        "worst": "worst",
        "source": "best",
        "max": "best",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized.endswith("p"):
        normalized = normalized[:-1]
    if not normalized.isdigit():
        raise argparse.ArgumentTypeError("quality 必须是 best、worst 或数字高度，例如 1080")
    return normalized


def build_format_selector(args: argparse.Namespace) -> str:
    if args.format:
        return args.format
    if args.audio_only:
        return "bestaudio/best"
    if args.quality == "best":
        return "bv*+ba/best"
    if args.quality == "worst":
        return "worst"
    height = args.quality
    return f"bv*[height<={height}]+ba/b[height<={height}]/best[height<={height}]/best"


def build_targets(args: argparse.Namespace) -> List[str]:
    targets = list(args.urls)
    for keyword in args.search or []:
        query = keyword.strip()
        if not query:
            continue
        targets.append(f"{args.search_provider}{args.search_count}:{query}")
    return targets


def parse_cookies_from_browser(value: Optional[str]) -> Optional[tuple]:
    if not value:
        return None
    parts = [part.strip() for part in value.split(":") if part.strip()]
    if not parts:
        return None
    browser = parts[0]
    if len(parts) == 1:
        return (browser,)
    if len(parts) == 2:
        return (browser, parts[1])
    if len(parts) == 3:
        return (browser, parts[1], parts[2])
    return (browser, parts[1], parts[2], parts[3])


def progress_hook(status: Dict[str, Any]) -> None:
    state = status.get("status")
    if state == "downloading":
        percent = status.get("_percent_str", "").strip()
        speed = status.get("_speed_str", "").strip()
        eta = status.get("_eta_str", "").strip()
        filename = Path(status.get("filename", "")).name
        message = f"\rDownloading {percent:>8}"
        if speed:
            message += f" | {speed}"
        if eta:
            message += f" | ETA {eta}"
        if filename:
            message += f" | {filename[:60]}"
        print(message, end="", flush=True)
    elif state == "finished":
        print("\nDownload finished. Processing file ...")


def ytdlp_options(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = ensure_output_dir(Path(args.output).expanduser().resolve())
    outtmpl = str(output_dir / args.template)

    opts: Dict[str, Any] = {
        "format": build_format_selector(args),
        "outtmpl": outtmpl,
        "noplaylist": not args.playlist,
        "ignoreerrors": args.ignore_errors,
        "retries": args.retries,
        "fragment_retries": args.fragment_retries,
        "continuedl": True,
        "concurrent_fragment_downloads": args.fragments,
        "progress_hooks": [progress_hook],
        "quiet": args.quiet,
        "no_warnings": args.no_warnings,
        "restrictfilenames": args.restrict_filenames,
        "windowsfilenames": True,
        "writethumbnail": args.thumbnail,
        "writesubtitles": args.subtitles,
        "writeautomaticsub": args.auto_subtitles,
        "subtitleslangs": args.subtitle_langs,
        "postprocessors": [],
    }

    if args.merge_format:
        opts["merge_output_format"] = args.merge_format
    if args.proxy:
        opts["proxy"] = args.proxy
    if args.rate_limit:
        opts["ratelimit"] = args.rate_limit
    if args.cookies:
        opts["cookiefile"] = args.cookies
    browser_cookie = parse_cookies_from_browser(args.cookies_from_browser)
    if browser_cookie:
        opts["cookiesfrombrowser"] = browser_cookie
    if args.date_after:
        opts["dateafter"] = args.date_after
    if args.date_before:
        opts["datebefore"] = args.date_before
    if args.max_downloads:
        opts["max_downloads"] = args.max_downloads
    if args.embed_metadata:
        opts["postprocessors"].append({"key": "FFmpegMetadata"})
    if args.embed_thumbnail:
        opts["writethumbnail"] = True
        opts["postprocessors"].append({"key": "EmbedThumbnail"})
    if args.audio_only:
        opts["postprocessors"].append(
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": args.audio_format,
                "preferredquality": args.audio_quality,
            }
        )
    if not opts["postprocessors"]:
        opts.pop("postprocessors")

    return opts


def print_environment_notes(args: argparse.Namespace) -> None:
    if args.quiet:
        return
    if args.audio_only or args.merge_format or args.embed_thumbnail or args.embed_metadata:
        if shutil.which("ffmpeg") is None:
            print(
                "提示：未在 PATH 中检测到 ffmpeg。下载仍可能开始，但合并、转码、嵌入封面/元数据可能失败。",
                file=sys.stderr,
            )


def run_download(urls: Iterable[str], args: argparse.Namespace) -> None:
    yt_dlp = import_ytdlp()
    opts = ytdlp_options(args)

    if args.print_options:
        import json

        print(json.dumps({k: str(v) for k, v in opts.items()}, ensure_ascii=False, indent=2))

    print_environment_notes(args)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download(list(urls))


def show_info(urls: Iterable[str], args: argparse.Namespace, list_formats: bool = False) -> None:
    yt_dlp = import_ytdlp()
    opts: Dict[str, Any] = {
        "quiet": args.quiet,
        "no_warnings": args.no_warnings,
        "noplaylist": not args.playlist,
        "skip_download": True,
    }
    if list_formats:
        opts["listformats"] = True
    if args.cookies:
        opts["cookiefile"] = args.cookies
    browser_cookie = parse_cookies_from_browser(args.cookies_from_browser)
    if browser_cookie:
        opts["cookiesfrombrowser"] = browser_cookie

    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in urls:
            info = ydl.extract_info(url, download=False)
            if list_formats:
                continue
            if not info:
                print(f"No info returned for {url}")
                continue
            if "entries" in info and info.get("entries") is not None:
                print(f"Playlist: {info.get('title', '(untitled)')}")
                for entry in info["entries"]:
                    if not entry:
                        continue
                    print(f"- {entry.get('title', '(untitled)')} | {entry.get('webpage_url') or entry.get('url')}")
            else:
                print(f"Title: {info.get('title', '(untitled)')}")
                print(f"ID: {info.get('id', '')}")
                print(f"Duration: {info.get('duration_string') or info.get('duration') or ''}")
                print(f"Uploader: {info.get('uploader', '')}")
                print(f"URL: {info.get('webpage_url', url)}")


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="一个基于 yt-dlp 的 YouTube 视频下载器。给出 URL 即可下载。",
    )
    parser.add_argument("urls", nargs="*", help="YouTube 视频/播放列表 URL，可一次传入多个")
    parser.add_argument(
        "-s",
        "--search",
        action="append",
        help='按关键词搜索并自动下载，例如 --search "python tutorial"，可重复使用',
    )
    parser.add_argument(
        "--search-count",
        type=positive_int,
        default=1,
        help="每个关键词自动下载的搜索结果数，默认：1",
    )
    parser.add_argument(
        "--search-provider",
        choices=["ytsearch", "ytsearchdate"],
        default="ytsearch",
        help="搜索方式：ytsearch 按相关性，ytsearchdate 按日期，默认：ytsearch",
    )
    parser.add_argument("-o", "--output", default="downloads", help="输出目录，默认：downloads")
    parser.add_argument(
        "-q",
        "--quality",
        type=parse_quality,
        default="best",
        help="视频质量：best、worst 或最大高度，例如 1080、720，默认：best",
    )
    parser.add_argument(
        "-f",
        "--format",
        help="直接传给 yt-dlp 的格式选择器，例如 '137+140' 或 'bv*+ba/best'",
    )
    parser.add_argument(
        "--merge-format",
        choices=["mp4", "mkv", "webm"],
        default="mp4",
        help="合并输出容器，默认：mp4",
    )
    template_help = f"输出文件名模板，默认：{DEFAULT_TEMPLATE}".replace("%", "%%")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help=template_help)
    parser.add_argument("--playlist", action="store_true", help="允许下载整个播放列表；默认只下载单个视频")
    parser.add_argument("--max-downloads", type=positive_int, help="最多下载多少个条目")
    parser.add_argument("--audio-only", action="store_true", help="只下载并提取音频")
    parser.add_argument(
        "--audio-format",
        default="mp3",
        choices=["best", "aac", "alac", "flac", "m4a", "mp3", "opus", "vorbis", "wav"],
        help="音频格式，默认：mp3",
    )
    parser.add_argument("--audio-quality", default="0", help="音频质量，0 最好，默认：0")
    parser.add_argument("--thumbnail", action="store_true", help="下载封面图")
    parser.add_argument("--embed-thumbnail", action="store_true", help="将封面嵌入媒体文件，需要 ffmpeg")
    parser.add_argument("--embed-metadata", action="store_true", help="嵌入标题、作者等元数据，需要 ffmpeg")
    parser.add_argument("--subtitles", action="store_true", help="下载人工字幕")
    parser.add_argument("--auto-subtitles", action="store_true", help="下载自动字幕")
    parser.add_argument(
        "--subtitle-langs",
        default=["zh-Hans", "zh-Hant", "en"],
        nargs="+",
        help="字幕语言列表，默认：zh-Hans zh-Hant en",
    )
    parser.add_argument("--cookies", help="cookies.txt 路径，用于需要登录/年龄验证的视频")
    parser.add_argument(
        "--cookies-from-browser",
        help="从浏览器读取 cookies，例如 chrome、edge、firefox 或 chrome:Profile 1",
    )
    parser.add_argument("--proxy", help="代理地址，例如 http://127.0.0.1:7890 或 socks5://127.0.0.1:1080")
    parser.add_argument("--rate-limit", help="限速，例如 2M、500K")
    parser.add_argument("--retries", type=positive_int, default=10, help="重试次数，默认：10")
    parser.add_argument("--fragment-retries", type=positive_int, default=10, help="分片重试次数，默认：10")
    parser.add_argument("--fragments", type=positive_int, default=4, help="并发分片数，默认：4")
    parser.add_argument("--date-after", help="只下载此日期之后的视频，格式 YYYYMMDD")
    parser.add_argument("--date-before", help="只下载此日期之前的视频，格式 YYYYMMDD")
    parser.add_argument("--restrict-filenames", action="store_true", help="文件名只使用安全 ASCII 字符")
    parser.add_argument("--ignore-errors", action="store_true", help="批量下载时跳过失败条目")
    parser.add_argument("--list-formats", action="store_true", help="列出可用格式，不下载")
    parser.add_argument("--info", action="store_true", help="显示视频/播放列表信息，不下载")
    parser.add_argument("--print-options", action="store_true", help="打印生成的 yt-dlp 选项，便于调试")
    parser.add_argument("--quiet", action="store_true", help="减少输出")
    parser.add_argument("--no-warnings", action="store_true", help="隐藏 yt-dlp 警告")
    parser.add_argument("--install-deps", action="store_true", help="安装/升级 yt-dlp 后退出")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.install_deps:
            install_dependencies()
            print("依赖安装完成。")
            return 0
        targets = build_targets(args)
        if not targets:
            parser.error("请提供至少一个 YouTube URL，或使用 --search 关键词搜索下载")
        if args.list_formats:
            show_info(targets, args, list_formats=True)
        elif args.info:
            show_info(targets, args, list_formats=False)
        else:
            run_download(targets, args)
        return 0
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    except DownloaderError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"下载失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
