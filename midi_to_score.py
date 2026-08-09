#!/usr/bin/env python3
"""Convert a monophonic Standard MIDI file to player.py JSON without dependencies."""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path


LOW_FREQUENCY = 411.3142
HIGH_FREQUENCY = 3160.6272


@dataclass(frozen=True)
class MidiEvent:
    tick: int
    track: int
    order: int
    kind: str
    channel: int = 0
    note: int = 0
    velocity: int = 0
    tempo: int = 500_000


@dataclass(frozen=True)
class NoteSpan:
    track: int
    channel: int
    note: int
    velocity: int
    start_tick: int
    end_tick: int


def read_vlq(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if offset >= len(data):
            raise ValueError("MIDI VLQ 意外结束")
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset
    raise ValueError("MIDI VLQ 超过 4 字节")


def parse_track(data: bytes, track_index: int) -> list[MidiEvent]:
    events = []
    offset = 0
    tick = 0
    running_status = None
    order = 0
    while offset < len(data):
        delta, offset = read_vlq(data, offset)
        tick += delta
        if offset >= len(data):
            break
        status = data[offset]
        if status & 0x80:
            offset += 1
            if status < 0xF0:
                running_status = status
        elif running_status is not None:
            status = running_status
        else:
            raise ValueError(f"轨道 {track_index} 缺少 running status")

        if status == 0xFF:
            if offset >= len(data):
                raise ValueError("损坏的 MIDI meta event")
            meta_type = data[offset]
            offset += 1
            length, offset = read_vlq(data, offset)
            payload = data[offset : offset + length]
            offset += length
            running_status = None
            if meta_type == 0x51 and length == 3:
                tempo = int.from_bytes(payload, "big")
                events.append(MidiEvent(tick, track_index, order, "tempo", tempo=tempo))
            elif meta_type == 0x2F:
                break
        elif status in (0xF0, 0xF7):
            length, offset = read_vlq(data, offset)
            offset += length
            running_status = None
        else:
            event_type = status & 0xF0
            channel = status & 0x0F
            data_length = 1 if event_type in (0xC0, 0xD0) else 2
            payload = data[offset : offset + data_length]
            if len(payload) != data_length:
                raise ValueError(f"轨道 {track_index} 的 channel event 不完整")
            offset += data_length
            if event_type == 0x90:
                note, velocity = payload
                kind = "on" if velocity else "off"
                events.append(MidiEvent(tick, track_index, order, kind, channel, note, velocity))
            elif event_type == 0x80:
                note, velocity = payload
                events.append(MidiEvent(tick, track_index, order, "off", channel, note, velocity))
        order += 1
    return events


def parse_midi(path: Path) -> tuple[int, list[list[MidiEvent]]]:
    data = path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("不是 Standard MIDI 文件")
    header_length = struct.unpack(">I", data[4:8])[0]
    if header_length < 6:
        raise ValueError("MIDI header 太短")
    _format, track_count, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("暂不支持 SMPTE time division")
    offset = 8 + header_length
    tracks = []
    for track_index in range(track_count):
        if data[offset : offset + 4] != b"MTrk":
            raise ValueError(f"找不到轨道 {track_index} 的 MTrk")
        length = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        start = offset + 8
        end = start + length
        tracks.append(parse_track(data[start:end], track_index))
        offset = end
    return division, tracks


def pair_notes(events: list[MidiEvent], track_index: int) -> list[NoteSpan]:
    active: dict[tuple[int, int], list[MidiEvent]] = {}
    notes = []
    for event in sorted(events, key=lambda item: (item.tick, item.order)):
        key = (event.channel, event.note)
        if event.kind == "on":
            active.setdefault(key, []).append(event)
        elif event.kind == "off" and active.get(key):
            started = active[key].pop(0)
            if event.tick > started.tick:
                notes.append(
                    NoteSpan(
                        track_index,
                        event.channel,
                        event.note,
                        started.velocity,
                        started.tick,
                        event.tick,
                    )
                )
    return sorted(notes, key=lambda item: (item.start_tick, item.end_tick, item.note))


def choose_track(tracks: list[list[MidiEvent]], requested: str) -> tuple[int, list[NoteSpan]]:
    summaries = [(index, pair_notes(events, index)) for index, events in enumerate(tracks)]
    print("MIDI 轨道：")
    for index, notes in summaries:
        if notes:
            low = min(note.note for note in notes)
            high = max(note.note for note in notes)
            print(f"  track {index}: {len(notes)} notes, MIDI {low}–{high}")
        else:
            print(f"  track {index}: 0 notes")
    if requested != "auto":
        index = int(requested)
        if index < 0 or index >= len(tracks):
            raise ValueError(f"轨道 {index} 不存在")
        return index, summaries[index][1]
    candidates = [(len(notes), index, notes) for index, notes in summaries if notes]
    if not candidates:
        raise ValueError("MIDI 中没有音符")
    _, index, notes = max(candidates)
    return index, notes


def tempo_map(tracks: list[list[MidiEvent]]) -> list[tuple[int, int]]:
    events = [event for track in tracks for event in track if event.kind == "tempo"]
    events.sort(key=lambda item: (item.tick, item.track, item.order))
    result = [(0, 500_000)]
    for event in events:
        if event.tick == result[-1][0]:
            result[-1] = (event.tick, event.tempo)
        else:
            result.append((event.tick, event.tempo))
    return result


def ticks_to_seconds(tick: int, division: int, tempos: list[tuple[int, int]]) -> float:
    seconds = 0.0
    previous_tick = 0
    current_tempo = tempos[0][1]
    for change_tick, tempo in tempos[1:]:
        if change_tick >= tick:
            break
        seconds += (change_tick - previous_tick) * current_tempo / division / 1_000_000.0
        previous_tick = change_tick
        current_tempo = tempo
    seconds += (tick - previous_tick) * current_tempo / division / 1_000_000.0
    return seconds


def auto_transpose(notes: list[NoteSpan]) -> int:
    low_midi = 69 + 12 * math.log2(LOW_FREQUENCY / 440.0)
    high_midi = 69 + 12 * math.log2(HIGH_FREQUENCY / 440.0)
    source_low = min(note.note for note in notes)
    source_high = max(note.note for note in notes)
    choices = []
    for shift in range(-48, 49, 12):
        shifted_low = source_low + shift
        shifted_high = source_high + shift
        if shifted_low >= low_midi and shifted_high <= high_midi:
            source_center = (shifted_low + shifted_high) / 2
            target_center = (low_midi + high_midi) / 2
            choices.append((abs(source_center - target_center), shift))
    if not choices:
        raise ValueError(
            f"MIDI 音域 {source_low}–{source_high} 无法仅靠八度移调放入 V100 音域"
        )
    return min(choices)[1]


def midi_note_name(note: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[note % 12]}{note // 12 - 1}"


def midi_frequency(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def flatten_monophonic(
    notes: list[NoteSpan],
    division: int,
    tempos: list[tuple[int, int]],
    transpose: int,
    strategy: str,
) -> list[dict]:
    boundaries = sorted({value for note in notes for value in (note.start_tick, note.end_tick)})
    events = []
    for start_tick, end_tick in zip(boundaries, boundaries[1:]):
        active = [note for note in notes if note.start_tick <= start_tick < note.end_tick]
        if active:
            if strategy == "highest":
                selected = max(active, key=lambda item: (item.note, item.start_tick))
            else:
                selected = max(active, key=lambda item: (item.start_tick, item.note))
            shifted_note = selected.note + transpose
            label = midi_note_name(shifted_note)
            frequency = midi_frequency(shifted_note)
        else:
            label = "rest"
            frequency = None
        start_seconds = ticks_to_seconds(start_tick, division, tempos)
        end_seconds = ticks_to_seconds(end_tick, division, tempos)
        duration = end_seconds - start_seconds
        if duration <= 0:
            continue
        if events and events[-1]["note"] == label:
            events[-1]["seconds"] = round(events[-1]["seconds"] + duration, 6)
        else:
            item = {"note": label, "seconds": round(duration, 6)}
            if frequency is not None:
                item["frequency_hz"] = round(frequency, 6)
            events.append(item)
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把单声道 MIDI 转成 V100 player.py 乐谱")
    parser.add_argument("midi", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--track", default="auto", help="默认自动选择音符最多的轨道")
    parser.add_argument("--transpose", default="auto", help="auto 或整数半音数")
    parser.add_argument("--strategy", choices=["latest", "highest"], default="latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    division, tracks = parse_midi(args.midi)
    track_index, notes = choose_track(tracks, args.track)
    if not notes:
        raise SystemExit(f"轨道 {track_index} 没有完整 Note On/Off")
    transpose = auto_transpose(notes) if args.transpose == "auto" else int(args.transpose)
    events = flatten_monophonic(
        notes,
        division,
        tempo_map(tracks),
        transpose,
        args.strategy,
    )
    output = args.output or args.midi.with_suffix(".score.json")
    payload = {
        "source_midi": str(args.midi),
        "track": track_index,
        "transpose_semitones": transpose,
        "events": events,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    note_events = sum(event["note"] != "rest" for event in events)
    total_seconds = sum(event["seconds"] for event in events)
    print(f"选择 track {track_index}，移调 {transpose:+d} 半音")
    print(f"输出 {note_events} 个发音段，时长 {total_seconds:.2f}s -> {output}")


if __name__ == "__main__":
    main()
