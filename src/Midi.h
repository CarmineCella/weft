#pragma once
//
// Midi.h  --  minimal MIDI file I/O for symbolic music data.
//
// Scope, deliberately narrow:
//
//   READ:   Standard MIDI File format 0 or 1.  Parses note-on /
//           note-off events into a (track-index x time-ordered) list
//           of MidiNote objects with absolute tick times.  Tempo and
//           ticks-per-quarter are kept.  Other meta events, sysex,
//           pitch-bend, control changes, program changes etc. are
//           skipped (their delta-time is still tracked so the file
//           clock stays correct).
//
//   WRITE:  Always emits Format 1 with the supplied tracks in order.
//           Each track gets a single tempo at time 0 (track 0 only),
//           then note-on / note-off pairs in absolute-time order,
//           encoded with variable-length delta times.
//
// MIDI's wire format is straightforward but pedantic:
//   - All multi-byte integers are big-endian.
//   - Delta times use a Variable-Length Quantity: each byte's low 7
//     bits are the value, the high bit is "continue".  We can store
//     up to 28-bit times this way.
//   - Channel events have a top-nibble status (0x80 .. 0xE0) and an
//     optional "running status" optimisation where the status byte
//     can be omitted if it matches the previous event.  Our reader
//     handles that.
//   - Meta events start with 0xFF and have their own length field.
//
// The parser is forgiving (skips unknown bytes within the declared
// track length) and the writer is rigid (well-formed output that any
// standard MIDI player will load).  We don't try to round-trip
// arbitrary MIDI files; we try to produce and consume the subset that
// makes sense for symbolic music data.
//
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft::midi {

// One note event in absolute-tick time.
struct Note {
    int      pitch;          // 0-127, MIDI key number
    int      velocity;       // 1-127 (0 in MIDI means note-off)
    int      start_tick;
    int      duration_tick;  // > 0
};

// One track is a time-sorted vector of notes.  In a chorale file each
// track is one voice (S, A, T, B).
using Track = std::vector<Note>;

struct File {
    std::vector<Track> tracks;
    int ticks_per_quarter = 480;    // standard default
    int us_per_quarter    = 500000; // 120 BPM (tempo meta), default
};

// ---- helpers --------------------------------------------------------------

namespace detail {

inline std::uint32_t read_be32(std::istream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("Midi: unexpected EOF reading 32-bit");
    return (std::uint32_t(b[0]) << 24) | (std::uint32_t(b[1]) << 16)
         | (std::uint32_t(b[2]) <<  8) |  std::uint32_t(b[3]);
}
inline std::uint16_t read_be16(std::istream& in) {
    unsigned char b[2];
    in.read(reinterpret_cast<char*>(b), 2);
    if (!in) throw std::runtime_error("Midi: unexpected EOF reading 16-bit");
    return (std::uint16_t(b[0]) << 8) | std::uint16_t(b[1]);
}
inline std::uint8_t read_u8(std::istream& in) {
    unsigned char b;
    in.read(reinterpret_cast<char*>(&b), 1);
    if (!in) throw std::runtime_error("Midi: unexpected EOF reading byte");
    return b;
}

// Variable-Length Quantity: 1-4 bytes, low 7 bits of each, high bit is
// "more bytes follow."
inline std::uint32_t read_vlq(std::istream& in) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        const std::uint8_t b = read_u8(in);
        v = (v << 7) | (b & 0x7F);
        if (!(b & 0x80)) return v;
    }
    throw std::runtime_error("Midi: VLQ exceeded 4 bytes");
}

inline void write_be32(std::ostream& out, std::uint32_t v) {
    const unsigned char b[4] = {
        static_cast<unsigned char>((v >> 24) & 0xFF),
        static_cast<unsigned char>((v >> 16) & 0xFF),
        static_cast<unsigned char>((v >>  8) & 0xFF),
        static_cast<unsigned char>(v & 0xFF)
    };
    out.write(reinterpret_cast<const char*>(b), 4);
}
inline void write_be16(std::ostream& out, std::uint16_t v) {
    const unsigned char b[2] = {
        static_cast<unsigned char>((v >> 8) & 0xFF),
        static_cast<unsigned char>(v & 0xFF)
    };
    out.write(reinterpret_cast<const char*>(b), 2);
}
inline void write_u8(std::ostream& out, std::uint8_t v) {
    out.write(reinterpret_cast<const char*>(&v), 1);
}
inline void write_vlq(std::ostream& out, std::uint32_t v) {
    // Emit between 1 and 4 bytes; the last has its high bit clear.
    unsigned char buf[4];
    int n = 0;
    buf[n++] = static_cast<unsigned char>(v & 0x7F);
    v >>= 7;
    while (v > 0 && n < 4) {
        buf[n++] = static_cast<unsigned char>((v & 0x7F) | 0x80);
        v >>= 7;
    }
    // Bytes were pushed least-significant first; emit reversed.
    for (int i = n - 1; i >= 0; --i) write_u8(out, buf[i]);
}

} // namespace detail

// ---- READ -----------------------------------------------------------------

inline File read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Midi: cannot open " + path);

    // --- Header chunk "MThd"
    char tag[5] = {0,0,0,0,0};
    in.read(tag, 4);
    if (std::string(tag, 4) != "MThd")
        throw std::runtime_error("Midi: not a MIDI file: " + path);

    const std::uint32_t hlen = detail::read_be32(in);
    if (hlen < 6) throw std::runtime_error("Midi: bad header length");
    const std::uint16_t format   = detail::read_be16(in);
    const std::uint16_t ntracks  = detail::read_be16(in);
    const std::uint16_t division = detail::read_be16(in);
    // Skip any extra header bytes.
    for (std::uint32_t k = 6; k < hlen; ++k) detail::read_u8(in);

    if (format != 0 && format != 1)
        throw std::runtime_error(
            "Midi: only Format 0 and Format 1 are supported (got "
            + std::to_string(format) + ")");
    if (division & 0x8000)
        throw std::runtime_error(
            "Midi: SMPTE timing not supported (division has top bit set)");

    File f;
    f.ticks_per_quarter = static_cast<int>(division);
    f.tracks.reserve(ntracks);

    // --- Track chunks "MTrk"
    for (std::uint16_t t = 0; t < ntracks; ++t) {
        in.read(tag, 4);
        if (std::string(tag, 4) != "MTrk")
            throw std::runtime_error("Midi: missing MTrk header on track " + std::to_string(t));

        const std::uint32_t tlen = detail::read_be32(in);
        const std::streampos track_end = in.tellg() + std::streamoff(tlen);

        Track track;
        std::vector<std::pair<int, int>> pending;   // (pitch -> start_tick) for active notes
        int abs_tick = 0;
        std::uint8_t running_status = 0;

        while (in.tellg() < track_end) {
            const std::uint32_t dt = detail::read_vlq(in);
            abs_tick += static_cast<int>(dt);

            std::uint8_t status = detail::read_u8(in);
            if (status < 0x80) {
                // Running status: status byte omitted, this byte is data.
                in.seekg(-1, std::ios::cur);
                status = running_status;
            } else {
                running_status = status;
            }

            if (status == 0xFF) {
                // Meta event.  Type byte + VLQ length + data.
                const std::uint8_t type = detail::read_u8(in);
                const std::uint32_t mlen = detail::read_vlq(in);
                if (type == 0x51 && mlen == 3) {
                    // Tempo: microseconds per quarter note (24-bit BE).
                    const std::uint8_t a = detail::read_u8(in);
                    const std::uint8_t b = detail::read_u8(in);
                    const std::uint8_t c = detail::read_u8(in);
                    f.us_per_quarter = (int(a) << 16) | (int(b) << 8) | int(c);
                } else {
                    for (std::uint32_t k = 0; k < mlen; ++k) detail::read_u8(in);
                }
                if (type == 0x2F) break;          // end of track
            } else if (status == 0xF0 || status == 0xF7) {
                // Sysex; skip the body.
                const std::uint32_t slen = detail::read_vlq(in);
                for (std::uint32_t k = 0; k < slen; ++k) detail::read_u8(in);
            } else {
                // Channel-voice event.  Top nibble = type.
                const std::uint8_t kind = status & 0xF0;
                if (kind == 0x80 || kind == 0x90) {
                    const int pitch = detail::read_u8(in);
                    const int vel   = detail::read_u8(in);
                    if (kind == 0x90 && vel > 0) {
                        // Note ON
                        pending.emplace_back(pitch, abs_tick);
                    } else {
                        // Note OFF (or note-on with velocity 0).
                        for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
                            if (it->first == pitch) {
                                const int start = it->second;
                                const int dur   = abs_tick - start;
                                if (dur > 0)
                                    track.push_back(Note{pitch, vel > 0 ? vel : 64,
                                                         start, dur});
                                pending.erase(std::next(it).base());
                                break;
                            }
                        }
                    }
                } else if (kind == 0xC0 || kind == 0xD0) {
                    // 1-byte data: program change, channel pressure.
                    detail::read_u8(in);
                } else {
                    // 2-byte data: CC, pitch bend, polyphonic key pressure.
                    detail::read_u8(in);
                    detail::read_u8(in);
                }
            }
        }
        // Any notes still "on" at end of track: drop them silently.
        // Real chorale files shouldn't have these.
        f.tracks.push_back(std::move(track));

        // Seek to declared end of track in case our reading missed bytes.
        in.seekg(track_end);
    }
    return f;
}

// ---- WRITE ----------------------------------------------------------------

inline void write(const std::string& path, const File& f) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Midi: cannot create " + path);

    // --- Header
    out.write("MThd", 4);
    detail::write_be32(out, 6);                                   // header length
    detail::write_be16(out, 1);                                   // format 1
    detail::write_be16(out, static_cast<std::uint16_t>(f.tracks.size()));
    detail::write_be16(out, static_cast<std::uint16_t>(f.ticks_per_quarter));

    // --- One track at a time.  We buffer each track into memory so we
    // can patch the length back into the chunk header at the end.
    for (std::size_t ti = 0; ti < f.tracks.size(); ++ti) {
        const Track& track = f.tracks[ti];

        // Convert notes to a time-sorted list of (tick, on/off, pitch, vel) events.
        struct Ev { int tick; int kind; int pitch; int vel; };  // kind: 1=on, 0=off
        std::vector<Ev> events;
        events.reserve(track.size() * 2);
        for (const Note& n : track) {
            events.push_back({n.start_tick, 1, n.pitch, n.velocity});
            events.push_back({n.start_tick + n.duration_tick, 0, n.pitch, n.velocity});
        }
        // Sort: by tick, then offs before ons at the same tick (so a
        // re-articulation closes the prior note before opening the new).
        std::sort(events.begin(), events.end(),
                  [](const Ev& a, const Ev& b) {
                      if (a.tick != b.tick) return a.tick < b.tick;
                      return a.kind < b.kind;       // 0 < 1
                  });

        std::ostringstream buf(std::ios::binary);

        // Track 0 gets the tempo meta first.
        if (ti == 0) {
            detail::write_vlq(buf, 0);
            detail::write_u8(buf, 0xFF);
            detail::write_u8(buf, 0x51);          // meta type: tempo
            detail::write_u8(buf, 0x03);          // length 3
            const std::uint32_t us = static_cast<std::uint32_t>(f.us_per_quarter);
            detail::write_u8(buf, static_cast<std::uint8_t>((us >> 16) & 0xFF));
            detail::write_u8(buf, static_cast<std::uint8_t>((us >>  8) & 0xFF));
            detail::write_u8(buf, static_cast<std::uint8_t>(us & 0xFF));
        }

        int prev_tick = 0;
        for (const Ev& e : events) {
            const std::uint32_t dt = static_cast<std::uint32_t>(e.tick - prev_tick);
            detail::write_vlq(buf, dt);
            const std::uint8_t status =
                static_cast<std::uint8_t>((e.kind == 1 ? 0x90 : 0x80) | 0); // channel 0
            detail::write_u8(buf, status);
            detail::write_u8(buf, static_cast<std::uint8_t>(e.pitch));
            detail::write_u8(buf, static_cast<std::uint8_t>(
                e.kind == 1 ? std::min(127, std::max(1, e.vel)) : 64));
            prev_tick = e.tick;
        }
        // End of track.
        detail::write_vlq(buf, 0);
        detail::write_u8(buf, 0xFF);
        detail::write_u8(buf, 0x2F);
        detail::write_u8(buf, 0x00);

        const std::string body = buf.str();
        out.write("MTrk", 4);
        detail::write_be32(out, static_cast<std::uint32_t>(body.size()));
        out.write(body.data(), body.size());
    }
}

} // namespace weft::midi
