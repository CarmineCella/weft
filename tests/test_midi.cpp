// test_midi.cpp  --  verify Midi.h: write -> read round-trips for one
// track, four tracks and a re-articulated pitch; tempo and resolution are
// kept; long delta times survive the variable-length encoding; running
// status and unknown events are handled on read; bad files are rejected.

#include "Midi.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace weft;

namespace {

int n_passed = 0, n_failed = 0;
void check(bool ok, const std::string& name) {
    if (ok) { ++n_passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++n_failed; std::cout << "  FAIL  " << name << "\n"; }
}

const std::string TMP = "test_midi_tmp.mid";

bool same_notes(const midi::Track& a, const midi::Track& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].pitch != b[i].pitch || a[i].start_tick != b[i].start_tick ||
            a[i].duration_tick != b[i].duration_tick)
            return false;
    return true;
}

void write_bytes(const std::string& path, const std::vector<unsigned char>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
}

} // namespace

void test_single_track() {
    std::cout << "single track round-trip:\n";
    midi::File f;
    f.ticks_per_quarter = 480;
    f.us_per_quarter    = 600000;                  // 100 BPM
    f.tracks.push_back({{60, 90, 0, 480}, {62, 90, 480, 240}, {64, 90, 720, 960}});
    midi::write(TMP, f);
    const midi::File g = midi::read(TMP);
    check(g.tracks.size() == 1, "one track comes back");
    check(g.ticks_per_quarter == 480, "ticks per quarter is kept");
    check(g.us_per_quarter == 600000, "tempo is kept");
    check(same_notes(f.tracks[0], g.tracks[0]), "pitches, starts and durations match");
    check(g.tracks[0][0].velocity == 90, "velocity is kept");
}

void test_four_tracks() {
    std::cout << "four tracks round-trip (a chorale's layout):\n";
    midi::File f;
    f.ticks_per_quarter = 480;
    const int pitches[4] = {72, 67, 60, 48};
    for (int v = 0; v < 4; ++v) {
        midi::Track t;
        for (int i = 0; i < 6; ++i)
            t.push_back({pitches[v] + (i % 3), 80, i * 480, 480});
        f.tracks.push_back(t);
    }
    midi::write(TMP, f);
    const midi::File g = midi::read(TMP);
    bool ok = (g.tracks.size() == 4);
    for (int v = 0; ok && v < 4; ++v) ok = same_notes(f.tracks[v], g.tracks[v]);
    check(ok, "all four voices come back unchanged and in order");
}

// The same pitch struck twice with no gap: the note-off of the first and
// the note-on of the second fall on the same tick.  The writer must emit
// the off first, or the reader would see one long note.
void test_rearticulation() {
    std::cout << "re-articulated pitch:\n";
    midi::File f;
    f.tracks.push_back({{60, 80, 0, 480}, {60, 80, 480, 480}, {60, 80, 960, 240}});
    midi::write(TMP, f);
    const midi::File g = midi::read(TMP);
    check(g.tracks.size() == 1 && g.tracks[0].size() == 3,
          "three separate notes come back, not one long one");
    check(same_notes(f.tracks[0], g.tracks[0]), "their starts and durations match");
}

// Delta times above 127 ticks need more than one byte; above 16383, three.
void test_long_delta_times() {
    std::cout << "long delta times (variable-length quantities):\n";
    midi::File f;
    f.tracks.push_back({{60, 80, 0, 100}, {61, 80, 200, 20000},
                        {62, 80, 2500000, 300}});
    midi::write(TMP, f);
    const midi::File g = midi::read(TMP);
    check(same_notes(f.tracks[0], g.tracks[0]),
          "1-, 2-, 3- and 4-byte delta times all round-trip");
}

// A hand-built file using running status (status byte omitted on the
// second and third events), a note-on with velocity 0 as note-off, and a
// control change and program change that must be skipped.
void test_reader_tolerance() {
    std::cout << "reader: running status, velocity-0 note-off, skipped events:\n";
    const std::vector<unsigned char> body = {
        0x00, 0xC0, 0x05,              // program change (skipped)
        0x00, 0xB0, 0x07, 0x64,        // control change (skipped)
        0x00, 0x90, 0x3C, 0x50,        // note on  C4
        0x60, 0x3C, 0x00,              // +96: running status, velocity 0 = off
        0x00, 0x3E, 0x50,              //      running status, note on D4
        0x60, 0x80, 0x3E, 0x40,        // +96: note off D4
        0x00, 0xFF, 0x2F, 0x00         // end of track
    };
    std::vector<unsigned char> file = {
        'M','T','h','d', 0,0,0,6, 0,0, 0,1, 0,0x60,       // format 0, 1 track, 96 tpq
        'M','T','r','k', 0,0,0, static_cast<unsigned char>(body.size())
    };
    file.insert(file.end(), body.begin(), body.end());
    write_bytes(TMP, file);
    const midi::File g = midi::read(TMP);
    const bool ok = g.tracks.size() == 1 && g.tracks[0].size() == 2 &&
                    g.tracks[0][0].pitch == 60 && g.tracks[0][0].start_tick == 0 &&
                    g.tracks[0][0].duration_tick == 96 &&
                    g.tracks[0][1].pitch == 62 && g.tracks[0][1].start_tick == 96 &&
                    g.tracks[0][1].duration_tick == 96;
    check(g.ticks_per_quarter == 96, "resolution read from the header");
    check(ok, "two notes recovered with the right starts and durations");
}

void test_errors() {
    std::cout << "bad input is rejected:\n";
    bool caught = false;
    try { midi::read("no_such_file.mid"); }
    catch (const std::runtime_error&) { caught = true; }
    check(caught, "a missing file throws");

    write_bytes(TMP, {'R','I','F','F', 0,0,0,0});
    caught = false;
    try { midi::read(TMP); } catch (const std::runtime_error&) { caught = true; }
    check(caught, "a file that is not MIDI throws");

    write_bytes(TMP, {'M','T','h','d', 0,0,0,6, 0,2, 0,1, 0,0x60});
    caught = false;
    try { midi::read(TMP); } catch (const std::runtime_error&) { caught = true; }
    check(caught, "format 2 is refused");
}

int main() {
    std::cout << "test_midi\n";
    test_single_track();
    test_four_tracks();
    test_rearticulation();
    test_long_delta_times();
    test_reader_tolerance();
    test_errors();
    std::remove(TMP.c_str());
    std::cout << n_passed << " passed, " << n_failed << " failed\n";
    return n_failed == 0 ? 0 : 1;
}
