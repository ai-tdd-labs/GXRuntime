// SPDX-License-Identifier: GPL-3.0-or-later
// dolgx_replay — Mode A headless digest replay of a .dolt trace.
// Re-decodes the recorded GX stream through RetailGxFrontend +
// ConsumingAuroraRenderSink (no Aurora, no GPU, no game boot) and emits one
// digest line per frame. Gates: --against-stats (the trace's own
// PRESENT_STATS, steady-state rules) or --digest <golden> (exact lines).

#include "dolruntime/aurora_recomp/replay.hpp"

#include "dolgx_replay_pixels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

int usage(std::FILE *out) {
  std::fprintf(
      out,
      "usage: dolgx_replay <trace.dolt> [options]\n"
      "  --against-stats        gate replay digests against the trace's own\n"
      "                         PRESENT_STATS (draws exact + vert-extent "
      "band;\n"
      "                         fails on >2 consecutive mismatch frames)\n"
      "  --digest <golden>      exact line-compare against a golden digest "
      "file\n"
      "  --write-digest <path>  write the digest lines to <path>\n"
      "  --write-parity-jsonl <path>\n"
      "                         write gc_gx_parity_trace_v1 frame/draw rows\n"
      "  --quiet                suppress per-frame digest lines on stdout\n"
      "  --histogram            print decode-event histograms (BP/CP/XF regs,\n"
      "                         TEV/genMode configs, tex/tlut/copy formats,\n"
      "                         draws) for M6 module ranking\n"
      "  --pixels               Mode B: replay through live Aurora (window +\n"
      "                         GPU) and emit per-frame pixel digests\n"
      "                         (Aurora-enabled builds only)\n"
      "  --core                 Mode B2: like --pixels but draws route\n"
      "                         through the gxcore Dolphin-ported core\n"
      "                         instead of the live Aurora gx layer\n"
      "  --pixel-digest <golden>       exact line-compare of pixel digests\n"
      "  --write-pixel-digest <path>   write pixel digest lines to <path>\n"
      "  --png-dir <dir>        with --pixels, dump one PNG per frame\n"
      "  --help                 this text\n"
      "exit: 0 ok, 1 replay/comparison failure, 2 usage or I/O error\n");
  return out == stdout ? 0 : 2;
}

// --histogram: module-demand tallies over the frontend's decode events
//. Ranks the M6 port queue; formats are named so the table maps
// straight onto Dolphin's TextureDecoder/TEV/copy modules.
struct Histogram {
  std::map<std::uint32_t, std::uint64_t> bp_regs;
  std::map<std::uint32_t, std::uint64_t> xf_regs;      // 0x1000-relative
  std::map<std::uint32_t, std::uint64_t> xf_mem_bases; // matrix/light memory
  std::map<std::uint32_t, std::uint64_t> cp_vcd_lo;    // distinct values
  std::map<std::uint32_t, std::uint64_t> tev_stage_counts;
  std::map<std::uint32_t, std::uint64_t> texgen_counts;
  std::map<std::uint32_t, std::uint64_t> ind_stage_counts;
  std::map<std::uint32_t, std::uint64_t> tex_formats;
  std::map<std::uint32_t, std::uint64_t> tlut_formats;
  std::map<std::uint32_t, std::uint64_t> copy_targets; // raw trigger bits3-6
  std::map<std::uint32_t, std::uint64_t> draw_prims;   // cmd&0xF8 per vtxfmt<<8
  std::uint64_t draw_verts = 0;
  std::uint64_t draws = 0;
  std::uint64_t display_lists = 0;
  std::uint64_t dl_bytes = 0;
  std::uint64_t indexed_xf_loads = 0;
  std::uint64_t indexed_spans = 0;
  std::uint64_t cull_all_writes = 0;
  std::uint64_t copies = 0;
  std::uint64_t copy_clears = 0;
  std::uint64_t copy_to_xfb = 0;
};

constexpr std::uint64_t kParityFnvBasis = 0xCBF29CE484222325ull;
constexpr std::uint64_t kParityFnvPrime = 1099511628211ull;

std::uint64_t parity_fnv(const void *data, std::size_t size) {
  auto hash = kParityFnvBasis;
  const auto *bytes = static_cast<const std::uint8_t *>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kParityFnvPrime;
  }
  return hash;
}

std::string json_string(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    switch (c) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 0x20u) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<unsigned>(c) << std::dec;
      } else {
        out << c;
      }
    }
  }
  out << '"';
  return out.str();
}

std::string hex64(std::uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::uppercase << std::setw(16)
      << std::setfill('0') << value;
  return out.str();
}

struct ParityFrameRange {
  std::uint32_t frame = 0;
  unsigned long long first_draw = 0;
  unsigned long long last_draw = 0;
  std::uint32_t draw_count = 0;
};

struct ParityWriter {
  struct DrawState {
    std::array<std::uint32_t, 256> bp{};
    std::array<bool, 256> bp_valid{};
    std::array<std::uint32_t, 2> vcd{};
    std::array<bool, 2> vcd_valid{};
    std::array<std::array<std::uint32_t, 3>, 8> vat{};
    std::array<std::array<bool, 3>, 8> vat_valid{};
  };

  struct GeometryObservations {
    bool valid = false;
    std::array<double, 3> object_min{};
    std::array<double, 3> object_max{};
    std::array<double, 3> world_min{};
    std::array<double, 3> world_max{};
    std::array<double, 4> clip_min{};
    std::array<double, 4> clip_max{};
    std::array<double, 2> uv_min{};
    std::array<double, 2> uv_max{};
    std::uint64_t world_hash = 0;
    std::uint64_t clip_hash = 0;
    bool clip_rejected = false;
    bool uv_valid = false;
    std::vector<std::array<double, 3>> world_samples;
    std::vector<std::array<double, 4>> clip_samples;
  };

  struct TextureSourceState {
    bool valid = false;
    std::uint32_t address = 0;
    std::uint32_t size = 0;
    std::uint32_t tlut_address = 0;
    std::uint32_t tlut_size = 0;
  };

  struct DrawSourceAuthority {
    std::array<bool, dolruntime::aurora_recomp::ConsumedDraw::kMaxTexmaps>
        texture{};
    std::array<bool, dolruntime::aurora_recomp::ConsumedDraw::kMaxTexmaps>
        tlut{};
  };

  std::ofstream out;
  DrawState state;
  std::deque<DrawState> pending_draw_states;
  std::array<TextureSourceState,
             dolruntime::aurora_recomp::ConsumedDraw::kMaxTexmaps>
      texture_sources{};
  std::deque<DrawSourceAuthority> pending_draw_sources;
  std::vector<ParityFrameRange> frames;
  // A mid-frame DFF starts with a zero-filled replay buffer, not an
  // authoritative MEM1 snapshot. Keep the byte ranges explicitly supplied by
  // MEM_UPDATE separate from their current values so a real all-zero texture
  // remains distinguishable from never-captured memory.
  std::map<std::uint32_t, std::uint32_t> initialized_mem1;

  ParityWriter(const char *path, const char *trace_path, const char *game_id)
      : out(path, std::ios::trunc) {
    if (out) {
      out << "{\"record\":\"meta\",\"schema\":\"gc_gx_parity_trace_v1\","
             "\"game_id\":"
          << json_string(game_id)
          << ",\"source\":{\"backend\":\"gxruntime_dolt_replay\",\"path\":"
          << json_string(trace_path) << "}}\n";
    }
  }

  void observe_event(const DolGxRecompTraceEvent &event) {
    if (event.kind == DOL_GX_RECOMP_EVENT_BP_REG && event.a < state.bp.size()) {
      state.bp[event.a] = event.b;
      state.bp_valid[event.a] = true;
    } else if (event.kind == DOL_GX_RECOMP_EVENT_CP_VCD &&
               event.a < state.vcd.size()) {
      state.vcd[event.a] = event.b;
      state.vcd_valid[event.a] = true;
    } else if (event.kind == DOL_GX_RECOMP_EVENT_CP_VAT &&
               event.a < state.vat.size() &&
               event.b < state.vat[event.a].size()) {
      state.vat[event.a][event.b] = event.c;
      state.vat_valid[event.a][event.b] = true;
    } else if (event.kind == DOL_GX_RECOMP_EVENT_TEXTURE &&
               event.a < texture_sources.size()) {
      texture_sources[event.a] = {
          .valid = true,
          .address = event.b,
          .size = event.c,
          .tlut_address = event.tlut_address,
          .tlut_size = event.tlut_entries * 2u,
      };
    } else if (event.kind == DOL_GX_RECOMP_EVENT_DRAW) {
      // ConsumingAuroraRenderSink completes a draw when the following draw
      // arrives. Queue the command-boundary state so the delayed draw callback
      // cannot accidentally observe BP/CP writes belonging to its successor.
      pending_draw_states.push_back(state);
      DrawSourceAuthority sources;
      for (std::size_t slot = 0; slot < texture_sources.size(); ++slot) {
        const TextureSourceState &source = texture_sources[slot];
        if (!source.valid)
          continue;
        sources.texture[slot] = has_mem1_bytes(source.address, source.size);
        sources.tlut[slot] =
            source.tlut_address >=
                dolruntime::aurora_recomp::kTmemSnapshotAddressBase ||
            has_mem1_bytes(source.tlut_address, source.tlut_size);
      }
      pending_draw_sources.push_back(sources);
    }
  }

  void observe_mem_update(std::uint32_t guest_address, std::uint32_t size) {
    if (size == 0u)
      return;
    std::uint32_t begin = dol_gx_recomp_guest_to_physical(guest_address);
    std::uint32_t end = begin + size;
    auto next = initialized_mem1.lower_bound(begin);
    if (next != initialized_mem1.begin()) {
      auto previous = std::prev(next);
      if (previous->second >= begin) {
        begin = previous->first;
        end = std::max(end, previous->second);
        next = initialized_mem1.erase(previous);
      }
    }
    while (next != initialized_mem1.end() && next->first <= end) {
      end = std::max(end, next->second);
      next = initialized_mem1.erase(next);
    }
    initialized_mem1.emplace(begin, end);
  }

  bool has_mem1_bytes(std::uint32_t guest_address, std::uint32_t size) const {
    if (size == 0u)
      return false;
    const std::uint32_t begin =
        dol_gx_recomp_guest_to_physical(guest_address);
    const std::uint32_t end = begin + size;
    auto range = initialized_mem1.upper_bound(begin);
    if (range == initialized_mem1.begin())
      return false;
    --range;
    return range->first <= begin && range->second >= end;
  }

  static void nullable_u32(std::ostream &stream, const DrawState &draw_state,
                           std::uint32_t index) {
    if (index < draw_state.bp.size() && draw_state.bp_valid[index])
      stream << draw_state.bp[index];
    else
      stream << "null";
  }

  static void bp_array(std::ostream &stream, const DrawState &draw_state,
                       std::uint32_t first, std::uint32_t stride,
                       std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::uint32_t index = first + i * stride;
      if (index >= draw_state.bp.size() || !draw_state.bp_valid[index]) {
        stream << "null";
        return;
      }
    }
    stream << '[';
    for (std::uint32_t i = 0; i < count; ++i) {
      if (i != 0)
        stream << ',';
      stream << draw_state.bp[first + i * stride];
    }
    stream << ']';
  }

  static void float_array(std::ostream &stream, const float *values,
                          std::size_t count) {
    stream << '[' << std::setprecision(9);
    for (std::size_t i = 0; i < count; ++i) {
      if (i != 0)
        stream << ',';
      stream << values[i];
    }
    stream << ']';
  }

  template <std::size_t N>
  static void double_rows(std::ostream &stream,
                          const std::vector<std::array<double, N>> &rows) {
    stream << '[';
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (i != 0)
        stream << ',';
      stream << '[' << std::setprecision(17);
      for (std::size_t component = 0; component < N; ++component) {
        if (component != 0)
          stream << ',';
        stream << rows[i][component];
      }
      stream << ']';
    }
    stream << ']';
  }

  template <std::size_t N>
  static void double_array(std::ostream &stream,
                           const std::array<double, N> &values) {
    stream << '[' << std::setprecision(17);
    for (std::size_t i = 0; i < N; ++i) {
      if (i != 0)
        stream << ',';
      stream << values[i];
    }
    stream << ']';
  }

  static std::uint32_t component_size(std::uint32_t type) {
    if (type <= 1u)
      return 1u;
    if (type <= 3u)
      return 2u;
    if (type <= 5u)
      return 4u;
    return 0u;
  }

  static bool decode_component(const std::uint8_t *data,
                               std::size_t available, std::uint32_t type,
                               std::uint32_t fraction, float *out) {
    const std::uint32_t size = component_size(type);
    if (out == nullptr || size == 0u || available < size)
      return false;
    const float scale = fraction == 0u ? 1.0f : std::ldexp(1.0f, fraction);
    switch (type) {
    case 0u:
      *out = static_cast<float>(data[0]) / scale;
      break;
    case 1u:
      *out = static_cast<float>(static_cast<std::int8_t>(data[0])) / scale;
      break;
    case 2u:
      *out = static_cast<float>((static_cast<std::uint32_t>(data[0]) << 8u) |
                                static_cast<std::uint32_t>(data[1])) /
             scale;
      break;
    case 3u: {
      const std::uint16_t bits =
          static_cast<std::uint16_t>((static_cast<std::uint32_t>(data[0]) << 8u) |
                                     static_cast<std::uint32_t>(data[1]));
      *out = static_cast<float>(static_cast<std::int16_t>(bits)) / scale;
      break;
    }
    case 4u:
    case 5u: {
      const std::uint32_t bits =
          (static_cast<std::uint32_t>(data[0]) << 24u) |
          (static_cast<std::uint32_t>(data[1]) << 16u) |
          (static_cast<std::uint32_t>(data[2]) << 8u) |
          static_cast<std::uint32_t>(data[3]);
      std::memcpy(out, &bits, sizeof bits);
      break;
    }
    default:
      return false;
    }
    return std::isfinite(*out);
  }

  // Keep the compact analyzer independent of host FMA contraction and loop
  // unrolling.  DFF's source-side adapter rounds every binary32 product and
  // addition explicitly; doing the same here prevents identical GX state
  // from producing lane-dependent clip bounds on different host compilers.
  static float round_f32(float value) {
    volatile float rounded = value;
    return rounded;
  }

  static float transform_row(const float *row, const std::array<float, 3> &v) {
    const float p0 = round_f32(row[0] * v[0]);
    const float p1 = round_f32(row[1] * v[1]);
    const float p2 = round_f32(row[2] * v[2]);
    const float s01 = round_f32(p0 + p1);
    const float s012 = round_f32(s01 + p2);
    return round_f32(s012 + row[3]);
  }

  static float project_two(float a, float x, float b, float y) {
    const float p0 = round_f32(a * x);
    const float p1 = round_f32(b * y);
    return round_f32(p0 + p1);
  }

  static float project_bias(float a, float x, float bias) {
    return round_f32(round_f32(a * x) + bias);
  }

  static GeometryObservations
  observe_geometry(const DrawState &draw_state,
                   const dolruntime::aurora_recomp::ConsumedDraw &draw) {
    GeometryObservations result;
    if (draw.vtx_fmt >= draw_state.vat.size() ||
        !draw_state.vcd_valid[0] ||
        !draw_state.vat_valid[draw.vtx_fmt][0] || draw.vertex_size == 0u ||
        draw.vertex_count == 0u)
      return result;

    const std::uint32_t vcd_lo = draw_state.vcd[0];
    if (((vcd_lo >> 9u) & 3u) != 1u)
      return result; // DFF compact geometry currently exposes direct positions.
    const std::uint32_t vat_a = draw_state.vat[draw.vtx_fmt][0];
    if ((vat_a & 1u) == 0u)
      return result; // Keep XY draws unavailable on both compact trace sides.
    const std::uint32_t position_type = (vat_a >> 1u) & 7u;
    const std::uint32_t position_fraction = (vat_a >> 4u) & 0x1Fu;
    const std::uint32_t value_size = component_size(position_type);
    if (value_size == 0u)
      return result;
    std::uint32_t position_offset = 0u;
    for (std::uint32_t attr = 0; attr < 9u; ++attr)
      position_offset += (vcd_lo >> attr) & 1u;
    const std::size_t required =
        static_cast<std::size_t>(position_offset) + 3u * value_size;
    if (required > draw.vertex_size ||
        draw.vertex_payload.size() <
            static_cast<std::size_t>(draw.vertex_count) * draw.vertex_size)
      return result;
    if ((draw.transform_flags &
         dolruntime::aurora_recomp::kDrawTransformProjectionValid) == 0u ||
        draw.current_pn_matrix >= DOL_GX_RECOMP_POSITION_MATRIX_COUNT ||
        (draw.position_matrix_valid_mask & (1u << draw.current_pn_matrix)) ==
            0u)
      return result;

    const float *matrix = draw.position_matrices[draw.current_pn_matrix];
    std::size_t tex0_offset = position_offset + 3u * value_size;
    std::uint32_t tex0_type = 0u;
    std::uint32_t tex0_fraction = 0u;
    std::uint32_t tex0_components = 0u;
    bool tex0_direct = false;
    if (draw_state.vcd_valid[1]) {
      const auto indexed_or_direct_size = [](std::uint32_t mode,
                                             std::size_t direct_size) {
        if (mode == 1u) return direct_size;
        if (mode == 2u) return std::size_t{1};
        if (mode == 3u) return std::size_t{2};
        return std::size_t{0};
      };
      const std::uint32_t normal_mode = (vcd_lo >> 11u) & 3u;
      if (normal_mode != 0u) {
        const std::uint32_t normal_type = (vat_a >> 10u) & 7u;
        const std::size_t normal_component_size = component_size(normal_type);
        const std::size_t normal_components = ((vat_a >> 9u) & 1u) ? 9u : 3u;
        tex0_offset += indexed_or_direct_size(
            normal_mode, normal_component_size * normal_components);
      }
      const auto color_size = [](std::uint32_t format) -> std::size_t {
        static constexpr std::size_t sizes[8] = {2u, 3u, 4u, 2u,
                                                  3u, 4u, 0u, 0u};
        return sizes[format & 7u];
      };
      const std::uint32_t color0_mode = (vcd_lo >> 13u) & 3u;
      const std::uint32_t color1_mode = (vcd_lo >> 15u) & 3u;
      tex0_offset += indexed_or_direct_size(
          color0_mode, color_size((vat_a >> 14u) & 7u));
      tex0_offset += indexed_or_direct_size(
          color1_mode, color_size((vat_a >> 18u) & 7u));
      const std::uint32_t tex0_mode = draw_state.vcd[1] & 3u;
      tex0_type = (vat_a >> 22u) & 7u;
      tex0_fraction = (vat_a >> 25u) & 0x1Fu;
      tex0_components = ((vat_a >> 21u) & 1u) ? 2u : 1u;
      tex0_direct = tex0_mode == 1u && component_size(tex0_type) != 0u &&
                    tex0_offset + tex0_components * component_size(tex0_type) <=
                        draw.vertex_size;
    }
    bool have_bounds = false;
    result.world_hash = 0xCBF29CE484222325ull;
    result.clip_hash = 0xCBF29CE484222325ull;
    auto mix_f32 = [](std::uint64_t &hash, float value) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      for (unsigned shift = 0; shift < 32; shift += 8) {
        hash ^= static_cast<std::uint8_t>(bits >> shift);
        hash *= 1099511628211ull;
      }
    };
    std::uint32_t shared_clip_mask = 0x3Fu;
    for (std::uint32_t vertex = 0; vertex < draw.vertex_count; ++vertex) {
      const std::size_t base =
          static_cast<std::size_t>(vertex) * draw.vertex_size +
          position_offset;
      std::array<double, 3> object{};
      for (std::uint32_t component = 0; component < object.size(); ++component) {
        const std::size_t offset = base + component * value_size;
        float decoded = 0.0f;
        if (!decode_component(draw.vertex_payload.data() + offset,
                              draw.vertex_payload.size() - offset,
                              position_type, position_fraction,
                              &decoded))
          return GeometryObservations{};
        object[component] = decoded;
      }
      if (tex0_direct) {
        std::array<double, 2> uv{};
        const std::size_t tex_component_size = component_size(tex0_type);
        for (std::uint32_t component = 0; component < tex0_components; ++component) {
          float decoded = 0.0f;
          const std::size_t offset =
              static_cast<std::size_t>(vertex) * draw.vertex_size + tex0_offset +
              component * tex_component_size;
          if (!decode_component(draw.vertex_payload.data() + offset,
                                draw.vertex_payload.size() - offset, tex0_type,
                                tex0_fraction, &decoded))
            return GeometryObservations{};
          uv[component] = decoded;
        }
        if (!result.uv_valid) {
          result.uv_min = result.uv_max = uv;
          result.uv_valid = true;
        } else {
          for (std::size_t axis = 0; axis < uv.size(); ++axis) {
            result.uv_min[axis] = std::min(result.uv_min[axis], uv[axis]);
            result.uv_max[axis] = std::max(result.uv_max[axis], uv[axis]);
          }
        }
      }
      if (!have_bounds) {
        result.object_min = object;
        result.object_max = object;
        have_bounds = true;
      } else {
        for (std::size_t axis = 0; axis < object.size(); ++axis) {
          result.object_min[axis] =
              std::min(result.object_min[axis], object[axis]);
          result.object_max[axis] =
              std::max(result.object_max[axis], object[axis]);
        }
      }
      const std::array<float, 3> object_f = {
          static_cast<float>(object[0]), static_cast<float>(object[1]),
          static_cast<float>(object[2])};
      const std::array<float, 3> world_f = {
          transform_row(matrix + 0u, object_f),
          transform_row(matrix + 4u, object_f),
          transform_row(matrix + 8u, object_f),
      };
      const std::array<double, 3> world = {
          world_f[0], world_f[1], world_f[2]};
      std::array<float, 4> clip_f{};
      if (draw.projection_type == 0u) {
        clip_f = {
            project_two(draw.projection[0], world_f[0], draw.projection[1],
                        world_f[2]),
            project_two(draw.projection[2], world_f[1], draw.projection[3],
                        world_f[2]),
            project_bias(draw.projection[4], world_f[2], draw.projection[5]),
            round_f32(-world_f[2])};
      } else {
        clip_f = {
            project_bias(draw.projection[0], world_f[0], draw.projection[1]),
            project_bias(draw.projection[2], world_f[1], draw.projection[3]),
            project_bias(draw.projection[4], world_f[2], draw.projection[5]),
            1.0f};
      }
      const std::array<double, 4> clip = {
          clip_f[0], clip_f[1], clip_f[2], clip_f[3]};
      for (float component : world_f)
        mix_f32(result.world_hash, component);
      for (float component : clip_f)
        mix_f32(result.clip_hash, component);
      result.clip_hash ^= 1u;
      result.clip_hash *= 1099511628211ull;
      if (vertex == 0u) {
        result.world_min = result.world_max = world;
        result.clip_min = result.clip_max = clip;
      } else {
        for (std::size_t axis = 0; axis < world.size(); ++axis) {
          result.world_min[axis] = std::min(result.world_min[axis], world[axis]);
          result.world_max[axis] = std::max(result.world_max[axis], world[axis]);
        }
        for (std::size_t axis = 0; axis < clip.size(); ++axis) {
          result.clip_min[axis] = std::min(result.clip_min[axis], clip[axis]);
          result.clip_max[axis] = std::max(result.clip_max[axis], clip[axis]);
        }
      }
      std::uint32_t mask = 0u;
      if (clip_f[3] - clip_f[0] < 0.0f) mask |= 0x01u;
      if (clip_f[0] + clip_f[3] < 0.0f) mask |= 0x02u;
      if (clip_f[3] - clip_f[1] < 0.0f) mask |= 0x04u;
      if (clip_f[1] + clip_f[3] < 0.0f) mask |= 0x08u;
      if (draw.projection_type != 0u && clip_f[2] > 0.000001f) mask |= 0x10u;
      if (clip_f[2] + clip_f[3] < -0.000001f) mask |= 0x20u;
      shared_clip_mask &= mask;
      if (result.world_samples.size() < 4u) {
        result.world_samples.push_back(world);
        result.clip_samples.push_back(clip);
      }
    }
    const bool perspective = draw.projection_type == 0u;
    const std::uint32_t z_raw =
        draw_state.bp_valid[0x40u] ? draw_state.bp[0x40u] : 0u;
    const bool hard_z = !perspective || (z_raw & 1u) != 0u ||
                        (z_raw & 0x10u) != 0u;
    const std::uint32_t hard_mask = (perspective ? 0x0Fu : 0u) |
                                    (hard_z ? 0x30u : 0u);
    result.clip_rejected =
        (perspective && result.clip_max[3] <= 0.0) ||
        (shared_clip_mask & hard_mask) != 0u;
    result.valid = have_bounds;
    return result;
  }

  void observe_draw(std::uint32_t frame, std::uint32_t frame_draw,
                    const dolruntime::aurora_recomp::ConsumedDraw &draw,
                    unsigned long long cumulative_draw) {
    if (!out)
      return;
    const DrawState draw_state =
        pending_draw_states.empty() ? state : pending_draw_states.front();
    if (!pending_draw_states.empty())
      pending_draw_states.pop_front();
    DrawSourceAuthority draw_sources;
    if (!pending_draw_sources.empty()) {
      draw_sources = pending_draw_sources.front();
      pending_draw_sources.pop_front();
    }
    if (frames.empty() || frames.back().frame != frame) {
      frames.push_back({.frame = frame,
                        .first_draw = cumulative_draw,
                        .last_draw = cumulative_draw,
                        .draw_count = 0});
    }
    frames.back().last_draw = cumulative_draw;
    ++frames.back().draw_count;

    const bool gen_valid = draw_state.bp_valid[0x00u];
    const std::uint32_t gen = draw_state.bp[0x00u];
    const bool texture_order_valid = draw_state.bp_valid[0x28u];
    const std::uint32_t texture_slot =
        texture_order_valid ? (draw_state.bp[0x28u] & 7u) : 0u;
    const bool texture_enabled =
        texture_order_valid && ((draw_state.bp[0x28u] >> 6u) & 1u) != 0u;
    const auto &texture =
        texture_slot < dolruntime::aurora_recomp::ConsumedDraw::kMaxTexmaps
            ? draw.textures[texture_slot]
            : draw.texture;
    const GeometryObservations geometry = observe_geometry(draw_state, draw);
    std::string texture_hash;
    std::string tlut_hash;
    const bool texture_source_captured =
        texture_slot < draw_sources.texture.size() &&
        draw_sources.texture[texture_slot];
    if (texture_enabled && texture_source_captured && texture.resolved &&
        texture.host_data != nullptr &&
        texture.size <= texture.host_available) {
      texture_hash = hex64(parity_fnv(texture.host_data, texture.size));
    }
    const std::size_t tlut_size =
        static_cast<std::size_t>(texture.tlut_entries) * 2u;
    const bool tlut_source_captured = texture_slot < draw_sources.tlut.size() &&
                                      draw_sources.tlut[texture_slot];
    if (texture_enabled && texture.has_tlut &&
        tlut_source_captured &&
        texture.tlut_host_data != nullptr &&
        tlut_size <= texture.tlut_host_available) {
      tlut_hash = hex64(parity_fnv(texture.tlut_host_data, tlut_size));
    }
    std::string payload_hash = hex64(
        parity_fnv(draw.vertex_payload.data(), draw.vertex_payload.size()));

    out << "{\"record\":\"draw\",\"ordinal\":" << cumulative_draw
        << ",\"source_draw\":" << cumulative_draw
        << ",\"source_frame\":" << frame << ",\"frame_ordinal\":" << frame
        << ",\"frame_draw\":" << frame_draw << ",\"copy_epoch\":" << frame
        << ",\"epoch_draw\":" << frame_draw
        << ",\"command\":{\"primitive\":" << ((draw.primitive >> 3u) & 7u)
        << ",\"vtxfmt\":" << draw.vtx_fmt << ",\"nverts\":" << draw.vertex_count
        << ",\"stride\":" << draw.vertex_size
        << "},\"gx_state\":{\"raw\":{"
           "\"gen_mode\":";
    nullable_u32(out, draw_state, 0x00u);
    out << ",\"blend_mode\":";
    nullable_u32(out, draw_state, 0x41u);
    out << ",\"z_mode\":";
    nullable_u32(out, draw_state, 0x40u);
    out << ",\"alpha_test\":";
    nullable_u32(out, draw_state, 0xF3u);
    out << ",\"pe_control\":";
    nullable_u32(out, draw_state, 0x43u);
    out << ",\"scissor_tl\":";
    nullable_u32(out, draw_state, 0x20u);
    out << ",\"scissor_br\":";
    nullable_u32(out, draw_state, 0x21u);
    out << "},\"tev_stages\":";
    if (gen_valid)
      out << (((gen >> 10u) & 0xFu) + 1u);
    else
      out << "null";
    out << ",\"cull_mode\":";
    if (gen_valid)
      out << ((gen >> 14u) & 3u);
    else
      out << "null";
    out << ",\"vcd_lo\":"
        << (draw_state.vcd_valid[0] ? std::to_string(draw_state.vcd[0])
                                    : "null")
        << ",\"vcd_hi\":"
        << (draw_state.vcd_valid[1] ? std::to_string(draw_state.vcd[1])
                                    : "null")
        << ",\"vat_a\":"
        << (draw.vtx_fmt < draw_state.vat.size() &&
                    draw_state.vat_valid[draw.vtx_fmt][0]
                ? std::to_string(draw_state.vat[draw.vtx_fmt][0])
                : "null")
        << ",\"vat_b\":"
        << (draw.vtx_fmt < draw_state.vat.size() &&
                    draw_state.vat_valid[draw.vtx_fmt][1]
                ? std::to_string(draw_state.vat[draw.vtx_fmt][1])
                : "null")
        << ",\"vat_c\":"
        << (draw.vtx_fmt < draw_state.vat.size() &&
                    draw_state.vat_valid[draw.vtx_fmt][2]
                ? std::to_string(draw_state.vat[draw.vtx_fmt][2])
                : "null")
        << "},\"texture\":{\"enabled\":"
        << (texture_enabled ? "true" : "false");
    if (texture_enabled) {
      out << ",\"slot\":" << texture_slot
          << ",\"address_phys\":" << (texture.address & 0x03FFFFFFu)
          << ",\"format\":" << texture.format << ",\"width\":" << texture.width
          << ",\"height\":" << texture.height << ",\"source_hash\":"
          << (texture_hash.empty() ? "null" : json_string(texture_hash))
          << ",\"source_hash_raw\":"
          << (texture_hash.empty() ? "null" : json_string(texture_hash))
          << ",\"tlut\":"
          << (texture.has_tlut
                  ? std::to_string(texture.tlut_address & 0x03FFFFFFu)
                  : "null")
          << ",\"tlut_address_phys\":"
          << (texture.has_tlut
                  ? std::to_string(texture.tlut_address & 0x03FFFFFFu)
                  : "null")
          << ",\"tlut_format\":"
          << (texture.has_tlut ? std::to_string(texture.tlut_format) : "null")
          << ",\"tlut_entries\":"
          << (texture.has_tlut ? std::to_string(texture.tlut_entries) : "null")
          << ",\"tlut_source_hash\":"
          << (tlut_hash.empty() ? "null" : json_string(tlut_hash));
    }
    out << "},\"matrix\":{\"projection_type\":";
    if ((draw.transform_flags &
         dolruntime::aurora_recomp::kDrawTransformProjectionValid) != 0u) {
      out << draw.projection_type << ",\"projection_coefficients\":";
      float_array(out, draw.projection, 6);
    } else {
      out << "null,\"projection_coefficients\":null";
    }
    // The parity schema uses a semantic 3x4 matrix slot. Keep GX's raw
    // word-addressed PNMTX register encoding internal to the DFF decoder.
    out << ",\"position_index\":" << draw.current_pn_matrix
        << ",\"position_values\":";
    if (draw.current_pn_matrix < DOL_GX_RECOMP_POSITION_MATRIX_COUNT &&
        (draw.position_matrix_valid_mask & (1u << draw.current_pn_matrix)) !=
            0u) {
      float_array(out, draw.position_matrices[draw.current_pn_matrix],
                  DOL_GX_RECOMP_POSITION_MATRIX_WORDS);
    } else {
      out << "null";
    }
    out << "},\"vertices\":{\"payload_hash\":" << json_string(payload_hash);
    if (geometry.valid) {
      out << ",\"object_bounds\":{\"min\":";
      double_array(out, geometry.object_min);
      out << ",\"max\":";
      double_array(out, geometry.object_max);
      out << "},\"world_bounds\":{\"min\":";
      double_array(out, geometry.world_min);
      out << ",\"max\":";
      double_array(out, geometry.world_max);
      out << "},\"world_hash\":" << json_string(hex64(geometry.world_hash))
          << ",\"world_samples\":";
      double_rows(out, geometry.world_samples);
      if (geometry.uv_valid) {
        out << ",\"uv_bounds\":{\"min\":";
        double_array(out, geometry.uv_min);
        out << ",\"max\":";
        double_array(out, geometry.uv_max);
        out << "}";
      }
    }
    out << "},\"post_clip\":{";
    if (geometry.valid) {
      out << "\"clip_hash\":" << json_string(hex64(geometry.clip_hash))
          << ",\"clip_bounds\":{\"min\":";
      double_array(out, geometry.clip_min);
      out << ",\"max\":";
      double_array(out, geometry.clip_max);
      out << "},\"clip_rejected\":"
          << (geometry.clip_rejected ? "true" : "false")
          << ",\"clip_samples\":";
      double_rows(out, geometry.clip_samples);
      out << ",\"projection_type\":" << draw.projection_type
          << ",\"derived_from\":\"gxruntime_direct_vertices\"";
    }
    out << "},\"fragment\":{\"tev_color_env\":";
    const std::uint32_t tev_stages =
        gen_valid ? (((gen >> 10u) & 0xFu) + 1u) : 0u;
    bp_array(out, draw_state, 0xC0u, 2u, tev_stages);
    out << ",\"tev_alpha_env\":";
    bp_array(out, draw_state, 0xC1u, 2u, tev_stages);
    out << ",\"tev_regs\":";
    bp_array(out, draw_state, 0xE0u, 1u, 8u);
    out << ",\"tev_ksel\":";
    bp_array(out, draw_state, 0xF6u, 1u, 8u);
    out << "},\"pixels\":{}}\n";
  }

  void
  write_frames(const std::vector<dolruntime::aurora_recomp::replay::FrameDigest>
                   &digests) {
    for (std::size_t ordinal = 0; ordinal < digests.size(); ++ordinal) {
      const auto &digest = digests[ordinal];
      const auto found =
          std::find_if(frames.begin(), frames.end(), [&](const auto &range) {
            return range.frame == digest.frame_index;
          });
      const unsigned long long first =
          found != frames.end() ? found->first_draw : 0;
      const unsigned long long last =
          found != frames.end() ? found->last_draw : 0;
      const std::uint32_t count = found != frames.end() ? found->draw_count : 0;
      out << "{\"record\":\"frame\",\"ordinal\":" << ordinal
          << ",\"source_frame\":" << digest.frame_index
          << ",\"copy_epoch\":" << digest.frame_index
          << ",\"draw_start\":" << first << ",\"draw_end\":" << last
          << ",\"draw_count\":" << count
          << ",\"unknown_opcodes\":0,"
             "\"gxruntime_content_hash\":"
          << json_string(hex64(digest.content_fnv))
          << ",\"gxruntime_state_hash\":"
          << json_string(hex64(digest.state_fnv)) << "}\n";
    }
  }
};

struct CliObservers {
  Histogram *histogram = nullptr;
  ParityWriter *parity = nullptr;
};

void histogram_observe(const DolGxRecompTraceEvent &event, void *user);

void cli_event_observer(const DolGxRecompTraceEvent &event, void *user) {
  auto *observers = static_cast<CliObservers *>(user);
  if (observers->histogram != nullptr)
    histogram_observe(event, observers->histogram);
  if (observers->parity != nullptr)
    observers->parity->observe_event(event);
}

void cli_draw_observer(std::uint32_t frame_index, std::uint32_t frame_draw,
                       const dolruntime::aurora_recomp::ConsumedDraw &draw,
                       unsigned long long cumulative_draw, void *user) {
  static_cast<ParityWriter *>(user)->observe_draw(frame_index, frame_draw, draw,
                                                  cumulative_draw);
}

void cli_mem_update_observer(std::uint32_t guest_address, std::uint32_t size,
                             void *user) {
  static_cast<ParityWriter *>(user)->observe_mem_update(guest_address, size);
}

void histogram_observe(const DolGxRecompTraceEvent &event, void *user) {
  auto *h = static_cast<Histogram *>(user);
  switch (event.kind) {
  case DOL_GX_RECOMP_EVENT_BP_REG: {
    const std::uint32_t reg = event.a;
    const std::uint32_t value = event.b;
    ++h->bp_regs[reg];
    if (reg == 0x00u) { // genMode
      ++h->texgen_counts[value & 0xFu];
      ++h->tev_stage_counts[((value >> 10u) & 0xFu) + 1u];
      ++h->ind_stage_counts[(value >> 16u) & 0x7u];
    } else if (reg == 0x52u) { // copy trigger
      ++h->copies;
      ++h->copy_targets[(value >> 3u) & 0xFu];
      if ((value >> 11u) & 1u)
        ++h->copy_clears;
      if ((value >> 14u) & 1u)
        ++h->copy_to_xfb;
    }
    break;
  }
  case DOL_GX_RECOMP_EVENT_XF_LOAD:
    if (event.a >= 0x1000u) {
      for (std::uint32_t i = 0; i < event.b; ++i)
        ++h->xf_regs[event.a - 0x1000u + i];
    } else {
      ++h->xf_mem_bases[event.a & 0xFF00u]; // bucket by 256-word region
    }
    break;
  case DOL_GX_RECOMP_EVENT_CP_VCD:
    if (event.a == 0u)
      ++h->cp_vcd_lo[event.b];
    break;
  case DOL_GX_RECOMP_EVENT_TEXTURE:
    ++h->tex_formats[event.d];
    break;
  case DOL_GX_RECOMP_EVENT_TLUT:
    ++h->tlut_formats[event.d];
    break;
  case DOL_GX_RECOMP_EVENT_DRAW:
    ++h->draws;
    ++h->draw_prims[(event.b << 8u) | event.a];
    h->draw_verts += event.c;
    break;
  case DOL_GX_RECOMP_EVENT_DISPLAY_LIST:
    ++h->display_lists;
    h->dl_bytes += event.b;
    break;
  case DOL_GX_RECOMP_EVENT_INDEXED_XF_LOAD:
    ++h->indexed_xf_loads;
    break;
  case DOL_GX_RECOMP_EVENT_INDEXED_SPAN:
    ++h->indexed_spans;
    break;
  case DOL_GX_RECOMP_EVENT_CULL_ALL:
    ++h->cull_all_writes;
    break;
  default:
    break;
  }
}

const char *tex_format_name(std::uint32_t format) {
  switch (format) {
  case 0x0:
    return "I4";
  case 0x1:
    return "I8";
  case 0x2:
    return "IA4";
  case 0x3:
    return "IA8";
  case 0x4:
    return "RGB565";
  case 0x5:
    return "RGB5A3";
  case 0x6:
    return "RGBA8";
  case 0x8:
    return "C4";
  case 0x9:
    return "C8";
  case 0xA:
    return "C14X2";
  case 0xE:
    return "CMPR";
  default:
    return "?";
  }
}

const char *tlut_format_name(std::uint32_t format) {
  switch (format) {
  case 0x0:
    return "IA8";
  case 0x1:
    return "RGB565";
  case 0x2:
    return "RGB5A3";
  default:
    return "?";
  }
}

void print_histogram(const Histogram &h) {
  std::printf("== histogram: draws %llu (verts %llu) display_lists %llu "
              "(%llu bytes) indexed_xf %llu indexed_spans %llu cull_all %llu\n",
              (unsigned long long)h.draws, (unsigned long long)h.draw_verts,
              (unsigned long long)h.display_lists,
              (unsigned long long)h.dl_bytes,
              (unsigned long long)h.indexed_xf_loads,
              (unsigned long long)h.indexed_spans,
              (unsigned long long)h.cull_all_writes);
  std::printf("== copies %llu (clear %llu, to_xfb %llu) targets:",
              (unsigned long long)h.copies, (unsigned long long)h.copy_clears,
              (unsigned long long)h.copy_to_xfb);
  for (const auto &[target, count] : h.copy_targets)
    std::printf(" %u:%llu", target, (unsigned long long)count);
  std::printf("\n== tex formats:");
  for (const auto &[format, count] : h.tex_formats)
    std::printf(" %s:%llu", tex_format_name(format), (unsigned long long)count);
  std::printf("\n== tlut formats:");
  for (const auto &[format, count] : h.tlut_formats)
    std::printf(" %s:%llu", tlut_format_name(format),
                (unsigned long long)count);
  std::printf("\n== genMode: tev_stages");
  for (const auto &[stages, count] : h.tev_stage_counts)
    std::printf(" %u:%llu", stages, (unsigned long long)count);
  std::printf(" | texgens");
  for (const auto &[texgens, count] : h.texgen_counts)
    std::printf(" %u:%llu", texgens, (unsigned long long)count);
  std::printf(" | ind_stages");
  for (const auto &[stages, count] : h.ind_stage_counts)
    std::printf(" %u:%llu", stages, (unsigned long long)count);
  std::printf("\n== draw prims (prim/vtxfmt:count):");
  for (const auto &[key, count] : h.draw_prims)
    std::printf(" %02X/%u:%llu", key & 0xFFu, key >> 8u,
                (unsigned long long)count);
  std::printf("\n== vcd_lo values:");
  for (const auto &[value, count] : h.cp_vcd_lo)
    std::printf(" 0x%X:%llu", value, (unsigned long long)count);
  std::printf("\n== bp regs:");
  for (const auto &[reg, count] : h.bp_regs)
    std::printf(" %02X:%llu", reg, (unsigned long long)count);
  std::printf("\n== xf regs (0x1000+):");
  for (const auto &[reg, count] : h.xf_regs)
    std::printf(" %02X:%llu", reg, (unsigned long long)count);
  std::printf("\n== xf mem regions (base&0xFF00):");
  for (const auto &[base, count] : h.xf_mem_bases)
    std::printf(" %04X:%llu", base, (unsigned long long)count);
  std::printf("\n");
}

} // namespace

int main(int argc, char **argv) {
  const char *trace_path = nullptr;
  const char *golden_path = nullptr;
  const char *write_path = nullptr;
  const char *parity_path = nullptr;
  bool against_stats = false;
  bool quiet = false;
  bool histogram = false;
  bool pixels = false;
  bool core = false;
  PixelReplayOptions pixel_options;

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (std::strcmp(arg, "--help") == 0)
      return usage(stdout);
    if (std::strcmp(arg, "--against-stats") == 0) {
      against_stats = true;
    } else if (std::strcmp(arg, "--quiet") == 0) {
      quiet = true;
    } else if (std::strcmp(arg, "--histogram") == 0) {
      histogram = true;
    } else if (std::strcmp(arg, "--digest") == 0 && i + 1 < argc) {
      golden_path = argv[++i];
    } else if (std::strcmp(arg, "--write-digest") == 0 && i + 1 < argc) {
      write_path = argv[++i];
    } else if (std::strcmp(arg, "--write-parity-jsonl") == 0 && i + 1 < argc) {
      parity_path = argv[++i];
    } else if (std::strcmp(arg, "--pixels") == 0) {
      pixels = true;
    } else if (std::strcmp(arg, "--core") == 0) {
      core = true;
    } else if (std::strcmp(arg, "--pixel-digest") == 0 && i + 1 < argc) {
      pixel_options.golden_path = argv[++i];
    } else if (std::strcmp(arg, "--write-pixel-digest") == 0 && i + 1 < argc) {
      pixel_options.write_path = argv[++i];
    } else if (std::strcmp(arg, "--png-dir") == 0 && i + 1 < argc) {
      pixel_options.png_dir = argv[++i];
    } else if (std::strcmp(arg, "--png-every") == 0 && i + 1 < argc) {
      pixel_options.png_every =
          static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg[0] != '-' && trace_path == nullptr) {
      trace_path = arg;
    } else {
      std::fprintf(stderr, "dolgx_replay: unknown argument '%s'\n", arg);
      return usage(stderr);
    }
  }
  if (trace_path == nullptr)
    return usage(stderr);

  if (pixels || core) {
    if (against_stats || golden_path != nullptr || write_path != nullptr ||
        parity_path != nullptr) {
      std::fprintf(stderr,
                   "dolgx_replay: --pixels/--core do not combine with Mode A "
                   "gates\n");
      return usage(stderr);
    }
    if (pixels && core) {
      std::fprintf(stderr, "dolgx_replay: --pixels and --core are exclusive\n");
      return usage(stderr);
    }
    pixel_options.quiet = quiet;
    if (core) {
#if DOLGX_REPLAY_HAS_CORE
      return dolgx_replay_core_main(trace_path, pixel_options);
#else
      std::fprintf(stderr,
                   "dolgx_replay: this build has no --core support (built "
                   "without Aurora; use the StrikersRecomp GUI tree's "
                   "dolgx_replay)\n");
      return 2;
#endif
    }
#if DOLGX_REPLAY_HAS_PIXELS
    return dolgx_replay_pixels_main(trace_path, pixel_options);
#else
    std::fprintf(stderr,
                 "dolgx_replay: this build has no --pixels support (built "
                 "without Aurora; use the StrikersRecomp GUI tree's "
                 "dolgx_replay)\n");
    return 2;
#endif
  }

  namespace replay = dolruntime::aurora_recomp::replay;
  namespace trace = dolruntime::aurora_recomp::trace;

  trace::TraceReader reader;
  if (!reader.open(trace_path)) {
    std::fprintf(stderr, "dolgx_replay: cannot open trace %s\n", trace_path);
    return 2;
  }
  const trace::TraceHeader &header = reader.header();
  char game_id[9] = {};
  std::memcpy(game_id, header.game_id, sizeof header.game_id);
  std::fprintf(stderr, "dolgx_replay: %s game_id=%s mem1=0x%08X version=%u\n",
               trace_path, game_id[0] != '\0' ? game_id : "(unset)",
               header.mem1_size, reader.version());

  Histogram hist;
  std::unique_ptr<ParityWriter> parity;
  if (parity_path != nullptr) {
    parity = std::make_unique<ParityWriter>(parity_path, trace_path,
                                            game_id[0] != '\0' ? game_id : "");
    if (!parity->out) {
      std::fprintf(stderr, "dolgx_replay: cannot open %s\n", parity_path);
      return 2;
    }
  }
  CliObservers observers{
      .histogram = histogram ? &hist : nullptr,
      .parity = parity.get(),
  };
  const replay::ReplayResult result = replay::replay_trace(
      reader, (histogram || parity) ? cli_event_observer : nullptr, &observers,
      parity ? cli_draw_observer : nullptr, parity.get(),
      parity ? cli_mem_update_observer : nullptr, parity.get());
  if (result.truncated)
    std::fprintf(stderr, "dolgx_replay: trace ends mid-record (interrupted "
                         "recording); replayed the complete prefix\n");

  std::vector<std::string> lines;
  lines.reserve(result.frames.size());
  for (const replay::FrameDigest &frame : result.frames)
    lines.push_back(replay::format_digest_line(frame));

  if (!quiet)
    for (const std::string &line : lines)
      std::printf("%s\n", line.c_str());

  if (write_path != nullptr) {
    std::ofstream out(write_path, std::ios::trunc);
    if (!out) {
      std::fprintf(stderr, "dolgx_replay: cannot write %s\n", write_path);
      return 2;
    }
    for (const std::string &line : lines)
      out << line << '\n';
  }

  if (histogram)
    print_histogram(hist);

  if (parity) {
    parity->write_frames(result.frames);
    parity->out.flush();
    if (!parity->out) {
      std::fprintf(stderr, "dolgx_replay: parity JSONL write failed: %s\n",
                   parity_path);
      return 2;
    }
  }

  if (!result.parse_ok) {
    std::fprintf(stderr, "dolgx_replay: FAIL %s\n", result.error.c_str());
    return 1;
  }

  int exit_code = 0;
  if (against_stats) {
    const replay::StatsCompareResult cmp =
        replay::compare_against_stats(result);
    std::fprintf(stderr,
                 "dolgx_replay: against-stats %s frames=%llu mismatches=%llu "
                 "worst_consecutive=%llu%s%s\n",
                 cmp.ok ? "OK" : "FAIL", cmp.frames_compared,
                 cmp.mismatch_frames, cmp.worst_consecutive,
                 cmp.detail.empty() ? "" : " first=", cmp.detail.c_str());
    if (!cmp.ok)
      exit_code = 1;
  }

  if (golden_path != nullptr) {
    std::ifstream golden(golden_path);
    if (!golden) {
      std::fprintf(stderr, "dolgx_replay: cannot open golden %s\n",
                   golden_path);
      return 2;
    }
    std::size_t line_index = 0;
    bool digest_ok = true;
    std::string golden_line;
    while (std::getline(golden, golden_line)) {
      if (line_index >= lines.size()) {
        std::fprintf(stderr,
                     "dolgx_replay: digest FAIL golden has %zu+ lines, "
                     "replay produced %zu\n",
                     line_index + 1u, lines.size());
        digest_ok = false;
        break;
      }
      if (golden_line != lines[line_index]) {
        std::fprintf(stderr,
                     "dolgx_replay: digest FAIL line %zu\n  golden: %s\n"
                     "  replay: %s\n",
                     line_index + 1u, golden_line.c_str(),
                     lines[line_index].c_str());
        digest_ok = false;
        break;
      }
      ++line_index;
    }
    if (digest_ok && line_index != lines.size()) {
      std::fprintf(stderr,
                   "dolgx_replay: digest FAIL golden has %zu lines, replay "
                   "produced %zu\n",
                   line_index, lines.size());
      digest_ok = false;
    }
    if (digest_ok)
      std::fprintf(stderr, "dolgx_replay: digest OK (%zu lines)\n", line_index);
    else
      exit_code = 1;
  }

  return exit_code;
}
