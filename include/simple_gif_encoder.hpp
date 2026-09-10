/**
 * @file simple_gif_encoder.hpp
 * @brief Small dependency-free GIF89a encoder for indexed animation frames.
 *
 * This private encoder deliberately supports the narrow artifact use case:
 * a global palette, full-size frames, and infinite looping.  Keeping it here
 * avoids making Python, ffmpeg, ImageMagick, or giflib a runtime dependency
 * of reproducible experiment artifacts.
 */

#ifndef DRO_MPC_SIMPLE_GIF_ENCODER_HPP
#define DRO_MPC_SIMPLE_GIF_ENCODER_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace dro_mpc {
namespace detail {
namespace gif {

struct Color {
    std::uint8_t red = 0;
    std::uint8_t green = 0;
    std::uint8_t blue = 0;
};

namespace internal {

inline void write_u16(std::ostream& out, std::uint16_t value) {
    out.put(static_cast<char>(value & 0xffu));
    out.put(static_cast<char>((value >> 8u) & 0xffu));
}

class BitWriter {
public:
    void write(std::uint16_t code, int bit_count) {
        bits_ |= static_cast<std::uint32_t>(code) << pending_bits_;
        pending_bits_ += bit_count;
        while (pending_bits_ >= 8) {
            bytes_.push_back(static_cast<std::uint8_t>(bits_ & 0xffu));
            bits_ >>= 8u;
            pending_bits_ -= 8;
        }
    }

    std::vector<std::uint8_t> finish() {
        if (pending_bits_ > 0) {
            bytes_.push_back(static_cast<std::uint8_t>(bits_ & 0xffu));
            bits_ = 0;
            pending_bits_ = 0;
        }
        return std::move(bytes_);
    }

private:
    std::vector<std::uint8_t> bytes_;
    std::uint32_t bits_ = 0;
    int pending_bits_ = 0;
};

inline std::vector<std::uint8_t> lzw_encode(
    const std::vector<std::uint8_t>& pixels, int minimum_code_size
) {
    if (pixels.empty()) return {};

    const int clear_code = 1 << minimum_code_size;
    const int end_code = clear_code + 1;
    constexpr int max_code = 4096;
    int code_size = minimum_code_size + 1;
    int next_code = end_code + 1;
    std::unordered_map<std::uint32_t, int> dictionary;
    dictionary.reserve(4096);
    BitWriter writer;

    const auto reset_dictionary = [&]() {
        dictionary.clear();
        code_size = minimum_code_size + 1;
        next_code = end_code + 1;
    };

    writer.write(static_cast<std::uint16_t>(clear_code), code_size);
    reset_dictionary();
    int prefix = pixels.front();
    for (std::size_t index = 1; index < pixels.size(); ++index) {
        const int suffix = pixels[index];
        const std::uint32_t key = (static_cast<std::uint32_t>(prefix) << 8u) |
                                  static_cast<std::uint32_t>(suffix);
        const auto found = dictionary.find(key);
        if (found != dictionary.end()) {
            prefix = found->second;
            continue;
        }

        writer.write(static_cast<std::uint16_t>(prefix), code_size);
        if (next_code < max_code) {
            const int inserted_code = next_code++;
            dictionary.emplace(key, inserted_code);
            // GIF's decoder adds the corresponding table entry after it
            // reads this emitted prefix. The code width therefore changes
            // when the first *new* code no longer fits, rather than when the
            // last old-width code is assigned.
            if (inserted_code == (1 << code_size) && code_size < 12) ++code_size;
        } else {
            writer.write(static_cast<std::uint16_t>(clear_code), code_size);
            reset_dictionary();
        }
        prefix = suffix;
    }
    writer.write(static_cast<std::uint16_t>(prefix), code_size);
    writer.write(static_cast<std::uint16_t>(end_code), code_size);
    return writer.finish();
}

inline void write_sub_blocks(std::ostream& out, const std::vector<std::uint8_t>& data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const std::size_t count = std::min<std::size_t>(255, data.size() - offset);
        out.put(static_cast<char>(count));
        out.write(reinterpret_cast<const char*>(data.data() + offset),
                  static_cast<std::streamsize>(count));
        offset += count;
    }
    out.put('\0');
}

}  // namespace internal

template <std::size_t PaletteSize>
class IndexedAnimationWriter {
public:
    IndexedAnimationWriter(std::ostream& out, std::uint16_t width, std::uint16_t height,
                           const std::array<Color, PaletteSize>& palette)
        : out_(out), width_(width), height_(height) {
        static_assert(PaletteSize >= 2 && PaletteSize <= 256,
                      "GIF palettes must contain between 2 and 256 colors");
        static_assert((PaletteSize & (PaletteSize - 1)) == 0,
                      "GIF palettes must have a power-of-two number of colors");
        if (width == 0 || height == 0) {
            throw std::invalid_argument("GIF dimensions must be non-zero");
        }

        int table_bits = 0;
        for (std::size_t count = PaletteSize; count > 2; count >>= 1u) ++table_bits;
        const int color_resolution = table_bits;
        const std::uint8_t packed = static_cast<std::uint8_t>(
            0x80u | (static_cast<std::uint8_t>(color_resolution) << 4u) |
            static_cast<std::uint8_t>(table_bits));

        out_.write("GIF89a", 6);
        internal::write_u16(out_, width_);
        internal::write_u16(out_, height_);
        out_.put(static_cast<char>(packed));
        out_.put('\0');  // background palette index
        out_.put('\0');  // pixel aspect ratio
        for (const auto& color : palette) {
            out_.put(static_cast<char>(color.red));
            out_.put(static_cast<char>(color.green));
            out_.put(static_cast<char>(color.blue));
        }

        // NETSCAPE2.0 application extension: loop forever.
        out_.put(static_cast<char>(0x21));
        out_.put(static_cast<char>(0xff));
        out_.put(static_cast<char>(0x0b));
        out_.write("NETSCAPE2.0", 11);
        out_.put(static_cast<char>(0x03));
        out_.put(static_cast<char>(0x01));
        internal::write_u16(out_, 0);
        out_.put('\0');
    }

    void write_frame(const std::vector<std::uint8_t>& pixels,
                     std::uint16_t delay_centiseconds) {
        if (finished_) throw std::logic_error("cannot append a GIF frame after finish");
        if (pixels.size() != static_cast<std::size_t>(width_) * height_) {
            throw std::invalid_argument("GIF frame does not match the logical screen size");
        }
        for (const std::uint8_t pixel : pixels) {
            if (pixel >= PaletteSize) {
                throw std::invalid_argument("GIF frame contains an out-of-palette index");
            }
        }

        // Graphic-control extension with no transparency or disposal override.
        out_.put(static_cast<char>(0x21));
        out_.put(static_cast<char>(0xf9));
        out_.put(static_cast<char>(0x04));
        out_.put('\0');
        internal::write_u16(out_, delay_centiseconds);
        out_.put('\0');
        out_.put('\0');

        out_.put(static_cast<char>(0x2c));
        internal::write_u16(out_, 0);
        internal::write_u16(out_, 0);
        internal::write_u16(out_, width_);
        internal::write_u16(out_, height_);
        out_.put('\0');  // global color table, non-interlaced

        int minimum_code_size = 1;
        while ((1 << minimum_code_size) < static_cast<int>(PaletteSize)) {
            ++minimum_code_size;
        }
        minimum_code_size = std::max(2, minimum_code_size);
        out_.put(static_cast<char>(minimum_code_size));
        internal::write_sub_blocks(out_, internal::lzw_encode(pixels, minimum_code_size));
    }

    void finish() {
        if (finished_) return;
        out_.put(static_cast<char>(0x3b));
        finished_ = true;
        if (!out_) throw std::runtime_error("failed while writing animated GIF");
    }

private:
    std::ostream& out_;
    std::uint16_t width_ = 0;
    std::uint16_t height_ = 0;
    bool finished_ = false;
};

}  // namespace gif
}  // namespace detail
}  // namespace dro_mpc

#endif  // DRO_MPC_SIMPLE_GIF_ENCODER_HPP
