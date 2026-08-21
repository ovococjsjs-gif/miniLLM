#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const bool add_special = llama_vocab_get_add_bos(vocab);
    int32_t count = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, true);
    if (count >= 0) {
        throw std::runtime_error("tokenizer sizing call unexpectedly succeeded");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-count));
    count = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        true);
    if (count < 0) {
        throw std::runtime_error("tokenization failed");
    }
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

static std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t count = llama_token_to_piece(
        vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    if (count < 0) {
        buffer.resize(static_cast<size_t>(-count));
        count = llama_token_to_piece(
            vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    }
    if (count < 0) {
        throw std::runtime_error("token to piece conversion failed");
    }
    return std::string(buffer.data(), static_cast<size_t>(count));
}

static std::string hex_encode(const std::string & value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(value.size() * 2);
    for (const unsigned char byte : value) {
        output.push_back(digits[byte >> 4]);
        output.push_back(digits[byte & 0x0f]);
    }
    return output;
}

static llama_token greedy_token(const std::vector<float> & logits) {
    return static_cast<llama_token>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

static std::vector<float> current_logits(llama_context * context, int32_t vocabulary) {
    const float * values = llama_get_logits_ith(context, -1);
    if (!values) {
        throw std::runtime_error("logits are unavailable");
    }
    return std::vector<float>(values, values + vocabulary);
}

static std::vector<uint8_t> save_sequence(
    llama_context * context, llama_state_seq_flags flags) {
    const size_t size = llama_state_seq_get_size_ext(context, 0, flags);
    if (!size) {
        throw std::runtime_error("sequence state size is zero");
    }
    std::vector<uint8_t> data(size);
    const size_t written = llama_state_seq_get_data_ext(context, data.data(), data.size(), 0, flags);
    if (written != size) {
        throw std::runtime_error("sequence state write was incomplete");
    }
    return data;
}

static void restore_sequence(
    llama_context * context,
    const std::vector<uint8_t> & data,
    llama_state_seq_flags flags) {
    const size_t consumed = llama_state_seq_set_data_ext(
        context, data.data(), data.size(), 0, flags);
    if (consumed != data.size()) {
        throw std::runtime_error("sequence state restore was incomplete");
    }
}

static uint32_t read_u32(const std::vector<uint8_t> & data, size_t offset) {
    if (offset + sizeof(uint32_t) > data.size()) {
        throw std::runtime_error("partial state header is truncated");
    }
    uint32_t value;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

static uint64_t read_u64(const std::vector<uint8_t> & data, size_t offset) {
    if (offset + sizeof(uint64_t) > data.size()) {
        throw std::runtime_error("partial state payload is truncated");
    }
    uint64_t value;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

static size_t recurrent_data_offset(const std::vector<uint8_t> & partial) {
    // Public llama.cpp sequence-state layout: magic, source seq-id, cell count,
    // then per-cell (position, n_seq_id, optional seq ids), followed by recurrent data.
    size_t cursor = 2 * sizeof(uint32_t);
    const uint32_t cell_count = read_u32(partial, cursor);
    cursor += sizeof(uint32_t);
    for (uint32_t cell = 0; cell < cell_count; ++cell) {
        cursor += sizeof(int32_t);  // position
        const uint32_t sequence_count = read_u32(partial, cursor);
        cursor += sizeof(uint32_t);
        cursor += static_cast<size_t>(sequence_count) * sizeof(int32_t);
        if (cursor > partial.size()) {
            throw std::runtime_error("partial state metadata is truncated");
        }
    }
    return cursor;
}

static std::vector<uint8_t> stale_recurrent_state(
    const std::vector<uint8_t> & before,
    const std::vector<uint8_t> & after,
    size_t & before_offset,
    size_t & after_offset) {
    before_offset = recurrent_data_offset(before);
    after_offset = recurrent_data_offset(after);
    if (before.size() - before_offset != after.size() - after_offset) {
        throw std::runtime_error("recurrent state payload sizes differ");
    }
    std::vector<uint8_t> output = after;
    std::copy(before.begin() + static_cast<std::ptrdiff_t>(before_offset), before.end(),
              output.begin() + static_cast<std::ptrdiff_t>(after_offset));
    return output;
}

struct tensor_segment {
    size_t offset;
    size_t bytes;
};

static std::vector<tensor_segment> recurrent_tensor_segments(
    const std::vector<uint8_t> & partial, size_t data_offset) {
    size_t cursor = data_offset;
    const uint32_t state_transposed = read_u32(partial, cursor);
    cursor += sizeof(uint32_t);
    const uint32_t layers = read_u32(partial, cursor);
    cursor += sizeof(uint32_t);
    if (state_transposed != 0 || layers != 24) {
        throw std::runtime_error("unexpected Qwen recurrent serialization header");
    }
    const uint32_t cell_count = read_u32(partial, 2 * sizeof(uint32_t));
    constexpr int recurrent_layers = 18;
    std::vector<tensor_segment> output;
    output.reserve(recurrent_layers * 2);
    for (int tensor = 0; tensor < recurrent_layers * 2; ++tensor) {
        const uint32_t type = read_u32(partial, cursor);
        cursor += sizeof(uint32_t);
        const uint64_t row_size = read_u64(partial, cursor);
        cursor += sizeof(uint64_t);
        const size_t bytes = static_cast<size_t>(row_size) * cell_count;
        if (type != 0 || bytes % sizeof(float) != 0 || cursor + bytes > partial.size()) {
            throw std::runtime_error("unexpected Qwen recurrent tensor encoding");
        }
        output.push_back({cursor, bytes});
        cursor += bytes;
    }
    if (cursor != partial.size()) {
        throw std::runtime_error("unparsed bytes remain in Qwen recurrent state");
    }
    return output;
}

static std::vector<uint8_t> blend_recurrent_state(
    const std::vector<uint8_t> & before,
    const std::vector<uint8_t> & after,
    double alpha) {
    const size_t before_data = recurrent_data_offset(before);
    const size_t after_data = recurrent_data_offset(after);
    const auto before_segments = recurrent_tensor_segments(before, before_data);
    const auto after_segments = recurrent_tensor_segments(after, after_data);
    if (before_segments.size() != after_segments.size()) {
        throw std::runtime_error("recurrent tensor inventories differ");
    }
    std::vector<uint8_t> output = after;
    for (size_t segment = 0; segment < before_segments.size(); ++segment) {
        const auto & left = before_segments[segment];
        const auto & right = after_segments[segment];
        if (left.bytes != right.bytes) {
            throw std::runtime_error("recurrent tensor sizes differ");
        }
        const size_t count = left.bytes / sizeof(float);
        for (size_t index = 0; index < count; ++index) {
            float before_value;
            float after_value;
            std::memcpy(
                &before_value,
                before.data() + left.offset + index * sizeof(float),
                sizeof(float));
            std::memcpy(
                &after_value,
                after.data() + right.offset + index * sizeof(float),
                sizeof(float));
            const float value = static_cast<float>(
                before_value + alpha * (after_value - before_value));
            std::memcpy(
                output.data() + right.offset + index * sizeof(float),
                &value,
                sizeof(float));
        }
    }
    return output;
}

static void decode_one(llama_context * context, llama_token token) {
    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(context, batch) != 0) {
        throw std::runtime_error("single-token decode failed");
    }
}

struct comparison {
    double kl;
    double rms;
    double max_abs;
    bool exact;
    llama_token argmax;
};

static comparison compare_logits(
    const std::vector<float> & oracle, const std::vector<float> & candidate) {
    if (oracle.size() != candidate.size()) {
        throw std::runtime_error("logit vectors differ in size");
    }
    const double oracle_max = *std::max_element(oracle.begin(), oracle.end());
    const double candidate_max = *std::max_element(candidate.begin(), candidate.end());
    double oracle_sum = 0.0;
    double candidate_sum = 0.0;
    double squared = 0.0;
    double max_abs = 0.0;
    for (size_t index = 0; index < oracle.size(); ++index) {
        oracle_sum += std::exp(static_cast<double>(oracle[index]) - oracle_max);
        candidate_sum += std::exp(static_cast<double>(candidate[index]) - candidate_max);
        const double difference = static_cast<double>(candidate[index]) - oracle[index];
        squared += difference * difference;
        max_abs = std::max(max_abs, std::abs(difference));
    }
    const double oracle_log_z = oracle_max + std::log(oracle_sum);
    const double candidate_log_z = candidate_max + std::log(candidate_sum);
    double kl = 0.0;
    for (size_t index = 0; index < oracle.size(); ++index) {
        const double log_p = static_cast<double>(oracle[index]) - oracle_log_z;
        const double log_q = static_cast<double>(candidate[index]) - candidate_log_z;
        const double probability = std::exp(log_p);
        kl += probability * (log_p - log_q);
    }
    return {
        kl,
        std::sqrt(squared / oracle.size()),
        max_abs,
        std::memcmp(oracle.data(), candidate.data(), oracle.size() * sizeof(float)) == 0,
        greedy_token(candidate),
    };
}

static void write_logits(const fs::path & path, const std::vector<float> & values) {
    std::ofstream output(path, std::ios::binary);
    output.write(
        reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write replay logits");
    }
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " MODEL.gguf OUTPUT_DIR PROMPT\n";
        return 2;
    }
    const std::string model_path = argv[1];
    const fs::path output_dir = argv[2];
    const std::string prompt = argv[3];
    fs::create_directories(output_dir);

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("failed to load model");
    }
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_seq_max = 1;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    context_params.offload_kqv = false;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        llama_model_free(model);
        throw std::runtime_error("failed to create context");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t vocabulary = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt);
    llama_batch prompt_batch = llama_batch_get_one(
        prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
    if (llama_decode(context, prompt_batch) != 0) {
        throw std::runtime_error("prompt decode failed");
    }
    const std::vector<uint8_t> partial_before = save_sequence(
        context, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const llama_token first_token = greedy_token(current_logits(context, vocabulary));
    decode_one(context, first_token);

    const std::vector<uint8_t> full_after = save_sequence(
        context, LLAMA_STATE_SEQ_FLAGS_NONE);
    const std::vector<uint8_t> partial_after = save_sequence(
        context, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const llama_token second_token = greedy_token(current_logits(context, vocabulary));
    size_t before_offset = 0;
    size_t after_offset = 0;
    const std::vector<uint8_t> stale_partial = stale_recurrent_state(
        partial_before, partial_after, before_offset, after_offset);

    // Oracle path: consume the same second token from the true post-event cache.
    decode_one(context, second_token);
    const std::vector<float> oracle_logits = current_logits(context, vocabulary);

    // Serialization control: full restore + unchanged partial restore must be exact.
    restore_sequence(context, full_after, LLAMA_STATE_SEQ_FLAGS_NONE);
    restore_sequence(context, partial_after, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    decode_one(context, second_token);
    const std::vector<float> control_logits = current_logits(context, vocabulary);

    // Stale recurrent control: preserve oracle attention KV/position, replace only
    // recurrent and convolution tensors with their pre-event values, then replay.
    restore_sequence(context, full_after, LLAMA_STATE_SEQ_FLAGS_NONE);
    restore_sequence(context, stale_partial, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    decode_one(context, second_token);
    const std::vector<float> stale_logits = current_logits(context, vocabulary);

    std::vector<double> blend_alphas = {0.25, 0.50, 0.75};
    std::vector<std::vector<float>> blend_logits;
    std::vector<comparison> blend_comparisons;
    for (const double alpha : blend_alphas) {
        const auto blended = blend_recurrent_state(partial_before, partial_after, alpha);
        restore_sequence(context, full_after, LLAMA_STATE_SEQ_FLAGS_NONE);
        restore_sequence(context, blended, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        decode_one(context, second_token);
        blend_logits.push_back(current_logits(context, vocabulary));
        blend_comparisons.push_back(compare_logits(oracle_logits, blend_logits.back()));
    }

    const comparison control = compare_logits(oracle_logits, control_logits);
    const comparison stale = compare_logits(oracle_logits, stale_logits);
    write_logits(output_dir / "oracle.logits.f32.bin", oracle_logits);
    write_logits(output_dir / "control.logits.f32.bin", control_logits);
    write_logits(output_dir / "stale.logits.f32.bin", stale_logits);
    for (size_t index = 0; index < blend_logits.size(); ++index) {
        const int percent = static_cast<int>(std::round(blend_alphas[index] * 100));
        write_logits(
            output_dir / ("blend-" + std::to_string(percent) + ".logits.f32.bin"),
            blend_logits[index]);
    }
    std::ofstream prompt_output(output_dir / "prompt.txt");
    prompt_output << prompt;

    std::ofstream metrics(output_dir / "metrics.tsv");
    metrics << std::setprecision(17);
    metrics << "prompt_tokens\t" << prompt_tokens.size() << '\n';
    metrics << "vocabulary\t" << vocabulary << '\n';
    metrics << "first_token\t" << first_token << '\n';
    metrics << "first_piece_hex\t" << hex_encode(token_piece(vocab, first_token)) << '\n';
    metrics << "second_token\t" << second_token << '\n';
    metrics << "second_piece_hex\t" << hex_encode(token_piece(vocab, second_token)) << '\n';
    metrics << "full_after_bytes\t" << full_after.size() << '\n';
    metrics << "partial_before_bytes\t" << partial_before.size() << '\n';
    metrics << "partial_after_bytes\t" << partial_after.size() << '\n';
    metrics << "partial_before_data_offset\t" << before_offset << '\n';
    metrics << "partial_after_data_offset\t" << after_offset << '\n';
    metrics << "control_exact\t" << (control.exact ? 1 : 0) << '\n';
    metrics << "control_kl\t" << control.kl << '\n';
    metrics << "control_rms\t" << control.rms << '\n';
    metrics << "control_max_abs\t" << control.max_abs << '\n';
    metrics << "control_argmax\t" << control.argmax << '\n';
    metrics << "stale_exact\t" << (stale.exact ? 1 : 0) << '\n';
    metrics << "stale_kl\t" << stale.kl << '\n';
    metrics << "stale_rms\t" << stale.rms << '\n';
    metrics << "stale_max_abs\t" << stale.max_abs << '\n';
    metrics << "stale_argmax\t" << stale.argmax << '\n';
    for (size_t index = 0; index < blend_comparisons.size(); ++index) {
        const int percent = static_cast<int>(std::round(blend_alphas[index] * 100));
        const std::string prefix = "blend_" + std::to_string(percent);
        const auto & item = blend_comparisons[index];
        metrics << prefix << "_kl\t" << item.kl << '\n';
        metrics << prefix << "_rms\t" << item.rms << '\n';
        metrics << prefix << "_max_abs\t" << item.max_abs << '\n';
        metrics << prefix << "_argmax\t" << item.argmax << '\n';
    }
    metrics << "oracle_argmax\t" << greedy_token(oracle_logits) << '\n';

    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    std::cout << "stale recurrent KL = " << stale.kl
              << ", serialization control exact = " << control.exact << '\n';
    return control.exact ? 0 : 3;
}
