#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct dataset_row {
    std::string id;
    std::string prompt;
    std::string answer;
};

struct hidden_capture {
    std::vector<float> values;
    int64_t embedding = 0;
    int64_t tokens = 0;
};

struct output_adapter {
    uint32_t embedding = 0;
    uint32_t hidden = 0;
    uint32_t candidates = 0;
    uint32_t gate_hidden = 0;
    float gain = 1.0F;
    float gate_threshold = 0.5F;
    std::vector<int32_t> token_ids;
    std::vector<float> mean;
    std::vector<float> scale;
    std::vector<float> weight1;
    std::vector<float> bias1;
    std::vector<float> weight2;
    std::vector<float> bias2;
    std::vector<float> gate_weight1;
    std::vector<float> gate_bias1;
    std::vector<float> gate_weight2;
    float gate_bias2 = 0.0F;
};

static std::vector<uint8_t> read_logical_tensor(ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error("h_pre_norm is not f32");
    }
    if (ggml_is_contiguous(tensor)) {
        std::vector<uint8_t> data(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
        return data;
    }
    const size_t row_bytes = static_cast<size_t>(tensor->ne[0]) * sizeof(float);
    const size_t row_count = static_cast<size_t>(ggml_nelements(tensor) / tensor->ne[0]);
    std::vector<uint8_t> data(row_bytes * row_count);
    const int dimensions = ggml_n_dims(tensor);
    for (size_t row = 0; row < row_count; ++row) {
        size_t remainder = row;
        size_t source_offset = 0;
        for (int dimension = 1; dimension < dimensions; ++dimension) {
            const size_t index = remainder % static_cast<size_t>(tensor->ne[dimension]);
            remainder /= static_cast<size_t>(tensor->ne[dimension]);
            source_offset += index * tensor->nb[dimension];
        }
        ggml_backend_tensor_get(
            tensor, data.data() + row * row_bytes, source_offset, row_bytes);
    }
    return data;
}

static bool capture_hidden(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<hidden_capture *>(user_data);
    const bool wanted = std::string(tensor->name) == "h_pre_norm";
    if (ask) {
        return wanted;
    }
    if (!wanted) {
        return true;
    }
    const auto bytes = read_logical_tensor(tensor);
    capture->embedding = tensor->ne[0];
    capture->tokens = ggml_nelements(tensor) / tensor->ne[0];
    capture->values.resize(bytes.size() / sizeof(float));
    std::memcpy(capture->values.data(), bytes.data(), bytes.size());
    return true;
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

static int hex_digit(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    throw std::runtime_error("invalid hexadecimal input");
}

static std::string hex_decode(const std::string & value) {
    if (value.size() % 2 != 0) {
        throw std::runtime_error("hexadecimal input has odd length");
    }
    std::string output(value.size() / 2, '\0');
    for (size_t index = 0; index < output.size(); ++index) {
        output[index] = static_cast<char>(
            (hex_digit(value[2 * index]) << 4) | hex_digit(value[2 * index + 1]));
    }
    return output;
}

static std::vector<dataset_row> read_dataset(const fs::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open dataset " + path.string());
    }
    std::vector<dataset_row> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const size_t first = line.find('\t');
        const size_t second = first == std::string::npos ? first : line.find('\t', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
            throw std::runtime_error("dataset row must have id, prompt hex, and answer hex");
        }
        rows.push_back({
            line.substr(0, first),
            hex_decode(line.substr(first + 1, second - first - 1)),
            hex_decode(line.substr(second + 1)),
        });
    }
    if (rows.empty()) {
        throw std::runtime_error("dataset is empty");
    }
    return rows;
}

static std::vector<llama_token> tokenize(
    const llama_vocab * vocab, const std::string & text, bool add_special) {
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

static std::string format_prompt(const llama_model * model, const std::string & prompt) {
    const std::string system =
        "You are a concise, accurate bilingual assistant. Answer directly without hidden reasoning.";
    const std::string user = prompt + "\n/no_think";
    const llama_chat_message messages[] = {
        {"system", system.c_str()},
        {"user", user.c_str()},
    };
    const char * chat_template = llama_model_chat_template(model, nullptr);
    int32_t count = llama_chat_apply_template(chat_template, messages, 2, true, nullptr, 0);
    if (count < 0) {
        throw std::runtime_error("chat template sizing failed");
    }
    std::vector<char> output(static_cast<size_t>(count) + 1);
    count = llama_chat_apply_template(
        chat_template, messages, 2, true, output.data(), static_cast<int32_t>(output.size()));
    if (count < 0 || static_cast<size_t>(count) >= output.size()) {
        throw std::runtime_error("chat template application failed");
    }
    return std::string(output.data(), static_cast<size_t>(count));
}

template <typename T>
static void write_values(std::ofstream & output, const T * values, size_t count) {
    output.write(
        reinterpret_cast<const char *>(values),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!output) {
        throw std::runtime_error("binary output failed");
    }
}

template <typename T>
static std::vector<T> read_vector(std::ifstream & input, size_t count) {
    std::vector<T> values(count);
    input.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) {
        throw std::runtime_error("adapter file is truncated");
    }
    return values;
}

static std::vector<int32_t> read_candidates(const fs::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open candidates " + path.string());
    }
    const auto bytes = input.tellg();
    if (bytes <= 0 || static_cast<size_t>(bytes) % sizeof(int32_t) != 0) {
        throw std::runtime_error("candidate token file has invalid size");
    }
    input.seekg(0);
    return read_vector<int32_t>(input, static_cast<size_t>(bytes) / sizeof(int32_t));
}

static void write_candidates(const fs::path & path, const std::vector<int32_t> & values) {
    std::ofstream output(path, std::ios::binary);
    write_values(output, values.data(), values.size());
}

static llama_batch all_output_batch(const std::vector<llama_token> & tokens) {
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t index = 0; index < batch.n_tokens; ++index) {
        batch.token[index] = tokens[static_cast<size_t>(index)];
        batch.pos[index] = index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = 1;
    }
    return batch;
}

static float logaddexp(float left, float right) {
    if (left == -std::numeric_limits<float>::infinity()) {
        return right;
    }
    const float maximum = std::max(left, right);
    return maximum + std::log(std::exp(left - maximum) + std::exp(right - maximum));
}

static int collect_mode(
    const std::string & model_path,
    const fs::path & dataset_path,
    const fs::path & output_dir,
    const fs::path * candidates_path) {
    fs::create_directories(output_dir);
    const auto rows = read_dataset(dataset_path);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("failed to load model");
    }
    hidden_capture capture;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 1024;
    context_params.n_batch = 1024;
    context_params.n_ubatch = 1024;
    context_params.n_seq_max = 1;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    context_params.cb_eval = capture_hidden;
    context_params.cb_eval_user_data = &capture;
    context_params.offload_kqv = false;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        llama_model_free(model);
        throw std::runtime_error("failed to create context");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t vocabulary = llama_vocab_n_tokens(vocab);
    std::vector<std::vector<llama_token>> answer_tokens;
    answer_tokens.reserve(rows.size());
    std::set<int32_t> candidate_set;
    for (const auto & row : rows) {
        auto tokens = tokenize(vocab, row.answer, false);
        tokens.push_back(llama_vocab_eos(vocab));
        for (llama_token token : tokens) {
            candidate_set.insert(token);
        }
        answer_tokens.push_back(std::move(tokens));
    }
    std::vector<int32_t> candidates;
    if (candidates_path) {
        candidates = read_candidates(*candidates_path);
        for (int32_t target : candidate_set) {
            if (std::find(candidates.begin(), candidates.end(), target) == candidates.end()) {
                throw std::runtime_error("validation target is absent from train candidates");
            }
        }
    } else {
        candidates.assign(candidate_set.begin(), candidate_set.end());
    }
    write_candidates(output_dir / "candidates.i32.bin", candidates);
    std::vector<uint8_t> is_candidate(static_cast<size_t>(vocabulary));
    for (int32_t token : candidates) {
        if (token < 0 || token >= vocabulary) {
            throw std::runtime_error("candidate token is outside vocabulary");
        }
        is_candidate[static_cast<size_t>(token)] = 1;
    }

    std::ofstream hidden_output(output_dir / "hidden.f32.bin", std::ios::binary);
    std::ofstream target_output(output_dir / "targets.i32.bin", std::ios::binary);
    std::ofstream base_output(output_dir / "base_candidate_logits.f32.bin", std::ios::binary);
    std::ofstream other_lse_output(output_dir / "other_logsumexp.f32.bin", std::ios::binary);
    std::ofstream other_max_output(output_dir / "other_max.f32.bin", std::ios::binary);
    std::ofstream index_output(output_dir / "index.tsv");
    index_output << "sample_id\tid\toffset\ttokens\tprompt_tokens\n";
    size_t offset = 0;
    for (size_t sample = 0; sample < rows.size(); ++sample) {
        llama_memory_clear(llama_get_memory(context), true);
        capture = {};
        const std::string formatted = format_prompt(model, rows[sample].prompt);
        auto prompt_tokens = tokenize(vocab, formatted, llama_vocab_get_add_bos(vocab));
        std::vector<llama_token> sequence = prompt_tokens;
        sequence.insert(sequence.end(), answer_tokens[sample].begin(), answer_tokens[sample].end() - 1);
        if (sequence.empty() || sequence.size() > context_params.n_batch) {
            throw std::runtime_error("teacher-forced sequence is outside context limits");
        }
        llama_batch batch = all_output_batch(sequence);
        const int decode_status = llama_decode(context, batch);
        llama_batch_free(batch);
        if (decode_status != 0) {
            throw std::runtime_error("teacher-forced decode failed");
        }
        if (capture.embedding != llama_model_n_embd(model) ||
            capture.tokens != static_cast<int64_t>(sequence.size())) {
            throw std::runtime_error("captured hidden tensor has unexpected shape");
        }
        const size_t count = answer_tokens[sample].size();
        const size_t first_hidden = prompt_tokens.size() - 1;
        for (size_t target_index = 0; target_index < count; ++target_index) {
            const size_t hidden_index = first_hidden + target_index;
            const float * hidden = capture.values.data() + hidden_index * capture.embedding;
            write_values(hidden_output, hidden, static_cast<size_t>(capture.embedding));
            const int32_t target = answer_tokens[sample][target_index];
            write_values(target_output, &target, 1);
            const float * logits = llama_get_logits_ith(context, static_cast<int32_t>(hidden_index));
            if (!logits) {
                throw std::runtime_error("per-token logits are unavailable");
            }
            float other_lse = -std::numeric_limits<float>::infinity();
            float other_max = -std::numeric_limits<float>::infinity();
            for (int32_t token = 0; token < vocabulary; ++token) {
                if (!is_candidate[static_cast<size_t>(token)]) {
                    other_lse = logaddexp(other_lse, logits[token]);
                    other_max = std::max(other_max, logits[token]);
                }
            }
            for (int32_t token : candidates) {
                write_values(base_output, logits + token, 1);
            }
            write_values(other_lse_output, &other_lse, 1);
            write_values(other_max_output, &other_max, 1);
        }
        index_output << sample << '\t' << rows[sample].id << '\t' << offset << '\t'
                     << count << '\t' << prompt_tokens.size() << '\n';
        offset += count;
    }
    std::ofstream summary(output_dir / "summary.tsv");
    summary << "records\t" << rows.size() << '\n';
    summary << "tokens\t" << offset << '\n';
    summary << "embedding\t" << llama_model_n_embd(model) << '\n';
    summary << "candidates\t" << candidates.size() << '\n';
    summary << "vocabulary\t" << vocabulary << '\n';
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    std::cout << "collected " << offset << " supervised tokens from " << rows.size()
              << " records\n";
    return 0;
}

static output_adapter read_adapter(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open adapter " + path.string());
    }
    char magic[8];
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, "AIRAODA2", 8) != 0) {
        throw std::runtime_error("invalid output adapter magic");
    }
    output_adapter adapter;
    input.read(reinterpret_cast<char *>(&adapter.embedding), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&adapter.hidden), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&adapter.candidates), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&adapter.gate_hidden), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&adapter.gain), sizeof(float));
    input.read(reinterpret_cast<char *>(&adapter.gate_threshold), sizeof(float));
    if (!input || !adapter.embedding || !adapter.hidden || !adapter.candidates ||
        !adapter.gate_hidden) {
        throw std::runtime_error("invalid output adapter dimensions");
    }
    adapter.token_ids = read_vector<int32_t>(input, adapter.candidates);
    adapter.mean = read_vector<float>(input, adapter.embedding);
    adapter.scale = read_vector<float>(input, adapter.embedding);
    adapter.weight1 = read_vector<float>(
        input, static_cast<size_t>(adapter.hidden) * adapter.embedding);
    adapter.bias1 = read_vector<float>(input, adapter.hidden);
    adapter.weight2 = read_vector<float>(
        input, static_cast<size_t>(adapter.candidates) * adapter.hidden);
    adapter.bias2 = read_vector<float>(input, adapter.candidates);
    adapter.gate_weight1 = read_vector<float>(
        input, static_cast<size_t>(adapter.gate_hidden) * adapter.embedding);
    adapter.gate_bias1 = read_vector<float>(input, adapter.gate_hidden);
    adapter.gate_weight2 = read_vector<float>(input, adapter.gate_hidden);
    input.read(reinterpret_cast<char *>(&adapter.gate_bias2), sizeof(float));
    if (!input) {
        throw std::runtime_error("adapter gate is truncated");
    }
    return adapter;
}

static float normalized_hidden(
    const float * hidden, const output_adapter & adapter, uint32_t index) {
    return (hidden[index] - adapter.mean[index]) * adapter.scale[index];
}

static float gate_probability(const float * hidden, const output_adapter & adapter) {
    std::vector<float> activation(adapter.gate_hidden);
    for (uint32_t row = 0; row < adapter.gate_hidden; ++row) {
        float value = adapter.gate_bias1[row];
        const float * weight =
            adapter.gate_weight1.data() + static_cast<size_t>(row) * adapter.embedding;
        for (uint32_t column = 0; column < adapter.embedding; ++column) {
            value += weight[column] * normalized_hidden(hidden, adapter, column);
        }
        activation[row] = std::tanh(value);
    }
    float value = adapter.gate_bias2;
    for (uint32_t column = 0; column < adapter.gate_hidden; ++column) {
        value += adapter.gate_weight2[column] * activation[column];
    }
    return 1.0F / (1.0F + std::exp(-value));
}

static void apply_adapter(
    float * logits, const float * hidden, const output_adapter & adapter) {
    std::vector<float> activation(adapter.hidden);
    for (uint32_t row = 0; row < adapter.hidden; ++row) {
        float value = adapter.bias1[row];
        const float * weight = adapter.weight1.data() + static_cast<size_t>(row) * adapter.embedding;
        for (uint32_t column = 0; column < adapter.embedding; ++column) {
            value += weight[column] * normalized_hidden(hidden, adapter, column);
        }
        activation[row] = value / (1.0F + std::exp(-value));
    }
    for (uint32_t row = 0; row < adapter.candidates; ++row) {
        float value = adapter.bias2[row];
        const float * weight = adapter.weight2.data() + static_cast<size_t>(row) * adapter.hidden;
        for (uint32_t column = 0; column < adapter.hidden; ++column) {
            value += weight[column] * activation[column];
        }
        logits[adapter.token_ids[row]] += adapter.gain * value;
    }
}

static llama_token greedy_token(const float * logits, int32_t vocabulary) {
    return static_cast<llama_token>(
        std::max_element(logits, logits + vocabulary) - logits);
}

static int generate_mode(
    const std::string & model_path,
    const fs::path & dataset_path,
    const fs::path & output_path,
    const fs::path * adapter_path,
    int max_tokens) {
    const auto rows = read_dataset(dataset_path);
    output_adapter adapter;
    if (adapter_path) {
        adapter = read_adapter(*adapter_path);
    }
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("failed to load model");
    }
    hidden_capture capture;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 1024;
    context_params.n_batch = 1024;
    context_params.n_ubatch = 1024;
    context_params.n_seq_max = 1;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    context_params.cb_eval = capture_hidden;
    context_params.cb_eval_user_data = &capture;
    context_params.offload_kqv = false;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        llama_model_free(model);
        throw std::runtime_error("failed to create context");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t vocabulary = llama_vocab_n_tokens(vocab);
    if (adapter_path && adapter.embedding != static_cast<uint32_t>(llama_model_n_embd(model))) {
        throw std::runtime_error("adapter embedding dimension does not match model");
    }
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("cannot open generation output");
    }
    output << "id\tanswer_hex\ttokens\tgate_probability\tadapter_active\n";
    for (const auto & row : rows) {
        llama_memory_clear(llama_get_memory(context), true);
        capture = {};
        const std::string formatted = format_prompt(model, row.prompt);
        auto prompt_tokens = tokenize(vocab, formatted, llama_vocab_get_add_bos(vocab));
        llama_batch prompt_batch = llama_batch_get_one(
            prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
        if (llama_decode(context, prompt_batch) != 0) {
            throw std::runtime_error("prompt decode failed");
        }
        float prompt_gate = 0.0F;
        bool adapter_active = false;
        if (adapter_path) {
            if (capture.tokens < 1 || capture.embedding != llama_model_n_embd(model)) {
                throw std::runtime_error("prompt hidden state unavailable for adapter gate");
            }
            const float * prompt_hidden = capture.values.data() +
                static_cast<size_t>(capture.tokens - 1) * capture.embedding;
            prompt_gate = gate_probability(prompt_hidden, adapter);
            adapter_active = prompt_gate >= adapter.gate_threshold;
        }
        std::string generated;
        int emitted = 0;
        for (; emitted < max_tokens; ++emitted) {
            float * logits = llama_get_logits_ith(context, -1);
            if (!logits || capture.tokens < 1 ||
                capture.embedding != llama_model_n_embd(model)) {
                throw std::runtime_error("generation logits or hidden state unavailable");
            }
            const float * hidden = capture.values.data() +
                static_cast<size_t>(capture.tokens - 1) * capture.embedding;
            if (adapter_active) {
                apply_adapter(logits, hidden, adapter);
            }
            const llama_token selected = greedy_token(logits, vocabulary);
            if (llama_vocab_is_eog(vocab, selected)) {
                break;
            }
            generated += token_piece(vocab, selected);
            capture = {};
            llama_token token = selected;
            llama_batch next = llama_batch_get_one(&token, 1);
            if (llama_decode(context, next) != 0) {
                throw std::runtime_error("generation decode failed");
            }
        }
        output << row.id << '\t' << hex_encode(generated) << '\t' << emitted << '\t'
               << prompt_gate << '\t' << (adapter_active ? 1 : 0) << '\n';
        output.flush();
        std::cout << row.id << " [gate=" << prompt_gate
                  << ", active=" << adapter_active << "]: " << generated << '\n';
    }
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    try {
        if (argc >= 5 && std::string(argv[1]) == "collect") {
            const fs::path * candidates = nullptr;
            fs::path candidate_storage;
            if (argc == 6) {
                candidate_storage = argv[5];
                candidates = &candidate_storage;
            } else if (argc != 5) {
                throw std::runtime_error(
                    "usage: qwen35-output-adapter collect MODEL DATA.tsv OUTPUT_DIR [CANDIDATES]");
            }
            return collect_mode(argv[2], argv[3], argv[4], candidates);
        }
        if (argc >= 5 && std::string(argv[1]) == "generate") {
            const fs::path * adapter = nullptr;
            fs::path adapter_storage;
            int max_tokens = 128;
            if (argc >= 6 && std::string(argv[5]) != "-") {
                adapter_storage = argv[5];
                adapter = &adapter_storage;
            }
            if (argc >= 7) {
                max_tokens = std::stoi(argv[6]);
            }
            if (argc < 5 || argc > 7 || max_tokens < 1 || max_tokens > 512) {
                throw std::runtime_error(
                    "usage: qwen35-output-adapter generate MODEL DATA.tsv OUTPUT.tsv [ADAPTER|-] [MAX_TOKENS]");
            }
            return generate_mode(argv[2], argv[3], argv[4], adapter, max_tokens);
        }
        std::cerr << "expected collect or generate mode\n";
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
