#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct capture_row {
    int stage;
    std::string name;
    std::string type;
    std::string shape;
    size_t elements;
    size_t bytes;
    size_t storage_bytes;
    bool contiguous;
    std::string file;
    double minimum;
    double maximum;
    double mean;
    double l2;
};

struct probe_state {
    fs::path output_dir;
    int stage = -1;
    std::vector<capture_row> captures;
};

static bool starts_with(const std::string & value, const std::string & prefix) {
    return value.rfind(prefix, 0) == 0;
}

static bool is_layer_tensor(const std::string & name, const std::string & base) {
    const std::string prefix = base + "-";
    if (!starts_with(name, prefix) || name.size() == prefix.size()) {
        return false;
    }
    return std::all_of(
        name.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
        name.end(),
        [](char value) { return value >= '0' && value <= '9'; });
}

static bool wanted_tensor(const std::string & name) {
    return is_layer_tensor(name, "state_predelta") ||
           is_layer_tensor(name, "new_state") ||
           is_layer_tensor(name, "conv_states") ||
           is_layer_tensor(name, "last_conv_states") ||
           is_layer_tensor(name, "l_out") || name == "h_pre_norm";
}

static std::string shape_string(const ggml_tensor * tensor) {
    std::ostringstream stream;
    const int dimensions = ggml_n_dims(tensor);
    for (int index = 0; index < dimensions; ++index) {
        if (index) {
            stream << 'x';
        }
        stream << tensor->ne[index];
    }
    return stream.str();
}

static std::string safe_name(std::string name) {
    for (char & value : name) {
        if (!(value >= 'A' && value <= 'Z') &&
            !(value >= 'a' && value <= 'z') &&
            !(value >= '0' && value <= '9') && value != '-' && value != '_') {
            value = '_';
        }
    }
    return name;
}

static std::vector<uint8_t> read_logical_tensor(ggml_tensor * tensor) {
    const size_t element_size = ggml_type_size(tensor->type);
    if (ggml_is_contiguous(tensor)) {
        std::vector<uint8_t> data(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
        return data;
    }
    if (ggml_is_quantized(tensor->type)) {
        throw std::runtime_error("non-contiguous quantized captures are unsupported");
    }

    const size_t row_bytes = static_cast<size_t>(tensor->ne[0]) * element_size;
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

static bool capture_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = static_cast<probe_state *>(user_data);
    const std::string name = tensor->name;
    const bool wanted = state->stage >= 0 && wanted_tensor(name);
    if (ask) {
        return wanted;
    }
    if (!wanted) {
        return true;
    }

    const size_t storage_bytes = ggml_nbytes(tensor);
    std::vector<uint8_t> data = read_logical_tensor(tensor);
    const size_t bytes = data.size();

    const std::string filename =
        "stage-" + std::to_string(state->stage) + "." + safe_name(name) + ".bin";
    const fs::path output = state->output_dir / filename;
    std::ofstream handle(output, std::ios::binary);
    if (!handle) {
        throw std::runtime_error("cannot write tensor capture " + output.string());
    }
    handle.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!handle) {
        throw std::runtime_error("failed while writing tensor capture " + output.string());
    }

    double minimum = std::numeric_limits<double>::quiet_NaN();
    double maximum = std::numeric_limits<double>::quiet_NaN();
    double mean = std::numeric_limits<double>::quiet_NaN();
    double l2 = std::numeric_limits<double>::quiet_NaN();
    if (tensor->type == GGML_TYPE_F32 && bytes == ggml_nelements(tensor) * sizeof(float)) {
        const auto * values = reinterpret_cast<const float *>(data.data());
        const size_t count = ggml_nelements(tensor);
        minimum = std::numeric_limits<double>::infinity();
        maximum = -std::numeric_limits<double>::infinity();
        double sum = 0.0;
        double squared = 0.0;
        for (size_t index = 0; index < count; ++index) {
            const double value = values[index];
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            sum += value;
            squared += value * value;
        }
        mean = count ? sum / static_cast<double>(count) : 0.0;
        l2 = std::sqrt(squared);
    }

    state->captures.push_back({
        state->stage,
        name,
        ggml_type_name(tensor->type),
        shape_string(tensor),
        static_cast<size_t>(ggml_nelements(tensor)),
        bytes,
        storage_bytes,
        ggml_is_contiguous(tensor),
        filename,
        minimum,
        maximum,
        mean,
        l2,
    });
    return true;
}

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

static void write_logits(
    const fs::path & output_dir,
    int stage,
    const float * logits,
    int32_t vocabulary,
    const llama_vocab * vocab,
    std::ofstream & token_log,
    llama_token & selected_token) {
    const std::string filename = "stage-" + std::to_string(stage) + ".logits.f32.bin";
    std::ofstream output(output_dir / filename, std::ios::binary);
    output.write(
        reinterpret_cast<const char *>(logits),
        static_cast<std::streamsize>(static_cast<size_t>(vocabulary) * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write logits");
    }

    std::vector<int32_t> indices(static_cast<size_t>(vocabulary));
    std::iota(indices.begin(), indices.end(), 0);
    constexpr size_t top_k = 10;
    std::partial_sort(
        indices.begin(),
        indices.begin() + top_k,
        indices.end(),
        [&](int32_t left, int32_t right) { return logits[left] > logits[right]; });
    selected_token = indices[0];
    token_log << stage << '\t' << selected_token << '\t'
              << hex_encode(token_piece(vocab, selected_token)) << '\t' << filename;
    token_log << std::setprecision(9);
    for (size_t rank = 0; rank < top_k; ++rank) {
        token_log << '\t' << indices[rank] << ':' << logits[indices[rank]];
    }
    token_log << '\n';
    token_log.flush();
}

static void write_capture_manifest(const probe_state & state) {
    std::ofstream output(state.output_dir / "captures.tsv");
    output << "stage\tname\ttype\tshape\telements\tbytes\tstorage_bytes\tcontiguous\tfile\tmin\tmax\tmean\tl2\n";
    output << std::setprecision(17);
    for (const auto & row : state.captures) {
        output << row.stage << '\t' << row.name << '\t' << row.type << '\t' << row.shape << '\t'
               << row.elements << '\t' << row.bytes << '\t' << row.storage_bytes << '\t'
               << (row.contiguous ? 1 : 0) << '\t' << row.file << '\t' << row.minimum << '\t'
               << row.maximum << '\t' << row.mean << '\t'
               << row.l2 << '\n';
    }
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL.gguf OUTPUT_DIR PROMPT [CONTINUATION_TOKENS]\n";
        return 2;
    }
    const std::string model_path = argv[1];
    const fs::path output_dir = argv[2];
    const std::string prompt = argv[3];
    const int continuation_tokens = argc == 5 ? std::stoi(argv[4]) : 1;
    if (continuation_tokens < 0 || continuation_tokens > 32) {
        throw std::invalid_argument("continuation token count must be between 0 and 32");
    }
    fs::create_directories(output_dir);

    probe_state state;
    state.output_dir = output_dir;

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
    context_params.cb_eval = capture_callback;
    context_params.cb_eval_user_data = &state;
    context_params.embeddings = false;
    context_params.no_perf = false;
    context_params.offload_kqv = false;

    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        llama_model_free(model);
        throw std::runtime_error("failed to create context");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt);
    if (prompt_tokens.empty() || prompt_tokens.size() > context_params.n_batch) {
        throw std::runtime_error("prompt token count is outside probe limits");
    }

    std::ofstream token_log(output_dir / "tokens.tsv");
    token_log << "stage\tselected_token\tselected_piece_hex\tlogits_file\ttop10_token_logit\n";
    std::ofstream prompt_log(output_dir / "prompt.txt");
    prompt_log << prompt;
    std::ofstream input_tokens(output_dir / "prompt_tokens.tsv");
    input_tokens << "index\ttoken\tpiece_hex\n";
    for (size_t index = 0; index < prompt_tokens.size(); ++index) {
        input_tokens << index << '\t' << prompt_tokens[index] << '\t'
                     << hex_encode(token_piece(vocab, prompt_tokens[index])) << '\n';
    }

    state.stage = 0;
    llama_batch prompt_batch = llama_batch_get_one(
        prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
    if (llama_decode(context, prompt_batch) != 0) {
        throw std::runtime_error("prompt decode failed");
    }

    const int32_t vocabulary = llama_vocab_n_tokens(vocab);
    llama_token selected = -1;
    const float * logits = llama_get_logits_ith(context, -1);
    if (!logits) {
        throw std::runtime_error("prompt logits are unavailable");
    }
    write_logits(output_dir, 0, logits, vocabulary, vocab, token_log, selected);

    std::string generated;
    for (int stage = 1; stage <= continuation_tokens; ++stage) {
        generated += token_piece(vocab, selected);
        state.stage = stage;
        llama_token current = selected;
        llama_batch next_batch = llama_batch_get_one(&current, 1);
        if (llama_decode(context, next_batch) != 0) {
            throw std::runtime_error("continuation decode failed");
        }
        logits = llama_get_logits_ith(context, -1);
        if (!logits) {
            throw std::runtime_error("continuation logits are unavailable");
        }
        write_logits(output_dir, stage, logits, vocabulary, vocab, token_log, selected);
    }

    std::ofstream summary(output_dir / "run.tsv");
    char model_description[256] = {};
    llama_model_desc(model, model_description, sizeof(model_description));
    summary << "model\t" << model_description << '\n';
    summary << "prompt_tokens\t" << prompt_tokens.size() << '\n';
    summary << "continuation_tokens\t" << continuation_tokens << '\n';
    summary << "vocabulary\t" << vocabulary << '\n';
    summary << "layers\t" << llama_model_n_layer(model) << '\n';
    summary << "embedding\t" << llama_model_n_embd(model) << '\n';
    summary << "generated_hex\t" << hex_encode(generated) << '\n';

    write_capture_manifest(state);
    llama_perf_context_print(context);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    std::cout << "captured " << state.captures.size() << " tensors across "
              << continuation_tokens + 1 << " stages\n";
    return 0;
}
