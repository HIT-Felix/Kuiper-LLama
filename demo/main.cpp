#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <cstdlib>
#include "model/llama3.h"

int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  const auto prompt_tokens = model.encode(sentence);
  const int32_t prompt_len = static_cast<int32_t>(prompt_tokens.size());
  LOG_IF(FATAL, prompt_tokens.empty()) << "The tokens is empty.";

  int32_t next = -1;
  std::vector<int32_t> words;
  STATUS_CHECK(model.prefill(prompt_tokens, next));
  int32_t pos = prompt_len;

  while (pos < total_steps) {
    if (model.is_sentence_ending(next)) {
      break;
    }
    words.push_back(next);
    const int32_t current_token = next;
    STATUS_CHECK(model.decode_step(current_token, pos, next));
    ++pos;
  }

  if (need_output) {
    printf("%s", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}


int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 5) {
    LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path [prompt] [max_steps]";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];
  const std::string sentence =
      argc >= 4 ? argv[3] : "hello";
  const int total_steps = argc >= 5 ? std::atoi(argv[4]) : 128;
  LOG_IF(FATAL, total_steps <= 0) << "The max_steps should be positive.";

  model::LLama2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path,
    checkpoint_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }

  auto start = std::chrono::steady_clock::now();
  printf("Generating...\n");
  printf("Prompt: %s\n", sentence.c_str());
  printf("Response: ");
  fflush(stdout);
  int steps = generate(model, sentence, total_steps, true);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  const auto& stats = model.generation_stats();
  const double prefill_tokens_per_s =
      stats.prefill_latency_ms > 0.0
          ? static_cast<double>(stats.prefill_tokens) * 1000.0 / stats.prefill_latency_ms
          : 0.0;
  const double decode_tokens_per_s =
      stats.decode_latency_ms > 0.0
          ? static_cast<double>(stats.decode_tokens) * 1000.0 / stats.decode_latency_ms
          : 0.0;
  printf("prefill_ms:%lf\ndecode_ms:%lf\nttft_ms:%lf\nprefill_tokens:%d\ndecode_tokens:%d\n"
         "prefill_tokens_per_s:%lf\ndecode_tokens_per_s:%lf\nallocated_blocks:%d\n"
         "block_size:%d\nprompt_last_block_tokens:%d\nprompt_tail_fragmentation:%lf\n"
         "kv_used_bytes:%zu\nkv_reserved_bytes:%zu\n",
         stats.prefill_latency_ms, stats.decode_latency_ms, stats.ttft_ms, stats.prefill_tokens,
         stats.decode_tokens, prefill_tokens_per_s, decode_tokens_per_s, stats.allocated_blocks,
         stats.block_size, stats.prompt_last_block_tokens, stats.prompt_tail_fragmentation,
         stats.kv_bytes_used, stats.kv_bytes_reserved);
  fflush(stdout);
  return 0;
}
