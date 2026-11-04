# 16. Post-training: kapel-train

A model that continues text is not yet an assistant that answers. Between the two comes post-training: the model learns the form of a conversation from examples (supervised fine-tuning), then which of two answers is better (DPO), then to find right answers by trying (GRPO). It costs far less computation than pre-training, and Smallem went through each method on a model of 29 million parameters. This chapter brings the same methods to published models, on the loop of chapter 15, with the adapters that let a single card adjust a model of several billion parameters.

Every method is a run of `kapel train`, its configuration choosing what a step learns from: tokens (`[data]`, pre-training and supervised fine-tuning), pairs of answers (`[dpo]`) or answers the model writes (`[grpo]`). `[lora]` freezes the model and trains adapters beside it, whatever the method. The objectives of DPO and GRPO are in `crates/kapel-train/src/trainer/post.rs`.

## 16.1 Conversations

A conversation becomes tokens through the model's chat template (section 9.5), and the loss counts only the tokens the assistant wrote: the model has to learn to answer, not to ask. An example is a JSON object of one of TRL's two forms (`chat.rs`):

- `{"messages": [...]}`: every answer of the assistant is learnt, with the text that ends its turn (`<|im_end|>` for Qwen, `<|eot_id|>` for Llama 3);
- `{"prompt": ..., "completion": ...}`: the completion is learnt, everything the template writes after the prompt. Both sides are messages, or both texts; a prompt of text is encoded with the tokenizer's own template (Llama's `<s>`), its completion followed by the end of text.

**Finding the answers.** An answer starts where the rendering of the messages before it, with the start of an answer (`add_generation_prompt`), ends: the template must render them the same with the answer after them, or the example is refused. Its end cannot be found the same way. Qwen 3's template writes an empty thinking block, `<think>\n\n</think>\n\n`, before the last answer of a conversation only: the rendering up to an earlier answer is not a prefix of the whole. So each answer is rendered once more with a mark appended to it, a character of the private use area (U+F8FF) that no example may hold. The marks are taken out, the text must be the plain rendering exactly, and each mark gives where its answer ends. The text that ends the turn, if it follows, is learnt with the answer.

The text is then encoded piece by piece, the assistant's pieces apart, with the special tokens the template writes read as such, as `transformers` reads them. A message that spells `<|im_start|>` gets the token: the examples of a fine-tuning are trusted data, as they are for TRL.

**Preparing.** `kapel prepare FILES --chat --tokenizer MODEL --output DIR` reads the examples of JSONL or Parquet files (a row of Parquet read as a JSON object, its nested lists of messages included), tokenizes them on every core as for text (section 15.13), leaves the exact duplicates out, and an example with nothing to learn as empty. The shards have a third file beside their index: the mask, one bit per token, 1 where the loss counts the token as a target (`dataset.rs`). No separator goes before a conversation: the template begins it, and the manifest says so (`"separated": false`). The loader reads the mask with the shard, and every target outside it becomes `IGNORE`. The test `conversations_keep_the_assistant_s_tokens_as_the_only_targets` prepares conversations of both forms from JSONL and Parquet, and counts, over one epoch of whole documents, exactly the assistant's tokens as targets.

**Rows.** With `data.pack = "documents"` and `data.mask`, conversations fill rows whole, each seeing only itself, positions from zero: no padding to compute, and the same function as one conversation a row. `"padded"` puts one document in each row, cut to its length or padded: the batches of `transformers` without packing, for comparing with TRL step for step. `"joined"` takes the same documents a pass and lays them in one row without their padding (section 15.13): the same training, without computing the padding.

**Against forgetting.** Smallem's model forgot much of what it knew of FineWeb-Edu when fine-tuned on conversations alone (its loss from 0.88 to 1.48 nats per byte), and kept it when half of each batch was ordinary text (0.89). That is `data.sources`: a source of conversations and one of text, each with its weight (section 15.13); the text has no mask, and its targets all count. `eval.control` names a text whose loss is measured every `eval.every` steps, as the validation's: what the fine-tuning makes the model forget, in the log as `control_loss`.

The 7,473 examples of GSM8K's training split, prepared with Qwen 3's tokenizer, have the tokens and the mask that TRL's tokenization gives them (the prompt rendered with the start of an answer, the whole conversation rendered, the difference learnt), for every one of them; 2.8 seconds, 955,000 of their 1.46 million tokens learnt.

## 16.2 LoRA and QLoRA

LoRA (Hu et al., 2021) freezes the model's weights and puts beside each chosen projection `W` a product of low rank, the only parameters trained:

```text
y = x · Wᵀ + s · (x · Aᵀ) · Bᵀ        A [r, in], B [out, r], s = α / r
```

For Qwen 3 0.6B with `r = 16` on every projection, 10 million parameters train instead of 600 million: their moments take 80 MB instead of 4.8 GB.

**Adding them** (`lora.rs`). `lora::add_adapters` freezes every parameter, then adds `A` and `B` for each projection of the blocks the targets name (`q_proj`, `v_proj`…, or `"all-linear"`; for GPT-2 `c_attn`, `c_proj`, `c_fc`). `B` starts at zero: the adapted model starts as the model, and the test `new_adapters_leave_the_model_as_it_was` checks the logits bit for bit for GPT-2, Llama and Qwen 3. `A` is drawn as PEFT draws it, Kaiming's uniform of `torch.nn.Linear`, between `±1/√in`. rsLoRA's scale, `α / √r`, is a setting. The groups of the optimizer apply after: a group may train a frozen parameter again, as PEFT's `modules_to_save` does.

**In the decoder.** The decoder of section 15.5 adds the product of an adapter wherever it finds one, by its name beside the projection's weight. The projections it computes as one product, the queries, keys and values together, the gate and the up projection together, keep their single product; their adapters' products are joined side by side, zeros for a projection without one, and added to it. The scale is applied to `x · Aᵀ`, `r` wide, rather than to the output, which is up to 200 times wider. The gradients reach `A` and `B` only; a frozen weight is a constant of the tape, which records nothing for it beyond its own product (section 15.1). The frozen weights stay in the computation's type, bf16, rather than in the `f32` of trainable parameters: half the memory of the model. The test `only_the_adapters_receive_gradients_and_they_are_right` checks the gradients of every factor against finite differences in `f64`.

**The embeddings.** Smallem added tokens of roles to its vocabulary, their rows of the table starting at the mean of the others: with the table frozen they stay identical, and the model cannot tell them apart. `lora.embedding` puts an adapter on the table too, as PEFT does: each row read is `E[i] + s · Aᵀ[i] · Bᵀ`, `Aᵀ [V, r]` gathered as the table is, its gradient rows (section 15.4). PEFT starts this `A` at zero and draws `B`.

**QLoRA** (Dettmers et al., 2023) quantizes the frozen projections: `lora.quantize = "q4_0"`, `"int4"` (groups of 64), `"q8_0"` or `"int8"`. Four bits and a scale for each block of 32 weights take 0.28 of the bytes of bf16. A quantized projection is computed by `QuantizedMatmul`, which reads the codes: with many rows, the card dequantizes pieces of the weights into the activations' type and multiplies them by the tiled product (section 7.9). Its backward pass, `g · W`, went through the host until this step, the weights dequantized there; it now dequantizes them on the card, all at once, and computes one product. The conformity suite checks it for every format with kernels. QLoRA's own format, NF4 with its constants quantized too, is not among Kapel's: Q4_0 and int4 are its formats of four bits.

**Merging.** `lora::merge` adds each adapter into its weight, `W ← W + s · B · A` (transposed for GPT-2's `[in, out]`), and removes the adapters: the model the adapted one computes, for inference or the Hub, which `kapel merge MODEL ADAPTER --output DIR` writes with the model's tokenizer. A quantized weight is decoded first, by `Cast`, which now converts quantized types to float ones (on the card by the same kernel). One subtlety: the embeddings' adapter changes the table, not the output of a model that ties them. The merged model keeps the table as it was for its output, `lm_head.weight`, and its `config.json` unties them; the test `the_merged_model_computes_what_the_adapted_one_does` checks the logits for the three architectures, embeddings adapted. Each merged weight keeps its type; `--dtype f32` widens a model of bf16 before the merge, so that `W + s · B · A` is rounded once: the adapter of the GRPO recipe at step 50 merged in `f32` gives PEFT's logits within 7 × 10⁻⁵, in bf16 within 1.05 of logits up to 37, the rounding of bf16 alone (`a_merge_into_f32_rounds_the_sums_of_a_bf16_model_once`).

**PEFT's files.** The factors are named as PEFT names them, without its prefix `base_model.model.`: `model.layers.0.self_attn.q_proj.lora_A.weight`. Only the embeddings' `A` is kept transposed, for its rows, and written `[r, V]` as PEFT has it. `lora::save` writes `adapter_model.safetensors` and `adapter_config.json`; `lora::load` reads them, PEFT's own included. A run with adapters saves only what it trains: its checkpoints are adapters of the model it started from (`trainable.safetensors` beside them for the other parameters it trains), and resume as any other. `tools/train/peft_parity.py` checks both ways, against PEFT 0.21: the adapters Kapel draws on every projection and on the embeddings, loaded by `PeftModel.from_pretrained`, and those PEFT draws with no factor at zero, read by Kapel; the logits differ by at most:

| Model | Kapel's adapter in PEFT | PEFT's adapter in Kapel |
| --- | --- | --- |
| Qwen 3 (`trl-internal-testing/tiny-Qwen3ForCausalLM`) | 3.4 × 10⁻⁸ | 9.3 × 10⁻⁹ |
| Llama (`hf-internal-testing/tiny-random-LlamaForCausalLM`) | 3.3 × 10⁻⁷ | 8.9 × 10⁻⁸ |
| GPT-2 (`hf-internal-testing/tiny-random-GPT2LMHeadModel`), `Conv1D` and tied embeddings | 2.2 × 10⁻⁷ | 8.9 × 10⁻⁸ |

**What a step costs.** With frozen weights, a product by a weight computes the gradient of its input only, not of the weight: a step costs two forward passes for them, three for the attention (`flops::frozen_step_flops`); the MFU of a run with adapters counts that.

## 16.3 The log-probabilities of tokens

Supervised fine-tuning needs one loss, the mean cross-entropy of the batch, and the fused loss of section 15.3 forms its gradients during the forward pass. DPO and GRPO weigh each sequence, each token, apart, by weights that depend on the log-probabilities themselves: the gradient of the logits is known only once the loss of the step is.

`linear_token_losses(h, w, targets, chunk)` (`loss.rs`) gives the cross-entropy of each position, `−ln p(target)`, 0 where the target is ignored, without forming the logits whole: `chunk` rows at a time, as the fused loss. Its backward pass forms each piece of logits again, then takes `CrossEntropy`'s derivative with the gradient of each position. That costs one product more than the fused loss, the price of not keeping logits of `N × V`. `Decoder::token_losses` gives them for a batch.

The losses of DPO and GRPO are then computed on the host, in `f64`, from these log-probabilities copied off the card: a few thousand numbers a step. Their derivatives come back to the card as the gradient of the per-token losses, and `Tape::backward_with` takes the backward pass from there. The losses, written as plain functions of numbers (`preference::dpo_terms`, `grpo::token_terms`), are tested against finite differences on their own.

## 16.4 DPO

DPO (Rafailov et al., 2023) learns from a comparison rather than an example: for a prompt, a preferred answer `y⁺` and a rejected one `y⁻`, and a frozen reference `π_ref`,

```text
loss = −ln σ(β · [(ln π(y⁺) − ln π_ref(y⁺)) − (ln π(y⁻) − ln π_ref(y⁻))])
```

The pairs (`preference.rs`) are JSON objects as TRL reads them: `prompt`, `chosen` and `rejected`, texts or messages, or `chosen` and `rejected` alone, whole conversations whose common beginning is the prompt. They are read whole into memory, each answer encoded with its prompt as a completion (section 16.1), and drawn in an order of their own each epoch, from the seed; the position in them is what a checkpoint saves. A pass holds `batch.rows / 2` pairs, each answer a row.

**The reference** is the model the run starts from. With adapters only, it is the model without them, which needs no copy: the decoder skips the adapters it does not find. Otherwise, when every weight trains or a group trains some, it is a frozen copy of the starting weights in the computation's type.

**The likelihood term.** Smallem's runs of DPO showed what Razin et al. (2024) call likelihood displacement: both answers grew less likely while their gap grew, the probability going to answers neither of them is, repetitions and invented numbers, and its accuracy fell from 57 % to 1.5 %. The term of RPO (Pang et al., 2024), the cross-entropy of the preferred answers by token, held them up: `dpo.nll` is its weight (TRL's `rpo_alpha`), 0 for DPO alone.

Each step logs, besides the loss, the share of pairs whose preferred answer has the larger implicit reward (`accuracy`), the mean rewards `β (ln π − ln π_ref)` of both answers and their margin, and `nll`. Pairs held out (`dpo.val`) are measured every `eval.every` steps the same way, without gradients and without the likelihood term: `val_loss`, `val_accuracy`, the mean rewards and `val_margin`, as TRL's evaluation reports them. The tests check that the loss is `ln 2` exactly when the model is its reference, that a run learns the preference of a small Qwen 3, and that a run of DPO killed and resumed computes the bits of one that was not.

## 16.5 GRPO

GRPO (Shao et al., *DeepSeekMath*, 2024) lets the model find right answers by itself. For each prompt the model writes a group of `G` answers; a program scores them; each answer's advantage is its reward against its group's, `(r − mean) / (std + 10⁻⁴)`, and each of its tokens weighs in the loss

```text
−min(ρ · A, clip(ρ, 1 − ε, 1 + ε_high) · A) + β · (π_ref/π − ln(π_ref/π) − 1)
```

`ρ = π / π_old` is the ratio of a token's probability now to that when its answer was written, bounded so that a step moves little; the second term is the divergence from the reference (Schulman's `k3`). With one step per generation (`grpo.iterations = 1`, as TRL's default), `π_old = π` and `ρ = 1`: the gradient is REINFORCE's, the advantage times that of `ln π`. With more, the log-probabilities when the answers were written are computed once, before the first of their steps. The test `a_ratio_out_of_bounds_stops_the_gradient` checks the bounds both ways; `with_its_own_answers_the_gradient_is_reinforce_s` the first case.

**Writing the answers.** The answers of a step are written by the inference engine of chapter 11, all at once (`grpo::generate`), from a model of the current weights: the adapters merged into it on the card (section 16.2), its float weights in bf16, its projections stacked (section 10.4). Their `G` answers share each prompt, and the engine computes it once (prefix sharing, section 11.3). The cache has a fixed size, `training.cache_gib` (4 GiB by default): the engine admits answers while its pages last. Each answer has a seed of its own drawn from the step's, so a resumed run writes the same answers. The answer's tokens include the one that ended it: the model learns to stop.

**Rewards** (`rewards.rs`) read an answer's text and the record its prompt came from. `math` gives 1 if the number the answer ends on, in its last `\boxed{}`, else after its last `####`, else its last number, is the number of the record's field (GSM8K's answers end with `#### 72`); `exact` compares the text; `regex` checks a format; `command` runs a program of one's own on each batch, a JSON object per line in, a number per line out. A program reaches them through the trait `Reward`. The rewards of a run add up with their weights, and each one's mean goes to the log.

**Averaging the loss.** The paper averages the losses of the tokens over each answer, then over the answers. A long answer weighs each of its tokens less, which favours long wrong answers: Liu et al. (*Dr. GRPO*, 2025) sum them instead, divided by a constant (the most tokens of an answer times the answers), and leave the advantages unscaled by the deviation. `grpo.loss` is `"grpo"`, `"dr_grpo"` or `"token"` (DAPO's mean over all the tokens of a step); `grpo.scale_rewards` overrides the scaling.

The test `grpo_makes_a_rewarded_answer_more_likely_and_a_task_is_measured` gives a small Qwen 3 a reward for one of its four most likely first tokens, sampled with `top_k = 4`: in 20 steps, the token becomes more than three times as likely, and the task measured along the run answers it.

## 16.6 Evaluations along a run

Every `eval.every` steps, a run measures, besides the validation loss:

- the loss on the text of control, `eval.control` (section 16.1);
- the accuracy of each task of `[[eval.tasks]]`: prompts whose answers a program checks, the rewards of GRPO, answered greedily by the engine from the current weights. The log has, for each task, its accuracy, the mean length of the answers, and the share that ended by themselves rather than at `max_tokens`.

`crates/kapel-train/examples/tasks.rs` measures the same after a run, for a model and an adapter, PEFT's or Kapel's: the evaluator of the measures below.

## 16.7 Measured

**Supervised fine-tuning with LoRA, against TRL.** Qwen 3 0.6B on GSM8K's 7,473 problems (`tools/train/gsm8k.py`), the completions GSM8K's answers: LoRA of rank 16, `α` 32, no dropout, on every projection; AdamW at 2 × 10⁻⁴, β 0.9 and 0.999, no decay, a linear decay without warmup; gradients clipped to 1; 16 examples a step, one a row (`pack = "padded"`), cut to 512 tokens; one epoch, 467 steps; bf16. TRL 1.16 with PEFT 0.21 and `transformers` 5.20, the same settings (`tools/train/trl_sft.py`), the batches in another order. The mean loss by window of 50 steps:

| Steps | Kapel | TRL |
| --- | --- | --- |
| 1–50 | 0.565 | 0.574 |
| 101–150 | 0.448 | 0.452 |
| 201–250 | 0.436 | 0.431 |
| 301–350 | 0.431 | 0.430 |
| 401–450 | 0.419 | 0.422 |

Both adapters were then answered greedily on the 1,319 problems of the test split, 512 tokens at most, by two evaluators: Kapel's engine in bf16, the adapters merged in `f32` (`examples/tasks.rs`), and `transformers`' `generate` in bf16, the adapters merged by PEFT:

| | Kapel's engine | `transformers` |
| --- | --- | --- |
| Qwen 3 0.6B, without thinking | 61.2 % | |
| Fine-tuned by Kapel | 48.3 % | 47.9 % |
| Fine-tuned by TRL | 47.5 % | 47.2 % |

Kapel is within 1 point of TRL, the target, by either evaluator: 0.8 and 0.7. Greedy answers in bf16 part ways as soon as one token differs: on the first 200 problems, the two evaluators give the same answer to TRL's adapter 127 times, and their accuracies over the whole test differ by 0.3. Both fine-tunings lower the accuracy of the model. GSM8K's answers are short, with its calculator's annotations (`<<48/2=24>>`): they teach the model a reasoning shorter than its own, 138 tokens an answer against 230.

The steps took 6 minutes 55 seconds in Kapel, 881 ms each at 9,300 tokens a second, and 6 minutes 22 in TRL. Kapel pads every row to 512 tokens and TRL each batch to its longest, so Kapel computes more tokens of padding. Kapel's peak was 13.3 GiB, its activations all kept; TRL's 2.5 GiB, recomputing them, its default (`training.checkpoint_activations` does the same in Kapel).

**Every weight.** The same fine-tuning without adapters (`recipes/sft-gsm8k`), AdamW at 10⁻⁵, the rest unchanged: GSM8K 50.0 % by Kapel's engine, the loss 0.408 over steps 401–450 (0.419 with LoRA). The steps took 402 seconds, at 9,518 tokens a second and 52.7 % of the card's peak; 21.3 GiB at the peak, AdamW's two moments in f32 beside 0.6 billion weights. More weights trained imitate GSM8K's answers more closely, and lose a little less of the model's own reasoning: 1.7 points more than LoRA, still 11 below the model before.

**DPO, against TRL.** Qwen 3 0.6B on 4,000 pairs of UltraFeedback that fit in 1,024 tokens of its template (`tools/train/ultrafeedback.py`, `recipes/dpo-ultrafeedback`): LoRA of rank 16, `α` 32, on every projection; AdamW at 5 × 10⁻⁵ with a linear decay; `β` 0.1; 8 pairs a step, one epoch, 500 steps; bf16. TRL 1.16 with the same settings (`tools/train/trl_dpo.py`), its batches in another order. The 500 pairs of the test split, measured every 100 steps (`dpo.val`):

| Step | Loss: Kapel | TRL | Accuracy: Kapel | TRL | Margin: Kapel | TRL |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 0.657 | 0.664 | 64.4 % | 63.5 % | 0.51 | 0.41 |
| 200 | 0.648 | 0.657 | 65.0 % | 63.3 % | 0.51 | 0.45 |
| 300 | 0.633 | 0.643 | 62.0 % | 65.1 % | 0.44 | 0.45 |
| 400 | 0.630 | 0.622 | 65.8 % | 65.9 % | 0.49 | 0.47 |
| 500 | 0.627 | 0.622 | 64.8 % | 65.1 % | 0.47 | 0.49 |

The training's means over steps 1–50 and 451–500: the loss 0.680 and 0.621 in Kapel, 0.697 and 0.644 in TRL; the accuracy 58.5 % and 65.0 %, against 59 % and 63.2 %. The two runs agree within what 500 pairs can tell (a point is 5 pairs). Kapel's steps are two passes of 4 pairs, as 16 rows of 1,024 tokens with every activation kept do not fit on the card: 2.4 seconds a step at 6,900 tokens a second, 13.3 GiB at the peak, 26 minutes in all against TRL's 28.

**Without padding.** The runs above pad every example to its row, as TRL pads without packing. With `data.pack = "joined"` (section 15.13), the same examples a pass are laid one after the other in a single row, and only their tokens are computed: a step of GSM8K's 16 examples is about 3,100 tokens instead of 8,192. The same runs again, the recipes as they now are:

| | Padded | Joined |
| --- | --- | --- |
| Every weight: GSM8K, loss 401–450 | 50.0 %, 0.408 | 50.0 %, 0.408 |
| Every weight: time, memory | 7 min 20 s, 21.3 GiB | 5 min 24 s, 15.9 GiB |
| LoRA: GSM8K, loss 401–450 | 48.3 %, 0.419 | 48.0 %, 0.419 |
| LoRA: time, memory | 7 min, 13.3 GiB | 4 min 25 s, 7.9 GiB |
| DPO: held-out loss and accuracy at step 500 | 0.627, 64.8 % | 0.629, 65.0 % |
| DPO: time, memory | 26 min, 13.3 GiB | 14 min, 11.5 GiB |

The fine-tuning with LoRA now takes less time than TRL's (6 min 22 s), the DPO half of it (28 min). The losses agree to the third decimal; the greedy answers part ways at the first token that differs, by a point at most here. The model's share of the card's peak falls (43 % for every weight, against 53 %): one row of 3,100 tokens makes smaller products than 16 of 512.

**GRPO on GSM8K.** Qwen 3 0.6B without thinking, on the problems of GSM8K's training split (`recipes/grpo-gsm8k`): each step 16 problems, 8 answers each of at most 512 tokens at temperature 1, the reward 1 for the right final number; LoRA of rank 16 on every projection, AdamW at 2 × 10⁻⁵; `β` 0.04, `ε` 0.2, the paper's averaging; 200 steps. The accuracy of the 1,319 problems of the test split, answered greedily every 50 steps, and the training's means by window:

| Steps | Reward | Tokens an answer | Divergence | GSM8K (test) |
| --- | --- | --- | --- | --- |
| 0 | | | | 61.2 % |
| 1–50 | 0.635 | 236 | 0.003 | 64.9 % |
| 51–100 | 0.668 | 234 | 0.008 | 67.1 % |
| 101–150 | 0.662 | 237 | 0.008 | 68.4 % |
| 151–200 | 0.703 | 238 | 0.009 | 69.3 % |

The accuracy on problems the run never saw rises at every measure, by 8.1 points in all, as DeepSeekMath's GRPO raised its instructed 7B model on GSM8K (82.9 % to 88.2 %); the answers do not grow longer, and the model stays close to its reference. No ratio was bounded: with one step per generation, the ratio is 1. The run took 1 hour 33 minutes, 28 seconds a step, half of it writing the answers (14 seconds for 128 answers, 30,000 tokens); 11.5 GiB at the peak. Dr. GRPO's averaging is written and tested, not measured.
