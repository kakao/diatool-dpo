# DiaTool-DPO: Multi-turn Direct Preference Optimization for Tool-augmented Language Models
* This repository is based on TRL(Transformer Reinforcement Learning) and functionary
```bibtex
@misc{vonwerra2022trl,
  author = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/trl}}
}
```
```bibtex
@software{MeetKai_Functionary_2024,
  author = {MeetKai},
  title = {Functionary},
  year = {2024},
  url = {https://github.com/MeetKai/functionary},
  version = {1.0.0}
}
```
* For installation, please follow the direction of TRL (included at the bottom of this README.md).
* Arguments added compared to TRL

| Argument | Meaning                | Default value if not specified |
|----------|------------------------|--------------------------------|
| gamma    | reward scaling factor  | 0.5                            |
| margin   | reward gap subtraction | 0.5                            |

* Running script
```
(이미지 사용 시 conda base 대신 기본 python 환경에 설치되어 있으므로 conda deactivate 먼저 하셔야합니다)
cd ~/trl
python -u -m accelerate.commands.launch
--config_file=/home/bc-user/trl/examples/accelerate_configs/deepspeed_zero3.yaml 
--num_processes 8
/home/bc-user/trl/examples/scripts/function_dpo.py 
--dataset_name={dataset_path}
--model_name_or_path={model_name_or_path_to_start_train}
--per_device_train_batch_size
1
--learning_rate
1e-7
--gradient_accumulation_steps
1
--logging_steps
10
--eval_steps
10
--output_dir={model_save_dir}
--warmup_steps
150
--report_to
wandb
--bf16
--logging_first_step
--no_remove_unused_columns
--max_length
8192
--max_prompt_length
4096
--do_eval
True
--eval_strategy
steps
--save_strategy
epoch
--save_steps
1
--beta
0.5
--gamma
0.5
--margin
2.0
--num_train_epochs
1.0
```
* Running script with example arguments
```
(이미지 사용 시 conda base 대신 기본 python 환경에 설치되어 있으므로 conda deactivate 먼저 하셔야합니다)
cd ~/trl
python -u -m accelerate.commands.launch
--config_file=/home/bc-user/trl/examples/accelerate_configs/deepspeed_zero3.yaml
--num_processes 8
/home/bc-user/trl/examples/scripts/function_dpo.py
--dataset_name=/data/llm-public_636/users/kong/data/dpo/diatool_dpo_dataset/glaive_ko_all/concat.jsonl.string
--model_name_or_path=/data/llm-public_636/users/hubert/workspace/kanana-fc/train/models/kanana-essence-8b-fc-v1.0.1-stage1-rc.20
--per_device_train_batch_size
1
--learning_rate
1e-7
--gradient_accumulation_steps
1
--logging_steps
10
--eval_steps
10
--output_dir={model_save_dir}
--warmup_steps
150
--report_to
wandb
--bf16
--logging_first_step
--no_remove_unused_columns
--max_length
8192
--max_prompt_length
4096
--do_eval
True
--eval_strategy
steps
--save_strategy
epoch
--save_steps
1
--beta
0.5
--gamma
0.5
--margin
2.0
--num_train_epochs
1.0
```
## Implementation Details
1. 데이터셋 파싱
* DiaTool-DPO 데이터셋은 tools 목록이 들어 있고, tool call을 위한 json 형태로 되어 있기 때문에 기존 DPO 데이터셋과 다른 파싱 과정을 거친다.
* 이 과정에서 SFT 모델의 tool call template을 tokenizer에서 읽어와 참조한다.
* 구현 위치: 
  * examples/scripts/function_dpo.py
  * trl/trainer/dpo_trainer.py: 
    * DPOTrainer.tokenize_row()
2. Multi-turn 데이터에서 agent turn을 label에서 masking하도록 agent turn과 assistant turn 구별되게 chunking하기
* 구현 위치: 
  * trl/trainer/dpo_trainer.py
    * get_masked_labels
    * get_prompt_template_from_tokenizer
    * get_prefix_assistant_token_ids
    * get_assistant_stop_token_ids
3. Turn 별로 reward 계산
* 구현 위치: 
  * trl/trainer/dpo_config.py: gamma, margin 추가
  * trl/trainer/dpo_trainer.py
    * DPOTrainer.get_batch_logps(): 2단계에서 구한 개별의 assistant turn에 turn order에 따른 scaling factor를 곱한다.
4. Turn 별 reward를 사용해 loss 계산
* 구현 위치:
  * trl/trainer/dpo_trainer.py
    * DPOTrainer.dpo_loss(): normalization, reward margin subtraction 구현
## 기타
* Dataset 경로: /data/llm-public_636/users/kong/data/dpo/diatool_dpo_dataset/glaive_ko_all/concat.jsonl.string
* 이미지: diatool_dpo
* 성능:
  * FCBench result file: /data/llm-public_636/users/kong/FunctionChat-Bench/output/others/glaive_korean_v3_dmpo_gamma0.5_beta0.5_margin2.0_lr1e-7

| Evaluation          | Model       | Call   | Competion | Slot    | Relevance | Avg(micro) |
|---------------------|-------------|--------|-----------|---------|-----------|------------|
| FCBench             | SFT-Only    | 0.8429 | 0.9571    | 0.6389  | 0.8261    | 0.8442     |
| FCBench             | SFT + DiaTool-DPO | 0.7714 | 0.8986    | 0.9167  | 0.9130    | 0.8500     |
| FCBench-calibration | SFT + DiaTool-DPO | 0.8571 | 0.9286    | 0.9167  | 0.9130    | 0.9045     |
    * SFT-Only: /data/nlp-public_338/models/decoder/internal/functionary-lmt-llama3-8b-sft-stage1-rc.30
    * SFT + DiaTool-DPO: /data/llm-public_636/users/kong/checkpoint/dmpo_attempt/diff_margin/glaive_korean_v3_dmpo_gamma0.5_beta0.5_margin2.0_lr1e-7_new_method2
* 위키
  * 인수인계: https://wiki.daumkakao.com/spaces/nLp/pages/1727260458/20250423_DiaTool_DPO+%EC%9D%B8%EC%88%98%EC%9D%B8%EA%B3%84
  * FCBench-calibration: call과 completion 수정한 내용. 모델이 뭐라고 대답해서 맞게 처리했는지
    * 파일 링크(wiki): https://wiki.daumkakao.com/spaces/nLp/pages/1734226022/FCBench+Dialog+Calibration
    * 액셀 파일 내 data/v3 컬럼 참조 
    * 예) XXX를 메모해줘 -> required field인 '메모 제목'을 자동생성하고 call이 일어나는게 정답으로 되어있지만, '메모 제목은 무엇으로 할까요?' 라고 물어보느라 call이 안 일어나는 경우에 대해 틀리지 않았다고 생각해 채점을 수정
* 이름 변경 관련
  * 위키나 파일 저장 경로에 보면 데이터를 v1, v2, v3이라 부르는 경우가 있는데, 이것은 논문에서 easy, hard, all에 대응함. 이름이 변경된 바 있음







<details>
<summary>TRL</summary>

<!-- summary 아래 한칸 공백 두어야함 -->
<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/trl_banner_dark.png">
</div>
## TRL
# TRL - Transformer Reinforcement Learning
> Full stack library to fine-tune and align large language models.

<p align="center">
    <a href="https://github.com/huggingface/trl/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/trl.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/trl/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/trl/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/trl/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/trl.svg">
    </a>
</p>


## What is it?

The `trl` library is a full stack tool to fine-tune and align transformer language and diffusion models using methods such as Supervised Fine-tuning step (SFT), Reward Modeling (RM) and the Proximal Policy Optimization (PPO) as well as Direct Preference Optimization (DPO). 

The library is built on top of the [`transformers`](https://github.com/huggingface/transformers) library and thus allows to use any model architecture available there.


## Highlights

- **`Efficient and scalable`**: 
    - [`accelerate`](https://github.com/huggingface/accelerate) is the backbone of `trl` which allows to scale model training from a single GPU to a large scale multi-node cluster with methods such as DDP and DeepSpeed.
    - [`PEFT`](https://github.com/huggingface/peft) is fully integrated and allows to train even the largest models on modest hardware with quantisation and methods such as LoRA or QLoRA.
    - [`unsloth`](https://github.com/unslothai/unsloth) is also integrated and allows to significantly speed up training with dedicated kernels.
- **`CLI`**: With the [CLI](https://huggingface.co/docs/trl/clis) you can fine-tune and chat with LLMs without writing any code using a single command and a flexible config system.
- **`Trainers`**: The Trainer classes are an abstraction to apply many fine-tuning methods with ease such as the [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer), [`DPOTrainer`](https://huggingface.co/docs/trl/trainer#trl.DPOTrainer), [`RewardTrainer`](https://huggingface.co/docs/trl/reward_trainer), [`PPOTrainer`](https://huggingface.co/docs/trl/trainer#trl.PPOTrainer), [`CPOTrainer`](https://huggingface.co/docs/trl/trainer#trl.CPOTrainer), and [`ORPOTrainer`](https://huggingface.co/docs/trl/trainer#trl.ORPOTrainer).
- **`AutoModels`**: The [`AutoModelForCausalLMWithValueHead`](https://huggingface.co/docs/trl/models#trl.AutoModelForCausalLMWithValueHead) & [`AutoModelForSeq2SeqLMWithValueHead`](https://huggingface.co/docs/trl/models#trl.AutoModelForSeq2SeqLMWithValueHead) classes add an additional value head to the model which allows to train them with RL algorithms such as PPO.
- **`Examples`**: Train GPT2 to generate positive movie reviews with a BERT sentiment classifier, full RLHF using adapters only, train GPT-j to be less toxic, [StackLlama example](https://huggingface.co/blog/stackllama), etc. following the [examples](https://github.com/huggingface/trl/tree/main/examples).

## Installation

### Python package
Install the library with `pip`:
```bash
pip install trl
```

### From source
If you want to use the latest features before an official release you can install from source:
```bash
pip install git+https://github.com/huggingface/trl.git
```

### Repository
If you want to use the examples you can clone the repository with the following command:
```bash
git clone https://github.com/huggingface/trl.git
```

## Command Line Interface (CLI)

You can use TRL Command Line Interface (CLI) to quickly get started with Supervised Fine-tuning (SFT), Direct Preference Optimization (DPO) and test your aligned model with the chat CLI: 

**SFT:**

```bash
trl sft --model_name_or_path facebook/opt-125m --dataset_name imdb --output_dir opt-sft-imdb
```

**DPO:**

```bash
trl dpo --model_name_or_path facebook/opt-125m --dataset_name trl-internal-testing/hh-rlhf-helpful-base-trl-style --output_dir opt-sft-hh-rlhf 
```

**Chat:**

```bash
trl chat --model_name_or_path Qwen/Qwen1.5-0.5B-Chat
```

Read more about CLI in the [relevant documentation section](https://huggingface.co/docs/trl/main/en/clis) or use `--help` for more details.

## How to use

For more flexibility and control over the training, you can use the dedicated trainer classes to fine-tune the model in Python.

### `SFTTrainer`

This is a basic example of how to use the `SFTTrainer` from the library. The `SFTTrainer` is a light wrapper around the `transformers` Trainer to easily fine-tune language models or adapters on a custom dataset.

```python
# imports
from datasets import load_dataset
from trl import SFTTrainer

# get dataset
dataset = load_dataset("imdb", split="train")

# get trainer
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)

# train
trainer.train()
```

### `RewardTrainer`

This is a basic example of how to use the `RewardTrainer` from the library. The `RewardTrainer` is a wrapper around the `transformers` Trainer to easily fine-tune reward models or adapters on a custom preference dataset.

```python
# imports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer

# load model and dataset - dataset needs to be in a specific format
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

...

# load trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

# train
trainer.train()
```

### `PPOTrainer`

This is a basic example of how to use the `PPOTrainer` from the library. Based on a query the language model creates a response which is then evaluated. The evaluation could be a human in the loop or another model's output.

```python
# imports
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch

# get models
model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
ref_model = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# initialize trainer
ppo_config = PPOConfig(batch_size=1, mini_batch_size=1)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(model, query_tensor)

# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model for one step with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
```

### `DPOTrainer`

`DPOTrainer` is a trainer that uses [Direct Preference Optimization algorithm](https://huggingface.co/papers/2305.18290). This is a basic example of how to use the `DPOTrainer` from the library. The `DPOTrainer` is a wrapper around the `transformers` Trainer to easily fine-tune reward models or adapters on a custom preference dataset.

```python
# imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

# load model and dataset - dataset needs to be in a specific format
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

...

# load trainer
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

# train
trainer.train()
```

## Development

If you want to contribute to `trl` or customizing it to your needs make sure to read the [contribution guide](https://github.com/huggingface/trl/blob/main/CONTRIBUTING.md) and make sure you make a dev install:

```bash
git clone https://github.com/huggingface/trl.git
cd trl/
make dev
```

## References

### Proximal Policy Optimisation
The PPO implementation largely follows the structure introduced in the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://huggingface.co/papers/1909.08593), [code](https://github.com/openai/lm-human-preferences)].

### Direct Preference Optimization
DPO is based on the original implementation of **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"** by E. Mitchell et al. \[[paper](https://huggingface.co/papers/2305.18290), [code](https://github.com/eric-mitchell/direct-preference-optimization)]


## Citation

```bibtex
@misc{vonwerra2022trl,
  author = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/trl}}
}
```

</details>
## LICENSE
License
This software is licensed under the Apache 2 license, quoted below.

Copyright 2025 Kakao Corp. http://www.kakaocorp.com

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
