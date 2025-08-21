# Train

- [ref, run command(maru-white-v2.3.0-split_00-5epoch)](https://www.notion.so/kakaobrain/maru-white-v2-3-0-split_00-5epoch-2b00981f70104a17bf6ea56a2eb67949?pvs=4)


### prerequisite

- mm-agent workspace
  - cuda12.1\_gemma\_trl\_functionary\_tooltalk 이미지 사용

- setting
  - pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel 이미지 사용  
  - install libraries

    ```shell
    # Install Dependencies
    $ pip install accelerate==0.27.2 transformers==4.38.1 bitsandbytes==0.41.1 scipy==1.11.3 sentencepiece==0.1.99 packaging==23.1 ninja==1.11.1 einops==0.7.0 wandb==0.15.11 jsonref==1.1.0 deepspeed==0.11.1 typer==0.9.0 tensorboard==2.15.1 wheel==0.42.0

    # Install Flash Attention 2
    $ pip install flash-attn==2.3.3 --no-build-isolation
    ```

### compute requirements

| Model    | Number of GPUs (A100-80GB) |
| :--------: | :-------: |
| maru-white-v2  | 4 |


### Train

[ref, README.md](README.md)

```
$ cd /path/to/functionary

# DeepSpeed ZeRO3 with accelerate launcher
# 4xA100 80GB, from the root directory of the repository
export WANDB_ENTITY=NAME_OF_ENTITY
export WANDB_PROJECT=functionary
accelerate launch --config_file "functionary/train/accelerate_configs/ds3_config.yaml" \
    --num_processes 4 \
    -m functionary.train.train \
    --model_name_or_path models/maru-white-v2.3.0 \
    --train_data_path train.jsonl \
    --eval_data_path val.jsonl \
    --bf16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --eval_steps 1 \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --learning_rate 9e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --optim "paged_adamw_32bit" \
    --gradient_checkpointing True \
    --output_dir output_model
```

### Test(Run Server)

```
$ cd /path/to/functionary

$ python3 server_vllm.py --model "output_model" --host 0.0.0.0
```
