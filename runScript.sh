#!/usr/bin/env bash
pip install accelerate
export HUGGINGFACE_TOKEN="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"

accelerate launch --config_file accelerate_config.yaml run.py --task_name long_term_forecast --is_training 1 --llm_model LLAMA --root_path ./Pecan/Data/ --data_path AllPecHourly.csv --model_id TimeLLM --model TimeLLM --batch_size 16 --data custom --features M --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 347 --dec_in 347 --c_out 347 --patience 8 --des 'Exp' --itr 1 --use_multi_gpu --devices "0,1"


