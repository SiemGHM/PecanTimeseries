[2025-02-05 21:09:59,988] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Cuda check
True
2
GPU 0: NVIDIA RTX 6000 Ada Generation
GPU 1: NVIDIA RTX 6000 Ada Generation
True
args. use_gpu
use multiGPU True
True
Inside the run file 0,1
['0', '1'] DEVICE IDS
Args in experiment:
Namespace(is_training=1, model_id='TimeLLM', model='TimeLLM', data='custom', root_path='./Pecan/Data/', data_path='AllPecHourly.csv', features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=96, label_len=48, pred_len=24, enc_in=347, dec_in=347, c_out=347, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=25, factor=3, distil=True, dropout=0.1, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=10, itr=1, train_epochs=10, batch_size=1, patience=8, learning_rate=0.0001, des='Exp', loss='MSE', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=True, devices='0,1', exp_name='MTSF', channel_independence=False, inverse=False, class_strategy='projection', target_root_path='./data/electricity/', target_data_path='electricity.csv', efficient_training=False, use_norm=True, partial_start_index=0, llm_layers=6, task_name='long_term_forecast', llm_dim=4096, patch_len=16, stride=8, llm_model='GPT2', prompt_domain=0, device_ids=[0, 1])
Use GPU: cuda:0
2048 ######################################################
CONFIG>LLMMOD GPT2
>>>>>>>start training : TimeLLM_TimeLLM_custom_M_ft96_sl48_ll24_pl512_dm8_nh2_el1_dl2048_df3_fctimeF_ebTrue_dtExp_projection_0>>>>>>>>>>>>>>>>>>>>>>>>>>
0   2016-01-01 06:00:00+00:00
1   2016-01-01 07:00:00+00:00
2   2016-01-01 08:00:00+00:00
3   2016-01-01 09:00:00+00:00
4   2016-01-01 10:00:00+00:00
Name: date, dtype: datetime64[ns, UTC]
datetime64[ns, UTC]
train 1900
1923   2016-03-21 09:00:00+00:00
1924   2016-03-21 10:00:00+00:00
1925   2016-03-21 11:00:00+00:00
1926   2016-03-21 12:00:00+00:00
1927   2016-03-21 13:00:00+00:00
Name: date, dtype: datetime64[ns, UTC]
datetime64[ns, UTC]
val 266
2212   2016-04-02 10:00:00+00:00
2213   2016-04-02 11:00:00+00:00
2214   2016-04-02 12:00:00+00:00
2215   2016-04-02 13:00:00+00:00
2216   2016-04-02 14:00:00+00:00
Name: date, dtype: datetime64[ns, UTC]
datetime64[ns, UTC]
test 554
LEARNING RATE 0.0001
	iters: 100, epoch: 1 | loss: 1.0000000
	speed: 0.2520s/iter; left time: 4763.2837s
	iters: 200, epoch: 1 | loss: 1.3906250
	speed: 0.2499s/iter; left time: 4697.4350s
	iters: 300, epoch: 1 | loss: 1.0468750
	speed: 0.2512s/iter; left time: 4697.4383s
	iters: 400, epoch: 1 | loss: 1.2812500
	speed: 0.2522s/iter; left time: 4691.3882s
	iters: 500, epoch: 1 | loss: 1.1171875
	speed: 0.2529s/iter; left time: 4678.2370s
	iters: 600, epoch: 1 | loss: 1.3593750
	speed: 0.2541s/iter; left time: 4676.3999s
	iters: 700, epoch: 1 | loss: 1.1406250
	speed: 0.2536s/iter; left time: 4641.7822s
	iters: 800, epoch: 1 | loss: 0.8554688
	speed: 0.2561s/iter; left time: 4660.4816s
	iters: 900, epoch: 1 | loss: 1.1250000
	speed: 0.2541s/iter; left time: 4599.2018s
	iters: 1000, epoch: 1 | loss: 1.2187500
	speed: 0.2544s/iter; left time: 4579.9598s
	iters: 1100, epoch: 1 | loss: 0.8437500
	speed: 0.2545s/iter; left time: 4556.4827s
	iters: 1200, epoch: 1 | loss: 1.5859375
	speed: 0.2561s/iter; left time: 4558.8187s
	iters: 1300, epoch: 1 | loss: 1.1796875
	speed: 0.2550s/iter; left time: 4513.5137s
