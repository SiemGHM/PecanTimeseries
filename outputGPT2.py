[2025-02-05 19:21:45,019] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
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
	iters: 100, epoch: 1 | loss: 0.6953125
	speed: 0.2539s/iter; left time: 4798.8427s
	iters: 200, epoch: 1 | loss: 1.0781250
	speed: 0.2511s/iter; left time: 4720.0612s
	iters: 300, epoch: 1 | loss: 0.7773438
	speed: 0.2525s/iter; left time: 4721.8307s
	iters: 400, epoch: 1 | loss: 1.1015625
	speed: 0.2529s/iter; left time: 4703.7781s
	iters: 500, epoch: 1 | loss: 0.8320312
	speed: 0.2532s/iter; left time: 4683.6154s
	iters: 600, epoch: 1 | loss: 1.0937500
	speed: 0.2535s/iter; left time: 4663.8251s
	iters: 700, epoch: 1 | loss: 0.8515625
	speed: 0.2546s/iter; left time: 4659.4612s
	iters: 800, epoch: 1 | loss: 0.5937500
	speed: 0.2573s/iter; left time: 4682.3131s
	iters: 900, epoch: 1 | loss: 0.9062500
	speed: 0.2543s/iter; left time: 4602.8665s
	iters: 1000, epoch: 1 | loss: 0.9648438
	speed: 0.2548s/iter; left time: 4586.1857s
	iters: 1100, epoch: 1 | loss: 0.6054688
	speed: 0.2545s/iter; left time: 4556.2585s
	iters: 1200, epoch: 1 | loss: 1.2109375
	speed: 0.2555s/iter; left time: 4547.8708s
	iters: 1300, epoch: 1 | loss: 0.8945312
	speed: 0.2544s/iter; left time: 4502.5384s
	iters: 1400, epoch: 1 | loss: 1.0390625
	speed: 0.2546s/iter; left time: 4481.8572s
	iters: 1500, epoch: 1 | loss: 1.0625000
	speed: 0.2550s/iter; left time: 4462.1117s
	iters: 1600, epoch: 1 | loss: 0.9453125
	speed: 0.2542s/iter; left time: 4424.0949s
	iters: 1700, epoch: 1 | loss: 0.6093750
	speed: 0.2549s/iter; left time: 4409.6142s
	iters: 1800, epoch: 1 | loss: 0.7968750
	speed: 0.2545s/iter; left time: 4378.5060s
	iters: 1900, epoch: 1 | loss: 0.8671875
	speed: 0.2552s/iter; left time: 4364.3604s
Epoch: 1 cost time: 482.9799118041992
Epoch: 1, Steps: 1900 | Train Loss: 0.9050051 Vali Loss: 0.7233546 Test Loss: 0.8027648
Validation loss decreased (inf --> 0.723355).  Saving model ...
Updating learning rate to 0.0001
	iters: 100, epoch: 2 | loss: 0.8476562
	speed: 1.4145s/iter; left time: 24047.9126s
	iters: 200, epoch: 2 | loss: 0.7500000
	speed: 0.2531s/iter; left time: 4278.3302s
	iters: 300, epoch: 2 | loss: 1.0234375
	speed: 0.2530s/iter; left time: 4249.9756s
	iters: 400, epoch: 2 | loss: 1.0859375
	speed: 0.2556s/iter; left time: 4269.5724s
	iters: 500, epoch: 2 | loss: 0.6875000
	speed: 0.2540s/iter; left time: 4217.1143s
	iters: 600, epoch: 2 | loss: 0.5898438
	speed: 0.2540s/iter; left time: 4190.8095s
	iters: 700, epoch: 2 | loss: 0.9609375
	speed: 0.2540s/iter; left time: 4166.3545s
	iters: 800, epoch: 2 | loss: 0.9921875
	speed: 0.2541s/iter; left time: 4141.3224s
	iters: 900, epoch: 2 | loss: 0.8945312
	speed: 0.2530s/iter; left time: 4099.0450s
	iters: 1000, epoch: 2 | loss: 0.7929688
	speed: 0.2543s/iter; left time: 4094.1238s
	iters: 1100, epoch: 2 | loss: 1.1328125
	speed: 0.2539s/iter; left time: 4062.9216s
	iters: 1200, epoch: 2 | loss: 0.9062500
	speed: 0.2547s/iter; left time: 4049.6452s
	iters: 1300, epoch: 2 | loss: 0.6562500
	speed: 0.2537s/iter; left time: 4008.2098s
	iters: 1400, epoch: 2 | loss: 0.9687500
	speed: 0.2537s/iter; left time: 3982.7868s
	iters: 1500, epoch: 2 | loss: 0.8398438
	speed: 0.2536s/iter; left time: 3957.0914s
	iters: 1600, epoch: 2 | loss: 0.6054688
	speed: 0.2531s/iter; left time: 3922.7651s
	iters: 1700, epoch: 2 | loss: 0.7929688
	speed: 0.2551s/iter; left time: 3929.5499s
	iters: 1800, epoch: 2 | loss: 0.5898438
	speed: 0.2544s/iter; left time: 3892.9197s
	iters: 1900, epoch: 2 | loss: 0.9882812
	speed: 0.2544s/iter; left time: 3867.1042s
Epoch: 2 cost time: 482.6253011226654
Epoch: 2, Steps: 1900 | Train Loss: 0.8707329 Vali Loss: 0.7435035 Test Loss: 0.8379794
EarlyStopping counter: 1 out of 8
Updating learning rate to 5e-05
	iters: 100, epoch: 3 | loss: 0.7578125
	speed: 1.4075s/iter; left time: 21254.1938s
	iters: 200, epoch: 3 | loss: 1.3281250
	speed: 0.2542s/iter; left time: 3812.5543s
	iters: 300, epoch: 3 | loss: 1.0234375
	speed: 0.2551s/iter; left time: 3801.4492s
	iters: 400, epoch: 3 | loss: 0.7968750
	speed: 0.2532s/iter; left time: 3747.5417s
	iters: 500, epoch: 3 | loss: 1.2265625
	speed: 0.2531s/iter; left time: 3721.3157s
	iters: 600, epoch: 3 | loss: 0.8242188
	speed: 0.2529s/iter; left time: 3693.2642s
	iters: 700, epoch: 3 | loss: 0.9375000
	speed: 0.2536s/iter; left time: 3677.4211s
	iters: 800, epoch: 3 | loss: 0.7500000
	speed: 0.2527s/iter; left time: 3639.4385s
	iters: 900, epoch: 3 | loss: 0.8828125
	speed: 0.2527s/iter; left time: 3614.2203s
	iters: 1000, epoch: 3 | loss: 1.0234375
	speed: 0.2532s/iter; left time: 3595.8472s
	iters: 1100, epoch: 3 | loss: 1.0078125
	speed: 0.2530s/iter; left time: 3567.2677s
	iters: 1200, epoch: 3 | loss: 0.8164062
	speed: 0.2545s/iter; left time: 3563.2699s
	iters: 1300, epoch: 3 | loss: 0.8203125
	speed: 0.2534s/iter; left time: 3522.3606s
	iters: 1400, epoch: 3 | loss: 0.9570312
	speed: 0.2534s/iter; left time: 3496.8589s
	iters: 1500, epoch: 3 | loss: 1.0156250
	speed: 0.2527s/iter; left time: 3462.7387s
	iters: 1600, epoch: 3 | loss: 0.7421875
	speed: 0.2528s/iter; left time: 3438.8527s
	iters: 1700, epoch: 3 | loss: 0.8085938
	speed: 0.2533s/iter; left time: 3420.0784s
	iters: 1800, epoch: 3 | loss: 0.9023438
	speed: 0.2534s/iter; left time: 3396.2142s
	iters: 1900, epoch: 3 | loss: 0.6015625
	speed: 0.2544s/iter; left time: 3383.2604s
Epoch: 3 cost time: 481.6236810684204
Epoch: 3, Steps: 1900 | Train Loss: 0.8503351 Vali Loss: 0.7130034 Test Loss: 0.7713588
Validation loss decreased (0.723355 --> 0.713003).  Saving model ...
Updating learning rate to 2.5e-05
	iters: 100, epoch: 4 | loss: 0.9335938
	speed: 1.4131s/iter; left time: 18653.9345s
	iters: 200, epoch: 4 | loss: 0.8164062
	speed: 0.2522s/iter; left time: 3303.8952s
	iters: 300, epoch: 4 | loss: 0.6015625
	speed: 0.2523s/iter; left time: 3279.5129s
	iters: 400, epoch: 4 | loss: 1.2109375
	speed: 0.2556s/iter; left time: 3296.8936s
	iters: 500, epoch: 4 | loss: 0.8593750
	speed: 0.2532s/iter; left time: 3240.8636s
	iters: 600, epoch: 4 | loss: 0.6250000
	speed: 0.2542s/iter; left time: 3228.3599s
	iters: 700, epoch: 4 | loss: 0.7304688
	speed: 0.2538s/iter; left time: 3197.7734s
	iters: 800, epoch: 4 | loss: 0.7773438
	speed: 0.2534s/iter; left time: 3167.1649s
	iters: 900, epoch: 4 | loss: 0.7734375
	speed: 0.2537s/iter; left time: 3145.9349s
	iters: 1000, epoch: 4 | loss: 0.9570312
	speed: 0.2535s/iter; left time: 3118.5357s
	iters: 1100, epoch: 4 | loss: 0.8398438
	speed: 0.2528s/iter; left time: 3083.8748s
	iters: 1200, epoch: 4 | loss: 1.2343750
	speed: 0.2546s/iter; left time: 3081.3466s
	iters: 1300, epoch: 4 | loss: 1.1484375
	speed: 0.2531s/iter; left time: 3037.3184s
	iters: 1400, epoch: 4 | loss: 0.7226562
	speed: 0.2533s/iter; left time: 3014.6006s
	iters: 1500, epoch: 4 | loss: 0.8085938
	speed: 0.2532s/iter; left time: 2988.1945s
	iters: 1600, epoch: 4 | loss: 0.9765625
	speed: 0.2536s/iter; left time: 2966.7971s
	iters: 1700, epoch: 4 | loss: 0.8671875
	speed: 0.2545s/iter; left time: 2952.5815s
	iters: 1800, epoch: 4 | loss: 1.2656250
	speed: 0.2534s/iter; left time: 2914.1770s
	iters: 1900, epoch: 4 | loss: 0.8671875
	speed: 0.2545s/iter; left time: 2901.0996s
Epoch: 4 cost time: 481.8166072368622
Epoch: 4, Steps: 1900 | Train Loss: 0.8442157 Vali Loss: 0.7071066 Test Loss: 0.7677504
Validation loss decreased (0.713003 --> 0.707107).  Saving model ...
Updating learning rate to 1.25e-05
	iters: 100, epoch: 5 | loss: 0.7500000
	speed: 1.4125s/iter; left time: 15962.1500s
	iters: 200, epoch: 5 | loss: 0.7539062
	speed: 0.2535s/iter; left time: 2839.7612s
	iters: 300, epoch: 5 | loss: 1.2578125
	speed: 0.2550s/iter; left time: 2830.6167s
	iters: 400, epoch: 5 | loss: 0.7148438
	speed: 0.2534s/iter; left time: 2787.5958s
	iters: 500, epoch: 5 | loss: 0.8750000
	speed: 0.2553s/iter; left time: 2782.9299s
	iters: 600, epoch: 5 | loss: 0.6210938
	speed: 0.2535s/iter; left time: 2738.5667s
	iters: 700, epoch: 5 | loss: 0.8437500
	speed: 0.2543s/iter; left time: 2720.8885s
	iters: 800, epoch: 5 | loss: 0.8476562
	speed: 0.2544s/iter; left time: 2696.5312s
	iters: 900, epoch: 5 | loss: 0.6015625
	speed: 0.2543s/iter; left time: 2670.5708s
	iters: 1000, epoch: 5 | loss: 0.8203125
	speed: 0.2540s/iter; left time: 2642.3151s
	iters: 1100, epoch: 5 | loss: 0.8164062
	speed: 0.2532s/iter; left time: 2608.4872s
	iters: 1200, epoch: 5 | loss: 0.5781250
	speed: 0.2541s/iter; left time: 2591.9165s
	iters: 1300, epoch: 5 | loss: 0.8671875
	speed: 0.2536s/iter; left time: 2561.9853s
	iters: 1400, epoch: 5 | loss: 0.6992188
	speed: 0.2542s/iter; left time: 2542.0982s
	iters: 1500, epoch: 5 | loss: 0.7187500
	speed: 0.2540s/iter; left time: 2514.6893s
	iters: 1600, epoch: 5 | loss: 0.7148438
	speed: 0.2530s/iter; left time: 2479.9805s
	iters: 1700, epoch: 5 | loss: 0.7500000
	speed: 0.2537s/iter; left time: 2461.3902s
	iters: 1800, epoch: 5 | loss: 0.7968750
	speed: 0.2546s/iter; left time: 2444.6320s
	iters: 1900, epoch: 5 | loss: 0.8632812
	speed: 0.2540s/iter; left time: 2413.2942s
Epoch: 5 cost time: 482.72309398651123
Epoch: 5, Steps: 1900 | Train Loss: 0.8423057 Vali Loss: 0.7056509 Test Loss: 0.7710233
Validation loss decreased (0.707107 --> 0.705651).  Saving model ...
Updating learning rate to 6.25e-06
	iters: 100, epoch: 6 | loss: 1.0078125
	speed: 1.4117s/iter; left time: 13271.2741s
	iters: 200, epoch: 6 | loss: 0.8398438
	speed: 0.2566s/iter; left time: 2386.2394s
	iters: 300, epoch: 6 | loss: 0.9257812
	speed: 0.2549s/iter; left time: 2345.6874s
	iters: 400, epoch: 6 | loss: 0.9414062
	speed: 0.2552s/iter; left time: 2322.3321s
	iters: 500, epoch: 6 | loss: 1.0312500
	speed: 0.2551s/iter; left time: 2296.4299s
	iters: 600, epoch: 6 | loss: 1.1406250
	speed: 0.2552s/iter; left time: 2271.1708s
	iters: 700, epoch: 6 | loss: 0.7421875
	speed: 0.2553s/iter; left time: 2246.9825s
	iters: 800, epoch: 6 | loss: 0.7500000
	speed: 0.2554s/iter; left time: 2222.3297s
	iters: 900, epoch: 6 | loss: 0.5820312
	speed: 0.2554s/iter; left time: 2197.0765s
	iters: 1000, epoch: 6 | loss: 0.5937500
	speed: 0.2548s/iter; left time: 2166.1186s
	iters: 1100, epoch: 6 | loss: 0.8789062
	speed: 0.2562s/iter; left time: 2152.7550s
	iters: 1200, epoch: 6 | loss: 0.9765625
	speed: 0.2549s/iter; left time: 2115.7112s
	iters: 1300, epoch: 6 | loss: 0.9335938
	speed: 0.2550s/iter; left time: 2091.0198s
	iters: 1400, epoch: 6 | loss: 0.8945312
	speed: 0.2554s/iter; left time: 2068.7208s
	iters: 1500, epoch: 6 | loss: 1.1875000
	speed: 0.2553s/iter; left time: 2042.4730s
	iters: 1600, epoch: 6 | loss: 0.5703125
	speed: 0.2566s/iter; left time: 2027.6299s
	iters: 1700, epoch: 6 | loss: 0.9531250
	speed: 0.2553s/iter; left time: 1991.4967s
	iters: 1800, epoch: 6 | loss: 0.7187500
	speed: 0.2552s/iter; left time: 1965.4671s
	iters: 1900, epoch: 6 | loss: 0.7929688
	speed: 0.2555s/iter; left time: 1941.9092s
Epoch: 6 cost time: 485.2269947528839
Epoch: 6, Steps: 1900 | Train Loss: 0.8426059 Vali Loss: 0.7069148 Test Loss: 0.7698831
EarlyStopping counter: 1 out of 8
Updating learning rate to 3.125e-06
	iters: 100, epoch: 7 | loss: 0.7539062
	speed: 1.4055s/iter; left time: 10542.4660s
	iters: 200, epoch: 7 | loss: 0.9804688
	speed: 0.2552s/iter; left time: 1888.8675s
	iters: 300, epoch: 7 | loss: 0.7695312
	speed: 0.2535s/iter; left time: 1851.1537s
	iters: 400, epoch: 7 | loss: 0.7734375
	speed: 0.2537s/iter; left time: 1826.9719s
	iters: 500, epoch: 7 | loss: 1.0781250
	speed: 0.2538s/iter; left time: 1802.3875s
	iters: 600, epoch: 7 | loss: 0.9414062
	speed: 0.2541s/iter; left time: 1779.0206s
	iters: 700, epoch: 7 | loss: 0.7343750
	speed: 0.2539s/iter; left time: 1752.5081s
	iters: 800, epoch: 7 | loss: 0.8632812
	speed: 0.2548s/iter; left time: 1733.0166s
	iters: 900, epoch: 7 | loss: 0.6953125
	speed: 0.2540s/iter; left time: 1702.1877s
	iters: 1000, epoch: 7 | loss: 1.0312500
	speed: 0.2551s/iter; left time: 1684.0291s
	iters: 1100, epoch: 7 | loss: 0.8789062
	speed: 0.2539s/iter; left time: 1650.4107s
	iters: 1200, epoch: 7 | loss: 0.6601562
	speed: 0.2538s/iter; left time: 1624.8030s
	iters: 1300, epoch: 7 | loss: 0.5859375
	speed: 0.2536s/iter; left time: 1597.6320s
	iters: 1400, epoch: 7 | loss: 0.8828125
	speed: 0.2546s/iter; left time: 1578.9558s
	iters: 1500, epoch: 7 | loss: 0.8320312
	speed: 0.2531s/iter; left time: 1544.3389s
	iters: 1600, epoch: 7 | loss: 1.0468750
	speed: 0.2538s/iter; left time: 1523.1131s
	iters: 1700, epoch: 7 | loss: 0.6914062
	speed: 0.2543s/iter; left time: 1500.6231s
	iters: 1800, epoch: 7 | loss: 0.9609375
	speed: 0.2537s/iter; left time: 1471.4657s
	iters: 1900, epoch: 7 | loss: 0.7500000
	speed: 0.2563s/iter; left time: 1461.3953s
Epoch: 7 cost time: 482.98682260513306
Epoch: 7, Steps: 1900 | Train Loss: 0.8427344 Vali Loss: 0.7077114 Test Loss: 0.7706939
EarlyStopping counter: 2 out of 8
Updating learning rate to 1.5625e-06
	iters: 100, epoch: 8 | loss: 0.8007812
	speed: 1.4038s/iter; left time: 7862.7447s
	iters: 200, epoch: 8 | loss: 0.6875000
	speed: 0.2520s/iter; left time: 1386.2713s
	iters: 300, epoch: 8 | loss: 0.8437500
	speed: 0.2531s/iter; left time: 1367.1282s
	iters: 400, epoch: 8 | loss: 0.8867188
	speed: 0.2539s/iter; left time: 1346.0537s
	iters: 500, epoch: 8 | loss: 0.9257812
	speed: 0.2560s/iter; left time: 1331.2498s
	iters: 600, epoch: 8 | loss: 0.7890625
	speed: 0.2539s/iter; left time: 1295.0762s
	iters: 700, epoch: 8 | loss: 0.8710938
	speed: 0.2538s/iter; left time: 1269.3225s
	iters: 800, epoch: 8 | loss: 0.7851562
	speed: 0.2530s/iter; left time: 1240.0315s
	iters: 900, epoch: 8 | loss: 0.8125000
	speed: 0.2536s/iter; left time: 1217.5325s
	iters: 1000, epoch: 8 | loss: 0.5546875
	speed: 0.2533s/iter; left time: 1190.8341s
	iters: 1100, epoch: 8 | loss: 0.8593750
	speed: 0.2532s/iter; left time: 1165.1326s
	iters: 1200, epoch: 8 | loss: 0.9140625
	speed: 0.2543s/iter; left time: 1144.5299s
	iters: 1300, epoch: 8 | loss: 0.7382812
	speed: 0.2541s/iter; left time: 1118.2363s
	iters: 1400, epoch: 8 | loss: 0.9414062
	speed: 0.2537s/iter; left time: 1091.2665s
	iters: 1500, epoch: 8 | loss: 0.7500000
	speed: 0.2528s/iter; left time: 1062.1857s
	iters: 1600, epoch: 8 | loss: 0.8710938
	speed: 0.2530s/iter; left time: 1037.4148s
	iters: 1700, epoch: 8 | loss: 1.1093750
	speed: 0.2534s/iter; left time: 1014.0148s
	iters: 1800, epoch: 8 | loss: 0.8671875
	speed: 0.2550s/iter; left time: 994.6699s
	iters: 1900, epoch: 8 | loss: 0.7265625
	speed: 0.2541s/iter; left time: 965.6462s
Epoch: 8 cost time: 481.9678473472595
Epoch: 8, Steps: 1900 | Train Loss: 0.8421937 Vali Loss: 0.7074375 Test Loss: 0.7717099
EarlyStopping counter: 3 out of 8
Updating learning rate to 7.8125e-07
	iters: 100, epoch: 9 | loss: 0.6210938
	speed: 1.4068s/iter; left time: 5206.4758s
	iters: 200, epoch: 9 | loss: 1.3828125
	speed: 0.2540s/iter; left time: 914.7414s
	iters: 300, epoch: 9 | loss: 0.6367188
	speed: 0.2537s/iter; left time: 888.2703s
	iters: 400, epoch: 9 | loss: 0.8554688
	speed: 0.2549s/iter; left time: 867.0006s
	iters: 500, epoch: 9 | loss: 0.7851562
	speed: 0.2533s/iter; left time: 836.1877s
	iters: 600, epoch: 9 | loss: 0.9335938
	speed: 0.2534s/iter; left time: 811.1824s
	iters: 700, epoch: 9 | loss: 0.9062500
	speed: 0.2539s/iter; left time: 787.3934s
	iters: 800, epoch: 9 | loss: 0.7343750
	speed: 0.2540s/iter; left time: 762.2617s
	iters: 900, epoch: 9 | loss: 0.8632812
	speed: 0.2538s/iter; left time: 736.3480s
	iters: 1000, epoch: 9 | loss: 0.8437500
	speed: 0.2535s/iter; left time: 710.0472s
	iters: 1100, epoch: 9 | loss: 0.7460938
	speed: 0.2542s/iter; left time: 686.6175s
	iters: 1200, epoch: 9 | loss: 1.0078125
	speed: 0.2551s/iter; left time: 663.5041s
	iters: 1300, epoch: 9 | loss: 0.6835938
	speed: 0.2534s/iter; left time: 633.8275s
	iters: 1400, epoch: 9 | loss: 0.5351562
	speed: 0.2533s/iter; left time: 608.2253s
	iters: 1500, epoch: 9 | loss: 0.7617188
	speed: 0.2539s/iter; left time: 584.1267s
	iters: 1600, epoch: 9 | loss: 0.9179688
	speed: 0.2531s/iter; left time: 557.0517s
	iters: 1700, epoch: 9 | loss: 0.6250000
	speed: 0.2558s/iter; left time: 537.3691s
	iters: 1800, epoch: 9 | loss: 0.8085938
	speed: 0.2538s/iter; left time: 507.9049s
	iters: 1900, epoch: 9 | loss: 1.0156250
	speed: 0.2547s/iter; left time: 484.2499s
Epoch: 9 cost time: 482.6481831073761
Epoch: 9, Steps: 1900 | Train Loss: 0.8428475 Vali Loss: 0.7071545 Test Loss: 0.7707952
EarlyStopping counter: 4 out of 8
Updating learning rate to 3.90625e-07
	iters: 100, epoch: 10 | loss: 0.9765625
	speed: 1.4071s/iter; left time: 2534.2391s
	iters: 200, epoch: 10 | loss: 0.5429688
	speed: 0.2529s/iter; left time: 430.2645s
	iters: 300, epoch: 10 | loss: 0.7304688
	speed: 0.2555s/iter; left time: 409.0539s
	iters: 400, epoch: 10 | loss: 0.5742188
	speed: 0.2533s/iter; left time: 380.1708s
	iters: 500, epoch: 10 | loss: 0.8359375
	speed: 0.2536s/iter; left time: 355.2600s
	iters: 600, epoch: 10 | loss: 0.7656250
	speed: 0.2534s/iter; left time: 329.7009s
	iters: 700, epoch: 10 | loss: 0.6250000
	speed: 0.2540s/iter; left time: 305.0063s
	iters: 800, epoch: 10 | loss: 1.3046875
	speed: 0.2531s/iter; left time: 278.6989s
	iters: 900, epoch: 10 | loss: 0.6328125
	speed: 0.2530s/iter; left time: 253.2299s
	iters: 1000, epoch: 10 | loss: 0.6640625
	speed: 0.2537s/iter; left time: 228.5506s
	iters: 1100, epoch: 10 | loss: 0.9687500
	speed: 0.2532s/iter; left time: 202.8397s
	iters: 1200, epoch: 10 | loss: 0.9960938
	speed: 0.2540s/iter; left time: 178.0535s
	iters: 1300, epoch: 10 | loss: 0.7343750
	speed: 0.2534s/iter; left time: 152.2792s
	iters: 1400, epoch: 10 | loss: 0.7382812
	speed: 0.2526s/iter; left time: 126.5672s
	iters: 1500, epoch: 10 | loss: 0.7031250
	speed: 0.2531s/iter; left time: 101.4867s
	iters: 1600, epoch: 10 | loss: 0.7890625
	speed: 0.2539s/iter; left time: 76.4287s
	iters: 1700, epoch: 10 | loss: 1.2656250
	speed: 0.2530s/iter; left time: 50.8593s
	iters: 1800, epoch: 10 | loss: 0.7929688
	speed: 0.2531s/iter; left time: 25.5656s
	iters: 1900, epoch: 10 | loss: 0.8437500
	speed: 0.2543s/iter; left time: 0.2543s
Epoch: 10 cost time: 481.70411109924316
Epoch: 10, Steps: 1900 | Train Loss: 0.8424250 Vali Loss: 0.7069041 Test Loss: 0.7702057
EarlyStopping counter: 5 out of 8
Updating learning rate to 1.953125e-07
>>>>>>>testing : TimeLLM_TimeLLM_custom_M_ft96_sl48_ll24_pl512_dm8_nh2_el1_dl2048_df3_fctimeF_ebTrue_dtExp_projection_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
2212   2016-04-02 10:00:00+00:00
2213   2016-04-02 11:00:00+00:00
2214   2016-04-02 12:00:00+00:00
2215   2016-04-02 13:00:00+00:00
2216   2016-04-02 14:00:00+00:00
Name: date, dtype: datetime64[ns, UTC]
datetime64[ns, UTC]
test 554
