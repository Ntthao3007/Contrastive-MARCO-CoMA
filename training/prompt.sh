CUDA_VISIBBLE_DEVICES=1 nohup python3 /home/ubuntu/20thao.nt/TST/MarcoDetoxification/training/finetune_bart.py > marco_v2.log &
CUDA_VISIBBLE_DEVICES=1 nohup python3 /home/ubuntu/20thao.nt/TST/MarcoDetoxification/training/finetune_bart_contrastive.py --contrastive_loss --add_negatives --alpha 0.5 > marcocontra_v1.log &
CUDA_VISIBBLE_DEVICES=1 python3 /home/ubuntu/20thao.nt/TST/MarcoDetoxification/training/finetune_bart.py 
CUDA_VISIBBLE_DEVICES=1  python eval_seq2seq_model.py --model_name /home/ubuntu/20thao.nt/TST/count-style-transfer/output_dir_marco/EXP1/best --save_path /home/ubuntu/20thao.nt/TST/count-style-transfer/output_dir_marco/contra_para --dataset 'paradetox' --fold 'test' --name 'test_eval'  --make_preds --evaluate 
