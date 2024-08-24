python run.py with data_root="/home/autolab/zhushatong/others/data/RSICD-new" num_gpus=1 num_nodes=1 per_gpu_batchsize=4 task_finetune_irtr_f30k_randaug test_only=True precision=32 load_path="/home/autolab/zhushatong/others/mlm_itm.ckpt" 

python run.py with data_root="/home/autolab/zhushatong/others/data/UCM_captions-new" num_gpus=1 num_nodes=1 per_gpu_batchsize=4 task_finetune_irtr_f30k_randaug test_only=True precision=32 load_path="/home/autolab/zhushatong/others/mlm_itm.ckpt" 

python run.py with data_root="/home/autolab/zhushatong/others/data/RSITMD-new" num_gpus=1 num_nodes=1 per_gpu_batchsize=4 task_finetune_irtr_f30k_randaug test_only=True precision=32 load_path="/home/autolab/zhushatong/others/f30k.ckpt" 

python run.py with data_root="/home/autolab/zhushatong/others/data/UCM_captions-new" num_gpus=1 num_nodes=1 per_gpu_batchsize=4 task_finetune_irtr_f30k_randaug test_only=True precision=32 load_path="/home/autolab/zhushatong/others/f30k.ckpt" 


conda create -n vilt python=3.8
pip install --upgrade pytorch_lightning
pytorch-lightning==1.9.3


CUDA_LAUNCH_BLOCKING=1 

export MASTER_ADDR=localhost
export MASTER_PORT=12355

export CUDA_VISIBLE_DEVICES=0