This repo is for the final project of Machine Learning in 2023 based on Simswap

We provide several models for face swap, including Deform, dancer, shifter(the results may be discouraging... perhaps we have no enough computation resources to train longer steps)

Run dancer:
```bash
python train.py --model_name dancer --dataset /content/TrainingData/vggface2_crop_arcfacealign_224 --lambda_cycle 0 --lambda_gp 0 --n_blocks 3 --n_layers 3 --model_freq 3000 --total_step 100000
```

Run shifter:
```bash
python train.py --model_name simplified --dataset /kaggle/working/TrainingData/vggface2_crop_arcfacealign_224 --lambda_gp 0 --n_blocks 6 --n_layers 5 --model_freq 3000 --total_step 100000 --kernel_type deform
```

Run deform:
```bash
python train.py --model_name simswap+=+ --name simswap --dataset /kaggle/working/TrainingData/vggface2_crop_arcfacealign_224 --lambda_gp 0 --n_blocks 6 --n_layers 3 --model_freq 20000 --total_step 100000 --kernel_type deform
```

In the wandb page, there are different running environments. You can find it by group the runs by batchSize. 4 is on Colab, with this repo, and 8 is on another group member's server, with the original simswap repo.
