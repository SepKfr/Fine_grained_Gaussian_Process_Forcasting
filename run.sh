for seed in $[RANDOM%10000] $[RANDOM%10000] $[RANDOM%10000]
do
  python train.py --exp_name traffic --model_name ATA_gp --attn_type ATA --denoising True --gp True --seed "${seed}" --cuda cuda:0
done