for seed in 4293 1692 3029
do
  python train.py --exp_name traffic --model_name ATA_gp --attn_type ATA --denoising True --gp True --seed 4293 --cuda cuda:0
done