for seed in $[RANDOM%10000] $[RANDOM%10000] $[RANDOM%10000]
do
  python train.py --exp_name solar --model_name ATA_gp --attn_type ATA --denoising True --gp True --seed "${seed}" --cuda cuda:0
  python train.py --exp_name solar --model_name ATA_iso --attn_type ATA --denoising True --gp False --seed "${seed}" --cuda cuda:0
  python train.py --exp_name solar --model_name ATA_no --attn_type ATA --denoising False --gp False --seed "${seed}" --cuda cuda:0
done