


python train_ATD.py  --run_name cifar10-svhn --model_type "fea" --training_type "adv" \
    --in_dataset cifar10-svhn --alpha 0.5 --batch_size 128 --num_epochs 20 --eps 0.0313 --attack_iters 10 \
    --regime Balanced --ood_dset cifar10-svhn --n_ood 64 --valsize 10000