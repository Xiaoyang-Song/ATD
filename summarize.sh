

# bash summarize.sh > summary.txt
python summarize.py --experiment mnist --regime Balanced
python summarize.py --experiment fashionmnist --regime Balanced
python summarize.py --experiment mnist-fashionmnist --regime Balanced
python summarize.py --experiment svhn --regime Balanced
python summarize.py --experiment cifar10-svhn --regime Balanced

python summarize.py --experiment fashionmnist-R2 --regime Imbalanced
python summarize.py --experiment svhn-R2 --regime Imbalanced