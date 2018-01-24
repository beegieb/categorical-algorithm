# Categorical Algorithm

## Install Dependencies 
```bash
git clone git@github.com:beegieb/categorical-algorithm.git

cd categorical-algorithm

# Best to do this in a virtualenv
pip install -r requirements.txt
```

## Training a model
```bash
mkdir checkpoints

python train.py --config=models/model01.json --game=breakout --checkpoint-dir=checkpoints
```

## Evaluating a model
```bash
python evaluate --config=models/model01.json --game=breakout --checkpoint-dir=checkpoints --eps=0.01 --render=True
```

## Contact deets
If you need any assistance or find any bugs, do not hesitate to make a pull request or reach out at: 

miroslaw AT gmail DOT com 
