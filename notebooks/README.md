# Colab notebooks

We could use the template to run our github repo scripts in colab.

Another way is that you can access the notebooks [here](https://colab.research.google.com/drive/1-vk1A_K_ZIW-WzjeY_DPpU40tZaG3TVo?usp=sharing) and run them directly. Please make sure to save the results.

## How to use

To run the scripts, say `neuralnets/batched/train_svrg.py`, you can run

```
!python neuralnets/batched/train_svrg.py --help
```

and we have the following command line arguments:

```
usage: train_svrg.py [-h] [--seed SEED] [--batch-size BATCH_SIZE] [--device DEVICE]
                     [--num_epochs NUM_EPOCHS]

options:
  -h, --help            show this help message and exit
  --seed SEED
  --batch-size BATCH_SIZE
  --device DEVICE
  --num_epochs NUM_EPOCHS
```

Note that they all have default values so none need to be passed.
