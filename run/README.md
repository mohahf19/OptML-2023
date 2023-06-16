## This directory contains the submission code for the project

To run the code, you can run:

```
python run/run_{algorithm}.py
```

where `algorithm` could be `sgd`, `saga`, or `svrg`. The available command line arguments are:

```
usage: run_{algorithm}.py [-h] [--num-steps NUM_STEPS] [--device DEVICE] [--test-every-x-steps TEST_EVERY_X_STEPS]
                  [--num-runs NUM_RUNS] [--seed SEED] [--num-parts NUM_PARTS] [--batch-size BATCH_SIZE]
                  [--lr LR]

options:
  -h, --help            show this help message and exit
  --num-steps NUM_STEPS
  --device DEVICE
  --test-every-x-steps TEST_EVERY_X_STEPS
  --num-runs NUM_RUNS
  --seed SEED
  --num-parts NUM_PARTS
  --batch-size BATCH_SIZE
  --lr LR
```

For testing, it is recommended to run with the following command:

```
python run/run_sgd.py --num_steps 200 --device mps --test-every-x-steps 10 --num-runs 2
```

For the tests, we run for 2000 steps, testing every 10 steps, and for 5 runs. IE, using the command:

```
python run/run_sgd.py --num_steps 2000 --device mps --test-every-x-steps 10 --num-runs 5
```

For the `sgd` and `svrg` runs, we run the following commands:

```
python run/run_{sgd/svrg}.py --num-steps 3000 --batch-size {BS} --device mps --test-every-x-steps 20 --lr 1e-2  --num-runs 5
```

where `BS` is 1, 16, 64 and 128. For `saga` we run the following command:

```
python run/run_saga.py --num-steps 3000 --batch-size 1 --device mps --test-every-x-steps 20 --lr 1e-2  --num-runs 5 --num-parts 10
```

and for more information, see the `run.ipynb` script in the top level directory.
