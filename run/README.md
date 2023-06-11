## This directory contains the submission code for the project

To run the code, you can run:

```
python run/run_sgd.py
```

depending on your script. You can add the `--help` flag to see the available options.

For testing, it is recommended to run with the following command:

```
python run/run_sgd.py --num_steps 200 --device mps --test-every-x-steps 10 --num-runs 2
```

For the tests, we run for 2000 steps, testing every 10 steps, and for 5 runs. IE, using the command:

```
python run/run_sgd.py --num_steps 2000 --device mps --test-every-x-steps 10 --num-runs 5
```
