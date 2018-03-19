### ECE-239AS-Project

#### Setup
- Download the data and place it in a directory called `project_datasets/`

#### Running on GPU
- ssh into gpu box: `ssh acm@131.179.54.92
- bug rohan/nikhil for password
- activate environment: "source ~/tensorflow/bin/activate"



Accuracies:

[0.38396624472573837, 0.38983050847457623, 0.3771186440677966, 0.3076923076923077, 0.3879310344827587, 0.3829787234042553, 0.3445378151260504, 0.3318965517241379, 0.35526315789473684], mean is 0.36235722084359534
➜  project git:(master) ✗ python3 lstm-gpu.py --hidden-dim 200 --bidirectional --lr 0.001