## Environment version

- Python 3.8.2
- CUDA 11.4
- MuJoCo210


## Dependencies

- torch=1.12.1
- gym=0.21.0
- mujoco_py=2.1.2.14
- hydra=1.2.0
- tqdm=4.64.1

## Benchmark on MuJuCo

### CAPQL

```
python main.py model.type=CAPQL training.alpha=0.1 name=HopperM-v0
```

alpha controls the strength of the augmentation effect (see Eq 8)

You can choose among the following environment name:

name = HopperM-v0, Walker2dM-v0, HalfCheetahM-v0, HumanoidEnvM-v0

### QENV_CTN

```
python main.py model.type=QENV_CTN name=HopperM-v0
```

You can choose among the following environment name:

name = HopperM-v0, Walker2dM-v0, HalfCheetahM-v0, HumanoidEnvM-v0
