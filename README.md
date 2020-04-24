### NetFrame is a flexible and scalable deep-learning framework to build segmentation model on large scale pathological images.

****

### Environment

python2.7

pip install -r requirements.txt


### Configuration

NetFame uses a configuration file of yaml to set up all the parameters.

|Argument |Comment |Example|
|:-----  |:-----|----- |
| random_mirror | Randomly mirrors the image | True |
| color_jitter | Randomly distorts color | True |
| random_blur | Randomly blurs the image | True |
| random_scale | Randomly scales the images between 0.75 to 1.25 times of the original size. | True |
| num_classes | Number of classes | 2 |
| learning_rate | Initial learning rate | 0.003 |
| lr_decay_step | The number of steps in a learning rate decay cycle | 15000 |
| lr_decay_ratio | Learning rate decay ratio | 0.5 |
| batch_size | Batch size | 32 |
| resnet_layer | Number of layers of the ResNet backbone | 50 |
| input_size | Input size | 320 |
| patch_size | Patch size | 320 |
| l2_loss_lambda | Factor for weight decay loss | !!float 1e-5 |
| restore_iters | Restored model corresponding to the iterations | 0 |
| log_label | Logging the metrics of the label | 1 |
| save_step | The number of steps model is saved in | 2000 |
| max_epoch | The maxmum training epoch | 10 |
| gpu | Specified the list of gpus to use | [0,1] |


### Build a New Model

#### Training

1. Create a configuration file setting all the parameters used for training; 

2. Implemente the `data` and `model` module under the `project` directory by inheriting the default parent class;

3. Run the training program, eg: 

    ```
    python main.py \
        --config_file ./config/stomach/config_stomach_v0.yaml \
        --version v0_0 \
        --mode train \
    ```

    
#### Inference

1. Change the value of argument `mode` from `train` to `test`;

2. Specify the number of iterations of the model to be tested;

3. Run the test program, eg: 

    ```
    python main.py \
        --config_file ./config/stomach/config_stomach_v0.yaml \
        --version v0_0 \
        --mode test \
        --restore_iters 50000
    ```
