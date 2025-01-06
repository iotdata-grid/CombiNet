# Edge-Cloud Collaborative Real-Time Video Object Detection for Industrial Surveillance Systems
The implementation of CombiNet, which propose an edge-cloud collaborative branchy CNN combining rapid edge object tracking and accurate edge-cloud collaborative object detection.


## Dependencies

1. Python 3.6+
2. Opencv
3. Pytorch 1.0 +
4. torch-vision
5. CUDA 10.0+



## Dataset

Download Imagenet VID 2015 dataset from [here](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php). This is the link for ILSVRC2017 as the link for ILSVRC2015 seems to down now, and get list of training, validation and test dataset in [here](https://drive.google.com/drive/folders/1g_d0Cok10C035IM-csxj5Y_3nh-qYG3x?usp=sharing)。
You can use your own custom dataset following the format of ILSVRC2017.


## Train

Make sure to be in python 3.6+ environment with all the dependencies installed. 

#### Detection Branch

The detection branch uses MobileNetV2-based SSD. 
The SSD come from below link. **Thanks for their open source work.**

- SSD:  [VIKRANT7/ssd](https://github.com/vikrant7/pytorch-looking-fast-and-slow)

```
$ cd ./CombiNet/cloud/ssd_cloud/
$ CUDA_VISIBLE_DEVICES=$GPU_ID python train.py 
```


#### Tracking Branch

The detection branch uses MobileNetV2-based SiameseRPN. 

The SiamRPN come from below link. **Thanks for their open source work.**

- SiamRPN:  [STVIR/pysot](https://github.com/STVIR/pysot)



```
$ cd ./CombiNet/edge/ssd_edge/
$ CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=2333 \
    --cfg config.yaml\
    ./tools/train.py 
```


## Evaluation

Before following steps, you should run **Emqx**  in docker to support the communication between edge and cloud container. The command is: 

```
$ docker run -d --name combinet_broker -p {YOUR_EMQX_PORT}:1883 emqx/emqx:v4.1.1
```

For the edge, we use the following commend to run the `./CombiNet/edge/ssd_edge/tools/eval.py` and the corresponding docker environment for starting the edge docker

```
$ cd ./CombiNet/edge/ssd_edge/
$ docker build -f Dockerfile -t combinet_edge:v1 .
$ . start_edge.sh
```

For the cloud, we use the following commend to run the `./CombiNet/cloud/ssd_cloud/eval.py` and the corresponding docker environment for starting the cloud docker and connecting with the edge

```
$ cd ./CombiNet/cloud/ssd_cloud/
$ docker build -f Dockerfile -t combinet_cloud:v1 .
$ . start_cloud.sh
```

## Main Results

<table>
    <tr>
        <th>Deployement</th><th>Methods</th><th>E2E Latency</th><th>mAP</th>
    </tr>
    <tr>
        <td rowspan="3">Edge-only</td><td>SSD</td><td>148ms</td><td>58.4</td>
    </tr>
    <tr>
        <td>Looking</td><td>139ms</td><td>59.1</td>
    </tr>
    <tr>
        <td>CombiNet</td><td>56ms</td><td>51.8</td>
    </tr>
    <tr>
        <td rowspan="3">Cloud-only</td><td>SSD</td><td>172ms</td><td>58.4</td>
    </tr>
    <tr>
        <td>Looking</td><td>205ms</td><td>59.1</td>
    </tr>
    <tr>
        <td>CombiNet</td><td>133ms</td><td>51.8</td>
    </tr>
    <tr>
        <td>Edge-cloud Collaboration</td><td>CombiNet</td><td>42ms</td><td>51.8</td>
    </tr>
</table>

