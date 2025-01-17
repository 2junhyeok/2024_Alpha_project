# Self Supervised Learning for Video Anomaly Detection

## Overview
이 프로젝트는 Jigsaw Learning을 활용한 비디오 이상 탐지(Video Anomaly Detection) 시스템을 구현합니다. 시공간적 특징을 학습하여 비디오 내의 이상 행동을 감지합니다.

## Method
### Core Concepts
* 기존 Jigsaw-VAD는 라벨이 없는 비디오 데이터에서 비정상적인 이벤트를 탐지하기 위해서, 공간적 직소 퍼즐과 시간적 직소 퍼즐을 해결하는 pretext task를 설계
* 그러나 Frame 단위에서만 Anomaly를 진행하여 정적인 이상치나, long-term에 취약함
* 이를 해결하고자 Frame 단위 시공간 Jigsaw에 Clip 단위까지 확장하여 DAD에 적용함

### Key Features
* **Clip 단위 Spatial**
  * 기존의 spatial은 한 장의 Frame을 대상으로 9개의 조각을 만들어 학습을 진행했지만, 본 연구에서는 Clip 자체를 9조각으로 나누어, 긴 time step을 고려한 Clip 단위 Spatial을 학습함
* **Clip 단위 Temporal**
  * 기존의 Temporal은 7장의 Frame을 대상으로 순서를 무작위로 섞어서 순서를 맞추는 방식으로 학습을 진행했지만, 본 연구에서는 7장의 Frame이 담긴 Clip을 대상으로 순서를 맞추는 방식으로 긴 time step도 학습함
* **4개의 Predictor**
  * Frame 단위와 Clip 단위 각각의 Spatial과 Temporal이 존재하기 때문에 4가지 출력을 생성하도록 학습을 진행함
* **Loss를 공유하는 Backbone**
  * Frame 단위와 Clip 단위는 입력의 형태가 다르기 때문에 두 가지 Task를 동시에 학습하기 위해서는 Backbone을 공유하면서 Loss를 공유하는 방식의 학습을 진행함

## Project Structure
```
📦 Video-Anomaly-Detection
├── 📂 models/               # Model implementation files
├── 📂 tool/                # Utility tools
├── 📄 aggregate.py         # Result aggregation and evaluation
├── 📄 data_preprocessing.py # Data preprocessing
├── 📄 dataset.py           # Dataset class implementation
├── 📄 gen_patches.py       # Patch generation
├── 📄 generate_frame_mask.py # Frame mask generation
├── 📄 main.py             # Main execution file
├── 📄 pkl.py              # Pickle file processing
└── 📄 restructure_dataset.py # Dataset structure reorganization
```

## Technical Requirements

### requirements
```bash
torch
torchvision
numpy
opencv-python
scipy
tqdm
```

## Dataset Structure
```
📦 DAD_Jigsaw
├── 📂 training/
│   ├── 📂 front_depth/
│   ├── 📂 front_IR/
│   ├── 📂 top_depth/
│   └── 📂 top_IR/
└── 📂 testing/
    ├── 📂 front_depth/
    ├── 📂 front_IR/
    ├── 📂 top_depth/
    └── 📂 top_IR/
```

## Installation Guide

1. Clone Repository
```bash
git clone https://github.com/yugwangyeol/Self-Supervised-Learning-for-Video-Anomaly-Detection.git
cd Self-Supervised-Learning-for-Video-Anomaly-Detection
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Setup Dataset Structure
```bash
python restructure_dataset.py
```

## Execution Guide

### 1. Data Preprocessing
```bash
python data_preprocessing.py
```

### 2. Frame Mask Generation
```bash
python generate_frame_mask.py
```

### 3. Model Training
```bash
python main.py --data_type top_depth --epochs 100 --batch_size 64
```

### 4. Evaluation
```bash
python aggregate.py --file [output_pkl_file] --dataset DAD_Jigsaw --frame_num [num_frames]
```

## Implementation Details

### Training Process
- Batch Size: 64
- Learning Rate: 1e-4
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

## Experimental Results
![Image](https://github.com/user-attachments/assets/cee8ae5d-71cc-4f3c-9eb2-39661368e0c8)

* 시공간적 특징을 동시에 고려하는 이중 분석 체계를 통해 안정적인 이상 탐지 성능 달성
* Driver Video Dataset이라는 세부 분야에서 준수한 성능을 달성함
* Self Supervised Learning을 도입하여 레이블이 있는 데이터 의존도를 낮춤
* 실시간 처리가 가능한 효율적인 시스템 구조 확립
* 실제 차량 환경에서의 즉각적인 위험감지 시스템으로 활용이 가능함
* 다양한 운전자 환경에 적용할 수 있는 확장성 확보
* 기존 시스템과의 통합이 쉬운 모듈화된 구조

## Conclusion
* 기존 모델에 비해 허용 가능한 오차 내에서 비슷한 성능
* 제안한 모델은 데이터에 대해 시간적, 공간적 특징을 더욱 다양하게 이해 가능
* combined 결과가 실제 적용 시에는 더 안정적인 결과를 제공할 수 있으리라 판단

## Contributors
소속: 국민대학교 AI빅데이터융합경영학과 Alpha Project  

팀원: 유광열, 김서령, 이준혁

## Ciation
```BiTex
@inproceedings{wang2022jigsaw-vad,
  title     = {Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles},
  author    = {Guodong Wang and Yunhong Wang and Jie Qin and Dongming Zhang and Xiuguo Bao and Di Huang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```
