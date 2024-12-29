FROM python:3.9-slim

WORKDIR /src

COPY . /src

RUN pip install --no-cache-dir -r requirements.cpu.txt

RUN echo '#!/bin/bash\n\
    python train-v2.py \
    --ds $DATASET \
    --bs $BATCH_SIZE \
    --le $LEARNING_RATE \
    --dpr $DROPOUT_RATE \
    --pt-max-epochs $MAX_PRETRAINING_EPOCHS \
    --ft-max-epochs $MAX_FINTUNING_EPOCHS \
    --cr $CORRUPTION_RATIO \
    --tau $TEMPERATURE' > /start.sh

RUN chmod +x /start.sh


ENV DATASET='dapt20' \
    BATCH_SIZE=128 \
    LEARNING_RATE=0.001 \
    MAX_PRETRAINING_EPOCHS=200 \
    MAX_FINTUNING_EPOCHS=50 \
    CORRUPTION_RATIO=0.1 \
    TEMPERATURE=0.1 \
    DROPOUT_RATE=0.1

CMD ["start.sh"]

## Install system dependencies
#RUN apt-get update && apt-get install -y \
#    git \
#    python3-pip \
#    python3-dev \
#    build-essential \
#    && rm -rf /var/lib/apt/lists/*

## Install PyTorch CPU version specifically
#RUN pip3 install --no-cache-dir \
#    torch==2.1.0+cpu \
#    torchvision==0.16.0+cpu \
#    -f https://download.pytorch.org/whl/cpu/torch_stable.html

## Install other Python packages
#RUN pip3 install --no-cache-dir \
#    numpy==1.24.3 \
#    pandas==2.1.1 \
#    scikit-learn==1.3.1 \
#    matplotlib==3.8.0 \
#    seaborn==0.13.0 \
#    tqdm==4.66.1 \
#    tensorboard==2.14.1
#
## Copy the source code
#COPY . /app/

# Create directories for data and models
#RUN mkdir -p /app/data /app/models /app/logs

# Create a script to start training
#RUN echo '#!/bin/bash\n\
#    python3 train.py \
#    --data_path $DATA_PATH \
#    --batch_size $BATCH_SIZE \
#    --learning_rate $LEARNING_RATE \
#    --epochs $EPOCHS \
#    --corruption_ratio $CORRUPTION_RATIO \
#    --noise_std $NOISE_STD \
#    --device "cpu" \
#    --model_save_path $MODEL_SAVE_PATH \
#    --log_dir $LOG_DIR\n' > /app/start_training.sh