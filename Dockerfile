FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/automated-petct-lesion-segmentation  && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm predict_cls.py /opt/algorithm/
COPY --chown=algorithm:algorithm predict_seg.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils_img.py /opt/algorithm/
COPY --chown=algorithm:algorithm nnunet /opt/algorithm/nnunet 
COPY --chown=algorithm:algorithm models /opt/algorithm/models
COPY --chown=algorithm:algorithm nnUNet_results /opt/algorithm/nnUNet_results

COPY --chown=algorithm:algorithm test/input/images /input/images

RUN python -m pip install --user -U pip && \
    python -m pip install --user -r requirements.txt && \
    mkdir -p /opt/algorithm/nnUNet_raw && \
    mkdir -p /opt/algorithm/nnUNet_preprocessed && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result

RUN pip uninstall -y opencv-python
RUN pip install opencv-python-headless


ENV nnUNet_raw="/opt/algorithm/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"

ENTRYPOINT ["python", "-m", "process", "$0", "$@"]
