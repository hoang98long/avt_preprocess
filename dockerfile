FROM continuumio/anaconda3

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY . /app/avt_preprocess

RUN conda create --name avt_preprocess python=3.8  # Thay python=3.8 bằng phiên bản Python mà bạn cần
RUN echo "conda activate avt_preprocess" >> ~/.bashrc
RUN conda init bash

COPY requirements.txt .
RUN conda run -n avt_preprocess pip install -r requirements.txt
RUN conda install -n avt_preprocess -c conda-forge gdal=3.2.1 sqlite=3.34.0
RUN conda install -n avt_preprocess -c conda-forge rasterio

CMD ["bash", "-c", "source activate avt_preprocess && cd /app/avt_preprocess && python main.py --config_file /app/config.json"]