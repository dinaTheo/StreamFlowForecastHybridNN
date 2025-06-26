FROM tensorflow/tensorflow:2.11.0-gpu

WORKDIR /app
COPY . /app

# RUN rm /etc/apt/sources.list.d/cuda.list || true

# RUN apt-get update && apt-get install -y libhdf5-dev libnetcdf-dev && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set matplotlib config directory to avoid permission issues
ENV MPLCONFIGDIR=/tmp

CMD ["python", "main.py"]
