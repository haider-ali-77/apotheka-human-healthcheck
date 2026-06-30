FROM nvcr.io/nvidia/clara-train-sdk:v3.1.01 
WORKDIR /opt/rai/MentalHealth/
COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y python3-venv
# RUN pip3 install virtualenv
ENV VIRTUAL_ENV =/opt/rai/MentalHealth/venv

RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade setuptools
RUN python -m pip install -r requirements.txt
COPY *.py ./
EXPOSE 5000
RUN mkdir /opt/rai/data
COPY data/vaSuicidePreventionInnovation.ods data/
CMD ["python","api.py"]
