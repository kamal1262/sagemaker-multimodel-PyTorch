FROM $BASE_IMAGE

COPY ./requirements.txt /var/requirements.txt

RUN pip install -r /var/requirements.txt
