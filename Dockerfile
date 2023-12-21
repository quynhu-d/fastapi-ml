FROM python:3.9.9-slim
WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt

EXPOSE 8008

CMD python src/app.py