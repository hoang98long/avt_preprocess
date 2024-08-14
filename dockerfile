FROM python:3.8 as requirements-stage

WORKDIR /app

COPY /home/avt/avt_preprocess /app/avt_preprocess

RUN chmod +x /app/avt_preprocess/main.exe

CMD ["./avt_preprocess/main.exe"]