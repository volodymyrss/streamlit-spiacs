FROM python:3.8

ADD app.py /app.py

RUN pip install --upgrade streamlit astropy matplotlib
RUN pip install --upgrade oda-api

ENTRYPOINT  streamlit run /app.py --server.port 8000
