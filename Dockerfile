FROM python:3.8

ADD app.py /app.py

RUN pip install --upgrade streamlit astropy matplotlib
