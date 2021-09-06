FROM python:3.8

ADD requirements.txt /requirements.txt

RUN pip install --upgrade -r requirements.txt

ADD app.py /app.py
ADD load_css.py /load_css.py
ADD ivis.py /ivis.py
ADD style.css /style.css

ENTRYPOINT  streamlit run /app.py --server.port 8000 --server.address 0.0.0.0
