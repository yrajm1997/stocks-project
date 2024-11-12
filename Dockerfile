FROM python:3.10

ADD requirements/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN rm requirements.txt

ADD figures/__init__.py figures/__init__.py

ADD stock_db.sqlite .

ADD app.py .

USER root

ENV OPENAI_KEY=enter-key-here

EXPOSE 8000

CMD ["chainlit", "run", "--host=0.0.0.0", "app.py"]
