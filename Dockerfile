FROM python:latest
COPY main.py lr.pickle /
RUN pip install gradio altair pandas scikit-learn numpy
CMD ["python", "./main.py"]