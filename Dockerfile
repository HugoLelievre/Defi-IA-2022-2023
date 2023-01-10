FROM python:latest
COPY main.py rf_model.pickle /
RUN pip install gradio altair pandas scikit-learn numpy
CMD ["python", "./main.py"]