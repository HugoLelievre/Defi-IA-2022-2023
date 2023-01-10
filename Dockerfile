FROM python:latest
COPY main.py rfmodel.pickle price_hotel_id.csv date_hotel_id.csv princing_requests.csv /
RUN pip install gradio altair pandas scikit-learn numpy
CMD ["python", "./main.py"]