FROM python:latest
COPY main.py rfmodel.pickle price_hotel_id.csv date_hotel_id.csv pricing_requests.csv /
RUN pip install gradio altair pandas scikit-learn numpy
CMD ["python", "./main.py"]
