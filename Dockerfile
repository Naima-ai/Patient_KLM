FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the endpoint needs
COPY klm_api.py .
COPY patient_klm_endpoint.py .
COPY data/patient_klm.db data/patient_klm.db

EXPOSE 8001

CMD ["python", "patient_klm_endpoint.py"]
