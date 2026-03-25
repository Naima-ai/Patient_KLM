FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all KLM files
COPY klm_api.py .
COPY patient_klm_endpoint.py .
COPY seed_demo2.py .
COPY entrypoint.sh .

# Copy the base database
COPY data/patient_klm.db data/patient_klm.db

# Copy Demo #2 JSON source files
COPY demo_2/ demo_2/

RUN chmod +x entrypoint.sh

EXPOSE 8001

ENV PATIENT_KLM_DB_PATH=/app/data/patient_klm.db

CMD ["./entrypoint.sh"]
