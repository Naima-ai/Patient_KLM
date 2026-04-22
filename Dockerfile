FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Core KLM files
COPY klm_api.py .
COPY patient_klm_endpoint.py .
COPY entrypoint.sh .

# Demo 2 seed
COPY seed_demo2.py .
COPY demo_2/ demo_2/

# Demo 3 seed + pipeline
COPY demo_3/ demo_3/

# PT-9921 dermatology patient
COPY seed_demo3.py .
COPY dermatology_dna.json .
COPY dermatology_ehr.json .

# Base database (P-001 data built by run_pipeline.py)
COPY data/patient_klm.db data/patient_klm.db

# Demo 3 pre-generated JSON files
COPY data/p003_ehr_records.json data/p003_ehr_records.json
COPY data/p003_genomic_profile.json data/p003_genomic_profile.json
COPY data/p003_patient_triples.json data/p003_patient_triples.json
COPY data/p003_pathology_triples.json data/p003_pathology_triples.json

RUN chmod +x entrypoint.sh

EXPOSE 8001

ENV PATIENT_KLM_DB_PATH=/app/data/patient_klm.db

CMD ["./entrypoint.sh"]
