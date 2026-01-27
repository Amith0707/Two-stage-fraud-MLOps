# Base image
FROM python:3.11
# working directory in container
WORKDIR /app
# Setting up requirements first
COPY requirements.txt .
# Installing dependencies
RUN pip install -r requirements.txt
# Copy rest now
COPY . .
# Expose port
EXPOSE 5500
# Initalize the application
CMD ["uvicorn","src.api.app:app","--host","0.0.0.0","--port","5500"] 