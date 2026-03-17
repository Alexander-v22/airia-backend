FROM python:3.11-slim
 
WORKDIR /app
 
# Install CPU-only PyTorch first before everything else
# This prevents pip from pulling in the full CUDA build
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
 
# Copy and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy app
COPY main.py .
 
EXPOSE 8000
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]