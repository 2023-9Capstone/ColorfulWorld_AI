FROM python:3

WORKDIR /ai-server

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN apk add libffi-dev
RUN pip install --no-cache-dir -r /requirements.txt

COPY . .

EXPOSE 5200

CMD ["python", "app.py", "flask","run","--port=5200"]
