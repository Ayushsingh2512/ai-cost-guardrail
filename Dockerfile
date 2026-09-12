FROM python:3.14-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY app ./app
COPY alembic ./alembic 
COPY alembic.ini ./
EXPOSE 8000
CMD [ "uv","run" ,"uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]