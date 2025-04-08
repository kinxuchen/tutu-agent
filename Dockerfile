FROM python:3.13.2

WORKDIR /app

RUN pip install poetry

COPY ./poetry.lock ./pyproject.toml ./pyproject.toml ./main.py ./llm.py ./constant.py ./.env ./
COPY ./agents ./agents/
COPY ./apis ./apis/
COPY ./entity ./entity/
COPY ./checkpointer ./checkpointer/
COPY ./components ./components/
COPY ./dto ./dto/
COPY ./services ./services/
COPY ./tools ./tools/

ENV host=0.0.0.0
ENV port=8002

RUN poetry install

CMD ["poetry", "run", "python", "main.py"]
