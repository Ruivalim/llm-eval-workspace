# Bancada de avaliação de LLM: frontend compilado pelo Bun, servido pela API FastAPI.
# Dois estágios pra imagem final não carregar node_modules nem toolchain JS.
FROM oven/bun:1.4 AS web
WORKDIR /web
COPY web/package.json web/bun.lock ./
RUN bun install --frozen-lockfile
COPY web/ ./
RUN bun run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md LICENSE ./
COPY llm_eval/ llm_eval/
RUN pip install --no-cache-dir .
# Prompts e categorias vão na imagem: mudar prompt é commit, não edição no volume.
COPY data/categories.yaml data/categories.yaml
COPY data/prompts/ data/prompts/
COPY --from=web /web/dist web/dist
# JS harness checks execute model output at runtime.
COPY --from=web /usr/local/bin/bun /usr/local/bin/bun
# Estado (sqlite, artefatos, traces, logs) fica em /data, montado como PVC.
ENV LLM_EVAL_DB=/data/llm_eval.db \
    LLM_EVAL_ARTIFACTS=/data/artifacts \
    LLM_EVAL_TRACES=/data/traces \
    LLM_EVAL_LOGS=/data/logs
RUN useradd -r -u 1000 -d /app app && mkdir -p /data && chown app /data
USER app
EXPOSE 8600
CMD ["uvicorn", "--factory", "llm_eval.api:create_app", "--host", "0.0.0.0", "--port", "8600"]
