# llm-eval-workspace: bancada local de avaliacao qualitativa de LLMs (FastAPI + React)
#
# `make` ou `make help` lista os alvos. Este Makefile é uma fachada: cada alvo
# delega pra ferramenta de verdade (uv, bun, pytest, ruff, vite). Regra nova
# entra com `## descrição` na mesma linha pra aparecer no help.

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables --no-builtin-rules --no-print-directory
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

# ---------------------------------------------------------------------------
# Variáveis (?= permite sobrescrever: `make dev PORT=8600`)
# ---------------------------------------------------------------------------

APP    ?= llm-eval-workspace
VENV   ?= venv
PY     := $(VENV)/bin/python
PORT   ?= 8600
# 127.0.0.1 por padrão: a API não tem autenticação e guarda chave de LLM. Expor na
# rede é decisão explícita (`make api HOST=0.0.0.0`), de preferência atrás de VPN.
HOST   ?= 127.0.0.1
WEB    := web
BUN    ?= bun
MODEL  ?=
LOCALE ?= en
EFFORT ?= default

# Cores só quando stdout é um terminal (pipe e CI ficam limpos)
BOLD  :=
CYAN  :=
GREEN :=
RESET :=
ifneq ($(shell [ -t 1 ] && echo tty),)
  BOLD  := $(shell tput bold 2>/dev/null)
  CYAN  := $(shell tput setaf 6 2>/dev/null)
  GREEN := $(shell tput setaf 2 2>/dev/null)
  RESET := $(shell tput sgr0 2>/dev/null)
endif

##@ Geral

.PHONY: help
help: ## Lista os alvos disponíveis
	@awk 'BEGIN { FS = ":.*##"; printf "\n$(BOLD)$(APP)$(RESET)\n\nUso: make $(CYAN)<alvo>$(RESET)\n" } \
	  /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) } \
	  /^[a-zA-Z0-9_.\/-]+:.*?##/ { printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2 } \
	  END { printf "\n" }' $(MAKEFILE_LIST)

.PHONY: setup
setup: ## Cria a venv (se faltar) e instala dependências com uv
	@echo "$(GREEN)▸ setup$(RESET)"
	test -x $(PY) || uv venv $(VENV)
	uv pip install --python $(PY) -e ".[dev]"
	cd $(WEB) && $(BUN) install

##@ Desenvolvimento

.PHONY: api
api: ## Sobe só a API em http://$(HOST):$(PORT) (serve web/dist se existir)
	@echo "$(GREEN)▸ api em $(HOST):$(PORT)$(RESET)"
	$(VENV)/bin/uvicorn --factory llm_eval.api:create_app --host $(HOST) --port $(PORT) --reload --reload-dir llm_eval

.PHONY: web
web: ## Sobe o Vite com hot reload em http://localhost:5173 (proxy /api -> :8600)
	@echo "$(GREEN)▸ web em :5173$(RESET)"
	cd $(WEB) && $(BUN) run dev

.PHONY: dev
dev: ## API + Vite juntos (Ctrl-C derruba os dois)
	@echo "$(GREEN)▸ dev: api $(HOST):$(PORT) + web :5173$(RESET)"
	@$(VENV)/bin/uvicorn --factory llm_eval.api:create_app --host $(HOST) --port $(PORT) --reload --reload-dir llm_eval & api=$$!; \
	  (cd $(WEB) && $(BUN) run dev) & web=$$!; \
	  cleanup() { trap - INT TERM EXIT; kill "$$api" "$$web" 2>/dev/null || true; wait "$$api" "$$web" 2>/dev/null || true; }; \
	  trap 'cleanup; exit 130' INT; trap 'cleanup; exit 143' TERM; trap cleanup EXIT; \
	  status=0; \
	  case "$$(uname -s)" in \
	    Linux) wait -n "$$api" "$$web" || status=$$?;; \
	    Darwin) \
	      while kill -0 "$$api" 2>/dev/null && kill -0 "$$web" 2>/dev/null; do sleep 0.5; done; \
	      if kill -0 "$$api" 2>/dev/null; then wait "$$web" || status=$$?; else wait "$$api" || status=$$?; fi;; \
	    *) wait "$$api" "$$web" || status=$$?;; \
	  esac; \
	  cleanup; exit "$$status"


.PHONY: battery
battery: ## Roda a bateria headless no modelo (MODEL=, LOCALE=en, EFFORT=default)
	@echo "$(GREEN)▸ battery $(MODEL) [$(LOCALE), effort $(EFFORT)]$(RESET)"
	$(PY) -m llm_eval.cli --model "$(MODEL)" --locale "$(LOCALE)" --effort "$(EFFORT)"

.PHONY: seed
seed: ## Recria o banco e reimporta os prompts de data/prompts
	@echo "$(GREEN)▸ seed$(RESET)"
	$(PY) -m llm_eval.db.init_db --reset

##@ Qualidade

.PHONY: fmt
fmt: ## Formata o código (modifica arquivos)
	@echo "$(GREEN)▸ fmt$(RESET)"
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .
	cd $(WEB) && $(BUN) run lint --fix

.PHONY: lint
lint: ## Lint estático, sem modificar nada
	@echo "$(GREEN)▸ lint$(RESET)"
	$(VENV)/bin/ruff format --check .
	$(VENV)/bin/ruff check .
	cd $(WEB) && $(BUN) run lint --deny-warnings && $(BUN) run typecheck

.PHONY: test
test: ## Roda os testes
	@echo "$(GREEN)▸ test$(RESET)"
	$(VENV)/bin/pytest
	cd $(WEB) && $(BUN) run test

.PHONY: check
check: lint test ## Lint e testes, o que roda antes de qualquer commit
	@echo "$(GREEN)✓ check ok$(RESET)"

.PHONY: security-scan
security-scan: ## Varre segredos no histórico inteiro (gitleaks + trufflehog)
	@echo "$(GREEN)▸ security-scan$(RESET)"
	@if command -v gitleaks >/dev/null; then \
	  gitleaks git --redact --verbose; \
	elif docker info >/dev/null 2>&1; then \
	  echo "gitleaks: rodando pela imagem Docker"; \
	  docker run --rm -v "$$PWD:/repo" ghcr.io/gitleaks/gitleaks:latest git /repo --redact --verbose; \
	else \
	  echo "sem gitleaks: https://github.com/gitleaks/gitleaks#installing"; exit 1; \
	fi
	@if command -v trufflehog >/dev/null; then \
	  trufflehog git "file://$$PWD" --only-verified --fail; \
	elif docker info >/dev/null 2>&1; then \
	  echo "trufflehog: rodando pela imagem Docker"; \
	  docker run --rm -v "$$PWD:/repo" trufflesecurity/trufflehog:latest git file:///repo --only-verified --fail; \
	else \
	  echo "sem trufflehog: https://github.com/trufflesecurity/trufflehog#installation"; exit 1; \
	fi

##@ Build

.PHONY: build
build: ## Compila o frontend em web/dist (a API serve isso em produção)
	@echo "$(GREEN)▸ build web$(RESET)"
	cd $(WEB) && $(BUN) run build

.PHONY: clean
clean: ## Remove caches gerados
	@echo "$(GREEN)▸ clean$(RESET)"
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info $(WEB)/dist
	find . -name __pycache__ -type d -prune -not -path './venv/*' -exec rm -rf {} +
