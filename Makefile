MAKEFLAGS += --silent

OPTIONS ?= --build --remove-orphans --force-recreate
APP ?= app

.PHONY: all setup healthcheck clean

all:
	docker-compose up $(OPTIONS) -d

setup:
	docker compose exec -it ollama ollama pull $(OLAMA_MODEL)
	docker compose exec -it ollama ollama list

%:
	docker-compose up $(OPTIONS) $@ -d
	docker-compose ps -a

healthcheck:
	docker inspect $(APP) --format "{{ (index (.State.Health.Log) 0).Output }}"

clean:
	docker-compose down --remove-orphans -v --rmi local

-include .env
