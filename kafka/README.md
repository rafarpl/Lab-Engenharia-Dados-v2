
# LAB KAFKA

---
## Disclaimer
> **Esta configuração é puramente para fins de desenvolvimento local e estudos**
> 

---


## Pré-requisitos?
* Docker
* Docker-Compose

---

## Criando o ambiente Kafka com o docker compose


```
docker compose up -d mc zookeeper kafka-broker

```

Listando o tópico criado
```sh
docker exec -it kafka-broker /bin/bash

kafka-topics --bootstrap-server localhost:9092 --list 

kafka-console-consumer --bootstrap-server localhost:9092 --topic sink-raw --from-beginning

```

[Kafka]https://github.com/Labdata-FIA/Engenharia-Dados/tree/main/21.Ingestao-Dados-Kafka
[Kafka-Connect]https://github.com/Labdata-FIA/Engenharia-Dados/tree/main/21.Ingestao-Dados-Kafka-Connect

