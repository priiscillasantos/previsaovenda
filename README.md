# Previsão de Renda: Análise Exploratória e Modelo Preditivo com Streamlit

Aplicação de Data Science para **análise exploratória e previsão de renda**, com modelo treinado em **scikit-learn** e **implantação em Streamlit**, seguindo as etapas do **CRISP-DM** (descritas neste README).

> **Destaque do projeto:** o Streamlit foi construído para **contar a história dos dados** (insights + contexto) e permitir **uso prático do modelo** por meio de um formulário de previsão.

---

## 🎥 Demonstração (Streamlit em funcionamento)

📽️ **Vídeo da aplicação:** 

[streamlit-st_pv-2026-01-18-10-06-27.webm](https://github.com/user-attachments/assets/24350c53-d45c-460c-a342-bf95649c2c5a)

---

## ✅ O que este projeto entrega

A aplicação em Streamlit possui páginas para:

- **Visão geral**: resumo do recorte filtrado (período, estatísticas e perfil dos dados)
- **Análises**: gráficos e comparações por grupos (tipo de renda, educação, etc.)
- **Relatório HTML (opcional)**: exibição de um relatório gerado no notebook (profiling)
- **Previsão**: formulário + inferência com o modelo treinado (`.pkl`)

---

## 1) Visão geral

O objetivo é entender o comportamento da variável **`renda`** a partir de informações cadastrais e socioeconômicas e construir um modelo preditivo capaz de estimá-la com base em atributos como:

- sexo  
- tipo de renda (assalariado, empresário etc.)  
- escolaridade  
- estado civil  
- tipo de residência  
- posse de veículo / imóvel  
- idade  
- tempo de emprego  
- quantidade de filhos  
- quantidade de pessoas na residência  
- período de referência (`data_ref`)  

Além da modelagem, o foco do projeto é a **implantação**: transformar a análise e o modelo em um app navegável, com narrativa e filtros.

---

## 2) Metodologia (CRISP-DM)

Este projeto foi desenvolvido com base no CRISP-DM:

### 2.1 Business Understanding
- **Problema:** estimar renda a partir de variáveis socioeconômicas.  
- **Utilidade:** apoiar análises e decisões (segmentação, perfil de renda, estimativas).

### 2.2 Data Understanding
- Leitura do dataset e checagem de estrutura.  
- Inspeção da distribuição de renda e presença de outliers.  
- Avaliação de variáveis categóricas (tipo de renda, educação etc.) e numéricas (idade, tempo de emprego etc.).

### 2.3 Data Preparation
- Tratamento de dados faltantes e ajustes de tipos.  
- Preparação de features para uso em modelo (pipeline).  
- Separação dos dados e preparação para treinamento.

### 2.4 Modeling
- Treinamento de modelo baseado em **RandomForest** usando **Pipeline (scikit-learn)**.  
- Persistência do modelo em arquivo `.pkl` para uso no app.

### 2.5 Evaluation
- Avaliação do desempenho com métricas calculadas no notebook.  
- Verificação lógica e visual dos resultados.

### 2.6 Deployment (Implantação)
- Construção do app em **Streamlit** com páginas de:
  - visão geral e resumo do recorte filtrado  
  - análises e gráficos  
  - relatório HTML (opcional)  
  - previsão de renda (formulário + inferência)

---

## 3) Dataset

- **Arquivo principal:** `input/previsao_de_renda.csv`

A aplicação exibe um panorama do recorte carregado, incluindo:
- período coberto (via `data_ref`)
- distribuição de renda
- comparações por categorias (ex.: tipo de renda, educação)
- checagem de qualidade (valores faltantes)

**Observação importante:**  
No recorte atual, foi identificada ausência relevante na coluna **`tempo_emprego`** (≈ **17%**). Isso pode impactar análises e o desempenho do modelo caso não seja tratado adequadamente.

---

## 4) Estrutura do projeto

Estrutura sugerida (compatível com a execução do Streamlit):

```text
previsaovenda/
├─ input/
│  └─ previsao_de_renda.csv
├─ output/
│  ├─ modelo_final_randomforest.pkl
│  └─ renda_analysis.html              # relatório HTML gerado no notebook
├─ projeto-2.ipynb                     # notebook principal (EDA + treino + avaliação)
├─ st_pv.py                            # aplicação Streamlit (implantação)
└─ README.md

```

---

## 5) Requisitos

- **Python:** 3.10+ (recomendado 3.11/3.12)
- Bibliotecas principais:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`
  - `matplotlib`
  - `streamlit`
  - `seaborn` (opcional para análises)

---

## 6) Como executar

### 6.1 Instalar dependências
Instale as bibliotecas necessárias no mesmo Python/ambiente que executará o Streamlit:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn

```
---

## 7) Valor do projeto (narrativa + uso prático)

A aplicação foi pensada para ir além de “gráficos soltos”:

resumo do recorte atual (filtros) e perfil dos dados

qualidade dos dados e impactos esperados (faltantes)

distribuição de renda (incluindo transformações quando necessário)

comparações por grupos (tipo de renda, educação, posse de imóvel/veículo etc.)

previsão prática: formulário com os principais atributos para estimar renda

---

## 8) Problemas comuns e como resolver

### 8.1 “No module named sklearn / matplotlib / joblib”

O Streamlit está rodando em um Python onde as libs não estão instaladas.
✅ Instale no mesmo ambiente que executa streamlit run.

### 8.2 “file is not defined”

Isso acontece quando um código feito para .py é executado no Jupyter.
✅ No app Streamlit use Path.cwd() ou caminhos relativos.

### 8.3 O terminal “não entra na pasta”

✅ Abra o VS Code direto na pasta do projeto (File > Open Folder) ou use cd até a raiz correta.

---

## 9) Autoria

Projeto desenvolvido como parte do curso Profissão: Cientista de Dados (EBAC), aplicando o ciclo CRISP-DM e técnicas de modelagem preditiva com implantação em Streamlit.

