# RTscore — Avaliação de Plausibilidade Cromatográfica para LC

Uma ferramenta para avaliar se estruturas moleculares candidatas são **cromatograficamente plausíveis** com base na relação entre **descritores moleculares** e **tempo de retenção observado (RT)** ou **índice de retenção (RI)**.

---

# Aplicação Web

O aplicativo foi implementado como uma **interface Streamlit** e foi projetado para apoiar **fluxos de trabalho de anotação em metabolômica**.

Você pode usar a aplicação diretamente online:

👉 **[Abrir RTscore](https://rt-score.streamlit.app/)**

Nenhuma instalação é necessária.

---

# Sumário

1. Conceito  
2. Por que a plausibilidade cromatográfica é importante  
3. Instalação  
4. Arquivos de entrada  
5. Executando a análise  
6. Interpretando cada aba  
7. Entendendo o suspicion score  
8. Interpretação crítica dos resultados  
9. Boas práticas  
10. Limitações  
11. Integração com pipelines de metabolômica  
12. Exemplo de workflow  

---

# 1. Conceito

Ao identificar compostos em metabolômica por LC-MS, o fluxo de trabalho usual depende de:

- massa exata  
- padrão isotópico  
- fragmentação MS/MS  
- correspondência em bancos de dados  

Entretanto, esses sinais **não garantem consistência cromatográfica**.

Duas moléculas com a mesma massa podem apresentar **comportamentos de retenção muito diferentes**.

Esta ferramenta avalia se uma estrutura candidata é **compatível com o sistema cromatográfico** comparando:

```

RT ou RI observado
vs
RT ou RI previsto

````

A previsão é derivada de **descritores moleculares**, como:

- logP  
- TPSA  
- doadores de ligação de hidrogênio  
- aceitadores de ligação de hidrogênio  
- ligações rotacionáveis  
- anéis aromáticos  
- massa molecular  

Esses descritores são calculados automaticamente usando **RDKit**.

---

# 2. Por que a plausibilidade cromatográfica é importante

Muitas anotações em metabolômica falham porque a estrutura:

- possui massa correta  
- corresponde a um banco de dados  

mas **eluí em uma região cromatográfica impossível**.

Exemplos:

| Composto | Comportamento esperado | RT observado |
|----------|-----------------------|--------------|
| Açúcar altamente polar | cedo | tarde |
| Lipídio hidrofóbico | tarde | cedo |

Essas discrepâncias podem indicar:

- candidato incorreto  
- aduto incorreto  
- contaminação espectral  
- co-eluição  
- viés de banco de dados  

A plausibilidade cromatográfica atua como um **terceiro filtro ortogonal**, após MS1 e MS/MS.

---

# 3. Instalação

Clone o repositório:

```bash
git clone https://github.com/yourrepo/RTscore
cd RTscore
````

Crie o ambiente:

```bash
conda create -n rtscore python=3.11
conda activate rtscore
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Dependências típicas:

```
streamlit
pandas
numpy
plotly
rdkit
pillow
```

Execute a aplicação:

```bash
streamlit run app_rtscore.py
```

---

# 4. Arquivos de entrada

Dois arquivos CSV são necessários.

---

# Dataset de Referência

Este dataset define o **comportamento cromatográfico de compostos conhecidos**.

Colunas obrigatórias:

| coluna   | descrição                 |
| -------- | ------------------------- |
| name     | nome do composto          |
| smiles   | estrutura molecular       |
| rt ou ri | referência cromatográfica |

Metadados opcionais:

```
class
mode
adduct
```

Exemplo:

```csv
name,smiles,rt,class
Caffeine,Cn1cnc2n(C)c(=O)n(C)c(=O)c12,1.92,alkaloid
Quercetin,O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12,4.25,flavonoid
```

### Comentário crítico

O dataset de referência **define o domínio do modelo**.

Um dataset ruim produz um modelo ruim.

Datasets adequados devem:

* representar a diversidade química do estudo
* conter ≥20 compostos
* cobrir o intervalo de RT de interesse

---

# Dataset de Candidatos

Contém estruturas candidatas para cada feature de MS.

Colunas obrigatórias:

| coluna         | descrição           |
| -------------- | ------------------- |
| feature_id     | feature de MS       |
| candidate_name | estrutura candidata |
| smiles         | estrutura molecular |

Opcionais:

```
observed_rt
observed_ri
candidate_class
rank_source
```

Exemplo:

```csv
feature_id,candidate_name,smiles,observed_rt
F001,Quercetin,O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12,4.22
```

---

# Dataset de Calibrantes (opcional)

Necessário apenas quando se utiliza **modo RI**.

Colunas:

```
rt,index
```

Exemplo:

```csv
rt,index
1.2,100
2.1,200
3.4,300
4.5,400
```

Esses compostos são usados para **interpolar valores de RI**.

---

# 5. Executando a análise

Etapas na interface:

### Passo 1 — Upload dos arquivos

Carregar:

* dataset de referência
* dataset de candidatos

Opcional:

* dataset de calibrantes

---

### Passo 2 — Escolher eixo de previsão

Opções:

```
Retention Time (RT)
Retention Index (RI)
```

### Modo RT

Usa diretamente tempos de retenção observados.

### Modo RI

Se RI não estiver disponível, o sistema calcula usando:

```
interpolação de RI a partir de calibrantes
```

Observação crítica:

Índices de retenção aumentam a **reprodutibilidade entre experimentos**.

---

### Passo 3 — Selecionar descritores

Descritores padrão:

```
MolLogP
TPSA
HBD
HBA
RotatableBonds
AromaticRingCount
MolWt
```

Insight crítico:

Descritores refletem a **fisicoquímica cromatográfica**.

---

### Passo 4 — Selecionar modelo

Dois modelos estão disponíveis:

**Score ponderado de descritores**

* robusto
* interpretável
* funciona bem com datasets pequenos

**Regressão linear**

* orientada pelos dados
* adaptável
* requer datasets maiores

---

### Passo 5 — Definir thresholds de suspeição

Exemplo:

```
Highly plausible < 1
Plausible < 2
Borderline < 3
Suspicious ≥ 3
```

---

### Passo 6 — Rodar análise

Clique em:

```
Run analysis
```

O app irá:

1. calcular descritores moleculares
2. ajustar o modelo de referência
3. prever RT ou RI
4. calcular desvios
5. calcular scores de suspeição
6. ranquear candidatos

---

# 6. Interpretando cada aba

## Overview

Mostra:

* tipo de modelo
* eixo usado
* desvio padrão residual
* resumo dos datasets

---

## Reference Model

### Observed vs Predicted

Padrão ideal:

```
pontos próximos à diagonal
```

---

### Residual Plot

Mostra erros do modelo.

---

## Prediction View

Mostra:

```
RT previsto vs RT observado
```

e a tabela de candidatos.

---

## Candidate Plausibility

A aba mais importante.

Visualizações principais:

* distribuição de scores de referência
* mapa de plausibilidade
* ranking de candidatos

---

## Structures

Mostra as estruturas químicas das moléculas.

---

## Export

Permite exportar:

* resultados de referência
* resultados de candidatos

---

# 7. Suspicion Score

Definido como:

```
abs_error / residual_sd
```

Significa quantos desvios padrão o candidato se afasta do RT esperado.

---

# 8. Interpretação crítica

A plausibilidade cromatográfica **não prova identidade**.

Ela apenas responde:

```
Esta estrutura é cromatograficamente consistente?
```

---

# 9. Boas práticas

Use:

* ≥20 compostos de referência
* compostos cobrindo todo o intervalo de RT
* condições LC consistentes

Evite:

* misturar colunas
* misturar gradientes
* misturar fases móveis

---

# 10. Limitações

### Dependência da coluna

Colunas LC diferentes alteram relações de RT.

### Dependência do gradiente

Gradientes diferentes alteram o comportamento de eluição.

---

# 11. Integração com pipelines de metabolômica

Pipeline típico:

```
detecção de features
→ predição de fórmula
→ busca em biblioteca espectral
→ filtro RTscore
→ ranking final
```

---

# 12. Exemplo de workflow

```
LC-HRMS
↓
detecção de features (MZmine)
↓
estruturas candidatas (GNPS / SIRIUS / bancos de dados)
↓
RTscore
↓
inspeção manual
↓
atribuição de nível de anotação
```

---

# Recomendação final

RTscore deve ser usado como **ferramenta de suporte à decisão**, não como identificador automático.

As anotações mais robustas combinam:

```
evidência espectral
plausibilidade cromatográfica
raciocínio químico
```

