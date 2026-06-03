# Testes experimentais para o relatório PA2 Galaxy MPI

Colocar estes ficheiros na pasta `projeto_corrigido`, ao lado de `main.py`.

## 1) Preparar terminal

```bat
conda activate py38
cd C:\Projetos\Pa\ProjetoPA2-Gif\projeto_corrigido
```

## 2) Registar perfil da máquina

Cada colega corre isto uma vez:

```bat
python machine_profile.py
```

## 3) Teste comum obrigatório para todos

Cada colega corre a mesma matriz:

```bat
python run_experiment_matrix.py --particles 500,1000,1500,2000 --ranks 1,2,4 --repeats 2 --years 1000 --dt 25 --with-plot-test
```

Se um PC for fraco, usar:

```bat
python run_experiment_matrix.py --particles 500,1000,1500 --ranks 1,2,4 --repeats 2 --years 1000 --dt 25 --with-plot-test
```

Se um PC tiver 8 cores reais, pode fazer extra:

```bat
python run_experiment_matrix.py --particles 1000,2000,3000 --ranks 1,2,4,8 --repeats 2 --years 1000 --dt 25
```

## 4) Agregar resultados

Juntem as pastas `results/NOME_DO_PC` de todos no mesmo computador e corram:

```bat
python merge_results.py
```

Isto gera:

```text
results\combined_runtime_results.csv
results\aggregated_scaling_results.csv
results\report_ready_tables.md
```

## 5) Teste final do GIF bonito

Só é preciso num computador, normalmente o mais forte:

```bat
mpiexec -n 4 python main.py --mode beauty --particles 2200 --years 12000 --dt 20 --plot-interval 200 --output-dir final_gif
```

## 6) Teste fiel ao enunciado, frame a cada 1000 anos

```bat
mpiexec -n 4 python main.py --mode report --particles 1800 --years 10000 --dt 25 --output-dir report_gif
```

## 7) Estimativa de escala 10^9

```bat
mpiexec -n 4 python main.py --scale-only --particles 1000000000 --years 1000000000 --dt 1000 --output-dir scale_estimate
```

# Orientação de Testes — Projeto PA2 Galaxy Simulation

Malta, para termos resultados comparáveis para o relatório, vamos todos testar da mesma forma.

## 1. Preparar o ambiente

Abram o **Anaconda Prompt** e façam:

```bat
conda activate py38
cd C:\Projetos\Pa\ProjetoPA2-Gif\projeto_corrigido
```

Antes dos testes, fechem programas pesados, por exemplo:

- Chrome com muitas tabs
- Jogos
- Discord
- Spotify
- Downloads
- IDEs pesadas
- Outros programas que usem muito CPU/RAM

Se for portátil, deixem ligado à corrente e em modo de energia de alto desempenho.

---

## 2. Registar informação do PC

Primeiro corram:

```bat
python machine_profile.py
```

Isto cria uma pasta em `results` com informação do vosso PC.

---

## 3. Fazer o teste principal

Depois corram este comando:

```bat
python run_experiment_matrix.py --particles 500,1000,1500,2000 --ranks 1,2,4 --repeats 2 --years 1000 --dt 25 --with-plot-test
```

Este é o teste mais importante.

Ele testa:

- 500, 1000, 1500 e 2000 partículas
- 1, 2 e 4 processos MPI
- 2 repetições para cada caso
- 1 teste extra com plotting

---

## 4. Se o PC for fraco

Se o teste estiver demasiado lento, usem este comando mais leve:

```bat
python run_experiment_matrix.py --particles 500,1000,1500 --ranks 1,2,4 --repeats 2 --years 1000 --dt 25 --with-plot-test
```

---

## 5. Se o PC for mais forte

Quem tiver 8 cores ou um PC melhor pode também correr este teste extra:

```bat
python run_experiment_matrix.py --particles 1000,2000,3000 --ranks 1,2,4,8 --repeats 2 --years 1000 --dt 25
```

Este teste é extra. Não substitui o teste principal.

---

## 6. Não mexer no GIF

Para estes testes não interessa o GIF bonito.

Estamos a medir performance, não visualização final.

O GIF final será feito só num PC depois.

---

## 7. Enviar resultados

No fim, enviem a vossa pasta dentro de:

```text
results
```

Exemplo:

```text
results\NOME_DO_VOSSO_PC
```

Essa pasta tem os CSVs que vamos juntar para o relatório.

---

## Objetivo dos testes

Queremos comparar:

- Tempo com 1, 2 e 4 processos
- Diferença entre PCs
- Speedup
- Eficiência
- Quando MPI compensa
- Quando o overhead de comunicação começa a pesar
- Diferença entre simulação sem plotting e com plotting

Assim conseguimos pôr no relatório resultados reais e discutir melhor o projeto.

---

## Notas importantes

Todos devem usar os mesmos parâmetros para os testes principais:

```text
particles = 500, 1000, 1500, 2000
ranks = 1, 2, 4
repeats = 2
years = 1000
dt = 25
```

Assim os resultados ficam comparáveis entre os vários computadores.
