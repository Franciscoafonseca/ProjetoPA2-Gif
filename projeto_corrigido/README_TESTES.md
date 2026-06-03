# README_TESTES.md

## Preparação

```bat
conda activate py38
cd "..\ProjetoPA2-Gif\projeto_corrigido"
```

---

## 1. Registar perfil do computador

Executar apenas uma vez:

```bat
python machine_profile.py
```

---

## 2. Teste principal (obrigatório para todos)

```bat
python run_experiment_matrix.py --particles 500,1000,1500,2000 --ranks 1,2,4 --repeats 3 --years 1000 --dt 25 --with-plot-test
```

Se o computador for mais fraco:

```bat
python run_experiment_matrix.py --particles 500,1000,1500 --ranks 1,2,4 --repeats 3 --years 1000 --dt 25 --with-plot-test
```

---

## 3. Teste extra (apenas PCs mais fortes)

```bat
python run_experiment_matrix.py --particles 1000,2000,3000 --ranks 1,2,4,8 --repeats 3 --years 1000 --dt 25
```

---

## 4. Enviar resultados

Enviar a pasta:

```text
results\NOME_DO_PC
```

---

## 5. Agregar resultados (apenas uma pessoa)

Depois de juntar todas as pastas:

```bat
python merge_results.py
```

Serão gerados:

```text
results\combined_runtime_results.csv
results\aggregated_scaling_results.csv
results\report_ready_tables.md
```

---

## 6. GIF final bonito (apenas um PC)

```bat
mpiexec -n 4 python main.py --mode beauty --particles 2200 --years 12000 --dt 20 --plot-interval 200 --output-dir final_gif
```

---

## 7. GIF fiel ao enunciado (apenas um PC)

```bat
mpiexec -n 4 python main.py --mode report --particles 1800 --years 10000 --dt 25 --output-dir report_gif
```

---

## 8. Estimativa para 10^9 partículas

```bat
mpiexec -n 4 python main.py --scale-only --particles 1000000000 --years 1000000000 --dt 1000 --output-dir scale_estimate
```

---

## O que cada colega tem de entregar

```text
results\NOME_DO_PC
```

Nada mais.
