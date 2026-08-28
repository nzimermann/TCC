# Avaliação no split de teste

- Pesos: `runs\detect\smoke_test\weights\best.pt`
- Imagens no split de teste: 4320 (avaliadas nesta rodada: 50)

## Métricas oficiais (Ultralytics `model.val(split='test')`)
- Pulado (rodando com `--limit`, é só o debug rápido por subgrupo).

## Checagem cruzada (matching por IoU>=0.5, conf>=0.25)
- TP=0 FP=0 FN=52
- Precision: nan  Recall: 0.000

## Por condição de chuva
| Grupo | Imagens | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| rain=não | 50 | 0 | 0 | 52 | nan | 0.000 |
| rain=sim | 0 | 0 | 0 | 0 | nan | nan |

## Por período do dia
| Grupo | Imagens | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| time=morning | 13 | 0 | 0 | 14 | nan | 0.000 |
| time=afternoon | 26 | 0 | 0 | 27 | nan | 0.000 |
| time=evening | 10 | 0 | 0 | 10 | nan | 0.000 |
| time=night | 1 | 0 | 0 | 1 | nan | 0.000 |

## Por tamanho da placa (área bbox/imagem — cortes: (0.0015, 0.008))
- Sem coluna de precision/FP aqui: falso positivo não corresponde a nenhuma placa real, então não tem "tamanho" próprio.
| Grupo | TP | FN | Recall |
|---|---|---|---|
| placa pequena | 0 | 46 | 0.000 |
| placa média | 0 | 0 | nan |
| placa grande | 0 | 6 | 0.000 |

## Por legibilidade da placa (`leg` do dataset original)
- `leg=0` (Illegible) não aparece: o Passo 2 já descartou toda imagem com alguma placa ilegível.
- Mesma ressalva do tamanho: sem precision/FP, falso positivo não tem legibilidade própria.
| Grupo | TP | FN | Recall |
|---|---|---|---|
| leg=1 (Poor) | 0 | 6 | 0.000 |
| leg=2 (Good) | 0 | 30 | 0.000 |
| leg=3 (Perfect) | 0 | 16 | 0.000 |
