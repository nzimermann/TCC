# Avaliação no split de teste

- Pesos: `models\plate_detector.pt`
- Imagens no split de teste: 4320 (avaliadas na checagem por subgrupo: 4320)

## Métricas oficiais (Ultralytics `model.val(split='test')`)
- Precision: 0.913
- Recall: 0.977
- mAP50: 0.984
- mAP50-95: 0.924

## Checagem cruzada (matching por IoU>=0.5, conf>=0.25)
- TP=4531 FP=668 FN=23
- Precision: 0.872  Recall: 0.995

## Por condição de chuva
| Grupo | Imagens | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| rain=não | 4237 | 4448 | 667 | 23 | 0.870 | 0.995 |
| rain=sim | 83 | 83 | 1 | 0 | 0.988 | 1.000 |

## Por período do dia
| Grupo | Imagens | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| time=morning | 1514 | 1613 | 218 | 7 | 0.881 | 0.996 |
| time=afternoon | 1504 | 1561 | 293 | 10 | 0.842 | 0.994 |
| time=evening | 744 | 772 | 103 | 5 | 0.882 | 0.994 |
| time=night | 558 | 585 | 54 | 1 | 0.915 | 0.998 |

## Por tamanho da placa (área bbox/imagem — cortes: (0.0015, 0.008))
- Sem coluna de precision/FP aqui: falso positivo não corresponde a nenhuma placa real, então não tem "tamanho" próprio.
| Grupo | TP | FN | Recall |
|---|---|---|---|
| placa pequena | 453 | 11 | 0.976 |
| placa média | 3447 | 11 | 0.997 |
| placa grande | 631 | 1 | 0.998 |

## Por legibilidade da placa (`leg` do dataset original)
- `leg=0` (Illegible) não aparece: o Passo 2 já descartou toda imagem com alguma placa ilegível.
- Mesma ressalva do tamanho: sem precision/FP, falso positivo não tem legibilidade própria.
| Grupo | TP | FN | Recall |
|---|---|---|---|
| leg=1 (Poor) | 760 | 19 | 0.976 |
| leg=2 (Good) | 1377 | 3 | 0.998 |
| leg=3 (Perfect) | 2394 | 1 | 1.000 |
