# EDA — LPLCv2 (annotations_v2.json)

- Total de imagens: **37099**
- Total de placas anotadas: **41487**
- Imagens ausentes em `data/images/` (referenciadas no JSON mas sem arquivo): **0**

## Legibilidade (`leg`)
- Illegible (0): 5362 (12.9%)
- Poor (1): 7520 (18.1%)
- Good (2): 10180 (24.5%)
- Perfect (3): 18425 (44.4%)

## Outros atributos
- `faulty=true`: 3690 imagens (9.9%)
- `rain=true`: 770 imagens (2.1%)
- Períodos do dia: {'night': 4146, 'evening': 9156, 'afternoon': 12799, 'morning': 10998}
- Dias distintos (`day`): [0, 1, 2]
- Câmeras distintas (`cam`, excluindo nulo): 865
- Imagens com `cam=null`: 2338
- Imagens por câmera — mediana: 21, p10: 2, p90: 81, max: 3603

## Placas por imagem
- Distribuição: {1: 33537, 2: 2962, 3: 446, 4: 101, 5: 37, 6: 14, 7: 1, 8: 1}

## Tamanho da bbox em relação à imagem (apenas leg > 0, dataset completo)
- Amostras: 36125
- Largura relativa — mediana: 0.0750, p10: 0.0430, p90: 0.1741
- Altura relativa — mediana: 0.0413, p10: 0.0260, p90: 0.0979
- Área relativa — mediana: 0.00306, p10: 0.00120, p90: 0.01657

## Efeito dos filtros propostos (Passo 2)
- Imagens descartadas por `faulty=true`: 3690
- Imagens descartadas por conter ao menos 1 placa `leg=0`: 4607
- Imagens restantes para treino: **28802** (77.6% do total)
- Placas restantes para treino: **30738** (74.1% do total)

## Figuras geradas
- `reports/figures/leg_distribution.png`
- `reports/figures/time_distribution.png`
- `reports/figures/bbox_area_ratio.png`
- `reports/figures/anns_per_image.png`