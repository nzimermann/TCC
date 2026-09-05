# TCC
Sistema de Controle de Acesso de Baixo Custo para Veículos Utilizando Visão Computacional na Borda

## Sobre

Detector de placas de carro (classe única) baseado em YOLO11n, treinado sobre o dataset [LPLCv2](https://github.com/lmlwojcik/LPLCv2-Dataset). O resultado final é um modelo `.pt` capaz de localizar a placa em uma imagem ou vídeo, testável ao vivo por webcam.

## Estrutura do repositório

```
data/
  raw/            # dataset bruto: images/ + annotations_v2.json (não versionado)
  yolo/           # dataset já convertido pro formato YOLO (gerado por script)
weights/          # checkpoint pré-treinado do Ultralytics (baixado automaticamente)
models/           # modelo final do projeto (plate_detector.pt)
outputs/
  runs/detect/    # saída de cada treino/avaliação do Ultralytics
  reports/        # relatórios e figuras gerados pelos scripts
src/
  data_prep/      # exploração, filtragem, split e conversão do dataset
  training/       # treino e exportação do modelo final
  evaluation/     # avaliação no split de teste
  inference/      # demo de detecção ao vivo pela webcam
configs/          # hiperparâmetros de treino
notebooks/        # notebook para treinar no Kaggle (GPU)
```

## Ambiente

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

O dataset (`data/raw/images/` + `data/raw/annotations_v2.json`) precisa ser obtido separadamente junto aos autores do LPLCv2 e colocado nesses caminhos — não é distribuído neste repositório. O checkpoint pré-treinado (`weights/yolo11n.pt`) é baixado automaticamente pelo Ultralytics na primeira vez que um script de treino roda.

## Sequência de execução

Os scripts em `src/` foram feitos para rodar nesta ordem, cada um consumindo a saída do anterior:

| # | Script | O que faz |
|---|---|---|
| 1 | `src/data_prep/eda.py` | Explora o dataset bruto (distribuições de legibilidade, câmera, tamanho de placa, condições de captura) e gera `outputs/reports/eda_summary.md` + gráficos em `outputs/reports/figures/`. |
| 2 | `src/data_prep/build_splits.py` | Aplica as regras de filtragem (`src/data_prep/filters.py`: descarta imagem com `faulty=true` ou com qualquer placa ilegível) e separa treino/val/teste **por câmera**, pra nenhuma câmera aparecer em mais de um split. Gera `data/yolo/splits.json`. |
| 3 | `src/data_prep/convert_annotations.py` | Converte as anotações do formato do LPLCv2 pro formato YOLO (`<classe> x_center y_center largura altura` normalizado) e monta `data/yolo/images/`, `data/yolo/labels/` e `data/yolo/data.yaml`. |
| — | `src/data_prep/visualize_annotations.py` *(opcional)* | Desenha as caixas de algumas imagens sorteadas, pra conferir visualmente que a interpretação das coordenadas está correta antes de treinar. |
| 4 | `src/training/train.py` | Treina o YOLO11n. Local, só é viável em modo `--smoke-test` (poucas imagens/épocas, só pra validar que o pipeline roda de ponta a ponta — sem GPU o treino completo é inviável). O treino de verdade, com o dataset inteiro, roda no Kaggle usando `notebooks/kaggle_train.ipynb` (que executa os passos 2–4 e o treino dentro de uma sessão com GPU). |
| 5 | `src/evaluation/evaluate.py` | Avalia o modelo treinado no split de teste: métricas oficiais do Ultralytics (precision, recall, mAP50, mAP50-95) e uma quebra por subgrupo (chuva, período do dia, tamanho da placa, legibilidade). Gera `outputs/reports/eval_summary.md`. |
| 6 | `src/training/export_model.py` | Promove o `best.pt` do treino escolhido para `models/plate_detector.pt` — o modelo oficial do projeto — conferindo antes que é mesmo um detector de placa válido. |

Comandos de cada um (todos aceitam `--help` pra ver as opções completas):

```bash
.venv/Scripts/python.exe src/data_prep/eda.py
.venv/Scripts/python.exe src/data_prep/build_splits.py
.venv/Scripts/python.exe src/data_prep/convert_annotations.py
.venv/Scripts/python.exe src/data_prep/visualize_annotations.py --n 5
.venv/Scripts/python.exe src/training/train.py --smoke-test
.venv/Scripts/python.exe src/evaluation/evaluate.py --weights outputs/runs/detect/full_run/weights/best.pt
.venv/Scripts/python.exe src/training/export_model.py --weights outputs/runs/detect/full_run/weights/best.pt
```

**Sobre o treino no Kaggle:** ao baixar `best.pt`/`last.pt` pela interface do Kaggle, o arquivo chega como `.zip` mesmo aparecendo como `.pt` na aba Output — isso é esperado (um `.pt` do PyTorch já é um arquivo zip por dentro). Basta renomear a extensão de volta pra `.pt`, não precisa extrair.

## Testar a detecção com a webcam

Depois que `models/plate_detector.pt` existir (passo 6), `src/inference/webcam_demo.py` abre a webcam (ou um vídeo/arquivo) e desenha ao vivo as placas detectadas:

```bash
.venv/Scripts/python.exe src/inference/webcam_demo.py
.venv/Scripts/python.exe src/inference/webcam_demo.py --source path/to/video.mp4
```

Pressione `q` ou `Esc` pra sair. Use `--imgsz 960` pra casar com a resolução usada no treino (o padrão é 640, mais rápido em CPU, mas o modelo foi treinado em 960).
