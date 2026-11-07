import json
import os
from pathlib import Path
import h5py
import matplotlib
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix)
from tensorflow import keras
from tensorflow.keras import layers

matplotlib.use("Agg") #faz o matplotlib salvar os gráficos em arquivo sem abrir janela
PONTAS_DEDOS = [4, 8, 12, 16, 20] #indices do mediapipe das pontas dos dedos

def normalizar_landmarks(landmarks) -> np.ndarray:
    pontos = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    pontos = pontos - pontos[0] #isso vai colocar o pulso como origem
    escala = float(np.linalg.norm(pontos[9]))
    if escala == 0:  # evita dividir por zero
        escala = 0.000001

    pontos = pontos / escala
    fechamento_dedos = [float(np.linalg.norm(pontos[p])) for p in PONTAS_DEDOS] #distância de cada ponta de dedo até o pulso, dedo dobrado fica com a distância menor
    polegar_indicador = float(np.linalg.norm(pontos[PONTAS_DEDOS[0]] - pontos[5])) #distância da ponta do polegar até a base do indicador (ponto 5)
    espacamentos = [
        float(np.linalg.norm(pontos[PONTAS_DEDOS[i]] - pontos[PONTAS_DEDOS[i + 1]]))
        for i in range(1, len(PONTAS_DEDOS) - 1)
    ]

    return np.concatenate(
        (
            pontos.ravel(),
            np.asarray(fechamento_dedos, dtype=np.float32),
            np.asarray([polegar_indicador], dtype=np.float32),
            np.asarray(espacamentos, dtype=np.float32),
        )
    )

def listar_classes(pasta_dataset: str | Path) -> list[tuple[str, list[Path]]]:
    classes = []
    for pasta in sorted(Path(pasta_dataset).iterdir()):
        if pasta.is_dir():
            imagens = sorted(pasta.glob("*.jpg")) #pega todas as fotos .jpg da pasta
            if imagens:
                classes.append((pasta.name.upper(), imagens)) #o nome da pasta vira a letra

    if len(classes) < 2:
        raise ValueError("Precisa ter pelo menos duas pastas com fotos")
    return classes

def extrair_dataset(detector, classes: list[tuple[str, list[Path]]], minimo_por_classe: int) -> dict[str, np.ndarray]:
    dados = {}
    for classe, imagens in classes:
        vetores = []
        for indice, caminho in enumerate(imagens, start=1):
            try:
                imagem = mp.Image.create_from_file(str(caminho))
                resultado = detector.detect(imagem)
            except Exception as erro:
                print(f"Não foi possível ler '{caminho.name}': {erro}")
                continue

            if resultado.hand_landmarks: #só usa as fotos em que o mediapipe achou a mão
                vetores.append(normalizar_landmarks(resultado.hand_landmarks[0]))

            if indice % 50 == 0 or indice == len(imagens):
                print(f"  {indice}/{len(imagens)}", end="\r", flush=True)

        if len(vetores) < minimo_por_classe:
            raise ValueError(
                f"Precisa ter pelo menos {minimo_por_classe} fotos válidas."
            )
        dados[classe] = np.asarray(vetores, dtype=np.float32)
    return dados

def criar_modelo(X_normalizacao: np.ndarray, numero_classes: int) -> keras.Model:
    normalizador = layers.Normalization()
    normalizador.adapt(X_normalizacao)
    #são 3 camadas com duas sendo com a função de ativação relu e uma sendo a saida com softmax que gera o vetor de probabilidade de saida
    modelo = keras.Sequential(
        [
            keras.Input(shape=(72,)), #63 coordenadas (21 pontos x 3) + 5 fechamentos + 1 polegar-indicador + 3 espaçamentos = 72 (só para lembrar vai que eu esqueço disso)
            normalizador, 
            layers.Dense(128, activation="relu"), 
            layers.Dense(64, activation="relu"),
            layers.Dense(numero_classes, activation="softmax"),
        ]
    )
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo

def separar_validacao(dados: dict[str, np.ndarray]):
    X_treino, y_treino, X_validacao, y_validacao = [], [], [], []
    for classe, vetores in dados.items():
        total = len(vetores)
        corte = int(total * 0.8)  # pega e faz o calculo para que 80% seja para treino
        X_treino.extend(vetores[:corte]) #pega até os 80% que é o valor do corte para o treino
        y_treino.extend([classe] * corte)
        X_validacao.extend(vetores[corte:]) #vai do corte até o final 20% para a validação
        y_validacao.extend([classe] * (total - corte))
    return (np.asarray(X_treino, dtype=np.float32),
        np.asarray(y_treino),
        np.asarray(X_validacao, dtype=np.float32),
        np.asarray(y_validacao),
    )

def salvar_relatorio_visual(historico, acuracia):
    """Aqui vamos gerar os gráficos da acurácia, perda e a matriz de confusão 
            
        Isso vai servir para validar o treinamento do modelo
        e vai me mostrar se eu vou precisar aumentar o numero de
        neuronios ou o numero de camadas
    """
    epocas = range(1, len(historico.history["loss"]) + 1)
    plt.figure(figsize=(17, 14))
    plt.suptitle(f"Acurácia na validação: {acuracia * 100:.2f}%", fontsize=16)
    #gráfico para mostrar a acurácia do modelo 
    plt.subplot(2, 2, 1)
    plt.plot(epocas, historico.history["accuracy"], label="Treino")
    plt.plot(epocas, historico.history["val_accuracy"], label="Validação")
    plt.title("Acurácia por época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)
    plt.legend()

    #gráfico para mostrar a perda do modelo
    plt.subplot(2, 2, 2)
    plt.plot(epocas, historico.history["loss"], label="Treino")
    plt.plot(epocas, historico.history["val_loss"], label="Validação")
    plt.title("Perda por época")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.legend()

    plt.tight_layout()
    plt.savefig("relatorio_treinamento.png")
    plt.close() #fechameneto do plt

def matriz_confusao_grafico(y_val, previsoes, classes):
    """Aqui vamos gerar o gráfico da matriz de confusão"""
    matriz = confusion_matrix(y_val, previsoes, labels=classes, normalize="true")

    plt.figure(figsize=(12, 10))
    tela = ConfusionMatrixDisplay(matriz, display_labels=classes)
    tela.plot(ax=plt.gca(), cmap="Blues", include_values=False)  # gca = pega o gráfico atual
    plt.title("Matriz de confusão")
    plt.xticks(rotation=45)
    plt.savefig("matriz_confusao.png")
    plt.close()

def grafico_f1_score(metricas, classes):
    """Aqui vamos calcular o f1-score"""
    f1 = []
    for letra in classes:
        f1.append(metricas[letra]["f1-score"] * 100)

    plt.figure(figsize=(14, 6))
    plt.bar(classes, f1)
    plt.title("F1-score de cada letra")
    plt.xlabel("Letra")
    plt.ylabel("F1-score (%)")
    plt.ylim(0, 100)
    plt.savefig("grafico_f1_score.png")
    plt.close()
   

keras.utils.set_random_seed(42) #semente fixa pra dar o mesmo resultado toda vez que treinar
classes = listar_classes("dataset")
caminho = "hand_landmarker.task" #precisa ter esse modelo do mediapipe se não da erro (meu Deus sou burro de mais)
opcoes = vision.HandLandmarkerOptions(
    base_options=BaseOptions(caminho),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5)
with vision.HandLandmarker.create_from_options(opcoes) as detector:
    dados = extrair_dataset(detector, classes, 20)

classes_ordenadas = sorted(dados.keys())
mapa_classe_indice = {classe: indice for indice, classe in enumerate(classes_ordenadas)}
numero_classes = len(classes_ordenadas)
X_treino, y_treino, X_val, y_val = separar_validacao(dados)
y_treino_idx = np.asarray([mapa_classe_indice[c] for c in y_treino])
y_val_idx = np.asarray([mapa_classe_indice[c] for c in y_val])
print(f"\nValidando com {len(X_treino)} amostras de treino e {len(X_val)} amostras separadas...")
modelo = criar_modelo(X_treino, numero_classes) #chamada para finalmente criar o modelo após criar eu vou chamar o .fit para treinar ele
parada_antecipada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
#função para treinar o modelo
historico = modelo.fit(
    X_treino,
    y_treino_idx,
    validation_data=(X_val, y_val_idx),
    epochs=30,  
    batch_size=32,
    callbacks=[parada_antecipada],
    verbose=2,
)
previsoes_idx = np.argmax(modelo.predict(X_val, verbose=0), axis=1)
previsoes = np.asarray([classes_ordenadas[i] for i in previsoes_idx])
acuracia = accuracy_score(y_val, previsoes)
metricas = classification_report(y_val, previsoes, labels=classes_ordenadas, zero_division=0, output_dict=True)
print(f"Acurácia: {acuracia * 100:.2f}%")
print(classification_report(y_val, previsoes, labels=classes_ordenadas, zero_division=0, digits=3))
salvar_relatorio_visual(historico,acuracia)
matriz_confusao_grafico(y_val, previsoes, classes_ordenadas)
grafico_f1_score(metricas, classes_ordenadas)
melhor_epoca = int(np.argmin(historico.history["val_loss"]) + 1)
caminho_modelo_libras = "modelo_libras_mlp.h5"
temporario = "modelo_libras_mlp.tmp.h5"
modelo.save(temporario) #salva o modelo em um arquivo temporario

with h5py.File(temporario, "a") as arquivo_h5: #abre o arquivo h5 para gravar novas informações
    arquivo_h5.attrs["classes"] = json.dumps(classes_ordenadas)
os.replace(temporario, caminho_modelo_libras)


