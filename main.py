import json
import cv2
import h5py
import mediapipe as mp
import numpy as np
from tensorflow import keras
from txt_area import AreaDeTexto
import math
import time 

# Constantes — precisam ser asmeesmas usadas no script de treino,
# senão o vetor de features fica diferente do que o modelo aprendeu

COOLDOWN_ESPACO = 0.9  # segundos entre espaços
COOLDOWN_APAGAR = 0.9  # segundos entre apagões
ultimo_apagar = 0.0
ultimo_espaco = 0.0
CAMINHO_MODELO = "modelo_libras_mlp.h5"
CAMINHO_DETECTOR = "hand_landmarker.task"
CONFIANCA_MINIMA = 0.70  # abaixo disso, ignora a previsao

#LETRAS_PERMITIDAS = {"H", "j"}  # letras de teste a serem exibidas
ALTURA_BARRA_TEXTO = 60 

PULSO = 0
BASE_DEDO_MEDIO = 9
PONTAS_DEDOS = [4, 8, 12, 16, 20]
BASE_INDICADOR = 5

def vetor(p1, p2):
    #saber para que lado o pnto está apontando, no caso o polegar
    # se está apontando para fora da mão ou não
    return (p2.x - p1.x, p2.y - p1.y)

def normalizar(v):
    #deixa as setas/direcoes sempre do mesmo tamanho
    # antes de comparar, sem isso uma mão mais perto da camera teria setas
    # maiores que uma mão mais longes
    norma = math.hypot(v[0], v[1])
    if norma < 1e-6:
        return (0.0, 0.0)
    return (v[0] / norma, v[1] / norma)

def produto_escalar(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def distancia(p1, p2):
    #para sabr se o dedo esta dobrado ou esticado, comparando
    # a ponta do dedo com o punho
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def gesto_apagar(landmarks):
    #usa posição do punho e distante do dedo polegar para determinar o apagar
    punho = landmarks[0]
    base_medio = landmarks[9]
    pontas = [8, 12, 16, 20]
    bases = [5, 9, 13, 17]

    eixo_mao = normalizar(vetor(punho, base_medio))
    tamanho_mao = distancia(punho, base_medio)   

    #checa se todos os dedos estão fechados
    for ponta, base in zip(pontas, bases):

        #checagem feita pela distancia entre punho e ponta dos dedos, quanto
        #mais longe, mais a probabilidade dos dedos nao estarem dobrados
        d_ponta = distancia(landmarks[ponta], punho)
        d_base = distancia(landmarks[base], punho)
        if d_ponta > d_base * 1.1:
            return False

   #checa se polega est esticado
    ponta_polegar = landmarks[4]
    base_polegar = landmarks[2]

    #polegar pego por distancia entre ponta e base
    comprimento_polegar = distancia(ponta_polegar, base_polegar)

    # se polegar esticado a 80% ou mais entende que ele está suficientemente
    #esticado, simbolizando o gestor de apagar/dislike
    if comprimento_polegar < tamanho_mao * 0.80:
        return False

    #eixo que o polegar se encontra,se estiver apotando fora/oposto da mão
    #entende que provavelmente é gesto apagar
    direcao_polegar = normalizar(vetor(base_polegar, ponta_polegar))
    alinhamento = produto_escalar(direcao_polegar, eixo_mao)
    if alinhamento <= 0.5:
        return False
    
    dist_polegar_indicador = distancia(landmarks[4], landmarks[6]) / tamanho_mao

    inclinacao_mao = abs(eixo_mao[0])  # componente x do eixo da mão, normalizado    

    return True  # true para gesto apagar, caso todas as condições forem satisfeitas

def gesto_espaco(landmarks):
    # mão aberta, 4 dedos esticados e juntos + polegar pouco afastado da palma
    punho = landmarks[0]
    tamanho_mao = distancia(punho, landmarks[9])
    if tamanho_mao < 1e-6:
        return False

    pontas = [8, 12, 16, 20]
    articulacoes = [6, 10, 14, 18]  #pip de cada dedo

    #dedos esticados, a ponta tem que estar mais long do punho que a articulação do meio
    for ponta, art in zip(pontas, articulacoes):
        if distancia(landmarks[ponta], punho) < distancia(landmarks[art], punho) * 1.1:
            return False

    # dedos juntados entre si
    for a, b in zip(pontas, pontas[1:]):
        if distancia(landmarks[a], landmarks[b]) / tamanho_mao > 0.45:
            return False

    # polegar consideravelmente afastado, ponta do polegar longe da base do indicador
    if distancia(landmarks[4], landmarks[5]) / tamanho_mao < 0.6:
        return False

    return True

def normalizar_landmarks(landmarks) -> np.ndarray:
    """mesma funçao usada no treino — gera o vetor de 72 features."""
    pontos = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

    #se o padrão de pontos de landmarks/esqueleto da mão não for o esperado, cai e erro
    if pontos.shape != (21, 3):
        raise ValueError(f"Esperados 21 pontos da mão; recebidos {pontos.shape}.")

    pontos = pontos - pontos[PULSO]
    escala = max(float(np.linalg.norm(pontos[BASE_DEDO_MEDIO])), 1e-6)
    pontos = pontos / escala

    fechamento_dedos = [float(np.linalg.norm(pontos[p])) for p in PONTAS_DEDOS]
    polegar_indicador = float(
        np.linalg.norm(pontos[PONTAS_DEDOS[0]] - pontos[BASE_INDICADOR])
    )
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
    ).astype(np.float32, copy=False)

#função recebe caminho do do modelo_libras_mlp.h5
def carregar_modelo(caminho: str):
    #carrega modelo keras e as classes salvas nos atributos do h5

    #aqui o modelo recebe o caminho do arquivo .h5 e carrega o modelo e as classes
    modelo = keras.models.load_model(caminho)

    #efetua leitura do arquivo de treino .h5
    with h5py.File(caminho, "r") as arquivo_h5:
        #classe recebe valor da lista de letras do arquivo de treino 
        classes = json.loads(arquivo_h5.attrs["classes"])
    return modelo, classes

#recebe modelo já carregado, a lista de letras e os ontos da mão(landmarks)
def prever_letra(modelo, classes, landmarks):
    #Roda modelo em uma mão detectada e retorna (letra, confianca/precisão)

    #transforma o landmarks em um array de 72 numeros que é o o que o medolo entende
    features = normalizar_landmarks(landmarks)

    #formata o array em um formato espera para receber esses dados
    features = np.expand_dims(features, axis=0)

    #recebe esse array para o modelo e retorna a letra prevista e a confiança dessa previsão
    probabilidades = modelo.predict(features, verbose=0)[0]
    indice = int(np.argmax(probabilidades))
    return classes[indice], float(probabilidades[indice])


def desenhar_barra_de_texto(frame, texto: str):
    altura, largura = frame.shape[:2]
 
    sobreposicao = frame.copy()
    cv2.rectangle(
        sobreposicao,
        (0, altura - ALTURA_BARRA_TEXTO),
        (largura, altura),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(sobreposicao, 0.6, frame, 0.4, 0, dst=frame)
 
    texto_exibido = texto if texto else "_"
    cv2.putText(
        frame,
        texto_exibido,
        (15, altura - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )


def main() -> None:
    modelo, classes = carregar_modelo(CAMINHO_MODELO)
    #print(f"Modelo carregado. Filtrando apenas: {sorted(LETRAS_PERMITIDAS)}")

    ultimo_apagar = 0.0

    ultimo_espaco = 0.0  # variável para controlar o cooldown do gesto de espaço

    area_de_texto = AreaDeTexto(frames_para_confirmar=15)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=CAMINHO_DETECTOR),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )

    cap = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # flip(frame, 1) espelha horizontalmente (efeito "espelho" da webcam)
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = detector.detect(mp_image)

            letra_valida_no_frame = None

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    letra = None
                    confianca = 0.0

                    if gesto_apagar(hand_landmarks):
                        agora = time.time()
                        if agora - ultimo_apagar > COOLDOWN_APAGAR:
                            area_de_texto.apagar_ultimo()  # método que você precisa ter/criar na classe AreaDeTexto
                            ultimo_apagar = agora
                        cv2.putText(frame, "APAGAR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

                    elif gesto_espaco(hand_landmarks):
                        agora = time.time()
                        if agora - ultimo_espaco > COOLDOWN_ESPACO:
                            area_de_texto.adicionar_espaco()  # já existe na sua classe
                            ultimo_espaco = agora
                        cv2.putText(frame, "ESPACO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 2)

                    else:                   
                        letra_valida_no_frame = None
                        try:
                            letra, confianca = prever_letra(modelo, classes, hand_landmarks)
                        except ValueError:
                            letra, confianca = None, 0.0
                    
                    if letra is not None and confianca >= CONFIANCA_MINIMA:
                        texto = f"{letra} ({confianca * 100:.0f}%)"
                        cv2.putText(
                            frame,
                            texto,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 0),
                            2,
                        )
                        letra_valida_no_frame = letra                

            area_de_texto.atualizar(letra_valida_no_frame)
 
            desenhar_barra_de_texto(frame, area_de_texto.obter_texto())

            cv2.imshow("Projeto LIBRAS", frame)

            if cv2.waitKey(1) & 0xFF == 27:  #sai do loop quando clica no esc
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()