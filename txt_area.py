class AreaDeTexto:
    
    ##acumula letras inseidas pelo usuario, no caso só é adicionada de fato somente quando a letra 
    #é repetida por alguns frames
    def __init__(self, frames_para_confirmar: int = 15):
        self.texto = "" #inicializa texto em branco
        self.frames_para_confirmar = frames_para_confirmar #aqui é quantidade MINIMA de frames que tem que ser parar ser capturado 
                                                            #e confirmar a letra sinalizada

        #inicializa variaveis de texto/ultima letra sem valores
        self._ultima_letra_vista = None #ultima letra detectada
        self._contador_estabilidade = 0 #diz quantos frames a letra sinalizada foi confrimada
        self._ultima_letra_confirmada = None #valor de ultima letr confirmada para inserir na caixa de texto (as que tem no minimo 15 frames capturados)

    #função para detectar letra letra e atualizar caixa de texto
    def atualizar(self, letra: str | None) -> None:        

        #se letra chegar vazio, todo o conjunto também fica
        if letra is None:
            self._ultima_letra_vista = None
            self._contador_estabilidade = 0
            self._ultima_letra_confirmada = None
            return

        #se a ultima letra detectada for igual a letra atual então estabilida soma seu valor atual +1 (exemplo: se 14, vira 15)
        #se não entende que é uma nova sequencia e atualiza ultima letra e reseta contagem de estabilidade com 1
        if letra == self._ultima_letra_vista:
            self._contador_estabilidade += 1
        else:
            self._ultima_letra_vista = letra
            self._contador_estabilidade = 1

        #verificação se contador chegou na quantidade correta de frames necessarios para confirmação
        confirmou_agora = self._contador_estabilidade == self.frames_para_confirmar
        eh_letra_nova = letra != self._ultima_letra_confirmada

        #montando text area de fato
        #se a letra tem o minimo de frames e é letra nova ele adiciona caixa de texto
        if confirmou_agora and eh_letra_nova:
            self.texto += letra
            self._ultima_letra_confirmada = letra


    #funções a trabalhar, NIVALDO pode tentar terminar esses caso queira
    #todos são conjuntos a serem usados nas chamadas
    def adicionar_espaco(self) -> None:
        if self.texto and not self.texto.endswith(" "):
            self.texto += " "

    def apagar_ultimo(self) -> None:
        self.texto = self.texto[:-1]

    def limpar(self) -> None:
        self.texto = ""
        self._ultima_letra_vista = None
        self._contador_estabilidade = 0
        self._ultima_letra_confirmada = None

    def obter_texto(self) -> str:
        return self.texto