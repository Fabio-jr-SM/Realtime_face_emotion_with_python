# Detecção de Idade em Tempo Real usando OpenCV e DeepFace

Este projeto utiliza as bibliotecas OpenCV e DeepFace para realizar a detecção de rostos e a estimativa de idade em tempo real. O programa captura vídeo da webcam, detecta rostos nos frames de vídeo e usa o DeepFace para estimar a idade de cada rosto detectado.

## Pré-requisitos

Antes de executar o código, certifique-se de ter as seguintes bibliotecas instaladas:

- OpenCV
- Matplotlib
- DeepFace

Você pode instalar essas bibliotecas usando pip:

```bash
pip install opencv-python matplotlib deepface
```

## Explicação do Código

```python
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# Carrega o classificador Haar Cascade pré-treinado para detecção de rostos
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Captura vídeo da webcam padrão (dispositivo 0)
cap = cv2.VideoCapture(0)

# Se a webcam padrão não estiver disponível, tenta o próximo dispositivo (dispositivo 1)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
# Lança um erro se nenhuma webcam estiver disponível
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Lê um frame da webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Analisa o frame para estimativa de idade usando DeepFace
    results = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
    
    # Extrai o primeiro resultado da análise
    result = results[0]
    
    # Converte o frame para escala de cinza para detecção de rostos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos no frame em escala de cinza
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        # Desenha um retângulo ao redor de cada rosto detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Define a fonte para exibir o texto
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Exibe a idade estimada no frame
    cv2.putText(frame,
                str(result['age']),
                (50, 50),
                font, 3,
                (0, 0, 255),
                2,
                cv2.LINE_4)

    # Mostra o frame com os retângulos desenhados e a informação de idade
    cv2.imshow('Demo Video', frame)

    # Sai do loop quando a tecla 'ESC' for pressionada
    if cv2.waitKey(5) == 27:
        break

# Libera o objeto de captura de vídeo e fecha todas as janelas do OpenCV
cap.release()
cv2.destroyAllWindows()
```

## Explicação Detalhada

1. **Importar Bibliotecas**:
   - `cv2`: Biblioteca OpenCV para tarefas de visão computacional.
   - `matplotlib.pyplot`: Biblioteca para plotagem (não utilizada neste código, mas importada).
   - `DeepFace`: Biblioteca para análise de rosto, incluindo estimativa de idade.

2. **Carregar Classificador Haar Cascade**:
   ```python
   faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   ```
   - Isso carrega um modelo pré-treinado para detectar rostos em imagens.

3. **Inicializar Webcam**:
   ```python
   cap = cv2.VideoCapture(0)
   if not cap.isOpened():
       cap = cv2.VideoCapture(1)
   if not cap.isOpened():
       raise IOError("Cannot open webcam")
   ```
   - Inicializa a webcam para capturar vídeo. Se a webcam padrão (dispositivo 0) não estiver disponível, tenta o próximo dispositivo (dispositivo 1).

4. **Processar Frames de Vídeo**:
   ```python
   while True:
       ret, frame = cap.read()
       if not ret:
           break
   ```
   - Esse loop captura frames da webcam até ocorrer um erro ou o loop ser interrompido.

5. **Analisar Frame para Estimativa de Idade**:
   ```python
   results = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
   result = results[0]
   ```
   - O frame é analisado usando o DeepFace para estimar a idade dos rostos detectados no frame.

6. **Converter Frame para Escala de Cinza**:
   ```python
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   ```
   - Converte o frame para escala de cinza para detecção de rostos.

7. **Detectar Rostos**:
   ```python
   faces = faceCascade.detectMultiScale(gray, 1.1, 4)
   for (x, y, w, h) in faces:
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
   ```
   - Detecta rostos no frame em escala de cinza e desenha retângulos ao redor deles.

8. **Exibir Idade no Frame**:
   ```python
   cv2.putText(frame, str(result['age']), (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
   ```
   - Exibe a idade estimada no frame.

9. **Mostrar o Frame**:
   ```python
   cv2.imshow('Demo Video', frame)
   ```
   - Mostra o frame processado em uma janela.

10. **Sair ao Pressionar 'ESC'**:
    ```python
    if cv2.waitKey(5) == 27:
        break
    ```
    - Sai do loop quando a tecla 'ESC' for pressionada.

11. **Liberar Recursos**:
    ```python
    cap.release()
    cv2.destroyAllWindows()
    ```
    - Libera a webcam e fecha todas as janelas do OpenCV.

## Exemplo de Saída do DeepFace

A análise do DeepFace retorna um dicionário com vários atributos. Aqui está um exemplo do que a saída pode conter:

```json
[
    {
        "emotion": {
            "angry": 0.0003,
            "disgust": 0.00000024, 
            "fear": 0.013, 
            "happy": 99.9466, 
            "sad": 0.000037, 
            "surprise": 0.0023, 
            "neutral": 0.0377
        }, 
        "dominant_emotion": "happy", 
        "region": {
            "x": 75, 
            "y": 47, 
            "w": 66, 
            "h": 66
        }, 
        "face_confidence": 0.94, 
        "age": 24, 
        "gender": {
            "Woman": 6.2591, 
            "Man": 93.7409
        }, 
        "dominant_gender": "Man", 
        "race": {
            "asian": 18.5888, 
            "indian": 14.5939, 
            "black": 44.0936, 
            "white": 2.1721, 
            "middle eastern": 1.8568, 
            "latino hispanic": 18.6949
        }, 
        "dominant_race": "black"
    }
]
```

Esta saída inclui informações sobre emoções, idade, gênero e raça para o rosto detectado. Neste projeto, estamos focando no atributo idade.

## Conclusão

Este projeto demonstra como usar OpenCV e DeepFace para realizar detecção de rostos e estimativa de idade em tempo real. O código captura vídeo da webcam, detecta rostos e estima a idade de cada rosto detectado, exibindo essa informação nos frames de vídeo.