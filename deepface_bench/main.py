from enum import StrEnum
from deepface import DeepFace

BEZOS_IMG1 = 'dataset/bezos/img20.jpg'
BEZOS_IMG2 = 'dataset/bezos/img21.jpg'

class Models(StrEnum):
    # convert this into enum label and value type
    FACE_NET = 'Facenet'
    FACE_NET_512 = 'Facenet512'
    ARC_FACE = 'ArcFace'


def bezos_check(model: Models):
    result = DeepFace.verify(BEZOS_IMG1, BEZOS_IMG2, model_name=model.value)
    debug(result)

def main():
    for model in Models:
        bezos_check(model)

if __name__ == '__main__':
    main()
