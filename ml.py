from fastai.vision.all import *
learn = load_learner('modeldatesF.pkl')
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    return print(learn.predict(img))

predict("test03.png")