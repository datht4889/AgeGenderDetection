import numpy as np

gender_label = ['Male', 'Female']
age_cat =   ['Baby', 'Infancy', 'Pre-school Age', 'Teenager', 'Young Adult', 'Early Middle Age', 'Middle Age', 'Elderly']
age_interval = ['[0-1]', '[2-6]', '[7-12]', '[13-18]', '[19-25]', '[26-35]', '[36-45]', '[46-59]', '[60-Older]']

def age_categorize(age):
    if age <=1:
        return 0
    elif age <=3:
        return 1
    elif age <=12:
        return 2
    elif age <=18:
        return 3
    elif age <=25:
        return 4
    elif age <=35:
        return 5
    elif age <=45:
        return 6
    elif age <=59:
        return 7
    else:
        return 8


def predict(image, model):
    img = np.expand_dims(image, axis=0)
    img = np.vstack([img])/255.
    pred = model.predict(img)
    # pred_age = int(pred[0])
    pred_age = age_interval[age_categorize(pred[0])]
    pred_gender = gender_label[np.argmax(pred[1][0])]
    pred_age_cat = age_cat[age_categorize(pred[0])]
    return pred_gender, pred_age, pred_age_cat