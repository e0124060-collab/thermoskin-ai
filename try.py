pred = model.predict(features)   # suppose 0=healthy, 1=inflamed
if pred == 0:
    result = "Healthy (no inflammation)"
else:
    result = "Inflammation detected"
prob = model.predict_proba(features)[0][1]  # probability of "inflamed" class
if prob > 0.5:
    result = "Inflamed"
else:
    result = "Healthy"
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(y_train)
print(le.classes_)    # e.g. array(['healthy', 'inflamed'], dtype='<U9')
print(le.transform(['healthy','inflamed']))  # e.g. [0, 1]
print(model.classes_)  # e.g. array(['healthy','inflamed'], dtype='<U9')
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['healthy','inflamed']))
probs = model.predict_proba(X_test)   # shape (n_samples, 2)
print(model.classes_)                # e.g. ['healthy','inflamed']
print(probs[:3])                     
for prob in probs[:3]:
    paired = list(zip(model.classes_, prob))
    print(paired)  # e.g. [('healthy', 0.7), ('inflamed', 0.3)]
import numpy as np
print(np.min(image), np.max(image))  # before any transform
img_norm = (image - image.min()) / (image.max() - image.min())
print(img_norm.min(), img_norm.max())  # should be 0.0, 1.0
import matplotlib.pyplot as plt
plt.hist(img_flattened, bins=50); plt.title("Pixel value distribution"); plt.show()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(10,4))
for ax, idx in zip(axes, [0, 5, 10]):  # example indices
    img = X_test[idx].reshape(img_height, img_width)
    true_label = le.inverse_transform([y_test[idx]])[0]
    pred_label = le.inverse_transform([y_pred[idx]])[0]
    prob_inf = model.predict_proba(X_test[idx].reshape(1,-1))[0][1]  # prob of inflamed
    ax.imshow(img, cmap='hot')
    ax.set_title(f"True: {true_label}\nPred: {pred_label} ({prob_inf:.2f})")
    ax.axis('off')
plt.show()
