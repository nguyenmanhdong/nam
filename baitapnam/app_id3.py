from flask import Flask, request, render_template
import pickle

# ------------------------
# Load mô hình ID3
# ------------------------
with open("mushroom_model.pkl", "rb") as f:
    tree = pickle.load(f)

# Hàm predict (giống trong train_model_ID3.py)
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = sample.get(feature, None)
    if value in tree[feature]:
        return predict(tree[feature][value], sample)
    else:
        return "e"  # fallback nếu không tìm thấy giá trị

# ------------------------
# Thuộc tính đơn giản để demo form (có thể mở rộng đủ 22)
# ------------------------
ATTRIBUTES = {
    "odor": ["a","l","c","y","f","m","n","p","s"],
    "cap-color": ["n","b","c","g","r","p","u","e","w","y"],
    "gill-color": ["k","n","b","h","g","r","o","p","u","e","w","y"]
}

LABELS = {
    "e": "🍽️ Ăn được",
    "p": "☠️ Độc"
}

# ------------------------
# Flask App
# ------------------------
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    raw = None
    form_data = {}
    if request.method == "POST":
        form_data = {k: request.form.get(k) for k in ATTRIBUTES.keys()}
        raw = predict(tree, form_data)
        result = LABELS.get(raw, raw)
    return render_template("index.html", attrs=ATTRIBUTES, result=result, raw=raw, form=form_data)

if __name__ == "__main__":
    app.run(debug=True)
