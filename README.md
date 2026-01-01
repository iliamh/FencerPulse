cat > README.md <<'MD'
# FencerPulse

این پروژه یک اپ ساده‌ی فارسی برای شمشیربازی است.  
چند عدد ساده وارد می‌کنی و اپ می‌گوید کدام اسلحه به تو نزدیک‌تر است (فلوره/اپه/سابر).

## اجرا (فقط همین‌ها)

```bash
git clone <REPO_URL>
cd FencerPulse
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_demo_data.py
python scripts/train_model.py
streamlit run streamlit_app.py
