import os
import streamlit as st
import pandas as pd
import altair as alt

from app.engines.recommender import WeaponRecommender

st.set_page_config(
    page_title="FencerPulse",
    page_icon="🤺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = os.path.join("models", "weapon_model.joblib")

def load_model():
    if os.path.exists(MODEL_PATH):
        return WeaponRecommender.load(MODEL_PATH)
    return None

def hbar(top3):
    df = pd.DataFrame({"گزینه": [x[0] for x in top3], "احتمال": [x[1] for x in top3]})
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("احتمال:Q", scale=alt.Scale(domain=[0,1])),
        y=alt.Y("گزینه:N", sort="-x"),
        tooltip=["گزینه", alt.Tooltip("احتمال:Q", format=".2f")]
    ).properties(height=160)

st.markdown("""
<style>
.block-container{max-width:1100px; padding-top:1.5rem;}
.hero{border-radius:20px; padding:18px 20px; background: radial-gradient(80% 120% at 10% 10%, rgba(80,120,255,0.30), rgba(0,0,0,0.0)), rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);}
.panel{border-radius:20px; padding:16px 18px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);}
.muted{opacity:0.75; font-size:0.93rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='hero'>
  <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:14px;'>
    <div>
      <div style='font-size:2.05rem; font-weight:850;'>FencerPulse — استعداد‌یابی شمشیربازی</div>
      <div class='muted'>فقط چند عدد ساده وارد کن. خروجی: پیشنهاد اسلحه + نمودار احتمال + دلایل قابل فهم.</div>
    </div>
    <div style='font-size:2.3rem;'>🤺</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

if "step" not in st.session_state:
    st.session_state.step = 1

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("مرحله ۱: ورود اطلاعات", use_container_width=True):
        st.session_state.step = 1
with c2:
    if st.button("مرحله ۲: نتیجه", use_container_width=True):
        st.session_state.step = 2
with c3:
    if st.button("نمونه آماده", use_container_width=True):
        st.session_state.step = 3

st.progress({1:0.33, 2:0.66, 3:1.0}[st.session_state.step])

model = load_model()
if model is None:
    st.error("مدل پیدا نشد. اول این‌ها را اجرا کن:\n\npython scripts/make_demo_data.py\npython scripts/train_model.py")
    st.stop()

def sample_input():
    return dict(
        age=17, height_cm=174.0, weight_kg=68.0, reach_cm=176.0,
        sprint_20m_s=3.15, reaction_ms=240.0, beep_level=9.5, jump_cm=52.0,
        weekly_training_h=4.0,
        dominant_hand="راست", injury="ندارد", goal="مسابقه", experience="متوسط",
    )

def run_infer(x):
    res = model.predict(x)
    st.success(f"پیشنهاد اصلی: **{res.primary}**  |  اطمینان: **{res.confidence:.0%}**")
    st.altair_chart(hbar(res.top3), use_container_width=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("چرا این گزینه؟ (دلایل)")
    for name, val in res.explanation_items:
        sign = "↑" if val >= 0 else "↓"
        st.write(f"- {sign} **{name}**  (اثر تقریبی: {val:+.2f})")
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("ورود اطلاعات")
    st.caption("اعداد تقریبی هم قابل قبول است. اگر نداری، روی «نمونه آماده» بزن.")

    left, right = st.columns(2)
    with left:
        age = st.number_input("سن", 10, 40, 17, 1)
        height_cm = st.number_input("قد (cm)", 120.0, 220.0, 174.0, 0.5)
        weight_kg = st.number_input("وزن (kg)", 30.0, 180.0, 68.0, 0.5)
        reach_cm = st.number_input("ریچ/طول دست (cm)", 120.0, 240.0, 176.0, 0.5)
        weekly_training_h = st.number_input("ساعت تمرین هفتگی", 0.0, 20.0, 4.0, 0.5)

    with right:
        sprint_20m_s = st.number_input("زمان ۲۰ متر (ثانیه)", 2.4, 7.0, 3.15, 0.01)
        reaction_ms = st.number_input("تست واکنش (ms)", 150.0, 600.0, 240.0, 1.0)
        beep_level = st.number_input("بیپ تست (Level)", 1.0, 16.0, 9.5, 0.1)
        jump_cm = st.number_input("پرش عمودی (cm)", 10.0, 110.0, 52.0, 1.0)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        dominant_hand = st.selectbox("دست غالب", ["راست", "چپ"])
    with d2:
        injury = st.selectbox("آسیب‌دیدگی", ["ندارد", "زانو", "مچ پا", "شانه", "مچ دست"])
    with d3:
        goal = st.selectbox("هدف", ["تفریح", "مسابقه", "بورسیه"])
    with d4:
        experience = st.selectbox("سطح تجربه", ["مبتدی", "متوسط", "حرفه‌ای"])

    st.write("")
    if st.button("محاسبه نتیجه", type="primary", use_container_width=True):
        st.session_state.last_input = dict(
            age=int(age), height_cm=float(height_cm), weight_kg=float(weight_kg), reach_cm=float(reach_cm),
            sprint_20m_s=float(sprint_20m_s), reaction_ms=float(reaction_ms), beep_level=float(beep_level),
            jump_cm=float(jump_cm), weekly_training_h=float(weekly_training_h),
            dominant_hand=dominant_hand, injury=injury, goal=goal, experience=experience,
        )
        st.session_state.step = 2
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.step == 2:
    x = st.session_state.get("last_input", sample_input())
    run_infer(x)

else:
    st.info("این یک نمونه‌ی آماده برای تست سریع است.")
    run_infer(sample_input())
