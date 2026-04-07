import streamlit as st
import pandas as pd
import pulp
import plotly.express as px
import google.generativeai as genai

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="DSS Marketing", layout="wide")
st.title("📊 DSS - Tối ưu ngân sách Marketing")

# ==============================
# SESSION
# ==============================
if "run" not in st.session_state:
    st.session_state.run = False

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Input")

budget = st.sidebar.number_input(
    "Tổng ngân sách",
    1000000000, 5000000000, 2500000000, step=100000000
)

max_fb = st.sidebar.number_input("Max Facebook", value=1000000000)
max_gg = st.sidebar.number_input("Max Google", value=800000000)
max_em = st.sidebar.number_input("Max Email", value=1500000000)

if st.sidebar.button("🚀 Chạy Solver"):
    st.session_state.run = True

# ==============================
# SOLVER
# ==============================
if st.session_state.run:

    prob = pulp.LpProblem("Marketing", pulp.LpMaximize)

    x_FB = pulp.LpVariable('FB', lowBound=150000000)
    x_GG = pulp.LpVariable('GG', lowBound=150000000)
    x_LI = pulp.LpVariable('LI', lowBound=150000000)
    x_EM = pulp.LpVariable('EM', lowBound=150000000)
    x_TT = pulp.LpVariable('TT', lowBound=150000000)

    prob += (
        (-783206 + 3.897*x_FB) +
        (-363867 + 5.081*x_GG) +
        (-3349816 + 5.602*x_LI) +
        (-1763155 + 7.153*x_EM) +
        (2115162 - 0.589*x_TT)
    )

    prob += x_FB + x_GG + x_LI + x_EM + x_TT <= budget
    prob += x_FB <= max_fb
    prob += x_GG <= max_gg
    prob += x_EM <= max_em

    prob.solve()

    if pulp.LpStatus[prob.status] == "Optimal":

        df = pd.DataFrame({
            "Kênh": ["Facebook", "Google", "LinkedIn", "Email", "TikTok"],
            "Ngân sách": [
                x_FB.varValue,
                x_GG.varValue,
                x_LI.varValue,
                x_EM.varValue,
                x_TT.varValue
            ]
        })

        revenue = pulp.value(prob.objective)

        col1, col2 = st.columns(2)
        col1.metric(" Doanh thu", f"{revenue:,.0f}")
        col2.metric(" Ngân sách", f"{df['Ngân sách'].sum():,.0f}")

        st.markdown("---")

        fig = px.pie(df, values="Ngân sách", names="Kênh", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.style.format({"Ngân sách": "{:,.0f}"}))

        # ==============================
        # AI GEMINI (API MỚI)
        # ==============================
        st.markdown("---")
        st.markdown("###  AI Tư vấn")

        api_key = st.text_input("Nhập Gemini API Key", type="password")

        if st.button("Phân tích AI"):

            if not api_key:
                st.warning(" Nhập API key trước!")
            else:
                with st.spinner("AI đang phân tích..."):
                    try:
                        genai.configure(api_key=api_key)

                        #  MODEL MỚI CHUẨN (KHÔNG LỖI)
                        model = genai.GenerativeModel("models/gemini-1.5-flash")

                        prompt = f"""
                        Ngân sách tổng: {budget:,.0f} VNĐ

                        Facebook: {x_FB.varValue:,.0f}
                        Google: {x_GG.varValue:,.0f}
                        LinkedIn: {x_LI.varValue:,.0f}
                        Email: {x_EM.varValue:,.0f}
                        TikTok: {x_TT.varValue:,.0f}

                        Doanh thu: {revenue:,.0f} VNĐ

                        Hãy:
                        1. Đánh giá phương án
                        2. Nêu rủi ro lớn nhất
                        3. Đề xuất cải thiện

                        Trả lời NGẮN GỌN, chuyên nghiệp như giám đốc marketing.
                        """

                        response = model.generate_content(prompt)

                        if response and hasattr(response, "text"):
                            st.success(" AI đã phân tích:")
                            st.write(response.text)
                        else:
                            st.warning("AI không trả lời.")

                    except Exception as e:
                        st.error(f" Lỗi: {str(e)}")

    else:
        st.error(" Không tìm được nghiệm tối ưu")
