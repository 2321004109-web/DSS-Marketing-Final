import streamlit as st
import pandas as pd
import pulp
import plotly.express as px
import google.generativeai as genai

# ==========================================
# 1. CONFIG
# ==========================================
st.set_page_config(page_title="DSS Marketing Optimization", layout="wide")
st.title("Hệ Hỗ Trợ Ra Quyết Định (DSS) - ABC Retail")
st.markdown("### Tối ưu hóa phân bổ ngân sách Marketing theo kịch bản")

# ==========================================
# 2. SESSION STATE
# ==========================================
if "da_chay_solver" not in st.session_state:
    st.session_state.da_chay_solver = False

# ==========================================
# 3. SIDEBAR
# ==========================================
st.sidebar.header("Điều chỉnh Kịch Bản (Inputs)")

tong_ngan_sach = st.sidebar.number_input(
    "Tổng ngân sách tối đa (VNĐ)",
    min_value=1000000000,
    max_value=5000000000,
    value=2500000000,
    step=100000000
)

st.sidebar.subheader("Giới hạn chi tiêu (Tránh bão hòa)")
max_fb = st.sidebar.number_input("Trần Facebook Ads", value=1000000000, step=100000000)
max_gg = st.sidebar.number_input("Trần Google Ads", value=800000000, step=100000000)
max_em = st.sidebar.number_input("Trần Email Marketing", value=1500000000, step=100000000)

st.sidebar.markdown("---")

if st.sidebar.button("Chạy Mô Hình Tối Ưu"):
    st.session_state.da_chay_solver = True

# ==========================================
# 4. SOLVER
# ==========================================
if st.session_state.da_chay_solver:

    prob = pulp.LpProblem("Toi_Uu_Doanh_Thu", pulp.LpMaximize)

    x_FB = pulp.LpVariable('Facebook_Ads', lowBound=0)
    x_GG = pulp.LpVariable('Google_Ads', lowBound=0)
    x_LI = pulp.LpVariable('LinkedIn_Ads', lowBound=0)
    x_EM = pulp.LpVariable('Email_Marketing', lowBound=0)
    x_TT = pulp.LpVariable('TikTok_Ads', lowBound=0)

    # Hàm doanh thu (từ regression)
    doanh_thu_FB = -783206 + 3.897 * x_FB
    doanh_thu_GG = -363867 + 5.081 * x_GG
    doanh_thu_LI = -3349816 + 5.602 * x_LI
    doanh_thu_EM = -1763155 + 7.153 * x_EM
    doanh_thu_TT = 2115162 - 0.589 * x_TT

    prob += (doanh_thu_FB + doanh_thu_GG + doanh_thu_LI + doanh_thu_EM + doanh_thu_TT)

    # Constraints
    prob += (x_FB + x_GG + x_LI + x_EM + x_TT <= tong_ngan_sach)
    prob += (x_FB <= max_fb)
    prob += (x_GG <= max_gg)
    prob += (x_EM <= max_em)

    # Min budget
    prob += (x_FB >= 150000000)
    prob += (x_GG >= 150000000)
    prob += (x_LI >= 150000000)
    prob += (x_EM >= 150000000)
    prob += (x_TT >= 150000000)

    prob.solve()

    if pulp.LpStatus[prob.status] == 'Optimal':

        df_kq = pd.DataFrame({
            "Kênh Marketing": ["Facebook Ads", "Google Ads", "LinkedIn Ads", "Email Marketing", "TikTok Ads"],
            "Ngân sách tối ưu (VNĐ)": [
                x_FB.varValue,
                x_GG.varValue,
                x_LI.varValue,
                x_EM.varValue,
                x_TT.varValue
            ]
        })

        tong_dt = pulp.value(prob.objective)

        col1, col2 = st.columns(2)
        col1.metric("Tổng Doanh Thu Kỳ Vọng", f"{tong_dt:,.0f} VNĐ")
        col2.metric("Tổng Ngân Sách", f"{df_kq['Ngân sách tối ưu (VNĐ)'].sum():,.0f} VNĐ")

        st.markdown("---")

        col_chart, col_table = st.columns([1.5, 1])

        with col_chart:
            fig = px.pie(df_kq,
                         values='Ngân sách tối ưu (VNĐ)',
                         names='Kênh Marketing',
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.dataframe(df_kq.style.format({"Ngân sách tối ưu (VNĐ)": "{:,.0f}"}))

        # ==========================================
        # 5. GEMINI AI
        # ==========================================
        st.markdown("---")
        st.markdown("###  Trợ lý AI Tư Vấn Chiến Lược")

        api_key_input = st.text_input("Nhập Gemini API Key:", type="password")

        if st.button("Phân tích bằng AI"):

            if not api_key_input:
                st.warning(" Vui lòng nhập API key!")
            else:
                with st.spinner("AI đang phân tích..."):

                    try:
                        genai.configure(api_key=api_key_input)

                        model = genai.GenerativeModel("gemini-1.5-flash")

                        prompt = f"""
                        Ngân sách tổng: {tong_ngan_sach:,.0f} VNĐ

                        Facebook: {x_FB.varValue:,.0f}
                        Google: {x_GG.varValue:,.0f}
                        LinkedIn: {x_LI.varValue:,.0f}
                        Email: {x_EM.varValue:,.0f}
                        TikTok: {x_TT.varValue:,.0f}

                        Doanh thu: {tong_dt:,.0f}

                        Hãy:
                        1. Đánh giá phương án
                        2. Nêu rủi ro lớn nhất
                        3. Đề xuất cải thiện

                        Ngắn gọn, chuyên nghiệp.
                        """

                        response = model.generate_content(prompt)

                        if response and hasattr(response, "text"):
                            st.success(" AI đã phân tích:")
                            st.write(response.text)
                        else:
                            st.warning("AI không trả kết quả.")

                    except Exception as e:
                        st.error(f" Lỗi: {str(e)}")

    else:
        st.error(" Không tìm được nghiệm tối ưu")
