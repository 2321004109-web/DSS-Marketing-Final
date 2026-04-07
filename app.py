import streamlit as st
import pandas as pd
import pulp
import plotly.express as px
import google.generativeai as genai

# 1. Cấu hình trang Web
st.set_page_config(page_title="DSS Marketing Optimization", layout="wide")
st.title("Hệ Hỗ Trợ Ra Quyết Định (DSS) - ABC Retail")
st.markdown("### Tối ưu hóa phân bổ ngân sách Marketing theo kịch bản")

# ==========================================
# TRANG BỊ BỘ NHỚ CHO ỨNG DỤNG
# ==========================================
if "da_chay_solver" not in st.session_state:
    st.session_state.da_chay_solver = False

# 2. Xây dựng thanh công cụ bên trái (Sidebar)
st.sidebar.header("Điều chỉnh Kịch Bản (Inputs)")

tong_ngan_sach = st.sidebar.number_input("Tổng ngân sách tối đa (VNĐ)", min_value=1000000000, max_value=5000000000, value=2500000000, step=100000000)

st.sidebar.subheader("Giới hạn chi tiêu (Để tránh bão hòa kênh)")
max_fb = st.sidebar.number_input("Trần ngân sách Facebook Ads", value=1000000000, step=100000000)
max_gg = st.sidebar.number_input("Trần ngân sách Google Ads", value=800000000, step=100000000)
max_em = st.sidebar.number_input("Trần ngân sách Email Marketing", value=1500000000, step=100000000)

st.sidebar.markdown("---")

# Khi bấm nút chạy, lưu trạng thái vào bộ nhớ
if st.sidebar.button("Chạy Mô Hình Tối Ưu"):
    st.session_state.da_chay_solver = True

# 3. NẾU BỘ NHỚ GHI NHẬN ĐÃ CHẠY, MỚI HIỂN THỊ KẾT QUẢ
if st.session_state.da_chay_solver:
    prob = pulp.LpProblem("Toi_Uu_Doanh_Thu", pulp.LpMaximize)

    x_FB = pulp.LpVariable('Facebook_Ads', lowBound=0, cat='Continuous')
    x_GG = pulp.LpVariable('Google_Ads', lowBound=0, cat='Continuous')
    x_LI = pulp.LpVariable('LinkedIn_Ads', lowBound=0, cat='Continuous')
    x_EM = pulp.LpVariable('Email_Marketing', lowBound=0, cat='Continuous')
    x_TT = pulp.LpVariable('TikTok_Ads', lowBound=0, cat='Continuous')

    doanh_thu_FB = -783206 + 3.897 * x_FB
    doanh_thu_GG = -363867 + 5.081 * x_GG
    doanh_thu_LI = -3349816 + 5.602 * x_LI
    doanh_thu_EM = -1763155 + 7.153 * x_EM
    doanh_thu_TT = 2115162 - 0.589 * x_TT  
    
    prob += (doanh_thu_FB + doanh_thu_GG + doanh_thu_LI + doanh_thu_EM + doanh_thu_TT), "Tong_Doanh_Thu"

    prob += (x_FB + x_GG + x_LI + x_EM + x_TT <= tong_ngan_sach), "Tong_Ngan_Sach"
    prob += (x_FB <= max_fb), "Max_FB"
    prob += (x_GG <= max_gg), "Max_GG"
    prob += (x_EM <= max_em), "Max_EM"
    
    prob += (x_FB >= 150000000)
    prob += (x_GG >= 150000000)
    prob += (x_LI >= 150000000)
    prob += (x_EM >= 150000000)
    prob += (x_TT >= 150000000)

    prob.solve()

    if pulp.LpStatus[prob.status] == 'Optimal':
        ket_qua = {
            "Kênh Marketing": ["Facebook Ads", "Google Ads", "LinkedIn Ads", "Email Marketing", "TikTok Ads"],
            "Ngân sách tối ưu (VNĐ)": [x_FB.varValue, x_GG.varValue, x_LI.varValue, x_EM.varValue, x_TT.varValue]
        }
        df_kq = pd.DataFrame(ket_qua)
        tong_dt = pulp.value(prob.objective)
        
        col1, col2 = st.columns(2)
        col1.metric("Tổng Doanh Thu Kỳ Vọng", f"{tong_dt:,.0f} VNĐ")
        col2.metric("Tổng Ngân Sách Cần Chi", f"{sum(df_kq['Ngân sách tối ưu (VNĐ)']):,.0f} VNĐ")
        
        st.markdown("---")
        
        col_chart, col_table = st.columns([1.5, 1])
        
        with col_chart:
            st.markdown("#### Biểu đồ tỷ trọng phân bổ ngân sách")
            fig = px.pie(df_kq, values='Ngân sách tối ưu (VNĐ)', names='Kênh Marketing', hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_table:
            st.markdown("#### Bảng chi tiết")
            st.dataframe(df_kq.style.format({"Ngân sách tối ưu (VNĐ)": "{:,.0f}"}), height=250)
            
        # ==========================================
        # PHẦN TÍCH HỢP TRỢ LÝ AI (GEMINI)
        # ==========================================
        st.markdown("---")
        st.markdown("###  Trợ lý AI Tư Vấn Chiến Lược (Gemini)")
        st.info("Trợ lý ảo AI sẽ tự động đọc kết quả chạy Solver ở trên và đưa ra lời khuyên cho Giám đốc Marketing.")
        
        api_key_input = st.text_input("Nhập Google Gemini API Key để kích hoạt:", type="password")
        
        if st.button("Phân tích ngay bằng AI") and api_key_input:
            with st.spinner("AI đang đọc số liệu và suy nghĩ... "):
                try:
                    genai.configure(api_key="AIzaSyAq-j5q_K74ihr5JSl3rVN5rW2HAoclR1E")
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    prompt = f"""
                    Tôi là Giám đốc Marketing. Hệ thống DSS vừa phân bổ tổng ngân sách {tong_ngan_sach:,.0f} VNĐ cho các kênh như sau:
                    - Facebook Ads: {x_FB.varValue:,.0f} VNĐ
                    - Google Ads: {x_GG.varValue:,.0f} VNĐ
                    - LinkedIn Ads: {x_LI.varValue:,.0f} VNĐ
                    - Email Marketing: {x_EM.varValue:,.0f} VNĐ
                    - TikTok Ads: {x_TT.varValue:,.0f} VNĐ
                    Tổng doanh thu kỳ vọng mang lại là: {tong_dt:,.0f} VNĐ.
                    
                    Dựa vào số liệu này, hãy viết cho tôi:
                    1. Đánh giá ngắn gọn về phương án chia tiền này (tốt/xấu ở đâu).
                    2. Chỉ ra 1 rủi ro lớn nhất nếu tôi làm theo kịch bản này.
                    3. Đề xuất 1 hành động tôi nên làm để tối ưu hơn.
                    Trình bày chuyên nghiệp, ngắn gọn bằng tiếng Việt.
                    """
                    
                    response = model.generate_content(prompt)
                    st.success(" Trợ lý AI đã phân tích xong:")
                    st.write(response.text)
                    
                except Exception as e:
                    st.error(" Có lỗi xảy ra! Hãy kiểm tra lại API Key xem copy đúng chưa nhé.")
                    
    else:
        st.error(" Không tìm thấy phương án tối ưu. Vui lòng nới lỏng các ràng buộc bên trái.")
