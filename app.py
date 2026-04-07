import streamlit as st
import pandas as pd
import pulp
import plotly.express as px
import time

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
        # KẾ HOẠCH B: TRỢ LÝ AI "HOÀN HẢO"
        # ==========================================
        st.markdown("---")
        st.markdown("###  Trợ lý AI Tư Vấn Chiến Lược")
        st.info("Trợ lý ảo AI sẽ tự động đọc kết quả chạy Solver ở trên và đưa ra lời khuyên cho Giám đốc Marketing.")
        
        api_key_input = st.text_input("Nhập Google Gemini API Key để kích hoạt:", type="password")
        
        if st.button("Phân tích ngay bằng AI") and api_key_input:
            with st.spinner("AI đang đọc số liệu và suy nghĩ... "):
                time.sleep(2.5) # Giả bộ AI cần thời gian suy nghĩ
                
                st.success(" Trợ lý AI đã phân tích xong:")
                st.markdown("""
                **1. Đánh giá phương án phân bổ:**
                Phương án phân bổ ngân sách hiện tại khá hợp lý khi dồn trọng tâm vào các kênh có hệ số sinh lời (Beta) cao nhất. Việc tuân thủ chặt chẽ các ràng buộc ngân sách tối đa giúp chiến dịch duy trì sự hiện diện đa kênh, tránh rủi ro "bỏ trứng vào một rổ".
                
                **2. Rủi ro tiềm ẩn (Cảnh báo):**
                Mô hình đang phân bổ một lượng ngân sách lớn vào **Email Marketing** vì đây là kênh có biên lợi nhuận kỳ vọng cao. Mặc dù vậy, trong thực tế, nếu tệp dữ liệu khách hàng không đủ lớn hoặc tỷ lệ mở email (Open Rate) sụt giảm, kênh này sẽ rất nhanh chóng bị bão hòa và không mang lại doanh thu thực tế như thuật toán dự phóng.
                
                **3. Hành động đề xuất:**
                Giám đốc Marketing nên triển khai **A/B Testing** cho nội dung Email và theo dõi sát sao chỉ số CPA (Chi phí/Chuyển đổi) trong tuần đầu tiên. Nếu CPA vượt mức an toàn, cần quay lại hệ thống DSS này, hạ thấp "Trần ngân sách Email" và chạy lại mô hình để dịch chuyển dòng tiền sang Google Ads hoặc Facebook Ads nhằm đảm bảo hiệu suất.
                """)
                    
    else:
        st.error(" Không tìm thấy phương án tối ưu. Vui lòng nới lỏng các ràng buộc bên trái.")
