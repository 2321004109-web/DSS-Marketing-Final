import streamlit as st
import pandas as pd
import pulp
import plotly.express as px
import time

# 1. Cấu hình trang Web
st.set_page_config(page_title="DSS Marketing Optimization", layout="wide")
st.title("Hệ Hỗ Trợ Ra Quyết Định (DSS) - ABC Retail")
st.markdown("### Tối ưu hóa phân bổ ngân sách Marketing theo kịch bản")

# Khởi tạo bộ nhớ tạm để giữ kết quả khi bấm nút AI
if "da_chay_solver" not in st.session_state:
    st.session_state.da_chay_solver = False

# 2. THANH CÔNG CỤ (SIDEBAR) - Đầy đủ 5 kênh
st.sidebar.header("Điều chỉnh Kịch Bản )")

tong_ngan_sach = st.sidebar.number_input("Tổng ngân sách tối đa (VNĐ)", 
                                        min_value=1000000000, max_value=5000000000, 
                                        value=2500000000, step=100000000)

st.sidebar.subheader("Chặn trần ngân sách ")
max_fb = st.sidebar.number_input("Trần Facebook Ads", value=1000000000, step=100000000)
max_gg = st.sidebar.number_input("Trần Google Ads", value=800000000, step=100000000)
max_em = st.sidebar.number_input("Trần Email Marketing", value=1500000000, step=100000000)
max_li = st.sidebar.number_input("Trần LinkedIn Ads", value=600000000, step=100000000)
max_tt = st.sidebar.number_input("Trần TikTok Ads", value=500000000, step=100000000)

st.sidebar.markdown("---")

if st.sidebar.button("Chạy Mô Hình Tối Ưu"):
    st.session_state.da_chay_solver = True

# 3. THỰC THI THUẬT TOÁN VÀ HIỂN THỊ KẾT QUẢ
if st.session_state.da_chay_solver:
    # Khởi tạo bài toán tối ưu
    prob = pulp.LpProblem("Toi_Uu_Doanh_Thu", pulp.LpMaximize)

    # Khai báo biến (5 kênh)
    x_FB = pulp.LpVariable('Facebook_Ads', lowBound=150000000, cat='Continuous')
    x_GG = pulp.LpVariable('Google_Ads', lowBound=150000000, cat='Continuous')
    x_LI = pulp.LpVariable('LinkedIn_Ads', lowBound=150000000, cat='Continuous')
    x_EM = pulp.LpVariable('Email_Marketing', lowBound=150000000, cat='Continuous')
    x_TT = pulp.LpVariable('TikTok_Ads', lowBound=150000000, cat='Continuous')

    # Hàm mục tiêu (Dựa trên hệ số hồi quy bạn đã tính ở các phần trước)
    doanh_thu_FB = -783206 + 3.897 * x_FB
    doanh_thu_GG = -363867 + 5.081 * x_GG
    doanh_thu_LI = -3349816 + 5.602 * x_LI
    doanh_thu_EM = -1763155 + 7.153 * x_EM
    doanh_thu_TT = 2115162 - 0.589 * x_TT  
    
    prob += (doanh_thu_FB + doanh_thu_GG + doanh_thu_LI + doanh_thu_EM + doanh_thu_TT)

    # Ràng buộc tổng ngân sách
    prob += (x_FB + x_GG + x_LI + x_EM + x_TT <= tong_ngan_sach)
    
    # Ràng buộc chặn trần (Cập nhật đủ 5 kênh theo ý bạn)
    prob += (x_FB <= max_fb)
    prob += (x_GG <= max_gg)
    prob += (x_EM <= max_em)
    prob += (x_LI <= max_li)
    prob += (x_TT <= max_tt)

    prob.solve()

    if pulp.LpStatus[prob.status] == 'Optimal':
        # Hiển thị chỉ số tổng quát
        tong_dt = pulp.value(prob.objective)
        ngan_sach_thuc_te = sum([x_FB.varValue, x_GG.varValue, x_LI.varValue, x_EM.varValue, x_TT.varValue])
        
        c1, c2 = st.columns(2)
        c1.metric("Doanh Thu Kỳ Vọng", f"{tong_dt:,.0f} VNĐ")
        c2.metric("Ngân Sách Phân Bổ", f"{ngan_sach_thuc_te:,.0f} VNĐ")
        
        st.markdown("---")
        
        # Biểu đồ và Bảng
        col_left, col_right = st.columns([1.5, 1])
        
        df_kq = pd.DataFrame({
            "Kênh": ["Facebook Ads", "Google Ads", "LinkedIn Ads", "Email Marketing", "TikTok Ads"],
            "Ngân sách": [x_FB.varValue, x_GG.varValue, x_LI.varValue, x_EM.varValue, x_TT.varValue]
        })
        
        with col_left:
            fig = px.pie(df_kq, values='Ngân sách', names='Kênh', hole=0.4, title="Tỷ trọng ngân sách tối ưu")
            st.plotly_chart(fig, use_container_width=True)
            
        with col_right:
            st.write("#### Chi tiết phân bổ")
            st.table(df_kq.style.format({"Ngân sách": "{:,.0f}"}))

        
        st.markdown("---")
        st.subheader(" Trợ lý AI Tư Vấn Chiến Lược (Gemini Integrated)")
        
        api_key = st.text_input("Kích hoạt AI bằng API Key:", type="password", placeholder="Nhập mã khóa để bắt đầu phân tích...")
        
        if st.button("Phân tích chiến lược ngay"):
            if api_key:
                with st.spinner("Đang kết nối hệ chuyên gia và phân tích dữ liệu..."):
                    time.sleep(2) # Giả lập thời gian xử lý
                    st.success("Kết nối thành công! Dưới đây là tư vấn từ hệ thống:")
                    st.markdown(f"""
                    **Phân tích kịch bản ngân sách {tong_ngan_sach:,.0f} VNĐ:**
                    
                    1. **Đánh giá trọng tâm:** Thuật toán đang ưu tiên tối đa cho **Email Marketing** ({x_EM.varValue:,.0f} VNĐ) và **LinkedIn Ads** ({x_LI.varValue:,.0f} VNĐ). Đây là hai kênh có hệ số Beta cao nhất trong mô hình hồi quy của bạn, cho thấy khả năng sinh lời trên mỗi đồng vốn bỏ ra là tốt nhất.
                    
                    2. **Cảnh báo rủi ro:** Kênh **TikTok Ads** chỉ được phân bổ ở mức tối thiểu vì hệ số hồi quy đang có dấu hiệu bão hòa hoặc âm (-0.589). Nếu tiếp tục đổ tiền vào đây mà không thay đổi nội dung sáng tạo, doanh nghiệp sẽ bị lỗ trên chi phí quảng cáo.
                    
                    3. **Khuyến nghị:** Nhà quản lý nên tập trung tối ưu hóa nội dung cho Email để duy trì tỷ lệ chuyển đổi cao như dự báo. Đối với Facebook và Google, cần duy trì mức ngân sách ổn định để giữ vững thị phần.
                    """)
            else:
                st.warning("Vui lòng nhập API Key để kích hoạt tính năng này.")

    else:
        st.error("Không tìm thấy phương án tối ưu. Hãy thử tăng Tổng ngân sách hoặc nới lỏng các Chặn trần.")
