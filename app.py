"""
FRUITY Dashboard - 발주의뢰 등록
================================
현대백화점 청과 수요 예측 기반 발주 지원 시스템

실행 방법:
    cd dashboard
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# 현재 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from components.header import render_header
from components.order_table import render_order_table
from data.mock_data import get_predictions_df
from data.supabase_client import (
    get_predictions_from_supabase,
    transform_supabase_to_display_df
)
import config

# 페이지 설정
st.set_page_config(
    page_title="FRUITY - 발주의뢰 등록",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일
st.markdown("""
<style>
    /* 전체 폰트 */
    .main {
        font-family: 'Malgun Gothic', sans-serif;
    }

    /* 헤더 스타일 */
    h2 {
        color: #1f4e79;
        border-bottom: 2px solid #1f4e79;
        padding-bottom: 10px;
    }

    /* 테이블 헤더 */
    .stMarkdown strong {
        color: #333;
    }

    /* 메트릭 카드 */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 10px;
        border-radius: 5px;
    }

    /* 버튼 스타일 */
    .stButton > button {
        border-radius: 5px;
    }

    /* 입력 필드 */
    .stNumberInput > div > div > input {
        text-align: center;
    }

    /* 확장 영역 */
    .report-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .chat-container {
        background-color: #e8f4ea;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    /* 하단 집계 */
    .footer-metrics {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """메인 앱 실행"""

    # 헤더 렌더링
    filters = render_header()

    # 예측 데이터 로드
    df = None

    if config.USE_SUPABASE:
        try:
            # Supabase에서 조회
            supabase_df = get_predictions_from_supabase(
                store_cd=filters['store'],
                prediction_date=filters['base_date'].strftime('%Y-%m-%d')
            )

            if supabase_df is not None and not supabase_df.empty:
                df = transform_supabase_to_display_df(supabase_df, filters['horizon'])

        except Exception as e:
            st.warning(f"Supabase 연결 실패: {e}")

    # Supabase 실패 또는 데이터 없음 → Mock 데이터 사용
    if df is None or df.empty:
        df = get_predictions_df(
            base_date=filters['base_date'],
            order_date=filters['order_date'],
            store_id=filters['store']
        )

    # 테이블 렌더링
    prediction_date_str = filters['base_date'].strftime('%Y-%m-%d')
    updated_df = render_order_table(df, filters['horizon'], prediction_date_str)

    # 저장 버튼 (하단)
    st.markdown("---")
    col1, col2, col3 = st.columns([3, 1, 1])

    with col2:
        if st.button("📥 임시저장", use_container_width=True):
            st.success("임시저장 완료!")

    with col3:
        if st.button("✅ 발주확정", type="primary", use_container_width=True):
            # 의뢰수량이 0인 항목 체크
            zero_items = updated_df[updated_df['의뢰수량'] == 0]
            if len(zero_items) > 0:
                st.warning(f"의뢰수량이 0인 항목이 {len(zero_items)}건 있습니다.")
            else:
                st.success("발주가 확정되었습니다!")
                st.balloons()


if __name__ == "__main__":
    main()
