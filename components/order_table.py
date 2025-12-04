"""
Order Table Component
=====================
발주의뢰 테이블 (수요 예측 + LLM 챗봇)
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import uuid
from pathlib import Path

# 상위 디렉토리 import
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.chatbot import get_chatbot
from data.supabase_client import (
    get_context_from_supabase,
    save_chat_history
)
import config


def render_order_table(df: pd.DataFrame, horizon: int, prediction_date: str = None) -> pd.DataFrame:
    """
    발주의뢰 테이블 렌더링

    Parameters
    ----------
    df : pd.DataFrame
        예측 데이터가 포함된 발주 목록
    horizon : int
        예측 horizon (1~4)
    prediction_date : str
        예측 기준일 (YYYY-MM-DD 형식)

    Returns
    -------
    pd.DataFrame
        수정된 발주 목록 (의뢰수량 포함)
    """
    st.markdown(f"### 발주의뢰 목록 ({len(df)}건) - t+{horizon} 예측")

    # session_state 초기화
    if 'expanded_rows' not in st.session_state:
        st.session_state.expanded_rows = set()
    if 'chat_rows' not in st.session_state:
        st.session_state.chat_rows = set()
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = {}
    if 'order_quantities' not in st.session_state:
        st.session_state.order_quantities = {row['단품코드']: 0 for _, row in df.iterrows()}

    # 테이블 헤더
    header_cols = st.columns([0.5, 1, 2, 0.7, 1, 1.2, 1.2, 1.2, 2, 0.8, 0.8, 1.5])
    headers = ['순번', '단품코드', '단품명', '단위', '의뢰수량',
               '예측값', '예측_min', '예측_max', '주요 영향 변수', '상세', '챗봇', '비고']

    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")

    st.markdown("---")

    # 각 행 렌더링
    updated_df = df.copy()

    for idx, row in df.iterrows():
        sku_code = row['단품코드']

        # 메인 행
        cols = st.columns([0.5, 1, 2, 0.7, 1, 1.2, 1.2, 1.2, 2, 0.8, 0.8, 1.5])

        # 순번
        cols[0].write(row['순번'])

        # 단품코드
        cols[1].write(sku_code)

        # 단품명
        cols[2].write(row['단품명'])

        # 단위
        cols[3].write(row['단위'])

        # 의뢰수량 (editable)
        order_qty = cols[4].number_input(
            label=f"qty_{sku_code}",
            label_visibility="collapsed",
            min_value=0,
            max_value=9999,
            value=st.session_state.order_quantities.get(sku_code, 0),
            step=1,
            key=f"order_qty_{sku_code}"
        )
        st.session_state.order_quantities[sku_code] = order_qty
        updated_df.at[idx, '의뢰수량'] = order_qty

        # 예측값 (강조)
        cols[5].markdown(f"**:blue[{row['예측값']}]**")

        # 예측_min
        cols[6].write(row['예측값_min'])

        # 예측_max
        cols[7].write(row['예측값_max'])

        # 주요 영향 변수 (Top 3)
        cols[8].write(row['예측설명'])

        # 상세 리포트 토글
        detail_btn = cols[9].button("📊", key=f"detail_{sku_code}", help="상세 리포트 보기")
        if detail_btn:
            if sku_code in st.session_state.expanded_rows:
                st.session_state.expanded_rows.remove(sku_code)
            else:
                st.session_state.expanded_rows.add(sku_code)

        # 챗봇 토글
        chat_btn = cols[10].button("💬", key=f"chat_{sku_code}", help="AI 챗봇 열기")
        if chat_btn:
            if sku_code in st.session_state.chat_rows:
                st.session_state.chat_rows.remove(sku_code)
            else:
                st.session_state.chat_rows.add(sku_code)
                # 채팅 기록 초기화
                if sku_code not in st.session_state.chat_messages:
                    st.session_state.chat_messages[sku_code] = []

        # 비고
        note = cols[11].text_input(
            label=f"note_{sku_code}",
            label_visibility="collapsed",
            value=row['비고'],
            key=f"note_{sku_code}",
            placeholder="메모..."
        )
        updated_df.at[idx, '비고'] = note

        # 상세 리포트 확장 영역
        if sku_code in st.session_state.expanded_rows:
            with st.container():
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                """, unsafe_allow_html=True)

                st.markdown(f"#### 📊 {row['단품명']} 상세 리포트")
                st.markdown(f"**예측 모델**: {row['예측모델']}")
                st.markdown(row['상세리포트'])

                st.markdown("</div>", unsafe_allow_html=True)

        # 챗봇 확장 영역
        if sku_code in st.session_state.chat_rows:
            render_chat_interface(sku_code, row['단품명'], horizon, prediction_date)

        st.markdown("---")

    # 하단 집계
    render_footer(updated_df)

    return updated_df


def render_chat_interface(sku_code: str, sku_name: str, horizon: int = 1, prediction_date: str = None):
    """
    SKU별 챗봇 인터페이스 렌더링

    Parameters
    ----------
    sku_code : str
        단품코드
    sku_name : str
        단품명
    horizon : int
        예측 horizon (1~4)
    prediction_date : str
        예측 기준일 (YYYY-MM-DD 형식)
    """
    # 세션 ID 초기화
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    # Context 조회 (Supabase) - 선택된 prediction_date 기준
    context = None
    if config.USE_SUPABASE and prediction_date:
        try:
            context = get_context_from_supabase(
                store_cd='210',
                sku_code=sku_code,
                prediction_date=prediction_date,
                horizon=f't+{horizon}'
            )
        except Exception as e:
            pass

    # Context가 없으면 기본값
    if not context:
        context = {
            'sku_code': sku_code,
            'sku_name': sku_name,
            'horizon': f't+{horizon}',
            'predicted_value': 'N/A',
            'pred_min': 'N/A',
            'pred_max': 'N/A',
            'model_name': 'Unknown'
        }

    # 챗봇 인스턴스
    chatbot = get_chatbot()

    with st.container():
        st.markdown(f"""
        <div style="background-color: #e8f4ea; padding: 15px; border-radius: 10px; margin: 10px 0;">
        """, unsafe_allow_html=True)

        st.markdown(f"#### 💬 {sku_name} AI 어시스턴트")
        st.caption("수요 예측에 대해 질문하거나, 발주량 조정 시나리오를 물어보세요.")
        st.caption("💡 대화는 서비스 개선을 위해 저장될 수 있습니다.")

        # 채팅 기록 표시
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_messages.get(sku_code, []):
                if msg['role'] == 'user':
                    st.markdown(f"**🧑 나**: {msg['content']}")
                else:
                    st.markdown(f"**🤖 AI**: {msg['content']}")

        # 입력 영역
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                label=f"chat_input_{sku_code}",
                label_visibility="collapsed",
                placeholder="질문을 입력하세요...",
                key=f"chat_input_{sku_code}"
            )

        with col2:
            send_btn = st.button("전송", key=f"send_{sku_code}")

        if send_btn and user_input:
            # 사용자 메시지 추가
            if sku_code not in st.session_state.chat_messages:
                st.session_state.chat_messages[sku_code] = []

            st.session_state.chat_messages[sku_code].append({
                'role': 'user',
                'content': user_input
            })

            # AI 응답 (실제 챗봇 또는 폴백)
            chat_history = st.session_state.chat_messages.get(sku_code, [])
            ai_response = chatbot.get_response(
                user_message=user_input,
                context=context,
                chat_history=chat_history[:-1]  # 현재 메시지 제외
            )

            st.session_state.chat_messages[sku_code].append({
                'role': 'assistant',
                'content': ai_response
            })

            # Supabase에 대화 저장
            if config.USE_SUPABASE:
                try:
                    prediction_date = context.get('prediction_date', datetime.now().strftime('%Y-%m-%d'))
                    save_chat_history(
                        store_cd='210',
                        sku_code=sku_code,
                        prediction_date=prediction_date,
                        horizon=f't+{horizon}',
                        user_message=user_input,
                        assistant_message=ai_response,
                        session_id=st.session_state.session_id
                    )
                except Exception as e:
                    pass

            # 리렌더링
            st.rerun()

        # 예시 질문 버튼
        st.markdown("**빠른 질문:**")
        example_cols = st.columns(3)

        examples = chatbot.get_quick_suggestions()

        for i, (col, example) in enumerate(zip(example_cols, examples)):
            if col.button(example, key=f"example_{sku_code}_{i}"):
                if sku_code not in st.session_state.chat_messages:
                    st.session_state.chat_messages[sku_code] = []

                st.session_state.chat_messages[sku_code].append({
                    'role': 'user',
                    'content': example
                })

                chat_history = st.session_state.chat_messages.get(sku_code, [])
                ai_response = chatbot.get_response(
                    user_message=example,
                    context=context,
                    chat_history=chat_history[:-1]
                )

                st.session_state.chat_messages[sku_code].append({
                    'role': 'assistant',
                    'content': ai_response
                })

                # Supabase에 대화 저장
                if config.USE_SUPABASE:
                    try:
                        prediction_date = context.get('prediction_date', datetime.now().strftime('%Y-%m-%d'))
                        save_chat_history(
                            store_cd='210',
                            sku_code=sku_code,
                            prediction_date=prediction_date,
                            horizon=f't+{horizon}',
                            user_message=example,
                            assistant_message=ai_response,
                            session_id=st.session_state.session_id
                        )
                    except Exception as e:
                        pass

                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def generate_dummy_response(sku_code: str, question: str) -> str:
    """
    더미 AI 응답 생성 (나중에 실제 LLM으로 교체)

    Parameters
    ----------
    sku_code : str
        단품코드
    question : str
        사용자 질문

    Returns
    -------
    str
        AI 응답
    """
    # 더미 응답 (실제로는 LLM API 호출)
    responses = {
        "예측 근거가 뭐야?": f"[{sku_code}] 예측은 최근 7일 판매 트렌드, 요일 효과, 날씨 예보, 시즌성을 종합 분석한 결과입니다. 특히 이번 주는 주말 효과로 평일 대비 15~20% 상승이 예상됩니다.",
        "공격적 발주 시 리스크는?": f"[{sku_code}] 예측값 대비 20% 이상 초과 발주 시, 재고 폐기 리스크가 약 12% 증가합니다. 신선식품 특성상 D+2 이후 품질 저하가 우려되므로, 예측 상한선(예측_max) 이내 발주를 권장합니다.",
        "작년 대비 트렌드는?": f"[{sku_code}] 전년 동기 대비 약 10~15% 판매량 증가 추세입니다. 주요 원인은 건강식품 트렌드 지속과 프리미엄 과일 선호도 상승입니다."
    }

    # 정확히 매칭되는 질문이 없으면 기본 응답
    for key, response in responses.items():
        if key in question:
            return response

    return f"[{sku_code}] 질문을 분석 중입니다. 해당 상품의 수요 예측은 XGBoost 모델 기반이며, 최근 판매 패턴과 외부 요인(날씨, 공휴일)을 반영했습니다. 더 구체적인 질문이 있으시면 말씀해주세요."


def render_footer(df: pd.DataFrame):
    """
    하단 집계 영역 렌더링

    Parameters
    ----------
    df : pd.DataFrame
        발주 목록
    """
    st.markdown("### 집계")

    col1, col2, col3, col4 = st.columns(4)

    total_items = len(df)
    total_order_qty = df['의뢰수량'].sum()
    total_pred_qty = df['예측값'].sum()
    order_vs_pred = (total_order_qty / total_pred_qty * 100) if total_pred_qty > 0 else 0

    col1.metric("의뢰건수", f"{total_items}건")
    col2.metric("의뢰수량 합계", f"{total_order_qty:,}개")
    col3.metric("예측수량 합계", f"{total_pred_qty:,}개")
    col4.metric("발주율", f"{order_vs_pred:.1f}%",
                delta=f"{total_order_qty - total_pred_qty:+,}개" if total_order_qty != total_pred_qty else None)
