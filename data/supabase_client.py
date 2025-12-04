"""
Supabase Client for Dashboard
=============================

Streamlit Cloud에서 Supabase 조회/저장 클라이언트
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import uuid

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[Warning] supabase-py not installed. Using mock data fallback.")


def get_supabase_client() -> Optional['Client']:
    """Supabase 클라이언트 생성"""
    if not SUPABASE_AVAILABLE:
        return None

    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.warning(f"Supabase 연결 실패: {e}")
        return None


@st.cache_data(ttl=300)
def get_predictions_from_supabase(
    store_cd: str,
    prediction_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """predictions_display에서 예측 데이터 조회

    Parameters
    ----------
    store_cd : str
        점포 코드
    prediction_date : Optional[str]
        예측 기준일 (없으면 최신 날짜)

    Returns
    -------
    Optional[pd.DataFrame]
        예측 데이터 DataFrame, 실패 시 None
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        query = client.table("predictions_display").select("*").eq("store_cd", store_cd)

        if prediction_date:
            query = query.eq("prediction_date", prediction_date)
        else:
            # 최신 날짜 조회
            query = query.order("prediction_date", desc=True)

        response = query.execute()

        if response.data:
            df = pd.DataFrame(response.data)
            return df
        return None

    except Exception as e:
        st.warning(f"예측 데이터 조회 실패: {e}")
        return None


def get_context_from_supabase(
    store_cd: str,
    sku_code: str,
    prediction_date: str,
    horizon: str
) -> Optional[Dict]:
    """predictions_context에서 챗봇 context 조회

    Parameters
    ----------
    store_cd : str
        점포 코드
    sku_code : str
        단품 코드
    prediction_date : str
        예측 기준일
    horizon : str
        예측 horizon (t+1, t+2, etc.)

    Returns
    -------
    Optional[Dict]
        Context 데이터, 실패 시 None
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        response = client.table("predictions_context") \
            .select("*") \
            .eq("store_cd", store_cd) \
            .eq("sku_code", sku_code) \
            .eq("prediction_date", prediction_date) \
            .eq("horizon", horizon) \
            .single() \
            .execute()

        if response.data:
            return response.data
        return None

    except Exception as e:
        # single()은 없으면 에러, 조용히 처리
        return None


def get_latest_context_for_sku(
    store_cd: str,
    sku_code: str,
    horizon: str
) -> Optional[Dict]:
    """SKU의 최신 context 조회

    Parameters
    ----------
    store_cd : str
        점포 코드
    sku_code : str
        단품 코드
    horizon : str
        예측 horizon

    Returns
    -------
    Optional[Dict]
        최신 Context 데이터
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        response = client.table("predictions_context") \
            .select("*") \
            .eq("store_cd", store_cd) \
            .eq("sku_code", sku_code) \
            .eq("horizon", horizon) \
            .order("prediction_date", desc=True) \
            .limit(1) \
            .execute()

        if response.data and len(response.data) > 0:
            return response.data[0]
        return None

    except Exception as e:
        return None


def save_chat_history(
    store_cd: str,
    sku_code: str,
    prediction_date: str,
    horizon: str,
    user_message: str,
    assistant_message: str,
    session_id: Optional[str] = None
) -> bool:
    """chat_history에 대화 저장

    Parameters
    ----------
    store_cd : str
        점포 코드
    sku_code : str
        단품 코드
    prediction_date : str
        예측 기준일
    horizon : str
        예측 horizon
    user_message : str
        사용자 질문
    assistant_message : str
        AI 응답
    session_id : Optional[str]
        세션 식별자

    Returns
    -------
    bool
        저장 성공 여부
    """
    client = get_supabase_client()
    if not client:
        return False

    try:
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        data = {
            "store_cd": store_cd,
            "sku_code": sku_code,
            "prediction_date": prediction_date,
            "horizon": horizon,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "session_id": session_id
        }

        client.table("chat_history").insert(data).execute()
        return True

    except Exception as e:
        print(f"[Error] Failed to save chat history: {e}")
        return False


def get_chat_history(
    sku_code: str,
    prediction_date: str,
    session_id: Optional[str] = None,
    limit: int = 20
) -> List[Dict]:
    """chat_history에서 대화 기록 조회

    Parameters
    ----------
    sku_code : str
        단품 코드
    prediction_date : str
        예측 기준일
    session_id : Optional[str]
        세션 식별자 (없으면 전체)
    limit : int
        최대 조회 건수

    Returns
    -------
    List[Dict]
        대화 기록 리스트
    """
    client = get_supabase_client()
    if not client:
        return []

    try:
        query = client.table("chat_history") \
            .select("*") \
            .eq("sku_code", sku_code) \
            .eq("prediction_date", prediction_date)

        if session_id:
            query = query.eq("session_id", session_id)

        response = query.order("created_at", desc=False).limit(limit).execute()

        return response.data if response.data else []

    except Exception as e:
        return []


def transform_supabase_to_display_df(
    supabase_df: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:
    """Supabase 데이터를 대시보드 표시용 DataFrame으로 변환

    Parameters
    ----------
    supabase_df : pd.DataFrame
        Supabase에서 조회한 데이터
    horizon : int
        선택한 horizon (1~4)

    Returns
    -------
    pd.DataFrame
        대시보드 표시용 DataFrame
    """
    horizon_str = f"t+{horizon}"

    # horizon 필터링
    filtered_df = supabase_df[supabase_df['horizon'] == horizon_str].copy()

    if filtered_df.empty:
        return pd.DataFrame()

    # 대시보드 포맷으로 변환
    rows = []
    for idx, row in filtered_df.iterrows():
        # top_features에서 Top 3 정렬하여 주요 영향 변수 생성
        short_reason = "-"
        if row.get('top_features'):
            top_features = row['top_features']
            if isinstance(top_features, dict) and top_features:
                # 절댓값 기준 정렬하여 Top 3 추출
                sorted_features = sorted(
                    top_features.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:3]
                top_3_names = [feat for feat, _ in sorted_features]
                short_reason = ", ".join(top_3_names)

        rows.append({
            '순번': len(rows) + 1,
            '단품코드': row['sku_code'],
            '단품명': row.get('sku_name', row['sku_code']),
            '단위': 'EA',
            '의뢰수량': 0,
            '예측값': int(row['predicted_value']),
            '예측값_min': int(row['pred_min']),
            '예측값_max': int(row['pred_max']),
            '예측모델': row.get('model_name', 'Unknown'),
            '예측설명': short_reason,
            '상세리포트': f"### {row.get('sku_name', row['sku_code'])} 상세 리포트\n\n"
                         f"**예측 모델**: {row.get('model_name', 'Unknown')}\n\n"
                         f"**예측값**: {int(row['predicted_value'])}개 "
                         f"({int(row['pred_min'])} ~ {int(row['pred_max'])})",
            '비고': ''
        })

    return pd.DataFrame(rows)
