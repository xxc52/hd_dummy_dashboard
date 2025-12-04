"""
AI Chatbot Module
=================

OpenAI GPT API를 사용한 예측값 Q&A 챗봇
"""

import streamlit as st
from typing import Dict, List, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Horizon별 Feature 정보 (규칙 기반 하드코딩)
COMMON_FEATURES = [
    'rolling_mean_6', 'rolling_std_6', 'rolling_max_6',
    'rolling_mean_27', 'rolling_std_27', 'rolling_max_27',
    'AVG_SELL_UPRC_t-1', 'MAX_SELL_UPRC_t-1', 'MIN_SELL_UPRC_t-1',
    'PCST_AMT_t-1', 'PRCH_QTY_t-1',
    'RMRGN_RATE_t-1', 'RMRGN_t-1',
    'SPR_AMT_t-1', 'TR_CNT_t-1', 'TSALE_AMT_t-1',
    'lag_1', 'lag_2'
]

HORIZON_FEATURES = {
    't+1': {
        'lag': ['lag_6', 'lag_13', 'lag_20', 'lag_27'],
        'weather': ['TEMP_AVG_t+1', 'HM_AVG_t+1', 'WIND_AVG_t+1', 'RN_DAY_t+1'],
        'calendar': ['is_hd_holiday_t+1', 'is_weekend_t+1',
                     'month_sin_t+1', 'month_cos_t+1',
                     'day_sin_t+1', 'day_cos_t+1',
                     'weekday_sin_t+1', 'weekday_cos_t+1'],
        'holiday': ['holiday_korea_t-1_t+1', 'holiday_korea_t_t+1',
                    'holiday_korea_t+1_t+1', 'holiday_christ_t-1_t+1',
                    'holiday_christ_t_t+1', 'holiday_newyear_t-1_t+1',
                    'holiday_newyear_t_t+1', 'holiday_etc_t+1']
    },
    't+2': {
        'lag': ['lag_5', 'lag_12', 'lag_19', 'lag_26'],
        'weather': ['TEMP_AVG_t+2', 'HM_AVG_t+2', 'WIND_AVG_t+2', 'RN_DAY_t+2'],
        'calendar': ['is_hd_holiday_t+2', 'is_weekend_t+2',
                     'month_sin_t+2', 'month_cos_t+2',
                     'day_sin_t+2', 'day_cos_t+2',
                     'weekday_sin_t+2', 'weekday_cos_t+2'],
        'holiday': ['holiday_korea_t-1_t+2', 'holiday_korea_t_t+2',
                    'holiday_korea_t+1_t+2', 'holiday_christ_t-1_t+2',
                    'holiday_christ_t_t+2', 'holiday_newyear_t-1_t+2',
                    'holiday_newyear_t_t+2', 'holiday_etc_t+2']
    },
    't+3': {
        'lag': ['lag_4', 'lag_11', 'lag_18', 'lag_25'],
        'weather': ['TEMP_AVG_t+3', 'HM_AVG_t+3', 'WIND_AVG_t+3', 'RN_DAY_t+3'],
        'calendar': ['is_hd_holiday_t+3', 'is_weekend_t+3',
                     'month_sin_t+3', 'month_cos_t+3',
                     'day_sin_t+3', 'day_cos_t+3',
                     'weekday_sin_t+3', 'weekday_cos_t+3'],
        'holiday': ['holiday_korea_t-1_t+3', 'holiday_korea_t_t+3',
                    'holiday_korea_t+1_t+3', 'holiday_christ_t-1_t+3',
                    'holiday_christ_t_t+3', 'holiday_newyear_t-1_t+3',
                    'holiday_newyear_t_t+3', 'holiday_etc_t+3']
    },
    't+4': {
        'lag': ['lag_3', 'lag_10', 'lag_17', 'lag_24'],
        'weather': ['TEMP_AVG_t+4', 'HM_AVG_t+4', 'WIND_AVG_t+4', 'RN_DAY_t+4'],
        'calendar': ['is_hd_holiday_t+4', 'is_weekend_t+4',
                     'month_sin_t+4', 'month_cos_t+4',
                     'day_sin_t+4', 'day_cos_t+4',
                     'weekday_sin_t+4', 'weekday_cos_t+4'],
        'holiday': ['holiday_korea_t-1_t+4', 'holiday_korea_t_t+4',
                    'holiday_korea_t+1_t+4', 'holiday_christ_t-1_t+4',
                    'holiday_christ_t_t+4', 'holiday_newyear_t-1_t+4',
                    'holiday_newyear_t_t+4', 'holiday_etc_t+4']
    }
}


# System Prompt
SYSTEM_PROMPT = """당신은 현대백화점 청과 수요 예측 시스템의 AI 어시스턴트입니다.
발주 담당자의 질문에 친절하고 전문적으로 답변합니다.

역할:
- 예측값의 근거 설명
- 발주량 조정 시나리오 분석
- 시장 트렌드 및 시즌성 정보 제공
- 모델 성능 및 신뢰구간 설명

답변 스타일:
- 간결하고 명확하게 (3-5문장)
- 숫자와 근거를 함께 제시
- 발주 담당자 관점에서 실용적인 조언
- 한국어로 답변

중요:
- SHAP 값(주요 영향 요인)을 언급할 때는 반드시 절댓값 기준 내림차순으로 1위, 2위, 3위 순서를 정확히 지켜서 답변하세요.
- 제공된 SHAP Top 10 리스트의 순서가 정렬되어 있지 않을 수 있으니, 값의 크기를 직접 비교하여 순위를 정하세요.
"""

# Feature 한글 매핑
FEATURE_DESCRIPTIONS = {
    'rolling_mean_6': '최근 6일 평균 판매량',
    'rolling_std_6': '최근 6일 판매량 표준편차',
    'rolling_max_6': '최근 6일 최대 판매량',
    'rolling_mean_27': '최근 27일 평균 판매량',
    'rolling_std_27': '최근 27일 판매량 표준편차',
    'rolling_max_27': '최근 27일 최대 판매량',
    'lag_1': '1일 전 판매량',
    'lag_2': '2일 전 판매량',
    'lag_6': '6일 전 판매량 (1주 전)',
    'lag_13': '13일 전 판매량 (2주 전)',
    'lag_20': '20일 전 판매량 (3주 전)',
    'lag_27': '27일 전 판매량 (4주 전)',
    'is_weekend_t+1': '주말 여부',
    'is_weekend_t+2': '주말 여부',
    'is_weekend_t+3': '주말 여부',
    'is_weekend_t+4': '주말 여부',
    'is_hd_holiday_t+1': '현대백화점 휴무일 여부',
    'TEMP_AVG_t+1': '예보 평균 기온',
    'TEMP_AVG_t+2': '예보 평균 기온',
    'RN_DAY_t+1': '예보 강수 여부',
    'RN_DAY_t+2': '예보 강수 여부',
    'HM_AVG_t+1': '예보 평균 습도',
    'WIND_AVG_t+1': '예보 평균 풍속',
    'AVG_SELL_UPRC_t-1': '전일 평균 판매단가',
    'MAX_SELL_UPRC_t-1': '전일 최대 판매단가',
    'MIN_SELL_UPRC_t-1': '전일 최소 판매단가',
    'TSALE_AMT_t-1': '전일 총 매출액',
    'TR_CNT_t-1': '전일 거래건수',
    'PCST_AMT_t-1': '전일 매입원가',
    'PRCH_QTY_t-1': '전일 매입수량',
    'RMRGN_RATE_t-1': '전일 매익률',
    'RMRGN_t-1': '전일 매익액',
    'SPR_AMT_t-1': '전일 할인금액',
    'month_sin_t+1': '월 주기 (sin)',
    'month_cos_t+1': '월 주기 (cos)',
    'day_sin_t+1': '일 주기 (sin)',
    'day_cos_t+1': '일 주기 (cos)',
    'weekday_sin_t+1': '요일 주기 (sin)',
    'weekday_cos_t+1': '요일 주기 (cos)',
    'holiday_korea_t-1_t+1': '전일 공휴일 여부',
    'holiday_korea_t_t+1': '당일 공휴일 여부',
    'holiday_korea_t+1_t+1': '익일 공휴일 여부',
}


class PredictionChatbot:
    """예측값 설명 챗봇"""

    def __init__(self):
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """OpenAI 클라이언트 초기화"""
        if not OPENAI_AVAILABLE:
            print("[Chatbot] openai package not installed")
            return

        try:
            # Streamlit secrets 접근 방식
            if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
                api_key = st.secrets["openai"]["api_key"]
                self.client = openai.OpenAI(api_key=api_key)
                print("[Chatbot] OpenAI client initialized successfully")
            else:
                print("[Chatbot] OpenAI API key not found in secrets")
        except Exception as e:
            print(f"[Chatbot] OpenAI client initialization failed: {e}")

    def format_context(self, context: Dict) -> str:
        """context를 챗봇 프롬프트 형식으로 변환"""
        if not context:
            return "예측 정보를 불러올 수 없습니다."

        # 기본 정보
        result = f"""
[현재 상품 정보]
- 단품코드: {context.get('sku_code', 'N/A')}
- 단품명: {context.get('sku_name', 'N/A')}
- 예측일: {context.get('prediction_date', 'N/A')}
- 예측 horizon: {context.get('horizon', 'N/A')}

[예측 결과]
- 예측값: {context.get('predicted_value', 'N/A')}개
- 신뢰구간: {context.get('pred_min', 'N/A')} ~ {context.get('pred_max', 'N/A')}개
- 예측 모델: {context.get('model_name', 'N/A')}
"""

        # 모델 성능
        if context.get('val_rmse') or context.get('test_rmse'):
            result += f"""
[모델 성능]
- Validation RMSE: {context.get('val_rmse', 'N/A')}
- Test RMSE: {context.get('test_rmse', 'N/A')}
"""

        # 하이퍼파라미터
        if context.get('hyperparameters'):
            params = context['hyperparameters']
            if isinstance(params, dict):
                params_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:5]])
                result += f"""
[주요 하이퍼파라미터]
{params_str}
"""

        # SHAP 값 (Top 10 영향 요인)
        if context.get('shap_values'):
            shap = context['shap_values']
            if isinstance(shap, dict):
                result += "\n[주요 영향 요인 - SHAP Top 10]\n"
                for i, (feature, value) in enumerate(list(shap.items())[:10], 1):
                    feature_name = FEATURE_DESCRIPTIONS.get(feature, feature)
                    result += f"{i}. {feature_name}: {value:.4f}\n"

        # Feature 값 (top_features_values: SHAP Top 10의 실제 값)
        if context.get('top_features_values'):
            fv = context['top_features_values']
            if isinstance(fv, dict):
                result += "\n[해당 시점 주요 Feature 값 (SHAP Top 10)]\n"
                for feature, value in list(fv.items())[:10]:
                    feature_name = FEATURE_DESCRIPTIONS.get(feature, feature)
                    result += f"- {feature_name}: {value}\n"

        # 최근 트렌드
        if context.get('recent_trend'):
            trend = context['recent_trend']
            if isinstance(trend, dict):
                result += f"""
[최근 판매 트렌드]
- 7일 평균: {trend.get('7d_avg', 'N/A')}개
- 7일 표준편차: {trend.get('7d_std', 'N/A')}개
- 전주 대비: {trend.get('vs_prev_week', 'N/A')}
"""

        # Feature 정보 (horizon별 변수 목록 - 규칙 기반 하드코딩)
        horizon = context.get('horizon', 't+1')
        if horizon in HORIZON_FEATURES:
            hf = HORIZON_FEATURES[horizon]
            total_count = len(COMMON_FEATURES) + len(hf['lag']) + len(hf['weather']) + len(hf['calendar']) + len(hf['holiday'])
            result += f"""
[{horizon} 예측에 사용된 변수 정보]
- 총 변수 수: {total_count}개
- 공통 변수 (rolling, 가격, lag_1/2 등): {len(COMMON_FEATURES)}개
  예: {', '.join(COMMON_FEATURES[:5])}...
- Lag 변수 (horizon별 다름): {hf['lag']}
- 날씨 변수: {hf['weather']}
- 캘린더 변수: {hf['calendar']}
- 휴일 변수: {hf['holiday']}
- 타겟 변수: target_{horizon}
"""

        return result

    def translate_feature(self, feature_name: str) -> str:
        """영문 feature명을 한글로 변환"""
        return FEATURE_DESCRIPTIONS.get(feature_name, feature_name)

    def get_response(
        self,
        user_message: str,
        context: Dict,
        chat_history: List[Dict]
    ) -> str:
        """GPT API 호출 및 응답 생성

        Parameters
        ----------
        user_message : str
            사용자 질문
        context : Dict
            예측 context 정보
        chat_history : List[Dict]
            이전 대화 기록

        Returns
        -------
        str
            AI 응답
        """
        if not self.client:
            print("[Chatbot] No client available, using fallback response")
            return self._get_fallback_response(user_message, context)

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[예측 정보]\n{self.format_context(context)}"}
            ]

            # 최근 대화 기록 추가 (최대 5개)
            for msg in chat_history[-5:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            # 현재 질문 추가
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model="gpt-5.1",
                messages=messages,
                max_completion_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[Chatbot] API call failed: {e}")
            return self._get_fallback_response(user_message, context)

    def _get_fallback_response(self, question: str, context: Dict) -> str:
        """API 실패 시 폴백 응답"""
        sku_code = context.get('sku_code', 'Unknown')
        sku_name = context.get('sku_name', 'Unknown')
        predicted_value = context.get('predicted_value', 'N/A')
        model_name = context.get('model_name', 'Unknown')

        # 키워드 기반 응답
        if "근거" in question or "왜" in question:
            shap = context.get('shap_values', {})
            if isinstance(shap, dict) and shap:
                top_features = list(shap.keys())[:3]
                features_str = ", ".join([
                    FEATURE_DESCRIPTIONS.get(f, f) for f in top_features
                ])
                return f"[{sku_name}] 예측값 {predicted_value}개는 {model_name} 모델 기반입니다. " \
                       f"주요 영향 요인: {features_str}. " \
                       f"최근 판매 트렌드와 외부 요인을 종합 분석한 결과입니다."

        if "리스크" in question or "공격" in question:
            pred_max = context.get('pred_max', predicted_value)
            return f"[{sku_name}] 예측 상한({pred_max}개) 초과 발주 시 재고 폐기 리스크가 증가합니다. " \
                   f"신선식품 특성상 예측 범위 내 발주를 권장드립니다."

        if "트렌드" in question or "작년" in question:
            trend = context.get('recent_trend', {})
            if isinstance(trend, dict):
                vs_prev = trend.get('vs_prev_week', 'N/A')
                return f"[{sku_name}] 전주 대비 {vs_prev} 변동을 보이고 있습니다. " \
                       f"현재 예측값은 최근 판매 추세를 반영한 결과입니다."

        # 기본 응답
        return f"[{sku_name}] 예측값 {predicted_value}개는 {model_name} 모델 기반입니다. " \
               f"더 구체적인 질문이 있으시면 말씀해주세요."

    def get_quick_suggestions(self) -> List[str]:
        """빠른 질문 제안"""
        return [
            "예측 근거가 뭐야?",
            "공격적 발주 시 리스크는?",
            "최근 판매 트렌드는?"
        ]


def get_chatbot() -> PredictionChatbot:
    """챗봇 인스턴스 반환 (캐싱)"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PredictionChatbot()
    return st.session_state.chatbot
