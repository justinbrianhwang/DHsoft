# -*- coding: utf-8 -*-
import os
import time
from dataclasses import dataclass

# 단가(USD per 1M tokens) – 필요시 .env로 빼도 됨
PRICE_PER_M_TOKEN_INPUT = 0.15
PRICE_PER_M_TOKEN_OUTPUT = 0.60

def _env_float(key, default):
    try:
        return float(os.getenv(key, default))
    except Exception:
        return float(default)

@dataclass
class BudgetState:
    monthly_budget: float
    daily_budget: float
    spent_today: float = 0.0
    spent_month: float = 0.0

class BudgetManager:
    """
    - 추정 비용 누적하고, 한도 초과 시 폴백 플래그 제공
    - 실제 사용: GPT 호출 직전 estimate → 초과면 skip/샘플링축소
    """
    def __init__(self):
        self.state = BudgetState(
            monthly_budget=_env_float("MONTHLY_BUDGET_USD", 30.0),
            daily_budget=_env_float("DAILY_BUDGET_USD", 1.0),
        )
        self.last_reset_day = time.strftime("%Y-%m-%d")
        self.last_reset_month = time.strftime("%Y-%m")

    def _rollover_if_needed(self):
        # 일/월 경계에서 spent 리셋 (간단 버전)
        today = time.strftime("%Y-%m-%d")
        month = time.strftime("%Y-%m")
        if today != self.last_reset_day:
            self.state.spent_today = 0.0
            self.last_reset_day = today
        if month != self.last_reset_month:
            self.state.spent_month = 0.0
            self.last_reset_month = month

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * PRICE_PER_M_TOKEN_INPUT + \
               (output_tokens / 1_000_000) * PRICE_PER_M_TOKEN_OUTPUT

    def can_spend(self, estimate_cost: float) -> bool:
        self._rollover_if_needed()
        if self.state.spent_today + estimate_cost > self.state.daily_budget:
            return False
        if self.state.spent_month + estimate_cost > self.state.monthly_budget:
            return False
        return True

    def commit(self, actual_cost: float):
        self._rollover_if_needed()
        self.state.spent_today += actual_cost
        self.state.spent_month += actual_cost

    def remaining_today(self) -> float:
        self._rollover_if_needed()
        return max(0.0, self.state.daily_budget - self.state.spent_today)

    def remaining_month(self) -> float:
        self._rollover_if_needed()
        return max(0.0, self.state.monthly_budget - self.state.spent_month)

