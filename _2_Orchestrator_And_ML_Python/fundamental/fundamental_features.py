"""Turn SECFileSourcer output into a small set of ML-ready features.

The ``FundamentalFeatureExtractor`` walks the three-statement DataFrames and
emits seven normalized floats in roughly [-1, 1]:

- gross_margin, operating_margin, net_margin
- revenue_growth_yoy
- roe (return on equity)
- current_ratio
- fcf_to_capex

Missing data → 0.0 for that key (a neutral value that doesn't push the model
in either direction).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalFeatureExtractor:
    KEYS = [
        "gross_margin",
        "operating_margin",
        "net_margin",
        "revenue_growth_yoy",
        "roe",
        "current_ratio",
        "fcf_to_capex",
    ]

    REVENUE_KEYS = ("Revenues", "RevenueFromContractWithCustomer", "SalesRevenue")
    COGS_KEYS = ("CostOfRevenue", "CostOfGoodsSold", "CostOfGoodsAndServicesSold")
    OP_INCOME_KEYS = ("OperatingIncomeLoss",)
    NET_INCOME_KEYS = ("NetIncomeLoss", "NetIncome")
    EQUITY_KEYS = ("StockholdersEquity",)
    CURRENT_ASSETS_KEYS = ("AssetsCurrent",)
    CURRENT_LIAB_KEYS = ("LiabilitiesCurrent",)
    OPERATING_CF_KEYS = ("NetCashProvidedByUsedInOperatingActivities", "CashFlowFromOperatingActivities")
    CAPEX_KEYS = ("PaymentsToAcquirePropertyPlantAndEquipment", "CapitalExpenditures")

    def extract(self, financial_model: Dict[str, pd.DataFrame], ticker: str = "") -> Dict[str, float]:
        """Return the 7-element feature dict. Every key always present."""
        out = {k: 0.0 for k in self.KEYS}

        income = financial_model.get("annual_income_statement", pd.DataFrame())
        balance = financial_model.get("annual_balance_sheet", pd.DataFrame())
        cash = financial_model.get("annual_cash_flow", pd.DataFrame())

        if income is None or income.empty:
            logger.debug("No annual income statement for %s; returning zeros", ticker)
            return out

        latest_income = self._latest_row(income)
        prev_income = self._prev_row(income)
        latest_balance = self._latest_row(balance) if balance is not None and not balance.empty else {}
        latest_cash = self._latest_row(cash) if cash is not None and not cash.empty else {}

        revenue = self._pick(latest_income, self.REVENUE_KEYS)
        cogs = self._pick(latest_income, self.COGS_KEYS)
        op_income = self._pick(latest_income, self.OP_INCOME_KEYS)
        net_income = self._pick(latest_income, self.NET_INCOME_KEYS)

        if revenue and revenue != 0:
            if cogs is not None:
                out["gross_margin"] = self._clamp((revenue - cogs) / abs(revenue))
            if op_income is not None:
                out["operating_margin"] = self._clamp(op_income / abs(revenue))
            if net_income is not None:
                out["net_margin"] = self._clamp(net_income / abs(revenue))

        prev_revenue = self._pick(prev_income, self.REVENUE_KEYS)
        if revenue and prev_revenue and prev_revenue != 0:
            growth = (revenue - prev_revenue) / abs(prev_revenue)
            out["revenue_growth_yoy"] = self._clamp(growth, lo=-1.0, hi=1.0)

        equity = self._pick(latest_balance, self.EQUITY_KEYS)
        if net_income is not None and equity and equity != 0:
            out["roe"] = self._clamp(net_income / abs(equity), lo=-1.0, hi=1.0)

        current_assets = self._pick(latest_balance, self.CURRENT_ASSETS_KEYS)
        current_liab = self._pick(latest_balance, self.CURRENT_LIAB_KEYS)
        if current_assets and current_liab:
            raw = current_assets / current_liab
            out["current_ratio"] = self._clamp((raw - 1.0) / 2.0, lo=-1.0, hi=1.0)

        op_cf = self._pick(latest_cash, self.OPERATING_CF_KEYS)
        capex = self._pick(latest_cash, self.CAPEX_KEYS)
        if op_cf is not None and capex and capex != 0:
            ratio = op_cf / abs(capex)
            out["fcf_to_capex"] = self._clamp((ratio - 1.0) / 2.0, lo=-1.0, hi=1.0)

        return out

    @staticmethod
    def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
        try:
            f = float(x)
            if f != f:  # NaN check
                return 0.0
            return max(lo, min(hi, f))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _pick(row_like, keys) -> Optional[float]:
        if row_like is None:
            return None
        for k in keys:
            if hasattr(row_like, "get") and k in row_like:
                try:
                    val = row_like[k]
                    if pd.notna(val):
                        return float(val)
                except Exception:
                    continue
            candidate = None
            if hasattr(row_like, "index"):
                iterable = row_like.index
            elif isinstance(row_like, dict):
                iterable = row_like.keys()
            else:
                continue
            for idx in iterable:
                if k.lower() in str(idx).lower():
                    candidate = idx
                    break
            if candidate is not None:
                try:
                    val = row_like[candidate]
                    if pd.notna(val):
                        return float(val)
                except Exception:
                    continue
        return None

    @staticmethod
    def _latest_row(df: pd.DataFrame):
        if df is None or df.empty:
            return None
        try:
            return df.sort_index(ascending=False).iloc[0]
        except Exception:
            return df.iloc[-1]

    @staticmethod
    def _prev_row(df: pd.DataFrame):
        if df is None or df.empty or len(df) < 2:
            return None
        try:
            return df.sort_index(ascending=False).iloc[1]
        except Exception:
            return df.iloc[-2]
