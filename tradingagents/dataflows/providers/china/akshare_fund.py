"""
AKShare 基金/ETF 数据提供器

封装 AKShare 中跟公募基金、ETF 相关的接口，所有接口都通过 fund_support
分支调用，保持与 AKShareProvider 同样的 connected 标志位与日志风格。

接口对照（7 个核心）:
  1. fund_etf_spot_em()                    -> ETF 实时行情（全市场）
  2. fund_etf_hist_em(symbol, ...)         -> ETF 历史行情（K 线）
  3. fund_open_fund_info_em(symbol, ind.)  -> 开放基金净值/分红/累计净值走势
  4. fund_portfolio_hold_em(symbol, date)  -> 基金重仓股持仓
  5. fund_name_em()                        -> 全市场基金代码 -> 名称/类型 映射
  6. fund_manager_em()                     -> 基金经理列表
  7. fund_individual_basic_info_xq(symbol) -> 基金基础档案（雪球源）

设计原则:
- 所有方法返回 (ok: bool, data: Any, err: str) 三元组，便于上层做容错
- 不做缓存（缓存交给 fund_data_service 层做），保持本层薄而透
- 接口失败只记录 debug 日志，不抛异常
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from tradingagents.utils.logging_manager import get_logger

logger = get_logger('agents')


FundResult = Tuple[bool, Any, str]


class AKShareFundProvider:
    """AKShare 公募基金/ETF 数据提供器。

    与 AKShareProvider 解耦，专门服务基金线。初始化时若 akshare 不可用，
    connected=False，所有调用都会返回 (False, None, err)。
    """

    def __init__(self) -> None:
        self.name = "AKShareFund"
        self.ak = None
        self.connected = False
        self._initialize()

    def _initialize(self) -> None:
        try:
            import akshare as ak
            self.ak = ak
            self.connected = True
            logger.info(f"✅ [{self.name}] AKShare 初始化成功，version={getattr(ak, '__version__', 'unknown')}")
        except ImportError as e:
            logger.warning(f"⚠️ [{self.name}] akshare 未安装: {e}")
            self.connected = False
        except Exception as e:
            logger.error(f"❌ [{self.name}] 初始化失败: {e}")
            self.connected = False

    # ----- 内部工具 -----

    def _guard(self) -> Optional[FundResult]:
        if not self.connected or self.ak is None:
            return False, None, "AKShareFund 未初始化（akshare 不可用）"
        return None

    @staticmethod
    def _is_etf_code(symbol: str) -> bool:
        """510/159/501/588 系列识别为 ETF（场内基金）。"""
        if not symbol or len(symbol) < 3:
            return False
        prefix = symbol[:3]
        return prefix in {"510", "511", "512", "513", "515", "516", "517", "518",
                          "159", "501", "502", "588", "560", "561", "562", "563"}

    @staticmethod
    def _is_open_fund_code(symbol: str) -> bool:
        """6 位数字、非 ETF 前缀的当作场外公募。"""
        return symbol.isdigit() and len(symbol) == 6

    # ----- 1. ETF 实时行情 -----

    def get_etf_spot_all(self) -> FundResult:
        """返回全市场 ETF 实时行情（DataFrame）。"""
        guard = self._guard()
        if guard:
            return guard
        try:
            df = self.ak.fund_etf_spot_em()
            if df is None or df.empty:
                return False, None, "fund_etf_spot_em 返回为空"
            return True, df, ""
        except Exception as e:
            logger.debug(f"[{self.name}] get_etf_spot_all 失败: {e}")
            return False, None, f"{type(e).__name__}: {e}"

    def get_etf_spot(self, symbol: str) -> FundResult:
        """单只 ETF 实时行情。"""
        ok, df, err = self.get_etf_spot_all()
        if not ok:
            return ok, df, err
        try:
            row = df[df["代码"] == symbol]
            if row.empty:
                return False, None, f"ETF {symbol} 未在快照中找到"
            return True, row.iloc[0].to_dict(), ""
        except Exception as e:
            return False, None, f"{type(e).__name__}: {e}"

    # ----- 2. ETF 历史 K 线 -----

    def get_etf_hist(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        period: str = "daily",
        adjust: str = "qfq",
    ) -> FundResult:
        """ETF 历史行情（K 线）。日期格式 YYYYMMDD。

        主源: 东方财富 fund_etf_hist_em（支持日期范围 + 复权，但 WSL 上常抖动）
        备源: 新浪 fund_etf_hist_sina（稳定，但返回全量 + 英文列名，需切片重命名）
        """
        guard = self._guard()
        if guard:
            return guard

        # 1) 主源：东方财富
        try:
            df = self.ak.fund_etf_hist_em(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            if df is not None and not df.empty:
                return True, df, ""
        except Exception as e:
            logger.debug(f"[{self.name}] fund_etf_hist_em({symbol}) 失败，转新浪: {e}")

        # 2) 备源：新浪（symbol 要加 sh/sz 前缀）
        try:
            sina_sym = ("sh" if symbol.startswith("5") else "sz") + symbol
            df = self.ak.fund_etf_hist_sina(symbol=sina_sym)
            if df is None or df.empty:
                return False, None, f"ETF {symbol} 新浪源也无数据"
            # 列名英文→中文，与东方财富格式对齐
            df = df.rename(columns={
                "date": "日期", "open": "开盘", "high": "最高",
                "low": "最低", "close": "收盘", "volume": "成交量",
                "amount": "成交额",
            })
            # 日期字符串切片 (新浪 date 是 datetime.date)
            df["日期"] = df["日期"].astype(str)
            sd_iso = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            ed_iso = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            df = df[(df["日期"] >= sd_iso) & (df["日期"] <= ed_iso)].reset_index(drop=True)
            if df.empty:
                return False, None, f"ETF {symbol} 日期范围内无数据"
            # 补充 涨跌幅 列（新浪不提供）
            df["涨跌幅"] = (df["收盘"].pct_change() * 100).round(2).fillna(0)
            return True, df, ""
        except Exception as e:
            logger.debug(f"[{self.name}] fund_etf_hist_sina({symbol}) 也失败: {e}")
            return False, None, f"主源+备源均失败: {type(e).__name__}: {e}"

    # ----- 3. 开放基金净值 -----

    def get_open_fund_nav(self, symbol: str) -> FundResult:
        """开放式基金单位净值走势。返回 DataFrame [净值日期, 单位净值, 日增长率]。"""
        guard = self._guard()
        if guard:
            return guard
        try:
            df = self.ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势")
            if df is None or df.empty:
                return False, None, f"基金 {symbol} 净值数据为空"
            return True, df, ""
        except Exception as e:
            logger.debug(f"[{self.name}] get_open_fund_nav({symbol}) 失败: {e}")
            return False, None, f"{type(e).__name__}: {e}"

    def get_open_fund_acc_nav(self, symbol: str) -> FundResult:
        """开放式基金累计净值走势。"""
        guard = self._guard()
        if guard:
            return guard
        try:
            df = self.ak.fund_open_fund_info_em(symbol=symbol, indicator="累计净值走势")
            if df is None or df.empty:
                return False, None, f"基金 {symbol} 累计净值数据为空"
            return True, df, ""
        except Exception as e:
            logger.debug(f"[{self.name}] get_open_fund_acc_nav({symbol}) 失败: {e}")
            return False, None, f"{type(e).__name__}: {e}"

    # ----- 4. 基金重仓持仓 -----

    def get_fund_holdings(self, symbol: str, date: str) -> FundResult:
        """
        基金重仓股持仓。
        date 是年份字符串 (如 "2024")。返回 DataFrame 含 [股票代码,股票名称,占净值比例,持股数,持仓市值,季度]。
        """
        guard = self._guard()
        if guard:
            return guard
        try:
            df = self.ak.fund_portfolio_hold_em(symbol=symbol, date=date)
            if df is None or df.empty:
                return False, None, f"基金 {symbol} {date}年持仓数据为空"
            return True, df, ""
        except Exception as e:
            logger.debug(f"[{self.name}] get_fund_holdings({symbol},{date}) 失败: {e}")
            return False, None, f"{type(e).__name__}: {e}"

    # ----- 5. 基金名称索引 -----

    def get_all_fund_names(self) -> FundResult:
        """全市场基金代码 -> 名称/类型 索引。返回 DataFrame [基金代码,基金简称,基金类型,...]。"""
        guard = self._guard()
        if guard:
            return guard
        try:
            df = self.ak.fund_name_em()
            if df is None or df.empty:
                return False, None, "fund_name_em 返回为空"
            return True, df, ""
        except Exception as e:
            logger.debug(f"[{self.name}] get_all_fund_names 失败: {e}")
            return False, None, f"{type(e).__name__}: {e}"

    def lookup_fund_name(self, symbol: str) -> FundResult:
        """单只基金名称/类型。"""
        ok, df, err = self.get_all_fund_names()
        if not ok:
            return ok, df, err
        try:
            row = df[df["基金代码"] == symbol]
            if row.empty:
                return False, None, f"基金 {symbol} 未在名称表中找到"
            return True, row.iloc[0].to_dict(), ""
        except Exception as e:
            return False, None, f"{type(e).__name__}: {e}"

    # ----- 6. 基金经理列表 -----

    def get_all_managers(self) -> FundResult:
        """全市场基金经理列表。返回 DataFrame。"""
        guard = self._guard()
        if guard:
            return guard
        try:
            df = self.ak.fund_manager_em()
            if df is None or df.empty:
                return False, None, "fund_manager_em 返回为空"
            return True, df, ""
        except Exception as e:
            logger.debug(f"[{self.name}] get_all_managers 失败: {e}")
            return False, None, f"{type(e).__name__}: {e}"

    # ----- 7. 基金档案（雪球） -----

    def get_fund_profile(self, symbol: str) -> FundResult:
        """基金基础档案。返回 DataFrame [item, value]。

        ETF (510/159/...) 雪球档案 API 不支持，会抛 KeyError: 'data'。
        因此 ETF 走 ETF spot 拼装，开放基金维持雪球。
        """
        guard = self._guard()
        if guard:
            return guard

        if self._is_etf_code(symbol):
            ok, row, err = self.get_etf_spot(symbol)
            if not ok or row is None:
                return False, None, f"ETF {symbol} 行情失败: {err}"
            # row 已经是 dict（get_etf_spot 内部 .iloc[0].to_dict()）
            keys = ["代码", "名称", "最新价", "涨跌幅", "成交额", "流通市值",
                    "IOPV实时估值", "基金折价率"]
            import pandas as pd
            records = [{"item": k, "value": row[k]} for k in keys if k in row]
            if not records:
                return False, None, f"ETF {symbol} 档案无可用字段"
            return True, pd.DataFrame(records), ""

        # 场外开放基金走雪球档案
        try:
            df = self.ak.fund_individual_basic_info_xq(symbol=symbol)
            if df is None or df.empty:
                return False, None, f"基金 {symbol} 档案数据为空"
            return True, df, ""
        except Exception as e:
            logger.debug(f"[{self.name}] get_fund_profile({symbol}) 失败: {e}")
            return False, None, f"{type(e).__name__}: {e}"


# 单例
_provider_instance: Optional[AKShareFundProvider] = None


def get_akshare_fund_provider() -> AKShareFundProvider:
    """返回 AKShareFundProvider 单例。"""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = AKShareFundProvider()
    return _provider_instance
