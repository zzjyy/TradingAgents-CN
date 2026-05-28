# TradingAgents/graph/signal_processing.py

from langchain_openai import ChatOpenAI

# 导入统一日志系统和图处理模块日志装饰器
from tradingagents.utils.logging_init import get_logger
from tradingagents.utils.tool_logging import log_graph_module
logger = get_logger("graph.signal_processing")


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    @log_graph_module("signal_processing")
    def process_signal(self, full_signal: str, stock_symbol: str = None) -> dict:
        """
        Process a full trading signal to extract structured decision information.

        Args:
            full_signal: Complete trading signal text
            stock_symbol: Stock symbol to determine currency type

        Returns:
            Dictionary containing extracted decision information
        """

        # 验证输入参数
        if not full_signal or not isinstance(full_signal, str) or len(full_signal.strip()) == 0:
            logger.error(f"❌ [SignalProcessor] 输入信号为空或无效: {repr(full_signal)}")
            return {
                'action': '持有',
                'target_price': None,
                'confidence': 0.5,
                'risk_score': 0.5,
                'reasoning': '输入信号无效，默认持有建议'
            }

        # 清理和验证信号内容
        full_signal = full_signal.strip()
        if len(full_signal) == 0:
            logger.error(f"❌ [SignalProcessor] 信号内容为空")
            return {
                'action': '持有',
                'target_price': None,
                'confidence': 0.5,
                'risk_score': 0.5,
                'reasoning': '信号内容为空，默认持有建议'
            }

        # 检测股票类型和货币
        from tradingagents.utils.stock_utils import StockUtils

        market_info = StockUtils.get_market_info(stock_symbol)
        is_china = market_info['is_china']
        is_hk = market_info['is_hk']
        currency = market_info['currency_name']
        currency_symbol = market_info['currency_symbol']

        # === start edit by my_fund_support ===
        # 基金/ETF 走基金专用提取逻辑
        try:
            from tradingagents.utils.stock_validator import StockDataPreparer
            is_fund = StockDataPreparer._is_fund_code(stock_symbol) if stock_symbol else False
        except Exception:
            is_fund = False

        if is_fund:
            logger.info(f"🔍 [SignalProcessor] 检测到基金/ETF 代码 {stock_symbol}，走基金语义分支")
            return self._process_fund_signal(full_signal, stock_symbol)
        # === end edit by my_fund_support ===

        logger.info(f"🔍 [SignalProcessor] 处理信号: 股票={stock_symbol}, 市场={market_info['market_name']}, 货币={currency}",
                   extra={'stock_symbol': stock_symbol, 'market': market_info['market_name'], 'currency': currency})

        messages = [
            (
                "system",
                f"""您是一位专业的金融分析助手，负责从交易员的分析报告中提取结构化的投资决策信息。

请从提供的分析报告中提取以下信息，并以JSON格式返回：

{{
    "action": "买入/持有/卖出",
    "target_price": 数字({currency}价格，**必须提供具体数值，不能为null**),
    "confidence": 数字(0-1之间，如果没有明确提及则为0.7),
    "risk_score": 数字(0-1之间，如果没有明确提及则为0.5),
    "reasoning": "决策的主要理由摘要"
}}

请确保：
1. action字段必须是"买入"、"持有"或"卖出"之一（绝对不允许使用英文buy/hold/sell）
2. target_price必须是具体的数字,target_price应该是合理的{currency}价格数字（使用{currency_symbol}符号）
3. confidence和risk_score应该在0-1之间
4. reasoning应该是简洁的中文摘要
5. 所有内容必须使用中文，不允许任何英文投资建议

特别注意：
- 股票代码 {stock_symbol or '未知'} 是{market_info['market_name']}，使用{currency}计价
- 目标价格必须与股票的交易货币一致（{currency_symbol}）

如果某些信息在报告中没有明确提及，请使用合理的默认值。""",
            ),
            ("human", full_signal),
        ]

        # 验证messages内容
        if not messages or len(messages) == 0:
            logger.error(f"❌ [SignalProcessor] messages为空")
            return self._get_default_decision()
        
        # 验证human消息内容
        human_content = messages[1][1] if len(messages) > 1 else ""
        if not human_content or len(human_content.strip()) == 0:
            logger.error(f"❌ [SignalProcessor] human消息内容为空")
            return self._get_default_decision()

        logger.debug(f"🔍 [SignalProcessor] 准备调用LLM，消息数量: {len(messages)}, 信号长度: {len(full_signal)}")

        try:
            response = self.quick_thinking_llm.invoke(messages).content
            logger.debug(f"🔍 [SignalProcessor] LLM响应: {response[:200]}...")

            # 尝试解析JSON响应
            import json
            import re

            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                logger.debug(f"🔍 [SignalProcessor] 提取的JSON: {json_text}")
                decision_data = json.loads(json_text)

                # 验证和标准化数据
                action = decision_data.get('action', '持有')
                if action not in ['买入', '持有', '卖出']:
                    # 尝试映射英文和其他变体
                    action_map = {
                        'buy': '买入', 'hold': '持有', 'sell': '卖出',
                        'BUY': '买入', 'HOLD': '持有', 'SELL': '卖出',
                        '购买': '买入', '保持': '持有', '出售': '卖出',
                        'purchase': '买入', 'keep': '持有', 'dispose': '卖出'
                    }
                    action = action_map.get(action, '持有')
                    if action != decision_data.get('action', '持有'):
                        logger.debug(f"🔍 [SignalProcessor] 投资建议映射: {decision_data.get('action')} -> {action}")

                # 处理目标价格，确保正确提取
                target_price = decision_data.get('target_price')
                if target_price is None or target_price == "null" or target_price == "":
                    # 如果JSON中没有目标价格，尝试从reasoning和完整文本中提取
                    reasoning = decision_data.get('reasoning', '')
                    full_text = f"{reasoning} {full_signal}"  # 扩大搜索范围
                    
                    # 增强的价格匹配模式
                    price_patterns = [
                        r'目标价[位格]?[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',  # 目标价位: 45.50
                        r'目标[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',         # 目标: 45.50
                        r'价格[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',         # 价格: 45.50
                        r'价位[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',         # 价位: 45.50
                        r'合理[价位格]?[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)', # 合理价位: 45.50
                        r'估值[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',         # 估值: 45.50
                        r'[¥\$](\d+(?:\.\d+)?)',                      # ¥45.50 或 $190
                        r'(\d+(?:\.\d+)?)元',                         # 45.50元
                        r'(\d+(?:\.\d+)?)美元',                       # 190美元
                        r'建议[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',        # 建议: 45.50
                        r'预期[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',        # 预期: 45.50
                        r'看[到至]\s*[¥\$]?(\d+(?:\.\d+)?)',          # 看到45.50
                        r'上涨[到至]\s*[¥\$]?(\d+(?:\.\d+)?)',        # 上涨到45.50
                        r'(\d+(?:\.\d+)?)\s*[¥\$]',                  # 45.50¥
                    ]
                    
                    for pattern in price_patterns:
                        price_match = re.search(pattern, full_text, re.IGNORECASE)
                        if price_match:
                            try:
                                target_price = float(price_match.group(1))
                                logger.debug(f"🔍 [SignalProcessor] 从文本中提取到目标价格: {target_price} (模式: {pattern})")
                                break
                            except (ValueError, IndexError):
                                continue

                    # 如果仍然没有找到价格，尝试智能推算
                    if target_price is None or target_price == "null" or target_price == "":
                        target_price = self._smart_price_estimation(full_text, action, is_china)
                        if target_price:
                            logger.debug(f"🔍 [SignalProcessor] 智能推算目标价格: {target_price}")
                        else:
                            target_price = None
                            logger.warning(f"🔍 [SignalProcessor] 未能提取到目标价格，设置为None")
                else:
                    # 确保价格是数值类型
                    try:
                        if isinstance(target_price, str):
                            # 清理字符串格式的价格
                            clean_price = target_price.replace('$', '').replace('¥', '').replace('￥', '').replace('元', '').replace('美元', '').strip()
                            target_price = float(clean_price) if clean_price and clean_price.lower() not in ['none', 'null', ''] else None
                        elif isinstance(target_price, (int, float)):
                            target_price = float(target_price)
                        logger.debug(f"🔍 [SignalProcessor] 处理后的目标价格: {target_price}")
                    except (ValueError, TypeError):
                        target_price = None
                        logger.warning(f"🔍 [SignalProcessor] 价格转换失败，设置为None")

                result = {
                    'action': action,
                    'target_price': target_price,
                    'confidence': float(decision_data.get('confidence', 0.7)),
                    'risk_score': float(decision_data.get('risk_score', 0.5)),
                    'reasoning': decision_data.get('reasoning', '基于综合分析的投资建议')
                }
                logger.info(f"🔍 [SignalProcessor] 处理结果: {result}",
                           extra={'action': result['action'], 'target_price': result['target_price'],
                                 'confidence': result['confidence'], 'stock_symbol': stock_symbol})
                return result
            else:
                # 如果无法解析JSON，使用简单的文本提取
                return self._extract_simple_decision(response)

        except Exception as e:
            logger.error(f"信号处理错误: {e}", exc_info=True, extra={'stock_symbol': stock_symbol})
            # 回退到简单提取
            return self._extract_simple_decision(full_signal)

    def _smart_price_estimation(self, text: str, action: str, is_china: bool) -> float:
        """智能价格推算方法"""
        import re
        
        # 尝试从文本中提取当前价格和涨跌幅信息
        current_price = None
        percentage_change = None
        
        # 提取当前价格
        current_price_patterns = [
            r'当前价[格位]?[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',
            r'现价[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',
            r'股价[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',
            r'价格[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',
        ]
        
        for pattern in current_price_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    current_price = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        # 提取涨跌幅信息
        percentage_patterns = [
            r'上涨\s*(\d+(?:\.\d+)?)%',
            r'涨幅\s*(\d+(?:\.\d+)?)%',
            r'增长\s*(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)%\s*的?上涨',
        ]
        
        for pattern in percentage_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    percentage_change = float(match.group(1)) / 100
                    break
                except ValueError:
                    continue
        
        # 基于动作和信息推算目标价
        if current_price and percentage_change:
            if action == '买入':
                return round(current_price * (1 + percentage_change), 2)
            elif action == '卖出':
                return round(current_price * (1 - percentage_change), 2)
        
        # 如果有当前价格但没有涨跌幅，使用默认估算
        if current_price:
            if action == '买入':
                # 买入建议默认10-20%涨幅
                multiplier = 1.15 if is_china else 1.12
                return round(current_price * multiplier, 2)
            elif action == '卖出':
                # 卖出建议默认5-10%跌幅
                multiplier = 0.95 if is_china else 0.92
                return round(current_price * multiplier, 2)
            else:  # 持有
                # 持有建议使用当前价格
                return current_price
        
        return None

    def _extract_simple_decision(self, text: str) -> dict:
        """简单的决策提取方法作为备用"""
        import re

        # 提取动作
        action = '持有'  # 默认
        if re.search(r'买入|BUY', text, re.IGNORECASE):
            action = '买入'
        elif re.search(r'卖出|SELL', text, re.IGNORECASE):
            action = '卖出'
        elif re.search(r'持有|HOLD', text, re.IGNORECASE):
            action = '持有'

        # 尝试提取目标价格（使用增强的模式）
        target_price = None
        price_patterns = [
            r'目标价[位格]?[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',  # 目标价位: 45.50
            r'\*\*目标价[位格]?\*\*[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',  # **目标价位**: 45.50
            r'目标[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',         # 目标: 45.50
            r'价格[：:]?\s*[¥\$]?(\d+(?:\.\d+)?)',         # 价格: 45.50
            r'[¥\$](\d+(?:\.\d+)?)',                      # ¥45.50 或 $190
            r'(\d+(?:\.\d+)?)元',                         # 45.50元
        ]

        for pattern in price_patterns:
            price_match = re.search(pattern, text)
            if price_match:
                try:
                    target_price = float(price_match.group(1))
                    break
                except ValueError:
                    continue

        # 如果没有找到价格，尝试智能推算
        if target_price is None:
            # 检测股票类型
            is_china = True  # 默认假设是A股，实际应该从上下文获取
            target_price = self._smart_price_estimation(text, action, is_china)

        return {
            'action': action,
            'target_price': target_price,
            'confidence': 0.7,
            'risk_score': 0.5,
            'reasoning': '基于综合分析的投资建议'
        }

    def _get_default_decision(self) -> dict:
        """返回默认的投资决策"""
        return {
            'action': '持有',
            'target_price': None,
            'confidence': 0.5,
            'risk_score': 0.5,
            'reasoning': '输入数据无效，默认持有建议'
        }

    # === start edit by my_fund_support ===
    # ------------------------------------------------------------------
    # 基金/ETF 信号处理
    # 基金语义与个股不同：
    #   - 动作: 申购 / 持有 / 赎回 / 转换（场外）或 买入 / 持有 / 卖出（ETF 场内同股票）
    #   - 目标量化指标: 目标仓位权重 (0-1)，而非目标价（场外基金 T+1 净值无连续价格）
    #   - ETF 仍可有目标价（因为有实时盘中价），但权重比价格更重要
    # ------------------------------------------------------------------

    def _process_fund_signal(self, full_signal: str, stock_symbol: str = None) -> dict:
        """从分析报告中提取基金/ETF 的结构化决策。"""
        is_etf = bool(stock_symbol and stock_symbol[:3] in (
            "510", "511", "512", "513", "515", "516", "517", "518",
            "159", "501", "502", "588", "560", "561", "562", "563"
        ))
        action_set = "申购 / 持有 / 赎回 / 转换" if not is_etf else "买入 / 持有 / 卖出"
        default_action = "持有"

        system_prompt = f"""您是一位专业的基金分析助手，负责从研究报告中提取结构化的基金投资决策。

基金代码 {stock_symbol or '未知'} 属于{'场内 ETF' if is_etf else '场外公募基金'}。

请以 JSON 返回，字段如下：

{{
    "action": "{action_set} 之一",
    "target_weight": 数字(0-1之间，建议在组合中的目标仓位比例),
    "confidence": 数字(0-1),
    "risk_score": 数字(0-1),
    "reasoning": "决策的主要理由（中文）"
}}

要点：
1. action 必须是上述选项之一，使用中文
2. target_weight 表示在用户基金组合中的目标占比（如 0.15 = 15%）
3. 若报告中未明确仓位建议，按"申购→0.1、加仓→0.2、持有→保持现状（用 0.0 占位）、减仓→-0.1、赎回→0.0"给出
4. 所有字段必须使用中文，禁止英文 buy/hold/sell
"""

        messages = [
            ("system", system_prompt),
            ("human", full_signal),
        ]

        try:
            response = self.quick_thinking_llm.invoke(messages).content
            import json, re

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return self._extract_simple_fund_decision(full_signal, is_etf)

            decision_data = json.loads(json_match.group())

            valid_actions = {"申购", "持有", "赎回", "转换"} if not is_etf else {"买入", "持有", "卖出"}
            action = decision_data.get('action', default_action)
            if action not in valid_actions:
                action_map = {
                    'buy': '申购', 'hold': '持有', 'sell': '赎回',
                    'BUY': '申购', 'HOLD': '持有', 'SELL': '赎回',
                    '买入': '申购', '卖出': '赎回',  # 场外基金把买卖映射为申赎
                }
                if is_etf:
                    action_map = {
                        'buy': '买入', 'hold': '持有', 'sell': '卖出',
                        'BUY': '买入', 'HOLD': '持有', 'SELL': '卖出',
                        '申购': '买入', '赎回': '卖出',
                    }
                action = action_map.get(action, default_action)

            try:
                target_weight = float(decision_data.get('target_weight', 0.0))
                target_weight = max(-1.0, min(1.0, target_weight))
            except (TypeError, ValueError):
                target_weight = 0.0

            result = {
                'action': action,
                'target_price': None,
                'target_weight': target_weight,
                'confidence': float(decision_data.get('confidence', 0.7)),
                'risk_score': float(decision_data.get('risk_score', 0.5)),
                'reasoning': decision_data.get('reasoning', '基于综合分析的基金投资建议'),
                'asset_type': 'etf' if is_etf else 'fund',
            }
            logger.info(f"🔍 [SignalProcessor/Fund] 处理结果: {result}")
            return result

        except Exception as e:
            logger.error(f"基金信号处理错误: {e}", exc_info=True)
            return self._extract_simple_fund_decision(full_signal, is_etf)

    def _extract_simple_fund_decision(self, text: str, is_etf: bool) -> dict:
        """基金信号的简单文本回退提取。"""
        import re

        if is_etf:
            valid = {'买入', '持有', '卖出'}
            default = '持有'
        else:
            valid = {'申购', '持有', '赎回', '转换'}
            default = '持有'

        action = default
        for word in valid:
            if word in text:
                action = word
                break
        # 兜底：从英文/股票术语映射
        if action == default:
            if re.search(r'买入|buy', text, re.IGNORECASE):
                action = '买入' if is_etf else '申购'
            elif re.search(r'卖出|sell', text, re.IGNORECASE):
                action = '卖出' if is_etf else '赎回'

        # 提取目标权重
        target_weight = 0.0
        weight_match = re.search(r'(?:目标|建议)?(?:仓位|权重)[：:]?\s*(\d+(?:\.\d+)?)\s*%?', text)
        if weight_match:
            try:
                w = float(weight_match.group(1))
                target_weight = w / 100 if w > 1 else w
                target_weight = max(-1.0, min(1.0, target_weight))
            except ValueError:
                pass

        return {
            'action': action,
            'target_price': None,
            'target_weight': target_weight,
            'confidence': 0.6,
            'risk_score': 0.5,
            'reasoning': '基于综合分析的基金投资建议（文本回退提取）',
            'asset_type': 'etf' if is_etf else 'fund',
        }
    # === end edit by my_fund_support ===
