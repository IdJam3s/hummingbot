import os
import sys
import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple
from typing import Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
#from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
#from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

@dataclass
class TradeMetrics:
    trading_pair: str
    direction: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    amount: float
    leverage: int
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    holding_period: Optional[timedelta]
    priority_level: int

@dataclass
class TradeOpportunity:
    connector: str
    trading_pair: str
    direction: str
    leverage: int
    priority: int

    @property
    def priority_score(self) -> int:
        direction_score = 0 if self.direction == "LONG" else 1
        return self.priority * 10 + direction_score

class PerformanceTracker:
    def __init__(self, strategy):
        self.strategy = strategy
        self.trades: List[TradeMetrics] = []
        self.current_trades: Dict[str, TradeMetrics] = {}
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'current_pnl': 0.0,
            'win_rate': 0.0,
            'priority_metrics': {
                'P1': {'count': 0, 'pnl': 0.0, 'win_rate': 0.0},
                'P2': {'count': 0, 'pnl': 0.0, 'win_rate': 0.0},
                'P3': {'count': 0, 'pnl': 0.0, 'win_rate': 0.0},
                'P4': {'count': 0, 'pnl': 0.0, 'win_rate': 0.0}
            }
        }
        self.trades_dir = "trades_history"
        os.makedirs(self.trades_dir, exist_ok=True)

    def track_trade(self, trade: TradeMetrics):
        """Track new trade"""
        trade_key = f"{trade.trading_pair}_{trade.entry_time.timestamp()}"
        self.current_trades[trade_key] = trade
        self.metrics['total_trades'] += 1

        # Update priority metrics
        priority = f"P{trade.priority_level}"
        self.metrics['priority_metrics'][priority]['count'] += 1

    def update_trade(self, trade_key: str, exit_price: float, exit_time: datetime):
        """Update trade on exit"""
        if trade_key in self.current_trades:
            trade = self.current_trades[trade_key]
            trade.exit_price = exit_price
            trade.exit_time = exit_time
            trade.holding_period = exit_time - trade.entry_time

            # Calculate PnL
            if trade.direction == "LONG":
                trade.pnl = (exit_price - trade.entry_price) * trade.amount * trade.leverage
            else:
                trade.pnl = (trade.entry_price - exit_price) * trade.amount * trade.leverage

            trade.pnl_percentage = (trade.pnl / (trade.entry_price * trade.amount)) * 100

            # Update metrics
            self.metrics['total_pnl'] += trade.pnl
            if trade.pnl > 0:
                self.metrics['winning_trades'] += 1
            else:
                self.metrics['losing_trades'] += 1

            # Update priority metrics
            priority = f"P{trade.priority_level}"
            self.metrics['priority_metrics'][priority]['pnl'] += trade.pnl

            # Calculate win rates
            self.metrics['win_rate'] = (self.metrics['winning_trades'] /
                                      self.metrics['total_trades'] * 100)

            # Archive trade
            self.trades.append(trade)
            del self.current_trades[trade_key]

            # Save trade data
            self.save_trade(trade)

    def save_trade(self, trade: TradeMetrics):
        """Save trade details to file"""
        trade_data = {
            'trading_pair': trade.trading_pair,
            'direction': trade.direction,
            'entry_time': str(trade.entry_time),
            'exit_time': str(trade.exit_time),
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'amount': trade.amount,
            'leverage': trade.leverage,
            'pnl': trade.pnl,
            'pnl_percentage': trade.pnl_percentage,
            'holding_period': str(trade.holding_period),
            'priority_level': trade.priority_level,
            'exit_reason': getattr(trade, 'exit_reason', 'Unknown'),
            'holding_period_minutes': (trade.exit_time - trade.entry_time).total_seconds() / 60 if trade.exit_time else None
        }

        filename = f"{self.trades_dir}/trade_{trade.entry_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(trade_data, f, indent=4)

    def get_metrics(self) -> dict:
        """Get current performance metrics"""
        return self.metrics

    def get_summary(self) -> str:
        """Get performance summary"""
        return f"""Performance Summary:
        Total Trades: {self.metrics['total_trades']}
        Win Rate: {self.metrics['win_rate']:.2f}%
        Total PnL: {self.metrics['total_pnl']:.2f} USDT
        Priority Performance:
        P1: {self.metrics['priority_metrics']['P1']['count']} trades, {self.metrics['priority_metrics']['P1']['pnl']:.2f} USDT
        P2: {self.metrics['priority_metrics']['P2']['count']} trades, {self.metrics['priority_metrics']['P2']['pnl']:.2f} USDT
        P3: {self.metrics['priority_metrics']['P3']['count']} trades, {self.metrics['priority_metrics']['P3']['pnl']:.2f} USDT
        P4: {self.metrics['priority_metrics']['P4']['count']} trades, {self.metrics['priority_metrics']['P4']['pnl']:.2f} USDT
        """

class JamesDirectionalV2ControllerConfig(DirectionalTradingControllerConfigBase):
    """Configuration for the James Directional Strategy ControllerConfig"""
    controller_name: str = "james_directional_v2"
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    #markets: Dict[str, List[str]] = {}
    candles_config: List[CandlesConfig] = []
    #controllers_config: List[str] = []

    # Exchange Configuration
    exchange: str = Field(
        default="binance_perpetual",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter paper trading exchange (binance_perpetual/bybit_perpetual_testnet)"
        )
    )

    # Trading Parameters
    trading_pairs: List[str] = Field(
        default=[
            "AAVE-USDT", "ADS-USDT", "ALGO-USDT", "ANKR-USDT",
            "IOTA-USDT", "JUP-USDT", "WIF-USDT", "DOGE-USDT",
            "SUSHI-USDT", "CAKE-USDT", "PNUT-USDT", "MEME-USDT",
            "MOODENG-USDT", "GOAT-USDT", "CRV-USDT", "BLZ-USDT",
            "BOME-USDT", "DOT-USDT", "FIL-USDT", "GLM-USDT",
            "KAVA-USDT", "UNI-USDT", "SOL-USDT", "EOS-USDT",
            "MANA-USDT", "DYDX-USDT"
        ],
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter comma-separated list of trading pairs"
        )
    )

    paper_trade_account_balance: Decimal = Field(
        default=Decimal("30"),
        gt=0,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter paper trading account balance in USDT (default: 30)"
        )
    )

    max_position_size_usd: Decimal = Field(
        default=Decimal("10"),
        gt=0,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter maximum position size in USDT (default: 10)"
        )
    )

    daily_max_trades: int = Field(
        default=10,
        gt=0,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter maximum number of trades per day (default: 10)"
        )
    )

    order_amount_quote: Decimal = Field(
        default=Decimal("10"),
        gt=0,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter fixed order amount in USDT (default: 10)"
        )
    )

    candles_interval: str = Field(
        default="5m",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter candle interval (default: 5m)"
        )
    )

    # Technical Indicator Parameters
    rsi_length: int = Field(
        default=14,
        gt=0,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter RSI period length (default: 14)"
        )
    )

    stoch_k_length: int = Field(
        default=5,
        gt=0,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter Stochastic %K length (default: 5)"
        )
    )

    stoch_d_length: int = Field(
        default=3,
        gt=0,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter Stochastic %D length (default: 3)"
        )
    )

    # Risk Management
    max_concurrent_trades: int = Field(
        default=2,
        gt=0,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter maximum number of concurrent trades (default: 2)"
        )
    )

    position_mode: PositionMode = Field(
        default=PositionMode.HEDGE,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter position mode (HEDGE/ONEWAY)"
        )
    )

    @validator('position_mode', pre=True)
    def validate_position_mode(cls, v: Union[str, PositionMode]) -> PositionMode:
        if isinstance(v, PositionMode):
            return v
        if isinstance(v, str):
            v = v.upper()
            if v in PositionMode.__members__:
                return PositionMode[v]
        raise ValueError(f"Invalid position mode: {v}. Must be one of {', '.join(PositionMode.__members__.keys())}")

    @validator('trading_pairs', pre=True)
    def validate_trading_pairs(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [pair.strip() for pair in v.split(',') if pair.strip()]
        if isinstance(v, list):
            return [pair.strip() for pair in v if isinstance(pair, str) and pair.strip()]
        raise ValueError("trading_pairs must be a comma-separated string or a list of strings")

class JamesDirectionalV2Strategy(DirectionalTradingControllerBase):
    """James Directional Strategy with Long/Short Trading and Priority System"""

    def get_candles(self, trading_pair: str):
        return self.candles_feed.get_candles(
            connector=self.config.exchange,
            trading_pair=trading_pair,
            interval=self.config.candles_interval
        )

    def __init__(self, config: JamesDirectionalV2ControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.ready_to_trade = False
        self.active_trades = {}

        self.trades_today = 0
        self.last_trade_reset = datetime.now().date()

        # Initialize performance tracking
        self.performance_tracker = PerformanceTracker(self)

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def on_start(self):
        """Strategy initialization"""
        self.logger.info("Starting James Directional Strategy...")

        # Apply strategy settings
        self.apply_initial_settings()

        # Mark as ready to trade
        self.ready_to_trade = True
        self.logger.info("James Directional Strategy is fully initialized and ready to trade.")

    def on_stop(self):
        """Strategy shutdown"""
        self.logger.info("Stopping strategy and closing positions...")
        self.close_all_positions()
        self.performance_tracker.save_report()

    def on_tick(self):
        """Main trading logic"""
        try:
            # Process existing positions first
            self.process_positions()

            # Then look for new opportunities
            self.execute_prioritized_trades()

        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")

    def start_paper_trading(self):
        """Initialize and start paper trading"""
        try:
            self.setup_paper_trading_logging()
            self.apply_initial_settings()
            self.ready_to_trade = True
            self.logger.info("Paper trading started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error starting paper trading: {str(e)}")
            return False

    def setup_paper_trading_logging(self):
        """Setup paper trading specific logging"""
        os.makedirs('logs/paper_trading_logs', exist_ok=True)
        file_handler = logging.FileHandler('logs/paper_trading_logs/trading.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def check_paper_trading_limits(self) -> bool:
        """Check paper trading limits before executing trades"""
        # Reset daily trade count if needed
        if datetime.now().date() > self.last_trade_reset:
            self.trades_today = 0
            self.last_trade_reset = datetime.now().date()

        # Check daily trade limit
        if self.trades_today >= self.config.daily_max_trades:
            self.logger.info("Daily trade limit reached")
            return False

        # Check position size limits
        total_position_value = sum(pos.value for pos in self.get_active_positions())
        if total_position_value >= self.config.max_position_size_usd:
            self.logger.info("Maximum position size reached")
            return False

        return True

    def calculate_indicators(self, candles: pd.DataFrame) -> Dict:
        """Calculate all required technical indicators"""
        try:
            # Check minimum required data points
            if len(candles) < 200:
                raise ValueError("Insufficient data points for indicator calculation (minimum 200 required)")

            indicators = {}

            # Calculate Moving Averages
            for period in [3, 6, 9, 13, 15, 20, 33, 50, 100, 200]:
                try:
                    indicators[f'ma{period}'] = ta.sma(candles['close'], length=period)
                    if period in [3, 5, 15]:
                        indicators[f'ema{period}'] = ta.ema(candles['close'], length=period)
                except Exception as ma_error:
                    raise ValueError(f"Error calculating MA/EMA for period {period}: {str(ma_error)}")

            # Calculate Ichimoku Cloud
            try:
                ichimoku = ta.ichimoku(candles['high'], candles['low'], candles['close'])
                indicators.update({
                    'conversion_line': ichimoku['ITS_9'],
                    'base_line': ichimoku['ITS_26'],
                    'leading_span_a': ichimoku['ITS_A'],
                    'leading_span_b': ichimoku['ITS_B']
                })
            except Exception as ichimoku_error:
                raise ValueError(f"Error calculating Ichimoku: {str(ichimoku_error)}")

            # Calculate RSI
            try:
                indicators['rsi'] = ta.rsi(candles['close'], length=self.config.rsi_length)
                indicators['rsi_ma'] = ta.sma(indicators['rsi'], length=self.config.rsi_length)
            except Exception as rsi_error:
                raise ValueError(f"Error calculating RSI: {str(rsi_error)}")

            # Calculate Stochastic
            try:
                stoch = ta.stoch(
                    candles['high'],
                    candles['low'],
                    candles['close'],
                    k=self.config.stoch_k_length,
                    d=self.config.stoch_d_length
                )
                indicators.update({
                    'stoch_k': stoch[f'STOCHk_{self.config.stoch_k_length}_{self.config.stoch_d_length}'],
                    'stoch_d': stoch[f'STOCHd_{self.config.stoch_k_length}_{self.config.stoch_d_length}']
                })
            except Exception as stoch_error:
                raise ValueError(f"Error calculating Stochastic: {str(stoch_error)}")

            # Calculate MACD
            try:
                macd = ta.macd(candles['close'])
                indicators.update({
                    'macd': macd['MACD_12_26_9'],
                    'macd_signal': macd['MACDs_12_26_9']
                })
            except Exception as macd_error:
                raise ValueError(f"Error calculating MACD: {str(macd_error)}")

            # Calculate Momentum
            try:
                indicators['momentum'] = ta.mom(candles['close'], length=14)
            except Exception as mom_error:
                raise ValueError(f"Error calculating Momentum: {str(mom_error)}")

            # Add close prices and volume
            indicators['close'] = candles['close']
            indicators['volume'] = candles['volume']

            # Validate all required indicators are present
            required_indicators = [
                'ma3', 'ma6', 'ma9', 'ma13', 'ma15', 'ma20', 'ma33', 'ma50', 'ma100', 'ma200',
                'ema3', 'ema5', 'ema15', 'rsi', 'rsi_ma', 'stoch_k', 'stoch_d', 'macd', 'macd_signal',
                'momentum', 'conversion_line', 'base_line', 'leading_span_a', 'leading_span_b'
            ]

            missing_indicators = [ind for ind in required_indicators if ind not in indicators]
            if missing_indicators:
                raise ValueError(f"Missing required indicators: {missing_indicators}")

            return indicators

        except Exception as e:
            self.logger.error(f"Error in calculate_indicators: {str(e)}")
            raise

    def check_long_conditions(self, indicators: Dict) -> Tuple[bool, int]:
        """Check all conditions for long entry"""
        # Condition 1A: All MAs angling up
        condition_1a = all([
            indicators['ma3'][-1] > indicators['ma3'][-2],     # 3MA up
            indicators['ema3'][-1] > indicators['ema3'][-2],   # 3EMA up
            indicators['ma6'][-1] > indicators['ma6'][-2],     # 6MA up
            indicators['ma9'][-1] > indicators['ma9'][-2],     # 9MA up
            indicators['ma13'][-1] > indicators['ma13'][-2],   # 13MA up
            indicators['ema15'][-1] > indicators['ema15'][-2], # 15EMA up
            indicators['ma15'][-1] > indicators['ma15'][-2],   # 15MA up
            indicators['ma33'][-1] > indicators['ma33'][-2],   # 15MA up
            indicators['volume'][-1] > indicators['volume'][-2],# Volume up
            indicators['leading_span_a'][-1] > indicators['leading_span_a'][-2],  # LSA up
            indicators['conversion_line'][-1] > indicators['conversion_line'][-2]  # CL up
        ])

        # Condition 1B: Price and CL conditions
        condition_1b = (
            indicators['close'][-1] > indicators['conversion_line'][-1] and  # Price above CL
            indicators['conversion_line'][-1] > indicators['base_line'][-1]  # CL above BL
        )

        # Condition 1C: MACD and other indicators
        condition_1c = (
            indicators['macd'][-1] > indicators['macd_signal'][-1] and  # MACD crossed above Signal
            all([
                indicators['macd'][-1] > indicators['macd'][-2],        # MACD up
                indicators['stoch_k'][-1] > indicators['stoch_k'][-2],  # %K up
                indicators['rsi'][-1] > indicators['rsi'][-2],          # RSI up
                indicators['momentum'][-1] > indicators['momentum'][-2]  # Momentum up
            ]) and
            indicators['rsi'][-1] > indicators['rsi_ma'][-1]            # RSI above RSI-MA
        )

        # Condition 2A: 6MA conditions
        condition_2a = (
            indicators['ma6'][-1] < indicators['ma3'][-1] and  # 6MA less than 3MA
            indicators['ma6'][-1] < indicators['ema3'][-1]     # 6MA less than 3EMA
        )

        if all([condition_1a, condition_1b, condition_1c, condition_2a]):
            leverage = self.calculate_long_leverage(indicators)
            return True, leverage

        return False, 0

    def check_short_conditions(self, indicators: Dict) -> Tuple[bool, int]:
        """Check conditions for short entry (Condition X)"""
        # Part A: Price and MA conditions
        condition_x_a = all([
            indicators['close'][-1] < indicators['conversion_line'][-1],  # Price below CL
            indicators['conversion_line'][-1] < indicators['base_line'][-1],  # CL below BL
            indicators['ma6'][-1] > indicators['ma3'][-1],  # 6MA above 3MA
            indicators['ma6'][-1] > indicators['ema3'][-1],  # 6MA above 3EMA
            indicators['ma3'][-1] < indicators['ma3'][-2],  # 3MA angling down
            indicators['ema3'][-1] < indicators['ema3'][-2]  # 3EMA angling down
        ])

        # Part B: Moving averages angling down
        condition_x_b = all([
            indicators[ma][-1] < indicators[ma][-2] for ma in [
                'ma20', 'ma15', 'ema15', 'ma13', 'ma9', 'ma6', 'leading_span_a'
            ]
        ])

        # Part C: Technical indicators angling down
        condition_x_c = all([
            indicators['stoch_k'][-1] < indicators['stoch_k'][-2],  # %K down
            indicators['rsi'][-1] < indicators['rsi'][-2],          # RSI down
            indicators['macd'][-1] < indicators['macd'][-2],        # MACD down
            indicators['momentum'][-1] < indicators['momentum'][-2]  # Momentum down
        ])

        if all([condition_x_a, condition_x_b, condition_x_c]):
            leverage = self.calculate_short_leverage(indicators)
            return True, leverage

        return False, 0


    def process_positions(self):
        """Process and manage existing positions"""
        try:
            for connector_pair, trade in list(self.active_trades.items()):  # Use list to avoid modification during iteration
                connector, pair = connector_pair.split('_')

                # Get current indicators
                candles = self.get_candles(connector, pair)
                if len(candles) < 200:  # Ensure enough data for indicators
                    continue

                indicators = self.calculate_indicators(candles)
                current_price = self.get_mid_price(connector, pair)

                # Check exit conditions based on position direction
                exit_triggered = False
                exit_reason = ""

                if trade.direction == "LONG":
                    if self.check_long_exit_conditions(indicators):
                        exit_triggered = True
                        exit_reason = self.determine_long_exit_reason(indicators)
                else:  # SHORT
                    if self.check_short_exit_conditions(indicators):
                        exit_triggered = True
                        exit_reason = self.determine_short_exit_reason(indicators)

                if exit_triggered:
                    self.execute_exit(connector, pair, trade, current_price, exit_reason)

        except Exception as e:
            self.logger.error(f"Error processing positions: {str(e)}")


    def check_long_exit_conditions(self, indicators: Dict) -> bool:
        """Check conditions for exiting long positions"""
        # Base condition (Required)
        base_condition = (
            (indicators['ma3'][-1] < indicators['ma3'][-2] or
             indicators['ema3'][-1] < indicators['ema3'][-2]) and
            indicators['stoch_k'][-1] < indicators['stoch_k'][-2] and
            indicators['rsi'][-1] < indicators['rsi'][-2] and
            indicators['momentum'][-1] < indicators['momentum'][-2] and
            indicators['macd'][-1] < indicators['macd'][-2] and
            indicators['stoch_d'][-1] > indicators['stoch_k'][-1]
        )

        if not base_condition:
            return False

        # Additional exit conditions (2-8)
        additional_conditions = [
            # Condition 2: 6MA crosses
            ((indicators['ma6'][-1] > indicators['ma3'][-1] and
              indicators['ma3'][-1] < indicators['ma3'][-2]) or
             (indicators['ma6'][-1] > indicators['ema3'][-1] and
              indicators['ema3'][-1] < indicators['ema3'][-2])),

            # Condition 3: Price below CL
            indicators['close'][-1] < indicators['conversion_line'][-1],

            # Condition 4: RSI conditions
            (indicators['rsi'][-1] < indicators['rsi_ma'][-1] and
             indicators['ma3'][-1] < indicators['ma3'][-2] and
             indicators['ema3'][-1] < indicators['ema3'][-2]),

            # Condition 5: Stochastic and MA conditions
            (indicators['stoch_k'][-1] < indicators['stoch_k'][-2] and
             indicators['stoch_d'][-1] < indicators['stoch_d'][-2] and
             indicators['ma3'][-1] < indicators['ma3'][-2] and
             indicators['ema3'][-1] < indicators['ema3'][-2]),

            # Condition 6: Combined conditions
            ((indicators['ema5'][-1] < indicators['ema5'][-2] or
              indicators['conversion_line'][-1] < indicators['conversion_line'][-2] or
              indicators['rsi_ma'][-1] < indicators['rsi_ma'][-2]) and
             indicators['ma3'][-1] < indicators['ma3'][-2] and
             indicators['ema3'][-1] < indicators['ema3'][-2] and
             indicators['stoch_k'][-1] < indicators['stoch_k'][-2] and
             indicators['stoch_d'][-1] < indicators['stoch_d'][-2]),

            # Condition 7: Consecutive 3EMA downs
            (indicators['ema3'][-1] < indicators['ema3'][-2] and
             indicators['ema3'][-2] < indicators['ema3'][-3]),

            # Condition 8: MACD and MA conditions
            (indicators['macd'][-1] < indicators['macd'][-2] and
             indicators['ma3'][-1] < indicators['ma3'][-2] and
             indicators['ema3'][-1] < indicators['ema3'][-2])
        ]

        # Return True if base condition AND any additional condition is met
        return base_condition and any(additional_conditions)


    def determine_long_exit_reason(self, indicators: Dict) -> str:
        """Determine which long exit condition was triggered"""
        # Base condition check
        base_condition = (
            (indicators['ma3'][-1] < indicators['ma3'][-2] or
             indicators['ema3'][-1] < indicators['ema3'][-2]) and
            indicators['stoch_k'][-1] < indicators['stoch_k'][-2] and
            indicators['rsi'][-1] < indicators['rsi'][-2] and
            indicators['momentum'][-1] < indicators['momentum'][-2] and
            indicators['macd'][-1] < indicators['macd'][-2] and
            indicators['stoch_d'][-1] > indicators['stoch_k'][-1]
        )

        if not base_condition:
            return "Base condition not met"

        # Check additional conditions
        # Exit Condition 2: 6MA condition
        if ((indicators['ma6'][-1] > indicators['ma3'][-1] and
             indicators['ma3'][-1] < indicators['ma3'][-2]) or
            (indicators['ma6'][-1] > indicators['ema3'][-1] and
             indicators['ema3'][-1] < indicators['ema3'][-2])):
            return "Exit Condition 2: 6MA crosses - Either above 3MA with 3MA down OR above 3EMA with 3EMA down"

        # Exit Condition 3: Price closes below CL
        if indicators['close'][-1] < indicators['conversion_line'][-1]:
            return "Exit Condition 3: Price below CL"

        # Exit Condition 4: RSI conditions
        if (indicators['rsi'][-1] < indicators['rsi_ma'][-1] and
            indicators['ma3'][-1] < indicators['ma3'][-2] and
            indicators['ema3'][-1] < indicators['ema3'][-2]):
            return "Exit Condition 4: RSI below RSI_MA"

        # Exit Condition 5: Stochastic and MA conditions
        if (indicators['stoch_k'][-1] < indicators['stoch_k'][-2] and
            indicators['stoch_d'][-1] < indicators['stoch_d'][-2] and
            indicators['ma3'][-1] < indicators['ma3'][-2] and
            indicators['ema3'][-1] < indicators['ema3'][-2]):
            return "Exit Condition 5: Stochastic and 3MA conditions"

        # Exit Condition 6: Combined conditions
        if ((indicators['ema5'][-1] < indicators['ema5'][-2] or
            indicators['conversion_line'][-1] < indicators['conversion_line'][-2] or
            indicators['rsi_ma'][-1] < indicators['rsi_ma'][-2]) and
            indicators['ma3'][-1] < indicators['ma3'][-2] and
            indicators['ema3'][-1] < indicators['ema3'][-2] and
            indicators['stoch_k'][-1] < indicators['stoch_k'][-2] and
            indicators['stoch_d'][-1] < indicators['stoch_d'][-2]):
            return "Exit Condition 6: Combined (5|CL|RSI_MA) conditions"

        # Exit Condition 7: Consecutive 3EMA downs
        if (indicators['ema3'][-1] < indicators['ema3'][-2] and
            indicators['ema3'][-2] < indicators['ema3'][-3]):
            return "Exit Condition 7: Consecutive 3EMA downs"

        # Exit Condition 8: MACD and MA conditions
        if (indicators['macd'][-1] < indicators['macd'][-2] and
            indicators['ma3'][-1] < indicators['ma3'][-2] and
            indicators['ema3'][-1] < indicators['ema3'][-2]):
            return "Exit Condition 8: MACD and 3MA conditions"

        # If base condition is met but no specific condition identified
        return "Multiple conditions met"

    def check_short_exit_conditions(self, indicators: Dict) -> bool:
        """Check conditions for exiting short positions"""
        # Condition A
        condition_a = all([
            indicators['ema5'][-1] > indicators['ema5'][-2],    # 5EMA up
            indicators['ma3'][-1] > indicators['ma3'][-2],      # 3MA up
            indicators['ema3'][-1] > indicators['ema3'][-2],    # 3EMA up
            indicators['rsi'][-1] > indicators['rsi'][-2],      # RSI up
            indicators['momentum'][-1] > indicators['momentum'][-2]  # Momentum up
        ])

        # Condition B
        condition_b = (
            indicators['close'][-1] > indicators['conversion_line'][-1] and  # Price above CL
            indicators['ma3'][-1] > indicators['ma3'][-2] and               # 3MA up
            indicators['ema3'][-1] > indicators['ema3'][-2]                 # 3EMA up
        )

        # Condition C
        condition_c = all([
            indicators['ma6'][-1] > indicators['ma6'][-2],   # 6MA up
            indicators['ma3'][-1] > indicators['ma3'][-2],   # 3MA up
            indicators['ema3'][-1] > indicators['ema3'][-2]  # 3EMA up
        ])

        # Exit if any condition is met
        return condition_a or condition_b or condition_c


    def determine_short_exit_reason(self, indicators: Dict) -> str:
        """Determine which short exit condition was triggered"""
        # Check Condition A
        condition_a = all([
            indicators['ema5'][-1] > indicators['ema5'][-2],
            indicators['ma3'][-1] > indicators['ma3'][-2],
            indicators['ema3'][-1] > indicators['ema3'][-2],
            indicators['rsi'][-1] > indicators['rsi'][-2],
            indicators['momentum'][-1] > indicators['momentum'][-2]
        ])
        if condition_a:
            return "Exit Condition A: 5EMA, 3MA, 3EMA, RSI, and Momentum all up"


        # Check Condition B
        condition_b = (
            indicators['close'][-1] > indicators['conversion_line'][-1] and
            indicators['ma3'][-1] > indicators['ma3'][-2] and
            indicators['ema3'][-1] > indicators['ema3'][-2]
        )
        if condition_b:
            return "Exit Condition B: Price above CL with 3MA and 3EMA up"

        # Check Condition C
        condition_c = all([
            indicators['ma6'][-1] > indicators['ma6'][-2],
            indicators['ma3'][-1] > indicators['ma3'][-2],
            indicators['ema3'][-1] > indicators['ema3'][-2]
        ])
        if condition_c:
            return "Exit Condition C: 6MA, 3MA, and 3EMA all up"

        return "Unknown exit condition"

    def execute_exit(self, connector: str, trading_pair: str, trade: TradeOpportunity,
                    current_price: Decimal, exit_reason: str):
        """Execute position exit"""
        try:
            # Get position details
            position = self.get_position(connector, trading_pair)
            if not position or position.amount == 0:
                return

            # Execute exit order
            order_id = self.place_order(
                connector_name=connector,
                trading_pair=trading_pair,
                order_type=OrderType.MARKET,
                side=TradeType.SELL if trade.direction == "LONG" else TradeType.BUY,
                amount=abs(position.amount),
                price=current_price
            )

            # Update trade metrics
            trade_key = f"{connector}_{trading_pair}"
            if trade_key in self.active_trades:
                self.performance_tracker.update_trade(
                    trade_key,
                    float(current_price),
                    datetime.now(),
                    exit_reason
                )

                # Remove from active trades
                del self.active_trades[trade_key]

            self.logger.info(
                f"Exited {trade.direction} position for {trading_pair} at {current_price}. "
                f"Reason: {exit_reason}"
            )

        except Exception as e:
            self.logger.error(f"Error executing exit for {trading_pair}: {str(e)}")


    def calculate_long_leverage(self, indicators: Dict) -> int:
        """Calculate leverage for long positions"""
        volume_ratio = indicators['volume'][-1] / indicators['volume'][-2]

        # Check for 30x leverage conditions
        if all([
            indicators['ma33'][-1] > indicators['ma33'][-2],      # 33MA up
            indicators['ma100'][-1] > indicators['ma100'][-2],    # 100MA up
            indicators['conversion_line'][-1] > indicators['ma9'][-1],  # CL > 9MA
            indicators['ma200'][-1] >= indicators['ma200'][-2],   # 200MA flat or up
            volume_ratio >= 5,                                     # Volume 5x
            self.check_complete_energy_alignment(indicators)       # Complete alignment
        ]):
            return 30

        # Check for 20x leverage conditions
        if all([
            indicators['ma33'][-1] > indicators['ma33'][-2],      # 33MA up
            indicators['ma100'][-1] > indicators['ma100'][-2],    # 100MA up
            indicators['conversion_line'][-1] > indicators['ma9'][-1],  # CL > 9MA
            indicators['ma200'][-1] >= indicators['ma200'][-2],   # 200MA flat or up
            volume_ratio >= 2.4,                                   # Volume 2.4x
            self.check_partial_energy_alignment(indicators)        # Partial alignment
        ]):
            return 20

        # Check for 10x leverage conditions
        if all([
            indicators['ma33'][-1] > indicators['ma33'][-2],      # 33MA up
            indicators['conversion_line'][-1] > indicators['ma9'][-1],  # CL > 9MA
            indicators['ma100'][-1] >= indicators['ma100'][-2],   # 100MA flat or up
            volume_ratio >= 1.4                                    # Volume 1.4x
        ]):
            return 10

        return 5

    def calculate_short_leverage(self, indicators: Dict) -> int:
        """Calculate leverage for short positions"""
        volume_ratio = indicators['volume'][-1] / indicators['volume'][-2]

        # 30x leverage conditions
        if all([
            indicators['conversion_line'][-1] < indicators['conversion_line'][-2],  # CL down
            indicators['stoch_d'][-1] < indicators['stoch_d'][-2],                 # %D down
            indicators['ma100'][-1] < indicators['ma100'][-2],                     # 100MA down
            indicators['ma200'][-1] < indicators['ma200'][-2],                     # 200MA down
            indicators['close'][-1] < indicators['ma20'][-1],                      # Price below 20MA
            indicators['ma20'][-1] < indicators['ma100'][-1],                      # 20MA < 100MA
            indicators['ma100'][-1] < indicators['ma200'][-1],                     # 100MA < 200MA
            indicators['base_line'][-1] < indicators['base_line'][-2],             # BL down
            indicators['leading_span_b'][-1] < indicators['leading_span_b'][-2],   # LSB down
            volume_ratio >= 5                                                      # Volume 5x
        ]):
            return 30

        # 20x leverage conditions
        if all([
            indicators['conversion_line'][-1] < indicators['conversion_line'][-2],  # CL down
            indicators['stoch_d'][-1] < indicators['stoch_d'][-2],                 # %D down
            indicators['ma100'][-1] < indicators['ma100'][-2],                     # 100MA down
            indicators['ma200'][-1] < indicators['ma200'][-2],                     # 200MA down
            indicators['close'][-1] < indicators['ma20'][-1],                      # Price below 20MA
            indicators['ma20'][-1] < indicators['ma100'][-1],                      # 20MA < 100MA
            indicators['ma100'][-1] < indicators['ma200'][-1],                     # 100MA < 200MA
            volume_ratio >= 2.4                                                    # Volume 2.4x
        ]):
            return 20

        # 10x leverage conditions
        if all([
            indicators['conversion_line'][-1] < indicators['conversion_line'][-2],  # CL down
            indicators['stoch_d'][-1] < indicators['stoch_d'][-2],                 # %D down
            volume_ratio >= 1.4                                                    # Volume 1.4x
        ]):
            return 10

        return 5

    def check_complete_energy_alignment(self, indicators: Dict) -> bool:
        """Check for complete energy alignment (Condition 3B)"""
        return (
            indicators['close'][-1] > indicators['ma9'][-1] >
            indicators['ema15'][-1] > indicators['ma20'][-1] >
            indicators['ma33'][-1] > indicators['ma50'][-1]
        )

    def check_partial_energy_alignment(self, indicators: Dict) -> bool:
        """Check for partial energy alignment (Condition 3A)"""
        return (
            indicators['close'][-1] > indicators['ma9'][-1] >
            indicators['ema15'][-1] > indicators['ma20'][-1] >
            indicators['ma33'][-1]
        )

    def execute_prioritized_trades(self):
        """Execute trades based on priority system"""
        available_slots = self.config.max_concurrent_trades - len(self.active_trades)
        if available_slots <= 0:
            return

        opportunities = self.scan_opportunities()
        if not opportunities:
            return

        for opportunity in opportunities[:available_slots]:
            try:
                self.execute_trade(opportunity)
            except Exception as e:
                self.logger.error(f"Error executing trade: {str(e)}")

    def scan_opportunities(self) -> List[TradeOpportunity]:
        """Scan all pairs for trading opportunities and prioritize them"""
        opportunities = []
        for trading_pair in self.config.trading_pairs:
            # Skip if in active trade
            if f"{self.config.exchange}_{trading_pair}" in self.active_trades:
                continue

            candles = self.get_candles(trading_pair)
            if len(candles) < 200:
                continue

            indicators = self.calculate_indicators(candles)

            # Check long opportunities first
            long_valid, long_leverage = self.check_long_conditions(indicators)
            if long_valid:
                opportunities.append(TradeOpportunity(
                    connector=self.config.exchange,
                    trading_pair=trading_pair,
                    direction="LONG",
                    leverage=long_leverage,
                    priority=self.get_leverage_priority(long_leverage)
                ))

            # Then check short opportunities
            short_valid, short_leverage = self.check_short_conditions(indicators)
            if short_valid:
                opportunities.append(TradeOpportunity(
                    connector=self.config.exchange,
                    trading_pair=trading_pair,
                    direction="SHORT",
                    leverage=short_leverage,
                    priority=self.get_leverage_priority(short_leverage)
                ))

        # Sort by priority_score (prioritizes higher leverage and longs over shorts)
        opportunities.sort(key=lambda x: x.priority_score)
        return opportunities

    def execute_trade(self, opportunity: TradeOpportunity) -> Optional[str]:
        """Execute trade with position sizing and leverage"""
        try:
            # Check paper trading limits first
            if not self.check_paper_trading_limits():
                return None

            current_price = self.get_mid_price(opportunity.connector, opportunity.trading_pair)
            position_size = self.config.order_amount_quote / current_price

            # Additional check for position size
            if position_size <= 0:
                self.logger.warning(f"Invalid position size for {opportunity.trading_pair}")
                return None

            # Set leverage
            self.set_leverage(opportunity.connector, opportunity.trading_pair, opportunity.leverage)

            # Execute order
            order_id = self.place_order(
                connector_name=opportunity.connector,
                trading_pair=opportunity.trading_pair,
                order_type=OrderType.MARKET,
                side=TradeType.BUY if opportunity.direction == "LONG" else TradeType.SELL,
                amount=position_size,
                price=current_price
            )

            # Track trade
            self.trades_today += 1
            trade_metrics = TradeMetrics(
                trading_pair=opportunity.trading_pair,
                direction=opportunity.direction,
                entry_time=datetime.now(),
                exit_time=None,
                entry_price=current_price,
                exit_price=None,
                amount=float(position_size),
                leverage=opportunity.leverage,
                pnl=None,
                pnl_percentage=None,
                holding_period=None,
                priority_level=opportunity.priority
            )

            self.performance_tracker.track_trade(trade_metrics)
            self.active_trades[f"{opportunity.connector}_{opportunity.trading_pair}"] = opportunity

            self.logger.info(
                f"Executed {opportunity.direction} trade for {opportunity.trading_pair} "
                f"with {opportunity.leverage}x leverage"
            )

        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return None

    def get_leverage_priority(self, leverage: int) -> int:
        """Convert leverage to priority level"""
        if leverage >= 30:
            return 1  # P1
        elif leverage >= 20:
            return 2  # P2
        elif leverage >= 10:
            return 3  # P3
        return 4     # P4

    def format_status(self) -> str:
        """Format strategy status with paper trading information"""
        if not self.ready_to_trade:
            return "Strategy not ready for trading."

        lines = []

        # Add paper trading status
        lines.extend([
            "\n===== Paper Trading Status =====",
            f"Exchange: {self.config.exchange}",
            f"Trades Today: {self.trades_today}/{self.config.daily_max_trades}",
            f"Active Positions: {len(self.active_trades)}",
        ])

        # Add balance information with error handling
        try:
            balance_df = self.get_balance_df()
            lines.extend(["", "  Balances:"] +
                        ["    " + line for line in balance_df.to_string(index=False).split("\n")])
        except Exception as balance_error:
            lines.extend([
                "",
                "  Balance Retrieval Error:",
                f"    Unable to fetch balance: {str(balance_error)}",
                "    Please check exchange connection."
            ])

        # Add active positions with error handling
        try:
            active_positions_found = False
            lines.extend(["", "  Active Positions:"])
            for connector_pair, trade in self.active_trades.items():
                try:
                    connector, pair = connector_pair.split('_')
                    position = self.get_position(connector, pair)
                    if position:
                        active_positions_found = True
                        lines.append(
                            f"    {pair}: {position.amount:.4f} @ {position.entry_price:.4f} "
                            f"({trade.direction}, {trade.leverage}x)"
                        )
                except Exception as position_error:
                    lines.append(
                        f"    Error retrieving position for {connector_pair}: {str(position_error)}"
                    )

            if not active_positions_found:
                lines.append("    No active positions.")

        except Exception as positions_error:
            lines.extend([
                "",
                "  Positions Retrieval Error:",
                f"    Unable to fetch positions: {str(positions_error)}",
                "    Please check exchange connection."
            ])

        # Add performance metrics with error handling
        try:
            if hasattr(self, 'performance_tracker'):
                metrics = self.performance_tracker.get_metrics()
                lines.extend([
                    "",
                    "  Performance Metrics:",
                    f"    Total Trades: {metrics.get('total_trades', 0)}",
                    f"    Win Rate: {metrics.get('win_rate', 0):.2f}%",
                    f"    Current PnL: {metrics.get('current_pnl', 0):.2f} USDT",
                    "",
                    "  Priority Performance:",
                    f"    P1 Trades: {metrics.get('priority_metrics', {}).get('P1', {}).get('count', 0)}",
                    f"    P2 Trades: {metrics.get('priority_metrics', {}).get('P2', {}).get('count', 0)}",
                    f"    P3 Trades: {metrics.get('priority_metrics', {}).get('P3', {}).get('count', 0)}",
                    f"    P4 Trades: {metrics.get('priority_metrics', {}).get('P4', {}).get('count', 0)}"
                ])
        except Exception as metrics_error:
            lines.extend([
                "",
                "  Metrics Retrieval Error:",
                f"    Unable to fetch performance metrics: {str(metrics_error)}",
                "    Please check tracking system."
            ])

        return "\n".join(lines)

    def get_paper_balance(self) -> Optional[Decimal]:
        """Get current paper trading balance"""
        try:
            balance = self.get_balance(self.config.exchange, "USDT")
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching paper balance: {str(e)}")
            return None


class BacktestingFramework:
    def __init__(self, strategy: JamesDirectionalV2Strategy, config: JamesDirectionalV2ControllerConfig):
        self.strategy = strategy
        self.config = config
        self.results_dir = "backtest_results"
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.results_dir, exist_ok=True)
        self.reset_metrics()
        self.trades = []
        self.active_trades = {}

    def reset_metrics(self):
        """Initialize backtest metrics"""
        self.trades = []
        self.equity_curve = []
        self.metrics = {
            'long_trades': {
                'total': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0
            },
            'short_trades': {
                'total': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0
            },
            'priority_performance': {
                'P1': {'count': 0, 'pnl': 0.0, 'win_rate': 0.0},
                'P2': {'count': 0, 'pnl': 0.0, 'win_rate': 0.0},
                'P3': {'count': 0, 'pnl': 0.0, 'win_rate': 0.0},
                'P4': {'count': 0, 'pnl': 0.0, 'win_rate': 0.0}
            }
        }

    async def run_backtest(self, start_date: datetime, end_date: datetime):
        """Run backtest over specified period"""
        self.start_date = start_date
        self.end_date = end_date
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")

        # Fetch historical data for all pairs
        data = {}
        for pair in self.config.trading_pairs:
            data[pair] = await self.fetch_historical_data(pair, start_date, end_date)

        # Run simulation
        current_time = start_date
        while current_time <= end_date:
            await self.process_timestamp(current_time, data)
            current_time += timedelta(minutes=5)

        # Calculate final metrics
        self.calculate_final_metrics()
        self.generate_report()
        self.save_results()

    def calculate_final_metrics(self):
        """Calculate comprehensive performance metrics"""
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        self.metrics['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else float('inf')

        # Calculate average holding period
        holding_periods = [t.holding_period.total_seconds() / 3600 for t in self.trades if t.holding_period]
        self.metrics['avg_holding_period_hours'] = np.mean(holding_periods) if holding_periods else 0

        # Calculate maximum drawdown
        self.metrics['max_drawdown'] = self.calculate_max_drawdown()

        # Calculate Sharpe Ratio
        returns = pd.Series([t.pnl_percentage for t in self.trades if t.pnl_percentage is not None])
        self.metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        equity = pd.Series([point['equity'] for point in self.equity_curve])
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        return float(drawdown.min())

    def generate_visualizations(self):
        """Generate comprehensive performance visualizations"""
        self.plot_equity_curve()
        self.plot_drawdown()
        self.plot_trade_distribution()
        self.plot_priority_performance()
        self.plot_monthly_performance()

    def plot_equity_curve(self):
        """Plot equity curve with drawdown"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        equity_data = pd.DataFrame(self.equity_curve)

        # Equity curve
        ax1.plot(equity_data['timestamp'], equity_data['equity'], label='Equity')
        ax1.set_title('Strategy Equity Curve')
        ax1.grid(True)
        ax1.legend()

        # Drawdown
        drawdown = self.calculate_drawdown_series(equity_data['equity'])
        ax2.fill_between(equity_data['timestamp'], drawdown, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/equity_curve.png")
        plt.close()

    def plot_trade_distribution(self):
        """Plot trade distribution by direction and priority"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # PnL distribution by direction
        trades_df = pd.DataFrame([{
            'PnL': t.pnl,
            'Direction': t.direction
        } for t in self.trades if t.pnl is not None])

        sns.histplot(data=trades_df, x='PnL', hue='Direction', bins=50, ax=ax1)
        ax1.set_title('PnL Distribution by Direction')

        # Performance by priority
        priority_data = pd.DataFrame(self.metrics['priority_performance']).T
        priority_data['win_rate'].plot(kind='bar', ax=ax2)
        ax2.set_title('Win Rate by Priority Level')

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/trade_distribution.png")
        plt.close()

    def generate_report(self) -> str:
        """Generate detailed performance report"""
        report = [
            "=== Performance Report ===",
            f"Backtest Period: {self.start_date} to {self.end_date}",
            "",
            "Overall Performance:",
            f"Total Trades: {len(self.trades)}",
            f"Profit Factor: {self.metrics['profit_factor']:.2f}",
            f"Maximum Drawdown: {self.metrics['max_drawdown']:.2%}",
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}",
            f"Average Holding Period: {self.metrics['avg_holding_period_hours']:.1f} hours",
            "",
            "Long Performance:",
            f"Win Rate: {self.metrics['long_trades']['win_rate']:.2%}",
            f"Average Profit: {self.metrics['long_trades']['avg_profit']:.2%}",
            f"Average Loss: {self.metrics['long_trades']['avg_loss']:.2%}",
            "",
            "Short Performance:",
            f"Win Rate: {self.metrics['short_trades']['win_rate']:.2%}",
            f"Average Profit: {self.metrics['short_trades']['avg_profit']:.2%}",
            f"Average Loss: {self.metrics['short_trades']['avg_loss']:.2%}",
            "",
            "Priority Performance:",
        ]

        for priority in ['P1', 'P2', 'P3', 'P4']:
            perf = self.metrics['priority_performance'][priority]
            report.extend([
                f"{priority}:",
                f"  Count: {perf['count']}",
                f"  Win Rate: {perf['win_rate']:.2%}",
                f"  Total PnL: {perf['pnl']:.2f}",
            ])

        return "\n".join(report)

    def save_results(self):
        """Save backtest results and generate report"""
        # Save metrics to JSON
        with open(f"{self.results_dir}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=4, default=str)

        # Save trade history
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'trading_pair': t.trading_pair,
            'direction': t.direction,
            'leverage': t.leverage,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'priority_level': t.priority_level
        } for t in self.trades])
        trades_df.to_csv(f"{self.results_dir}/trade_history.csv", index=False)

        # Save performance report
        report = self.generate_report()
        with open(f"{self.results_dir}/performance_report.txt", 'w') as f:
            f.write(report)


def validate_config(config: JamesDirectionalV2ControllerConfig) -> Tuple[bool, str]:
    """Validate strategy configuration"""
    try:
        # Validate exchange
        if not config.exchange.endswith(('_perpetual', '_testnet')):
            return False, "Exchange must be a paper trading venue (use binance_paper_trade or bybit_perpetual_testnet)"

        # Validate trading pairs
        for pair in config.trading_pairs:
            if not pair.endswith('-USDT'):
                return False, f"Invalid trading pair format: {pair}. Must end with -USDT"

        # Validate amounts
        if config.order_amount_quote < Decimal("10"):
            return False, "Minimum order amount must be at least 10 USDT"

        if config.paper_trade_account_balance < config.order_amount_quote:
            return False, "Paper trading balance must be greater than order amount"

        return True, "Configuration validated successfully"
    except Exception as e:
        return False, f"Configuration validation error: {str(e)}"

def test_exchange_connection(strategy: JamesDirectionalV2Strategy) -> Tuple[bool, str]:
    """Test connection to exchange"""
    try:
        # Test market data access
        for pair in strategy.config.trading_pairs:
            price = strategy.get_mid_price(strategy.config.exchange, pair)
            if price is None or price == Decimal('0'):
                return False, f"Unable to fetch price for {pair}"

        # Test account data access
        balance = strategy.get_paper_balance()
        if balance is None:
            return False, "Unable to fetch paper trading balance"

        return True, "Exchange connection successful"
    except Exception as e:
        return False, f"Exchange connection error: {str(e)}"

def initialize_logging():
    """Initialize logging configuration"""
    log_dir = "logs/paper_trading_logs"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/strategy_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def print_startup_summary(strategy: JamesDirectionalV2Strategy):
    """Print strategy startup summary"""
    print("\n=== Strategy Startup Summary ===")
    print(f"Exchange: {strategy.config.exchange}")
    print(f"Trading Pairs: {', '.join(strategy.config.trading_pairs)}")
    print(f"Initial Balance: {strategy.get_paper_balance()} USDT")
    print(f"Order Amount: {strategy.config.order_amount_quote} USDT")
    print(f"Max Daily Trades: {strategy.config.daily_max_trades}")
    print(f"Position Mode: {strategy.config.position_mode}")
    print("=============================\n")

if __name__ == "__main__":
    try:
        # Initialize logging
        logger = initialize_logging()
        logger.info("Starting strategy initialization...")

        # Create and validate configuration
        config = JamesDirectionalV2ControllerConfig()
        valid_config, config_msg = validate_config(config)
        if not valid_config:
            raise ValueError(f"Configuration error: {config_msg}")
        logger.info("Configuration validated successfully")

        # Initialize strategy
        strategy = JamesDirectionalV2Strategy(config)
        logger.info("Strategy instance created")

        # Test exchange connection
        connected, conn_msg = test_exchange_connection(strategy)
        if not connected:
            raise ConnectionError(f"Exchange connection error: {conn_msg}")
        logger.info("Exchange connection verified")

        # Start strategy (IMPORTANT: This was missing)
        strategy.on_start()

        # Start paper trading
        if strategy.start_paper_trading():
            print_startup_summary(strategy)
            logger.info("Paper trading started successfully")

            # Run strategy
            strategy.run()
        else:
            raise RuntimeError("Failed to start paper trading")

    except Exception as e:
        logger.error(f"Strategy initialization failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
