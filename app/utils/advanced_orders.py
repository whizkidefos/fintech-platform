from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List
from datetime import datetime
from enum import Enum

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    OCO = "OCO"
    BRACKET = "BRACKET"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"

class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date

@dataclass
class OrderParams:
    symbol: str
    side: str
    order_type: OrderType
    size: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    expire_time: Optional[datetime] = None
    trailing_percent: Optional[Decimal] = None
    iceberg_qty: Optional[Decimal] = None
    min_qty: Optional[Decimal] = None
    
class AdvancedOrderExecutor:
    def __init__(self, market_data, risk_manager):
        self.market_data = market_data
        self.risk_manager = risk_manager
        
    async def execute_trailing_stop(self, params: OrderParams):
        """Execute a trailing stop order"""
        current_price = await self.market_data.get_latest_price(params.symbol)
        trail_amount = current_price * (params.trailing_percent / Decimal('100'))
        
        if params.side == 'SELL':
            trigger_price = current_price - trail_amount
        else:
            trigger_price = current_price + trail_amount
            
        return {
            'type': 'TRAILING_STOP',
            'trigger_price': trigger_price,
            'trail_amount': trail_amount
        }
        
    async def execute_iceberg_order(self, params: OrderParams):
        """Execute an iceberg order (large order split into smaller ones)"""
        remaining_qty = params.size
        visible_qty = params.iceberg_qty or (params.size / Decimal('10'))
        orders = []
        
        while remaining_qty > 0:
            qty = min(visible_qty, remaining_qty)
            order = {
                'type': 'LIMIT',
                'price': params.price,
                'quantity': qty
            }
            orders.append(order)
            remaining_qty -= qty
            
        return {
            'type': 'ICEBERG',
            'total_quantity': params.size,
            'visible_quantity': visible_qty,
            'orders': orders
        }
        
    async def execute_twap_order(self, params: OrderParams):
        """Execute a Time-Weighted Average Price order"""
        if not params.expire_time:
            raise ValueError("TWAP orders require an expiration time")
            
        now = datetime.utcnow()
        time_intervals = (params.expire_time - now).total_seconds() / 3600  # hours
        qty_per_order = params.size / Decimal(str(time_intervals))
        
        return {
            'type': 'TWAP',
            'interval_quantity': qty_per_order,
            'intervals': int(time_intervals),
            'end_time': params.expire_time
        }
        
    async def execute_bracket_order(self, params: OrderParams, take_profit: Decimal, stop_loss: Decimal):
        """Execute a bracket order (entry + take profit + stop loss)"""
        orders = []
        
        # Main entry order
        entry_order = {
            'type': params.order_type.value,
            'price': params.price,
            'quantity': params.size
        }
        orders.append(entry_order)
        
        # Take profit order
        tp_order = {
            'type': 'LIMIT',
            'price': take_profit,
            'quantity': params.size,
            'side': 'SELL' if params.side == 'BUY' else 'BUY'
        }
        orders.append(tp_order)
        
        # Stop loss order
        sl_order = {
            'type': 'STOP',
            'price': stop_loss,
            'quantity': params.size,
            'side': 'SELL' if params.side == 'BUY' else 'BUY'
        }
        orders.append(sl_order)
        
        return {
            'type': 'BRACKET',
            'orders': orders,
            'take_profit': take_profit,
            'stop_loss': stop_loss
        }
        
    async def execute_vwap_order(self, params: OrderParams):
        """Execute a Volume-Weighted Average Price order"""
        volume_profile = await self.market_data.get_volume_profile(params.symbol)
        total_volume = sum(v for _, v in volume_profile)
        orders = []
        
        remaining_qty = params.size
        for price, volume in volume_profile:
            volume_percent = Decimal(str(volume)) / Decimal(str(total_volume))
            qty = params.size * volume_percent
            
            if qty > remaining_qty:
                qty = remaining_qty
                
            order = {
                'type': 'LIMIT',
                'price': Decimal(str(price)),
                'quantity': qty
            }
            orders.append(order)
            remaining_qty -= qty
            
            if remaining_qty <= 0:
                break
                
        return {
            'type': 'VWAP',
            'orders': orders
        }

    def validate_order_params(self, params: OrderParams) -> bool:
        """Validate order parameters"""
        if params.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not params.price:
            return False
            
        if params.order_type == OrderType.STOP and not params.stop_price:
            return False
            
        if params.order_type == OrderType.TRAILING_STOP and not params.trailing_percent:
            return False
            
        if params.time_in_force == TimeInForce.GTD and not params.expire_time:
            return False
            
        return True
