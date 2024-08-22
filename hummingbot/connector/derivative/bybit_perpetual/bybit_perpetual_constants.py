from hummingbot.core.data_type.common import OrderType, PositionMode
from hummingbot.core.data_type.in_flight_order import OrderState

EXCHANGE_NAME = "bybit_perpetual"

DEFAULT_DOMAIN = "bybit_perpetual_main"

DEFAULT_TIME_IN_FORCE = "GoodTillCancel"

REST_URLS = {"bybit_perpetual_main": "https://api.bybit.com/",
             "bybit_perpetual_testnet": "https://api-testnet.bybit.com/"}
WSS_NON_LINEAR_PUBLIC_URLS = {"bybit_perpetual_main": "wss://stream.bybit.com/realtime",
                              "bybit_perpetual_testnet": "wss://stream-testnet.bybit.com/realtime"}
WSS_NON_LINEAR_PRIVATE_URLS = WSS_NON_LINEAR_PUBLIC_URLS
WSS_LINEAR_PUBLIC_URLS = {"bybit_perpetual_main": "wss://stream.bybit.com/realtime_public",
                          "bybit_perpetual_testnet": "wss://stream-testnet.bybit.com/realtime_public"}
WSS_LINEAR_PRIVATE_URLS = {"bybit_perpetual_main": "wss://stream.bybit.com/realtime_private",
                           "bybit_perpetual_testnet": "wss://stream-testnet.bybit.com/realtime_private"}


# unit in millisecond and default value is 5,000) to specify how long an HTTP request is valid.
# It is also used to prevent replay attacks.
# https://bybit-exchange.github.io/docs/v5/guide#parameters-for-authenticated-endpoints
X_API_RECV_WINDOW = str(50000)

X_API_SIGN_TYPE = str(2)


REST_API_VERSION = "v2"

HBOT_BROKER_ID = "Hummingbot"

MAX_ID_LEN = 36
SECONDS_TO_WAIT_TO_RECEIVE_MESSAGE = 30
POSITION_IDX_ONEWAY = 0
POSITION_IDX_HEDGE_BUY = 1
POSITION_IDX_HEDGE_SELL = 2

ORDER_TYPE_MAP = {
    OrderType.LIMIT: "Limit",
    OrderType.MARKET: "Market",
}

POSITION_MODE_API_ONEWAY = 0
POSITION_MODE_API_HEDGE = 3
POSITION_MODE_MAP = {
    PositionMode.ONEWAY: POSITION_MODE_API_ONEWAY,
    PositionMode.HEDGE: POSITION_MODE_API_HEDGE,
}

# REST API Public Endpoints
LINEAR_MARKET = "linear"
NON_LINEAR_MARKET = "non_linear"

LATEST_SYMBOL_INFORMATION_ENDPOINT = {
    LINEAR_MARKET: f"{REST_API_VERSION}/public/tickers",
    NON_LINEAR_MARKET: f"{REST_API_VERSION}/public/tickers"}
QUERY_SYMBOL_ENDPOINT = {
    LINEAR_MARKET: "/v5/market/instruments-info",
    NON_LINEAR_MARKET: "/v5/market/instruments-info"}
ORDER_BOOK_ENDPOINT = {
    LINEAR_MARKET: f"{REST_API_VERSION}/public/orderBook/L2",
    NON_LINEAR_MARKET: f"{REST_API_VERSION}/public/orderBook/L2"}
SERVER_TIME_PATH_URL = {
    LINEAR_MARKET: "/v5/market/time",
    NON_LINEAR_MARKET: "/v5/market/time"
}

# REST API Private Endpoints
SET_LEVERAGE_PATH_URL = {
    LINEAR_MARKET: "v5/position/set-leverage",
    NON_LINEAR_MARKET: "v5/position/set-leverage"}
GET_LAST_FUNDING_RATE_PATH_URL = {
    LINEAR_MARKET: "/v5/account/transaction-log",
    NON_LINEAR_MARKET: "/v5/account/contract-transaction-log"}
GET_PREDICTED_FUNDING_RATE_PATH_URL = {
    LINEAR_MARKET: "/private/linear/funding/predicted-funding",
    NON_LINEAR_MARKET: f"{REST_API_VERSION}/private/funding/predicted-funding"
}
GET_POSITIONS_PATH_URL = {
    LINEAR_MARKET: "private/linear/position/list",
    NON_LINEAR_MARKET: f"{REST_API_VERSION}/private/position/list"}
PLACE_ACTIVE_ORDER_PATH_URL = {
    LINEAR_MARKET: "private/linear/order/create",
    NON_LINEAR_MARKET: f"{REST_API_VERSION}/private/order/create"}
CANCEL_ACTIVE_ORDER_PATH_URL = {
    LINEAR_MARKET: "private/linear/order/cancel",
    NON_LINEAR_MARKET: f"{REST_API_VERSION}/private/order/cancel"}
CANCEL_ALL_ACTIVE_ORDERS_PATH_URL = {
    LINEAR_MARKET: "private/linear/order/cancelAll",
    NON_LINEAR_MARKET: f"{REST_API_VERSION}/private/order/cancelAll"}
QUERY_ACTIVE_ORDER_PATH_URL = {
    LINEAR_MARKET: "private/linear/order/search",
    NON_LINEAR_MARKET: f"{REST_API_VERSION}/private/order"}
USER_TRADE_RECORDS_PATH_URL = {
    LINEAR_MARKET: "v5/execution/list",
    NON_LINEAR_MARKET: "v5/execution/list"}
GET_WALLET_BALANCE_PATH_URL = {
    LINEAR_MARKET: "/v5/account/wallet-balance",
    NON_LINEAR_MARKET: "/v5/account/wallet-balance"}
SET_POSITION_MODE_URL = {
    LINEAR_MARKET: "/v5/position/switch-mode"}

# Funding Settlement Time Span
FUNDING_SETTLEMENT_DURATION = (5, 5)  # seconds before snapshot, seconds after snapshot

# WebSocket Public Endpoints
WS_PING_REQUEST = "ping"
WS_ORDER_BOOK_EVENTS_TOPIC = "orderBook_200.100ms"
WS_TRADES_TOPIC = "trade"
WS_INSTRUMENTS_INFO_TOPIC = "instrument_info.100ms"
WS_AUTHENTICATE_USER_ENDPOINT_NAME = "auth"
WS_SUBSCRIPTION_POSITIONS_ENDPOINT_NAME = "position"
WS_SUBSCRIPTION_ORDERS_ENDPOINT_NAME = "order"
WS_SUBSCRIPTION_EXECUTIONS_ENDPOINT_NAME = "execution"
WS_SUBSCRIPTION_WALLET_ENDPOINT_NAME = "wallet"

# Order Statuses
ORDER_STATE = {
    "Created": OrderState.OPEN,
    "New": OrderState.OPEN,
    "Filled": OrderState.FILLED,
    "PartiallyFilled": OrderState.PARTIALLY_FILLED,
    "Cancelled": OrderState.CANCELED,
    "PendingCancel": OrderState.PENDING_CANCEL,
    "Rejected": OrderState.FAILED,
}

GET_LIMIT_ID = "GETLimit"
POST_LIMIT_ID = "POSTLimit"
GET_RATE = 49  # per second
POST_RATE = 19  # per second

NON_LINEAR_PRIVATE_BUCKET_100_LIMIT_ID = "NonLinearPrivateBucket100"
NON_LINEAR_PRIVATE_BUCKET_600_LIMIT_ID = "NonLinearPrivateBucket600"
NON_LINEAR_PRIVATE_BUCKET_75_LIMIT_ID = "NonLinearPrivateBucket75"
NON_LINEAR_PRIVATE_BUCKET_120_B_LIMIT_ID = "NonLinearPrivateBucket120B"
NON_LINEAR_PRIVATE_BUCKET_120_C_LIMIT_ID = "NonLinearPrivateBucket120C"

LINEAR_PRIVATE_BUCKET_100_LIMIT_ID = "LinearPrivateBucket100"
LINEAR_PRIVATE_BUCKET_600_LIMIT_ID = "LinearPrivateBucket600"
LINEAR_PRIVATE_BUCKET_75_LIMIT_ID = "LinearPrivateBucket75"
LINEAR_PRIVATE_BUCKET_120_A_LIMIT_ID = "LinearPrivateBucket120A"

# Request error codes
RET_CODE_OK = 0

RET_CODE_MODE_POSITION_NOT_EMPTY = 110024
RET_CODE_MODE_NOT_MODIFIED = 110025
RET_CODE_MODE_ORDER_NOT_EMPTY = 110028
RET_CODE_HEDGE_NOT_SUPPORTED = 110029

RET_CODE_LEVERAGE_NOT_MODIFIED = 110043

RET_CODE_PARAMS_ERROR = 10001
RET_CODE_API_KEY_INVALID = 10003
RET_CODE_AUTH_TIMESTAMP_ERROR = 10021
RET_CODE_ORDER_NOT_EXISTS = 20001
RET_CODE_API_KEY_EXPIRED = 33004
RET_CODE_POSITION_ZERO = 130125
