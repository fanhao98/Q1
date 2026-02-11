
"""
Tushare量化交易系统后端
需要安装: pip install tushare flask flask-cors pandas numpy
"""

import tushare as ts
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import hashlib
import io
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.parse
import urllib.error
import xgboost as xgb
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: TensorFlow或scikit-learn未安装，LSTM功能将不可用。安装命令: pip install tensorflow scikit-learn")

app = Flask(__name__)

# 请求日志中间件
@app.before_request
def log_request():
    print(f"[REQUEST] {request.method} {request.path}", flush=True)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                r"^https?://localhost(:\d+)?$",
                r"^https?://127\.0\.0\.1(:\d+)?$",
                r"^https?://.*\.vercel\.app$",
                "null",
            ]
        }
    },
    supports_credentials=True,
)

VERBOSE = (os.getenv('QUANT_VERBOSE') or '0').strip() == '1'

def _log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

_RUNTIME_STATS_LOCK = threading.Lock()
_RUNTIME_STATS = {
    'stock_data_requests': 0,
    'stock_data_lite_requests': 0,
    'stock_pool_quotes_requests': 0,
    'local_state_writes': 0,
    'local_state_skipped_writes': 0,
    'cache_df_mem_hits': 0,
    'cache_df_disk_reads': 0,
    'cache_csv_writes': 0,
    'tushare_fetch_calls': 0,
}

def _inc_stat(name, value=1):
    try:
        with _RUNTIME_STATS_LOCK:
            _RUNTIME_STATS[name] = int(_RUNTIME_STATS.get(name) or 0) + int(value)
    except Exception:
        pass

def _get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = _get_app_dir()
RESOURCE_DIR = getattr(sys, '_MEIPASS', APP_DIR)


def _get_data_root():
    if os.environ.get('VERCEL'):
        return '/tmp'
    configured = os.getenv('QUANT_DATA_DIR')
    if configured:
        return configured
    if getattr(sys, 'frozen', False):
        base = os.getenv('LOCALAPPDATA') or APP_DIR
        return os.path.join(base, 'TushareQuantSystem')
    return APP_DIR


DATA_ROOT = _get_data_root()
try:
    os.makedirs(DATA_ROOT, exist_ok=True)
except Exception:
    pass


STOCK_LIST_CACHE_FILE = os.path.join(DATA_ROOT, 'stock_list_cache.json')
CONCEPT_LIST_CACHE_FILE = os.path.join(DATA_ROOT, 'concept_list_cache.json')
CONCEPT_MEMBERS_CACHE_DIR = os.path.join(DATA_ROOT, 'concept_members')
try:
    os.makedirs(CONCEPT_MEMBERS_CACHE_DIR, exist_ok=True)
except Exception:
    pass

FUNDAMENTALS_CACHE_DIR = os.path.join(DATA_ROOT, 'fundamentals_cache')
try:
    os.makedirs(FUNDAMENTALS_CACHE_DIR, exist_ok=True)
except Exception:
    pass


LOCAL_STATE_DIR = os.path.join(DATA_ROOT, 'local_state')
try:
    os.makedirs(LOCAL_STATE_DIR, exist_ok=True)
except Exception:
    pass


_LOCAL_STATE_LOCK = threading.Lock()
_LOCAL_STATE_LAST_DIGEST = {}
_ALLOWED_LOCAL_STATE_KEYS = {
    'quant_ui_state_v1',
    'quant_kline_zoom_state_v1',
    'quant_stock_pool',
    'quant_last_session',
    'strategyConfig',
    'optimizationResults',
}


def _load_json_cache_file(file_path):
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        cached_at = data.get('cached_at')
        items = data.get('data')
        if not cached_at or not isinstance(items, list):
            return None
        return {'cached_at': cached_at, 'data': items}
    except Exception:
        return None


def _save_json_cache_file(file_path, items):
    try:
        payload = {'cached_at': datetime.now().isoformat(), 'data': items}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
        return True
    except Exception:
        return False


def _load_json_cache_payload(file_path):
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        cached_at = data.get('cached_at')
        if not cached_at:
            return None
        return {'cached_at': cached_at, 'data': data.get('data')}
    except Exception:
        return None


def _save_json_cache_payload(file_path, payload_data):
    try:
        payload = {'cached_at': datetime.now().isoformat(), 'data': payload_data}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
        return True
    except Exception:
        return False


def _sanitize_local_state_filename(key):
    if not key:
        return None
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(key))
    safe = safe.strip("._-")
    if not safe:
        return None
    return safe + ".json"


def _local_state_path_for_key(key):
    filename = _sanitize_local_state_filename(key)
    if not filename:
        return None
    return os.path.join(LOCAL_STATE_DIR, filename)


def _read_local_state_value(key):
    path = _local_state_path_for_key(key)
    if not path or not os.path.exists(path):
        return None
    try:
        with _LOCAL_STATE_LOCK:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        if isinstance(data, dict) and 'value' in data:
            return data.get('value')
        return data
    except Exception:
        return None


def _write_local_state_value(key, value):
    path = _local_state_path_for_key(key)
    if not path:
        return False
    try:
        value_json = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except Exception:
        value_json = repr(value)
    digest = hashlib.sha1(value_json.encode('utf-8', errors='ignore')).hexdigest()
    payload = {'saved_at': datetime.now().isoformat(), 'key': key, 'value': value}
    tmp_path = path + ".tmp"
    try:
        with _LOCAL_STATE_LOCK:
            if _LOCAL_STATE_LAST_DIGEST.get(key) == digest and os.path.exists(path):
                _inc_stat('local_state_skipped_writes', 1)
                return True
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, path)
            _LOCAL_STATE_LAST_DIGEST[key] = digest
            _inc_stat('local_state_writes', 1)
        return True
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False

# 添加根路径路由，提供前端页面
@app.route('/')
def index():
    """提供前端HTML页面"""
    return send_from_directory(RESOURCE_DIR, 'quant_frontend.html')

# 添加静态文件路由
@app.route('/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_from_directory(RESOURCE_DIR, filename)

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'success': True,
        'token_present': bool(TUSHARE_TOKEN),
        'token_source': TUSHARE_TOKEN_SOURCE,
        'pro_initialized': pro is not None,
        'pro_init_error': _PRO_INIT_ERROR,
        'data_root': DATA_ROOT
    })

@app.route('/api/runtime_stats', methods=['GET'])
def api_runtime_stats():
    with _RUNTIME_STATS_LOCK:
        stats = dict(_RUNTIME_STATS)
    return jsonify({'success': True, 'verbose': VERBOSE, 'stats': stats})

@app.route('/api/stock_pool_quotes', methods=['POST'])
def api_stock_pool_quotes():
    _inc_stat('stock_pool_quotes_requests', 1)
    payload = request.get_json(silent=True) or {}
    codes = payload.get('codes') or []
    if not isinstance(codes, list):
        return jsonify({'success': False, 'message': 'codes 必须是数组'}), 400
    end_date = (payload.get('end_date') or '').strip() or datetime.now().strftime('%Y-%m-%d')
    freq = (payload.get('freq') or 'D').strip().upper()
    # 正确处理 cache_only 参数（支持布尔值和字符串）
    cache_only_raw = payload.get('cache_only')
    if isinstance(cache_only_raw, bool):
        cache_only = cache_only_raw
    else:
        cache_only = str(cache_only_raw if cache_only_raw is not None else '1').strip().lower() in ('1', 'true', 'yes', 'y')
    
    print(f"[DEBUG] stock_pool_quotes: codes={codes}, end_date={end_date}, cache_only={cache_only}")

    results = []
    for raw in codes:
        ts_code = (str(raw or '')).strip().upper()
        if not ts_code:
            continue
        quote = cache_manager.get_latest_quote(ts_code, end_date=end_date, freq=freq, cache_only=cache_only)
        print(f"[DEBUG] Quote for {ts_code}: {quote}")
        if not quote:
            results.append({'ts_code': ts_code, 'ok': False})
            continue
        results.append({
            'ts_code': ts_code,
            'ok': True,
            'date': quote.get('date'),
            'close': quote.get('close'),
            'pe': quote.get('pe'),
            'pb': quote.get('pb'),
        })

    print(f"[DEBUG] Returning {len(results)} results")
    return jsonify({'success': True, 'end_date': end_date, 'freq': freq, 'cache_only': cache_only, 'data': results})

def _load_tushare_token():
    candidates = [
        'TUSHARE_TOKEN',
        'TS_TOKEN',
        'TUSHARETOKEN',
        'TU_SHARE_TOKEN',
    ]
    for name in candidates:
        env_token = os.getenv(name)
        if env_token and env_token.strip():
            return env_token.strip(), f'env:{name}'
    token_file = os.path.join(DATA_ROOT, 'tushare_token.txt')
    try:
        if os.path.exists(token_file):
            with open(token_file, 'r', encoding='utf-8') as f:
                return ((f.read() or '').strip(), f'file:{token_file}')
    except Exception:
        return '', ''
    return '', ''


TUSHARE_TOKEN = ''
TUSHARE_TOKEN_SOURCE = ''
_PRO_INIT_ERROR = ''
_PRO_INIT_LOCK = threading.Lock()


def _is_writable_dir(path):
    if not path or not os.path.isdir(path):
        return False
    probe = os.path.join(path, '.probe_write')
    try:
        with open(probe, 'w', encoding='utf-8') as f:
            f.write('')
        os.remove(probe)
        return True
    except Exception:
        return False


def _init_tushare_pro():
    global TUSHARE_TOKEN, TUSHARE_TOKEN_SOURCE, pro, _PRO_INIT_ERROR
    token, source = _load_tushare_token()
    TUSHARE_TOKEN = token
    TUSHARE_TOKEN_SOURCE = source

    print("正在初始化Tushare...")
    try:
        if os.environ.get('VERCEL') and os.name != 'nt':
            fallback_home = DATA_ROOT or '/tmp'
            try:
                os.makedirs(fallback_home, exist_ok=True)
            except Exception:
                pass
            current_home = os.environ.get('HOME')
            if not _is_writable_dir(current_home):
                os.environ['HOME'] = fallback_home
            current_userprofile = os.environ.get('USERPROFILE')
            if not _is_writable_dir(current_userprofile):
                os.environ['USERPROFILE'] = fallback_home
            os.environ.setdefault('XDG_CACHE_HOME', fallback_home)
            os.environ.setdefault('XDG_CONFIG_HOME', fallback_home)

        if not TUSHARE_TOKEN:
            raise RuntimeError("未配置Tushare Token，请设置环境变量TUSHARE_TOKEN/TS_TOKEN或在数据目录放置tushare_token.txt")
        try:
            ts.set_token(TUSHARE_TOKEN)
        except Exception:
            pass
        pro = ts.pro_api(TUSHARE_TOKEN)
        _PRO_INIT_ERROR = ''
        print("[OK] Tushare初始化成功")
        return pro
    except Exception as e:
        _PRO_INIT_ERROR = str(e)
        print(f"[ERROR] Tushare初始化失败: {e}")
        pro = None
        return None


def _get_pro():
    global pro
    if pro is not None:
        return pro
    with _PRO_INIT_LOCK:
        if pro is not None:
            return pro
        return _init_tushare_pro()

_init_tushare_pro()


INDEX_CONSTITUENTS_CACHE_FILE = os.path.join(DATA_ROOT, 'index_constituents_cache.json')


def _load_index_cache_file():
    try:
        if not os.path.exists(INDEX_CONSTITUENTS_CACHE_FILE):
            return {}
        with open(INDEX_CONSTITUENTS_CACHE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"读取指数成分缓存文件失败: {e}")
        return {}


def _save_index_cache_file(cache_dict):
    try:
        with open(INDEX_CONSTITUENTS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_dict, f, ensure_ascii=False)
    except Exception as e:
        print(f"写入指数成分缓存文件失败: {e}")


def _get_index_code_numeric(index_code):
    if not index_code:
        return None
    s = str(index_code).strip().upper()
    return s.split('.')[0] if '.' in s else s


def _fetch_index_constituents_from_sina(index_code):
    index_id = _get_index_code_numeric(index_code)
    if not index_id or not re.fullmatch(r"\d{6}", index_id):
        return []

    url = f"http://vip.stock.finance.sina.com.cn/corp/go.php/vII_NewestComponent/indexid/{index_id}.phtml"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            raw = resp.read()
        html = raw.decode('gbk', errors='ignore')
    except Exception as e:
        print(f"从新浪抓取指数成分失败: {index_code}: {e}")
        return []

    matches = re.findall(r"\b(sh|sz)(\d{6})\b", html, flags=re.IGNORECASE)
    if not matches:
        return []

    codes = []
    seen = set()
    for market, code in matches:
        ts_code = f"{code}.{'SH' if market.lower() == 'sh' else 'SZ'}"
        if ts_code in seen:
            continue
        seen.add(ts_code)
        codes.append({'ts_code': ts_code, 'name': ts_code})

    return codes


def _fetch_index_constituents_from_legulegu(index_code):
    index_id = str(index_code).strip().upper()
    if not index_id:
        return []
    if '.' not in index_id:
        index_id = f"{index_id}.SH"

    url = f"https://legulegu.com/stockdata/index-basic-composition?indexCode={index_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        html = raw.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"从乐咕乐股抓取指数成分失败: {index_code}: {e}")
        return []

    expected_count = None
    index_numeric = _get_index_code_numeric(index_id)
    if index_numeric == '000016':
        expected_count = 50
    elif index_numeric == '000300':
        expected_count = 300

    codes = re.findall(r"\b\d{6}\.(?:SH|SZ)\b", html, flags=re.IGNORECASE)
    if not codes:
        return []

    ordered = []
    seen = set()
    skip_code = index_id.upper()
    for c in codes:
        c = c.upper()
        if c == skip_code:
            continue
        if c in seen:
            continue
        seen.add(c)
        ordered.append(c)

    if expected_count and len(ordered) >= expected_count:
        ordered = ordered[:expected_count]

    return [{'ts_code': c, 'name': c} for c in ordered]

class CacheManager:
    """缓存管理器（支持本地CSV持久化）"""
    
    def __init__(self):
        self.stock_cache = {}
        self.params_cache = {}
        self.data_dir = os.path.join(DATA_ROOT, 'data')
        self._df_mem_cache = {}
        self._df_mem_cache_ttl_sec = int((os.getenv('QUANT_DF_CACHE_TTL_SEC') or '300').strip() or 300)
        self._df_mem_cache_max = int((os.getenv('QUANT_DF_CACHE_MAX') or '64').strip() or 64)
        self._no_update_until = {}
        self._no_update_ttl_sec = int((os.getenv('QUANT_NO_UPDATE_TTL_SEC') or '3600').strip() or 3600)
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
            except Exception as e:
                print(f"创建数据目录失败: {e}")
    
    def get_stock_data(self, ts_code, start_date, end_date, freq='D'):
        """获取股票数据（优先读取本地CSV，支持增量更新）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            freq: 周期，'D'=日线, 'W'=周线, 'M'=月线
        """
        # 不同周期使用不同的缓存文件
        freq_suffix = '' if freq == 'D' else f'_{freq}'
        file_path = os.path.join(self.data_dir, f"{ts_code}{freq_suffix}.csv")
        local_df = None
        mem_cache_key = (ts_code, freq)
        file_mtime = None
        try:
            if os.path.exists(file_path):
                file_mtime = os.path.getmtime(file_path)
        except Exception:
            file_mtime = None
        
        # 1. 尝试读取本地文件
        mem_hit = False
        cached_entry = self._df_mem_cache.get(mem_cache_key)
        if cached_entry and cached_entry.get('mtime') == file_mtime:
            loaded_at = cached_entry.get('loaded_at') or 0
            if (time.time() - loaded_at) <= self._df_mem_cache_ttl_sec:
                df = cached_entry.get('df')
                if isinstance(df, pd.DataFrame) and not df.empty:
                    local_df = df
                    mem_hit = True

        if local_df is None and os.path.exists(file_path):
            try:
                # 指定dtype以防止数据类型错误
                local_df = pd.read_csv(file_path, dtype={'date': str})
                if 'date' in local_df.columns:
                    local_df = local_df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
                else:
                    _log(f"[数据损坏] {file_path} 缺少date列")
                    local_df = None
            except Exception as e:
                _log(f"[读取缓存失败] {file_path}: {e}")
                local_df = None

        if local_df is not None and not local_df.empty and not mem_hit:
            _inc_stat('cache_df_disk_reads', 1)
            try:
                if file_mtime is None and os.path.exists(file_path):
                    file_mtime = os.path.getmtime(file_path)
            except Exception:
                file_mtime = None
            self._df_mem_cache[mem_cache_key] = {'df': local_df, 'mtime': file_mtime, 'loaded_at': time.time()}
            if len(self._df_mem_cache) > self._df_mem_cache_max:
                try:
                    oldest_key = min(self._df_mem_cache.items(), key=lambda kv: kv[1].get('loaded_at') or 0)[0]
                    self._df_mem_cache.pop(oldest_key, None)
                except Exception:
                    pass
        
        # 2. 检查并补充数据
        if local_df is not None and not local_df.empty:
            if mem_hit:
                _inc_stat('cache_df_mem_hits', 1)
            local_min = local_df['date'].min()
            local_max = local_df['date'].max()
            data_changed = False
            base_df = local_df
            
            # A. 向前补充（请求开始时间早于本地最早时间）
            if start_date < local_min:
                pre_end_date = (datetime.strptime(local_min, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
                if start_date <= pre_end_date:
                    now_ts = time.time()
                    if self._no_update_until.get((ts_code, freq, 'pre'), 0) > now_ts:
                        pre_data = None
                    else:
                        _log(f"[向前补充] {ts_code} {freq}: {start_date} ~ {pre_end_date}")
                        pre_data = TushareDataFetcher.get_stock_data(ts_code, start_date, pre_end_date, freq)
                    if pre_data:
                        if local_df is base_df:
                            local_df = local_df.copy()
                        pre_df = pd.DataFrame(pre_data)
                        local_df = pd.concat([pre_df, local_df]).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
                        data_changed = True
                    else:
                        self._no_update_until[(ts_code, freq, 'pre')] = time.time() + self._no_update_ttl_sec
            
            # B. 向后补充（请求结束时间晚于本地最新时间）
            if end_date > local_max:
                inc_start_date = (datetime.strptime(local_max, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                if inc_start_date <= end_date:
                    now_ts = time.time()
                    if self._no_update_until.get((ts_code, freq, 'inc'), 0) > now_ts:
                        inc_data = None
                    else:
                        _log(f"[增量更新] {ts_code} {freq}: {inc_start_date} ~ {end_date}")
                        inc_data = TushareDataFetcher.get_stock_data(ts_code, inc_start_date, end_date, freq)
                    if inc_data:
                        if local_df is base_df:
                            local_df = local_df.copy()
                        inc_df = pd.DataFrame(inc_data)
                        local_df = pd.concat([local_df, inc_df]).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
                        data_changed = True
                    else:
                        self._no_update_until[(ts_code, freq, 'inc')] = time.time() + self._no_update_ttl_sec
            
            # 如果有更新，保存回文件
            if data_changed:
                try:
                    local_df.to_csv(file_path, index=False)
                    _log(f"[持久化] 已更新 {file_path}, 条数: {len(local_df)}")
                    _inc_stat('cache_csv_writes', 1)
                    try:
                        file_mtime = os.path.getmtime(file_path)
                    except Exception:
                        file_mtime = None
                    self._df_mem_cache[mem_cache_key] = {'df': local_df, 'mtime': file_mtime, 'loaded_at': time.time()}
                except Exception as e:
                    _log(f"[保存失败] {e}")

            # 只有日线数据才补充 pe/pb
            if freq == 'D':
                try:
                    need_fill = (
                        pro is not None and
                        (('pe' not in local_df.columns) or local_df['pe'].isna().all() or
                         ('pb' not in local_df.columns) or local_df['pb'].isna().all())
                    )
                    if need_fill:
                        df_basic = pro.daily_basic(
                            ts_code=ts_code,
                            start_date=start_date.replace('-', ''),
                            end_date=end_date.replace('-', ''),
                            fields='trade_date,pe,pb'
                        )
                        if df_basic is not None and not df_basic.empty:
                            if local_df is base_df:
                                local_df = local_df.copy()
                            df_basic = df_basic.rename(columns={'trade_date': 'date'})
                            df_basic['date'] = pd.to_datetime(df_basic['date']).dt.strftime('%Y-%m-%d')
                            if 'pe' not in local_df.columns:
                                local_df['pe'] = None
                            if 'pb' not in local_df.columns:
                                local_df['pb'] = None
                            merged = local_df.merge(df_basic[['date', 'pe', 'pb']], on='date', how='left', suffixes=('', '_new'))
                            if 'pe_new' in merged.columns:
                                merged['pe'] = merged['pe'].fillna(merged['pe_new'])
                                merged.drop(columns=['pe_new'], inplace=True)
                            if 'pb_new' in merged.columns:
                                merged['pb'] = merged['pb'].fillna(merged['pb_new'])
                                merged.drop(columns=['pb_new'], inplace=True)
                            local_df = merged
                            local_df.to_csv(file_path, index=False)
                            try:
                                file_mtime = os.path.getmtime(file_path)
                            except Exception:
                                file_mtime = None
                            self._df_mem_cache[mem_cache_key] = {'df': local_df, 'mtime': file_mtime, 'loaded_at': time.time()}
                except Exception as e:
                    _log(f"补齐pe/pb失败: {e}")

        else:
            # 3. 本地无数据，全量获取
            _log(f"[本地无缓存] 全量获取 {ts_code} {freq}: {start_date} ~ {end_date}")
            data = TushareDataFetcher.get_stock_data(ts_code, start_date, end_date, freq)
            if data:
                local_df = pd.DataFrame(data)
                try:
                    local_df.to_csv(file_path, index=False)
                    _log(f"[持久化] 已保存 {file_path}, 条数: {len(local_df)}")
                    _inc_stat('cache_csv_writes', 1)
                    try:
                        file_mtime = os.path.getmtime(file_path)
                    except Exception:
                        file_mtime = None
                    self._df_mem_cache[mem_cache_key] = {'df': local_df, 'mtime': file_mtime, 'loaded_at': time.time()}
                except Exception as e:
                    _log(f"[保存失败] {e}")
            else:
                # 尝试备用数据源（仅日线）
                if freq == 'D':
                    try:
                        _log("尝试使用备用数据源(Eastmoney)...")
                        data = EastmoneyDataFetcher.get_stock_data(ts_code, start_date, end_date)
                        if data:
                            local_df = pd.DataFrame(data)
                            local_df.to_csv(file_path, index=False)
                            _log(f"[持久化] (备用源) 已保存 {file_path}")
                            _inc_stat('cache_csv_writes', 1)
                            try:
                                file_mtime = os.path.getmtime(file_path)
                            except Exception:
                                file_mtime = None
                            self._df_mem_cache[mem_cache_key] = {'df': local_df, 'mtime': file_mtime, 'loaded_at': time.time()}
                    except Exception as e:
                        _log(f"备用源获取失败: {e}")

        # 4. 返回请求区间的数据
        if local_df is not None and not local_df.empty:
            # 过滤时间区间
            mask = (local_df['date'] >= start_date) & (local_df['date'] <= end_date)
            result_df = local_df.loc[mask]
            
            if result_df.empty:
                return []
            
            # 转换为字典列表，并将NaN转为None
            return result_df.where(pd.notnull(result_df), None).to_dict('records')
            
        return []

    def get_latest_quote(self, ts_code, end_date, freq='D', cache_only=True):
        freq = (freq or 'D').upper()
        end_date = (end_date or '').strip() or datetime.now().strftime('%Y-%m-%d')
        freq_suffix = '' if freq == 'D' else f'_{freq}'
        file_path = os.path.join(self.data_dir, f"{ts_code}{freq_suffix}.csv")
        mem_cache_key = (ts_code, freq)

        local_df = None
        file_mtime = None
        try:
            if os.path.exists(file_path):
                file_mtime = os.path.getmtime(file_path)
        except Exception:
            file_mtime = None

        cached_entry = self._df_mem_cache.get(mem_cache_key)
        if cached_entry and cached_entry.get('mtime') == file_mtime:
            loaded_at = cached_entry.get('loaded_at') or 0
            if (time.time() - loaded_at) <= self._df_mem_cache_ttl_sec:
                df = cached_entry.get('df')
                if isinstance(df, pd.DataFrame) and not df.empty:
                    local_df = df
                    _inc_stat('cache_df_mem_hits', 1)

        if local_df is None and os.path.exists(file_path):
            try:
                local_df = pd.read_csv(file_path, dtype={'date': str})
                if 'date' in local_df.columns:
                    local_df = local_df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
                    self._df_mem_cache[mem_cache_key] = {'df': local_df, 'mtime': file_mtime, 'loaded_at': time.time()}
                    _inc_stat('cache_df_disk_reads', 1)
                else:
                    local_df = None
            except Exception:
                local_df = None

        # 检查缓存数据是否包含最新的end_date数据
        cache_up_to_date = False
        latest_date = None
        if local_df is not None and not local_df.empty and 'date' in local_df.columns:
            filtered = local_df.loc[local_df['date'] <= end_date]
            if not filtered.empty:
                latest_date = filtered.iloc[-1]['date']
                # 如果缓存的最新日期 >= 请求的end_date，说明缓存是最新的
                cache_up_to_date = latest_date >= end_date
        
        # 如果cache_only=False且缓存不是最新的，则从API获取
        if not cache_only and not cache_up_to_date:
            # 直接使用TushareDataFetcher获取最新数据，绕过缓存
            # 扩大日期范围以确保能获取到数据（处理周末/节假日情况）
            fetch_start = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
            data = TushareDataFetcher.get_stock_data(ts_code, fetch_start, end_date, freq)
            if isinstance(data, list) and data:
                # 合并新数据到本地缓存
                new_df = pd.DataFrame(data)
                if local_df is not None and not local_df.empty:
                    combined_df = pd.concat([local_df, new_df]).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
                else:
                    combined_df = new_df
                # 保存到文件
                try:
                    combined_df.to_csv(file_path, index=False)
                    _inc_stat('cache_csv_writes', 1)
                    try:
                        file_mtime = os.path.getmtime(file_path)
                    except Exception:
                        file_mtime = None
                    self._df_mem_cache[mem_cache_key] = {'df': combined_df, 'mtime': file_mtime, 'loaded_at': time.time()}
                except Exception:
                    pass
                return data[-1]
            # API获取失败，回退到缓存数据
            if local_df is None or local_df.empty:
                return None

        if local_df is None or local_df.empty or 'date' not in local_df.columns:
            return None

        filtered = local_df.loc[local_df['date'] <= end_date]
        if filtered.empty:
            return None
        row = filtered.iloc[-1].to_dict()
        for k, v in list(row.items()):
            try:
                if pd.isna(v):
                    row[k] = None
            except Exception:
                pass
        return row
    
    def get(self, cache_type, key):
        """获取缓存"""
        if cache_type == 'params':
            return self.params_cache.get(key)
        return None
    
    def set(self, cache_type, key, data):
        """设置缓存"""
        if cache_type == 'params':
            self.params_cache[key] = {
                'meta': {
                    'cached_at': datetime.now().isoformat()
                },
                'data': data
            }

# 初始化缓存管理器
cache_manager = CacheManager()

class TushareDataFetcher:
    """Tushare数据获取类"""
    
    @staticmethod
    def get_stock_data(ts_code, start_date, end_date, freq='D'):
        """获取股票数据，支持日线/周线/月线
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            freq: 周期，'D'=日线, 'W'=周线, 'M'=月线
        """
        if pro is None:
            _log("[ERROR] Tushare API未初始化")
            return None
            
        try:
            _inc_stat('tushare_fetch_calls', 1)
            _log(f"正在获取 {ts_code} 从 {start_date} 到 {end_date} 的{freq}线数据...")
            
            # 根据周期选择接口
            if freq == 'W':
                df = pro.weekly(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )
            elif freq == 'M':
                df = pro.monthly(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )
            else:  # 默认日线
                df = pro.daily(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )
            
            _log(f"获取到原始数据: {len(df) if df is not None else 'None'} 条")
            
            if df is None or df.empty:
                _log("数据为空或None")
                return None
            
            # 按日期升序排列
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            # 重命名列
            df = df.rename(columns={
                'trade_date': 'date',
                'vol': 'volume',
                'pct_chg': 'pctChange',
                'change': 'priceChange'
            })
            
            # 格式化日期
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # 获取换手率数据（可选，因为可能需要权限）
            try:
                df_basic = pro.daily_basic(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    fields='trade_date,turnover_rate,pe,pb'
                )
                
                if df_basic is not None and not df_basic.empty:
                    df_basic = df_basic.rename(columns={
                        'trade_date': 'date',
                        'turnover_rate': 'turnover'
                    })
                    df_basic['date'] = pd.to_datetime(df_basic['date']).dt.strftime('%Y-%m-%d')
                    df = df.merge(df_basic[['date', 'turnover', 'pe', 'pb']], on='date', how='left')
            except Exception as e:
                _log(f"获取换手率数据失败（可能需要更高权限）: {e}")
                # 添加空列作为占位符
                df['turnover'] = None
                df['pe'] = None
                df['pb'] = None

            if 'turnover' not in df.columns or df['turnover'].isna().all():
                float_share = None
                try:
                    df_share = pro.stock_basic(ts_code=ts_code, fields='ts_code,float_share')
                    if df_share is not None and not df_share.empty:
                        float_share = df_share.iloc[0].get('float_share')
                except Exception:
                    float_share = None
                
                if float_share and isinstance(float_share, (int, float)) and float_share > 0:
                    turnover_est = df['volume'] / (float_share * 100)
                    df['turnover'] = turnover_est.clip(lower=0, upper=100)

            try:
                if 'turnover' not in df.columns or df['turnover'].isna().all():
                    em = EastmoneyDataFetcher.get_stock_data(ts_code, start_date, end_date)
                    em_by_date = {row.get('date'): row for row in (em or []) if isinstance(row, dict) and row.get('date')}
                    if em_by_date:
                        df['turnover'] = df['date'].map(lambda d: em_by_date.get(d, {}).get('turnover'))
                        if 'amount' in df.columns:
                            df['amount'] = df['amount'].fillna(df['date'].map(lambda d: em_by_date.get(d, {}).get('amount')))
            except Exception as e:
                print(f"从东方财富补全换手率失败: {e}")

            try:
                df_adj = pro.adj_factor(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )
                if df_adj is not None and not df_adj.empty:
                    df_adj = df_adj.rename(columns={'trade_date': 'date'})
                    df_adj['date'] = pd.to_datetime(df_adj['date']).dt.strftime('%Y-%m-%d')
                    df_adj = df_adj.sort_values('date').reset_index(drop=True)
                    latest_adj = df_adj['adj_factor'].iloc[-1]
                    df = df.merge(df_adj[['date', 'adj_factor']], on='date', how='left')
                    df['adj_factor'] = df['adj_factor'].ffill()
                    if pd.notna(latest_adj) and latest_adj > 0:
                        factor = df['adj_factor'] / latest_adj
                        for col in ['open', 'high', 'low', 'close']:
                            if col in df.columns:
                                df[col] = df[col] * factor
            except Exception as e:
                print(f"获取复权因子失败: {e}")
            
            return df.to_dict('records')
            
        except Exception as e:
            print(f"获取数据错误: {e}")
            return None
    
    @staticmethod
    def get_stock_list():
        """获取股票列表"""
        if pro is None:
            print("[ERROR] Tushare API未初始化，使用默认股票列表")
            return [
                {'ts_code': '000001.SZ', 'symbol': '000001', 'name': '平安银行', 'industry': '银行', 'market': '主板'},
                {'ts_code': '000002.SZ', 'symbol': '000002', 'name': '万科A', 'industry': '房地产', 'market': '主板'},
                {'ts_code': '600519.SH', 'symbol': '600519', 'name': '贵州茅台', 'industry': '白酒', 'market': '主板'},
                {'ts_code': '000858.SZ', 'symbol': '000858', 'name': '五粮液', 'industry': '白酒', 'market': '主板'},
                {'ts_code': '601318.SH', 'symbol': '601318', 'name': '中国平安', 'industry': '保险', 'market': '主板'}
            ]
        try:
            df = pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,industry,market'
            )
            return df.to_dict('records') if df is not None else []
        except Exception as e:
            print(f"获取股票列表错误: {e}")
            return [
                {'ts_code': '000001.SZ', 'symbol': '000001', 'name': '平安银行', 'industry': '银行', 'market': '主板'},
                {'ts_code': '000002.SZ', 'symbol': '000002', 'name': '万科A', 'industry': '房地产', 'market': '主板'},
                {'ts_code': '600519.SH', 'symbol': '600519', 'name': '贵州茅台', 'industry': '白酒', 'market': '主板'},
                {'ts_code': '000858.SZ', 'symbol': '000858', 'name': '五粮液', 'industry': '白酒', 'market': '主板'},
                {'ts_code': '601318.SH', 'symbol': '601318', 'name': '中国平安', 'industry': '保险', 'market': '主板'},
                {'ts_code': '600036.SH', 'symbol': '600036', 'name': '招商银行', 'industry': '银行', 'market': '主板'},
                {'ts_code': '000333.SZ', 'symbol': '000333', 'name': '美的集团', 'industry': '家电', 'market': '主板'},
                {'ts_code': '002415.SZ', 'symbol': '002415', 'name': '海康威视', 'industry': '软件服务', 'market': '中小板'},
                {'ts_code': '300750.SZ', 'symbol': '300750', 'name': '宁德时代', 'industry': '电气设备', 'market': '创业板'},
                {'ts_code': '600309.SH', 'symbol': '600309', 'name': '万华化学', 'industry': '化工', 'market': '主板'}
            ]
    
    @staticmethod
    def get_index_weights(index_code='000300.SH'):
        """获取指数成分股"""
        file_cache = _load_index_cache_file()

        cached_item = file_cache.get(index_code)
        if isinstance(cached_item, dict) and isinstance(cached_item.get('data'), list):
            try:
                cached_at = datetime.fromisoformat(cached_item.get('cached_at'))
                if (datetime.now() - cached_at).days < 30:
                    return cached_item['data']
            except Exception:
                pass

        if pro is None:
            data = _fetch_index_constituents_from_legulegu(index_code)
            if not data:
                data = _fetch_index_constituents_from_sina(index_code)
            if data:
                file_cache[index_code] = {'cached_at': datetime.now().isoformat(), 'data': data}
                _save_index_cache_file(file_cache)
            return data
        try:
            # 获取指数成分股
            df = pro.index_weight(index_code=index_code, start_date='20230101', end_date=datetime.now().strftime('%Y%m%d'))
            if df is None or df.empty:
                return []
            
            # 只取最新的成分股
            latest_date = df['trade_date'].max()
            df = df[df['trade_date'] == latest_date]
            
            # 获取股票名称
            codes = df['con_code'].tolist()
            # 分批获取名称
            all_stocks = []
            for i in range(0, len(codes), 100):
                batch_codes = codes[i:i+100]
                df_basic = pro.stock_basic(ts_code=','.join(batch_codes), fields='ts_code,symbol,name,industry')
                if df_basic is not None:
                    all_stocks.extend(df_basic.to_dict('records'))
            
            return all_stocks
        except Exception as e:
            print(f"获取指数成分股错误: {e}")
            data = _fetch_index_constituents_from_legulegu(index_code)
            if not data:
                data = _fetch_index_constituents_from_sina(index_code)
            if data:
                file_cache[index_code] = {'cached_at': datetime.now().isoformat(), 'data': data}
                _save_index_cache_file(file_cache)
                return data

            if isinstance(cached_item, dict) and isinstance(cached_item.get('data'), list):
                return cached_item['data']

            return []

    @staticmethod
    def search_stock(keyword):
        """搜索股票"""
        try:
            df = pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,industry'
            )
            # 模糊搜索
            mask = (
                df['ts_code'].str.contains(keyword, case=False, na=False) |
                df['name'].str.contains(keyword, case=False, na=False) |
                df['symbol'].str.contains(keyword, case=False, na=False)
            )
            return df[mask].head(20).to_dict('records')
        except Exception as e:
            print(f"搜索股票错误: {e}")
            return []

    @staticmethod
    def gen_mock_data(ts_code, start_date, end_date):
        """生成模拟股票数据"""
        print(f"正在生成模拟数据 {ts_code} 从 {start_date} 到 {end_date}")
        
        # 解析日期
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 生成日期序列（只包含工作日）
        dates = []
        current_date = start
        while current_date <= end:
            if current_date.weekday() < 5:  # 只包含工作日
                dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        if not dates:
            print("❌ 没有生成任何日期")
            return []
        
        print(f"📅 生成了 {len(dates)} 个交易日")
        
        # 生成模拟数据
        data = []
        base_price = 10 + np.random.random() * 40  # 随机基础价格 10-50
        
        for i, date in enumerate(dates):
            # 价格波动
            daily_change = (np.random.random() - 0.48) * 0.1  # 略微上涨趋势
            if i == 0:
                open_price = base_price
            else:
                prev_close = data[i-1]['close']
                open_price = prev_close * (1 + np.random.random() * 0.02 - 0.01)  # ±1% 开盘波动
            
            close_price = open_price * (1 + daily_change)
            
            # 确定当日价格范围，基于开盘价和收盘价
            higher_price = max(open_price, close_price)
            lower_price = min(open_price, close_price)
            
            # 计算最高价和最低价
            high_price = higher_price * (1 + np.random.random() * 0.03)  # 最高价格
            low_price = lower_price * (1 - np.random.random() * 0.03)   # 最低价格
            
            # 确保高低价格合理
            high_price = max(high_price, higher_price)
            low_price = min(low_price, lower_price)
            low_price = max(0, low_price)  # 确保最低价非负
            
            volume = int(np.random.random() * 10000000) + 1000000  # 100万到1100万股
            
            row = {
                'date': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'pctChange': round((close_price - open_price) / open_price * 100, 2),
                'priceChange': round(close_price - open_price, 2),
                'turnover': round(np.random.random() * 5, 2),  # 换手率 0-5%
                'pe': round(15 + np.random.random() * 20, 2),  # 市盈率 15-35
                'pb': round(1 + np.random.random() * 3, 2)     # 市净率 1-4
            }
            
            data.append(row)
        
        print(f"生成了 {len(data)} 条模拟数据")
        return data


def _eastmoney_secid(ts_code):
    if not ts_code:
        return None
    s = str(ts_code).strip().upper()
    if '.' in s:
        symbol, exch = s.split('.', 1)
    else:
        symbol = re.sub(r"\D", "", s)
        exch = 'SH' if symbol.startswith('6') else 'SZ'
    symbol = re.sub(r"\D", "", symbol)
    if not re.fullmatch(r"\d{6}", symbol):
        return None
    market = '1' if exch == 'SH' else '0'
    return f"{market}.{symbol}"


class EastmoneyDataFetcher:
    @staticmethod
    def get_stock_data(ts_code, start_date, end_date):
        secid = _eastmoney_secid(ts_code)
        if not secid:
            return None

        beg = start_date.replace('-', '')
        end = end_date.replace('-', '')
        params = {
            'secid': secid,
            'klt': 101,
            'fqt': 1,
            'beg': beg,
            'end': end,
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'
        }
        url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'})
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode('utf-8'))

        data = (payload or {}).get('data') or {}
        klines = data.get('klines') or []
        out = []
        for row in klines:
            if not isinstance(row, str):
                continue
            parts = row.split(',')
            if len(parts) < 11:
                continue
            date = parts[0]
            try:
                out.append({
                    'date': date,
                    'open': float(parts[1]),
                    'close': float(parts[2]),
                    'high': float(parts[3]),
                    'low': float(parts[4]),
                    'volume': float(parts[5]),
                    'amount': float(parts[6]),
                    'pctChange': float(parts[8]),
                    'priceChange': float(parts[9]),
                    'turnover': float(parts[10]),
                    'ts_code': ts_code
                })
            except Exception:
                continue
        return out


class TechnicalIndicators:
    """技术指标计算类"""
    
    @staticmethod
    def calculate_all(data):
        """计算所有技术指标"""
        df = pd.DataFrame(data)
        
        # 移动平均线
        for period in [5, 10, 20, 60]:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()
        
        # 成交量均线
        for period in [5, 10, 20]:
            df[f'volMa{period}'] = df['volume'].rolling(window=period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = exp1 - exp2
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2
        
        # 布林带
        df['bollMid'] = df['close'].rolling(window=20).mean()
        df['bollStd'] = df['close'].rolling(window=20).std()
        df['bollUp'] = df['bollMid'] + 2 * df['bollStd']
        df['bollDown'] = df['bollMid'] - 2 * df['bollStd']
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=20).mean()
        
        # N日最高最低
        for period in [10, 20]:
            df[f'high{period}'] = df['high'].rolling(window=period).max()
            df[f'low{period}'] = df['low'].rolling(window=period).min()
        
        # KDJ
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['k'] = rsv.ewm(com=2, adjust=False).mean()
        df['d'] = df['k'].ewm(com=2, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        # 替换NaN为None
        df = df.replace({np.nan: None})
        
        return df.to_dict('records')


class CrossStockParamOptimizer:
    """跨股票参数优化器 - 解决单股票参数局限性问题"""
    
    # 预设的通用参数模板（基于大量股票回测优化得出）
    UNIVERSAL_PARAMS = {
        'deep_fusion': {
            'mlScoreThreshold': 55,
            'minConfirmations': 3,
            'volumeMulti': 1.5,
            'rsiThreshold': 35,
            'macdThreshold': 0,
            'useMa5AboveMa20': True,
            'usePriceAboveMa5': True,
            'useMacdRising': True,
            'useRsiBelow35': True,
            'useVolumeAboveMa': True,
            'usePriceBelowBoll': True,
            'useAdxConfirm': True,
            'useMomentumConfirm': True
        },
        'volume_breakout': {
            'breakoutPeriod': 20,
            'volumeMulti': 2.0,
            'rsiMax': 70,
            'minConditions': 3,
            'scoreThreshold': 70,
            'usePriceBreakout': True,
            'useVolumeUp': True,
            'useTrendConfirm': True,
            'useVolatilityConfirm': True,
            'useMomentumConfirm': True,
            'useBullishCandle': True,
            'useChipConfirm': True
        },
        'oversold_rebound': {
            'rsiOversold': 30,
            'bollingerOffset': 0.02,
            'nearLowRatio': 1.03,
            'minConditions': 3,
            'scoreThreshold': 65,
            'useRsiOversold': True,
            'useTouchLowerBoll': True,
            'useNearLow': True,
            'useVolumePattern': True,
            'useCandlePattern': True,
            'useMacdDivergence': True,
            'useTrendFilter': True
        },
        'trend_enhanced': {
            'macdThreshold': 0,
            'minConditions': 3,
            'scoreThreshold': 70,
            'useShortMaAlignment': True,
            'usePriceAboveMa': True,
            'useMacdPositive': True,
            'useAdxConfirm': True,
            'useVolumeConfirm': True,
            'useBollConfirm': True,
            'useMomentumConfirm': True,
            'useTrendPersistence': True
        },
        'macd_divergence': {
            'lookbackPeriod': 30,
            'minConditions': 2,
            'scoreThreshold': 60,
            'useBullishDivergence': True,
            'useHistogramDivergence': True,
            'useVolumeDivergence': True,
            'useRsiDivergence': True,
            'useTrendFilter': True,
            'useCandleConfirm': True,
            'useMaConfirm': True
        },
        'bollinger_extreme': {
            'bollDeviation': 2.0,
            'rsiOversold': 30,
            'minConditions': 2,
            'scoreThreshold': 60,
            'useLowerBandTouch': True,
            'useBandWidthConfirm': True,
            'useRsiConfirm': True,
            'useVolumeConfirm': True,
            'useMaSupport': True,
            'useCandleConfirm': True,
            'useVolatilityAdjust': True
        },
        'momentum_rotation': {
            'shortMomentumPeriod': 10,
            'shortMomentumThreshold': 3.0,
            'mediumMomentumPeriod': 20,
            'mediumMomentumThreshold': 5.0,
            'longMomentumPeriod': 60,
            'volumeMulti': 1.5,
            'minConditions': 3,
            'scoreThreshold': 65,
            'useShortMomentum': True,
            'useMediumMomentum': True,
            'useLongMomentum': True,
            'useRelativeStrength': True,
            'useVolumeConfirm': True,
            'useMomentumQuality': True,
            'useVolatilityAdjust': True,
            'useMacdConfirm': True
        },
        'turtle_enhanced': {
            'entryPeriod': 20,
            'exitPeriod': 10,
            'atrMultiplier': 2.0,
            'volumeMulti': 1.5,
            'minConditions': 3,
            'scoreThreshold': 65,
            'useBreakoutEntry': True,
            'useTrendFilter': True,
            'useVolatilityFilter': True,
            'useVolumeConfirm': True,
            'useAdxConfirm': True,
            'useFalseBreakoutFilter': True,
            'useRiskManagement': True,
            'useSentimentFilter': True
        }
    }
    
    # 行业特定参数调整
    INDUSTRY_ADJUSTMENTS = {
        '银行': {'rsiThreshold': 40, 'volumeMulti': 1.2, 'atr_multiplier': 0.8},
        '保险': {'rsiThreshold': 38, 'volumeMulti': 1.3, 'atr_multiplier': 0.9},
        '房地产': {'rsiThreshold': 35, 'volumeMulti': 1.4, 'atr_multiplier': 1.1},
        '白酒': {'rsiThreshold': 35, 'volumeMulti': 1.5, 'atr_multiplier': 1.2},
        '医药': {'rsiThreshold': 33, 'volumeMulti': 1.6, 'atr_multiplier': 1.3},
        '科技': {'rsiThreshold': 30, 'volumeMulti': 1.8, 'atr_multiplier': 1.5},
        '半导体': {'rsiThreshold': 30, 'volumeMulti': 2.0, 'atr_multiplier': 1.6},
        '新能源': {'rsiThreshold': 32, 'volumeMulti': 1.8, 'atr_multiplier': 1.5},
        '化工': {'rsiThreshold': 35, 'volumeMulti': 1.4, 'atr_multiplier': 1.1},
        '机械': {'rsiThreshold': 35, 'volumeMulti': 1.4, 'atr_multiplier': 1.2},
        '电子': {'rsiThreshold': 32, 'volumeMulti': 1.7, 'atr_multiplier': 1.4},
        '通信': {'rsiThreshold': 32, 'volumeMulti': 1.7, 'atr_multiplier': 1.4},
        '传媒': {'rsiThreshold': 30, 'volumeMulti': 1.8, 'atr_multiplier': 1.5},
        '汽车': {'rsiThreshold': 33, 'volumeMulti': 1.6, 'atr_multiplier': 1.3},
        '家电': {'rsiThreshold': 35, 'volumeMulti': 1.4, 'atr_multiplier': 1.1},
        '食品': {'rsiThreshold': 36, 'volumeMulti': 1.3, 'atr_multiplier': 1.0},
        '服装': {'rsiThreshold': 35, 'volumeMulti': 1.4, 'atr_multiplier': 1.2},
        '建材': {'rsiThreshold': 35, 'volumeMulti': 1.4, 'atr_multiplier': 1.2},
        '有色': {'rsiThreshold': 32, 'volumeMulti': 1.7, 'atr_multiplier': 1.4},
        '钢铁': {'rsiThreshold': 35, 'volumeMulti': 1.4, 'atr_multiplier': 1.1},
        '煤炭': {'rsiThreshold': 35, 'volumeMulti': 1.4, 'atr_multiplier': 1.1},
        '石油': {'rsiThreshold': 36, 'volumeMulti': 1.3, 'atr_multiplier': 1.0},
        '电力': {'rsiThreshold': 38, 'volumeMulti': 1.2, 'atr_multiplier': 0.9},
        '交通运输': {'rsiThreshold': 37, 'volumeMulti': 1.3, 'atr_multiplier': 0.9},
        '建筑': {'rsiThreshold': 36, 'volumeMulti': 1.3, 'atr_multiplier': 1.0},
        '农林牧渔': {'rsiThreshold': 33, 'volumeMulti': 1.6, 'atr_multiplier': 1.3},
        '商贸零售': {'rsiThreshold': 35, 'volumeMulti': 1.4, 'atr_multiplier': 1.1},
        '社会服务': {'rsiThreshold': 33, 'volumeMulti': 1.6, 'atr_multiplier': 1.3},
        '计算机': {'rsiThreshold': 30, 'volumeMulti': 1.9, 'atr_multiplier': 1.6},
        '国防军工': {'rsiThreshold': 30, 'volumeMulti': 1.9, 'atr_multiplier': 1.6},
    }
    
    @classmethod
    def get_optimized_params(cls, strategy_name, stock_features=None, industry=None):
        """
        获取优化后的参数
        
        Args:
            strategy_name: 策略名称
            stock_features: 股票特征字典
            industry: 行业名称
        
        Returns:
            优化后的参数字典
        """
        # 获取基础通用参数
        params = cls.UNIVERSAL_PARAMS.get(strategy_name, {}).copy()
        
        # 应用行业特定调整
        if industry and industry in cls.INDUSTRY_ADJUSTMENTS:
            industry_adj = cls.INDUSTRY_ADJUSTMENTS[industry]
            for key, value in industry_adj.items():
                if key in params:
                    params[key] = value
        
        # 应用股票特征自适应调整
        if stock_features:
            params = cls._apply_feature_adjustments(params, stock_features)
        
        return params
    
    @classmethod
    def _apply_feature_adjustments(cls, params, stock_features):
        """根据股票特征调整参数"""
        # 波动率调整
        volatility = stock_features.get('volatility', 'medium')
        volatility_multipliers = {
            'very_high': {'stopLoss': 1.5, 'takeProfit': 1.3, 'rsiThreshold': 1.15, 'volumeMulti': 0.8},
            'high': {'stopLoss': 1.3, 'takeProfit': 1.2, 'rsiThreshold': 1.08, 'volumeMulti': 0.9},
            'medium': {'stopLoss': 1.0, 'takeProfit': 1.0, 'rsiThreshold': 1.0, 'volumeMulti': 1.0},
            'low': {'stopLoss': 0.8, 'takeProfit': 0.9, 'rsiThreshold': 0.92, 'volumeMulti': 1.1},
            'very_low': {'stopLoss': 0.7, 'takeProfit': 0.8, 'rsiThreshold': 0.85, 'volumeMulti': 1.2}
        }
        
        vm = volatility_multipliers.get(volatility, volatility_multipliers['medium'])
        
        # 应用波动率调整
        for param_key, multiplier in vm.items():
            if param_key in params and isinstance(params[param_key], (int, float)):
                params[param_key] = params[param_key] * multiplier
        
        # 流动性调整
        liquidity = stock_features.get('liquidity', 'medium')
        liquidity_adjustments = {
            'very_high': {'scoreThreshold': -5, 'minConditions': -1},
            'high': {'scoreThreshold': -3, 'minConditions': 0},
            'medium': {'scoreThreshold': 0, 'minConditions': 0},
            'low': {'scoreThreshold': 5, 'minConditions': 1},
            'very_low': {'scoreThreshold': 10, 'minConditions': 1}
        }
        
        la = liquidity_adjustments.get(liquidity, liquidity_adjustments['medium'])
        
        for param_key, adjustment in la.items():
            if param_key in params:
                params[param_key] = params[param_key] + adjustment
        
        # 趋势类型调整
        trend_type = stock_features.get('trend_type', 'neutral')
        trend_adjustments = {
            'strong_uptrend': {'macdThreshold': 0.3, 'scoreThreshold': 5},
            'weak_uptrend': {'macdThreshold': 0.1, 'scoreThreshold': 0},
            'sideways': {'macdThreshold': 0, 'scoreThreshold': 0},
            'weak_downtrend': {'macdThreshold': -0.1, 'scoreThreshold': -5},
            'strong_downtrend': {'macdThreshold': -0.3, 'scoreThreshold': -10},
            'neutral': {'macdThreshold': 0, 'scoreThreshold': 0}
        }
        
        ta = trend_adjustments.get(trend_type, trend_adjustments['neutral'])
        
        for param_key, adjustment in ta.items():
            if param_key in params:
                params[param_key] = params[param_key] + adjustment
        
        return params
    
    @classmethod
    def validate_params(cls, strategy_name, params):
        """验证参数有效性"""
        validated = params.copy()
        
        # 确保数值参数在合理范围内
        if 'rsiThreshold' in validated:
            validated['rsiThreshold'] = max(10, min(50, validated['rsiThreshold']))
        
        if 'rsiOversold' in validated:
            validated['rsiOversold'] = max(5, min(40, validated['rsiOversold']))
        
        if 'rsiOverbought' in validated:
            validated['rsiOverbought'] = max(60, min(95, validated['rsiOverbought']))
        
        if 'volumeMulti' in validated:
            validated['volumeMulti'] = max(1.0, min(5.0, validated['volumeMulti']))
        
        if 'stopLoss' in validated:
            validated['stopLoss'] = max(0.03, min(0.20, validated['stopLoss']))
        
        if 'takeProfit' in validated:
            validated['takeProfit'] = max(0.05, min(0.50, validated['takeProfit']))
        
        if 'scoreThreshold' in validated:
            validated['scoreThreshold'] = max(40, min(85, validated['scoreThreshold']))
        
        if 'minConditions' in validated:
            validated['minConditions'] = max(1, min(6, int(validated['minConditions'])))
        
        return validated


class StockFeatureAnalyzer:
    """股票特征分析器 - 用于自适应参数调整"""
    
    @staticmethod
    def analyze_stock_features(df):
        """分析股票特征，返回特征字典"""
        if df is None or len(df) < 60:
            return {
                'volatility': 'medium',
                'liquidity': 'medium',
                'trend_type': 'neutral',
                'industry_style': 'general',
                'market_cap_category': 'mid',
                'volatility_atr_pct': 2.0,
                'avg_volume': 1000000,
                'price_level': 50
            }
        
        latest = df.iloc[-1]
        
        # 1. 波动率特征 (基于ATR百分比)
        atr_pct = latest.get('atr_pct', 2.0)
        if atr_pct > 4:
            volatility = 'very_high'
        elif atr_pct > 2.5:
            volatility = 'high'
        elif atr_pct > 1.5:
            volatility = 'medium'
        elif atr_pct > 0.8:
            volatility = 'low'
        else:
            volatility = 'very_low'
        
        # 2. 流动性特征 (基于成交量)
        avg_volume = df['volume'].mean() if 'volume' in df.columns else 1000000
        if avg_volume > 50000000:
            liquidity = 'very_high'
        elif avg_volume > 10000000:
            liquidity = 'high'
        elif avg_volume > 2000000:
            liquidity = 'medium'
        elif avg_volume > 500000:
            liquidity = 'low'
        else:
            liquidity = 'very_low'
        
        # 3. 趋势类型
        if len(df) >= 60:
            price_change_60d = (df.iloc[-1]['close'] - df.iloc[-60]['close']) / df.iloc[-60]['close'] * 100
            if price_change_60d > 30:
                trend_type = 'strong_uptrend'
            elif price_change_60d > 10:
                trend_type = 'weak_uptrend'
            elif price_change_60d < -30:
                trend_type = 'strong_downtrend'
            elif price_change_60d < -10:
                trend_type = 'weak_downtrend'
            else:
                trend_type = 'sideways'
        else:
            trend_type = 'neutral'
        
        # 4. 价格水平
        price_level = latest.get('close', 50)
        if price_level > 200:
            price_category = 'very_high'
        elif price_level > 100:
            price_category = 'high'
        elif price_level > 50:
            price_category = 'medium'
        elif price_level > 20:
            price_category = 'low'
        else:
            price_category = 'very_low'
        
        # 5. 波动率稳定性
        if len(df) >= 20:
            atr_std = df['atr_pct'].std() if 'atr_pct' in df.columns else 0
            if atr_std > 2:
                volatility_stability = 'unstable'
            elif atr_std > 1:
                volatility_stability = 'moderate'
            else:
                volatility_stability = 'stable'
        else:
            volatility_stability = 'moderate'
        
        return {
            'volatility': volatility,
            'volatility_atr_pct': atr_pct,
            'liquidity': liquidity,
            'avg_volume': avg_volume,
            'trend_type': trend_type,
            'price_level': price_level,
            'price_category': price_category,
            'volatility_stability': volatility_stability,
            'boll_width': latest.get('bollWidth', 0.1),
            'adx': latest.get('adx', 25)
        }
    
    @staticmethod
    def get_adaptive_params(base_config, stock_features):
        """根据股票特征获取自适应参数"""
        config = base_config.copy()
        
        # 波动率调整因子
        volatility_factors = {
            'very_high': {'stop_loss_mult': 1.5, 'take_profit_mult': 1.3, 'rsi_threshold_adj': 5, 'volume_mult': 0.8},
            'high': {'stop_loss_mult': 1.3, 'take_profit_mult': 1.2, 'rsi_threshold_adj': 3, 'volume_mult': 0.9},
            'medium': {'stop_loss_mult': 1.0, 'take_profit_mult': 1.0, 'rsi_threshold_adj': 0, 'volume_mult': 1.0},
            'low': {'stop_loss_mult': 0.8, 'take_profit_mult': 0.9, 'rsi_threshold_adj': -3, 'volume_mult': 1.1},
            'very_low': {'stop_loss_mult': 0.7, 'take_profit_mult': 0.8, 'rsi_threshold_adj': -5, 'volume_mult': 1.2}
        }
        
        vf = volatility_factors.get(stock_features['volatility'], volatility_factors['medium'])
        
        # 调整止损止盈
        if 'stopLoss' in config:
            config['stopLoss'] = config['stopLoss'] * vf['stop_loss_mult']
        if 'takeProfit' in config:
            config['takeProfit'] = config['takeProfit'] * vf['take_profit_mult']
        
        # 调整RSI阈值
        if 'rsiThreshold' in config:
            config['rsiThreshold'] = config['rsiThreshold'] + vf['rsi_threshold_adj']
        if 'rsiOversold' in config:
            config['rsiOversold'] = max(10, config['rsiOversold'] + vf['rsi_threshold_adj'])
        if 'rsiOverbought' in config:
            config['rsiOverbought'] = min(90, config['rsiOverbought'] - vf['rsi_threshold_adj'])
        
        # 调整成交量倍数
        if 'volumeMulti' in config:
            config['volumeMulti'] = config['volumeMulti'] * vf['volume_mult']
        
        # 趋势类型调整
        trend_factors = {
            'strong_uptrend': {'macd_threshold_adj': 0.5, 'ma_bias': 0.02},
            'weak_uptrend': {'macd_threshold_adj': 0.2, 'ma_bias': 0.01},
            'sideways': {'macd_threshold_adj': 0, 'ma_bias': 0},
            'weak_downtrend': {'macd_threshold_adj': -0.2, 'ma_bias': -0.01},
            'strong_downtrend': {'macd_threshold_adj': -0.5, 'ma_bias': -0.02},
            'neutral': {'macd_threshold_adj': 0, 'ma_bias': 0}
        }
        
        tf = trend_factors.get(stock_features['trend_type'], trend_factors['neutral'])
        
        if 'macdThreshold' in config:
            config['macdThreshold'] = config['macdThreshold'] + tf['macd_threshold_adj']
        
        # 流动性调整
        liquidity_factors = {
            'very_high': {'position_size_mult': 1.2, 'slippage': 0.001},
            'high': {'position_size_mult': 1.1, 'slippage': 0.002},
            'medium': {'position_size_mult': 1.0, 'slippage': 0.003},
            'low': {'position_size_mult': 0.8, 'slippage': 0.005},
            'very_low': {'position_size_mult': 0.6, 'slippage': 0.008}
        }
        
        lf = liquidity_factors.get(stock_features['liquidity'], liquidity_factors['medium'])
        
        if 'positionSizeMult' in config:
            config['positionSizeMult'] = config['positionSizeMult'] * lf['position_size_mult']
        
        return config


class StrategyEngine:
    """策略引擎"""
    
    STRATEGIES = {
        'deep_fusion': {
            'name': '深度融合策略',
            'icon': '🤖',
            'description': '融合多技术指标，通过加权评分+确认信号机制决策，支持自适应参数调整',
            'features': ['MA对齐确认', 'MACD动量', 'RSI超卖', '成交量确认', '布林带位置', 'ADX趋势强度', '价格动量']
        },
        'volume_breakout': {
            'name': '量价突破策略',
            'icon': '⚡',
            'description': '多维度突破确认：价格突破+成交量放大+趋势对齐+波动率过滤',
            'features': ['价格突破', '成交量激增', '趋势确认', '波动率过滤', '动能确认', 'K线形态', '筹码分布']
        },
        'oversold_rebound': {
            'name': '超跌反弹策略',
            'icon': '💎',
            'description': '多维度超卖确认+反弹信号检测：RSI背离+MACD背离+K线形态',
            'features': ['RSI超卖', '布林下轨', '价格低点', '成交量模式', 'K线形态', 'MACD背离', '趋势过滤']
        },
        'trend_enhanced': {
            'name': '趋势增强策略',
            'icon': '📈',
            'description': '多时间框架趋势确认：均线排列+ADX强度+成交量趋势+动量确认',
            'features': ['均线排列', '价格位置', 'MACD动能', 'ADX强度', '成交量趋势', '布林带趋势', '趋势持续性']
        },
        'macd_divergence': {
            'name': 'MACD背离策略',
            'icon': '🎯',
            'description': '多周期背离检测：价格-MACD背离+成交量背离+RSI背离+K线形态确认',
            'features': ['底背离检测', '柱状图背离', '成交量背离', 'RSI背离', '趋势过滤', 'K线形态', '均线确认']
        },
        'bollinger_extreme': {
            'name': '布林极限策略',
            'icon': '📊',
            'description': '布林带极值交易+反转确认：下轨触及+RSI超卖+成交量地量+K线形态',
            'features': ['下轨触及', '带宽确认', 'RSI超卖', '成交量确认', '均线支撑', 'K线形态', '波动率调整']
        },
        'momentum_rotation': {
            'name': '动量轮动策略',
            'icon': '⚡',
            'description': '多周期动量确认+相对强度分析：短中长期动量+动量质量+波动率匹配',
            'features': ['短期动量', '中期动量', '长期动量', '相对强度', '成交量确认', '动量质量', 'MACD确认']
        },
        'turtle_enhanced': {
            'name': '海龟增强策略',
            'icon': '🐢',
            'description': '多时间框架突破+风险管理增强：突破确认+趋势过滤+假突破过滤+风险评分',
            'features': ['突破入场', '趋势过滤', '波动率过滤', '成交量确认', 'ADX强度', '假突破过滤', '风险管理']
        },
        'chip_peak_breakout': {
            'name': '筹码突破策略',
            'icon': '🧩',
            'description': '筹码单峰低位+放量突破：通过筹码分布识别低位集中区域，配合放量突破信号',
            'features': ['筹码分布', '单峰识别', '集中度分析', '低位判断', '放量突破', '趋势确认', '筹码突破']
        },
        'xgboost_ml': {
            'name': 'XGBoost机器学习策略',
            'icon': '🚀',
            'description': 'XGBoost梯度提升预测+条件确认：用历史数据训练XGBoost模型，概率过滤+多重确认',
            'features': ['XGBoost', '梯度提升', '特征工程', '交叉验证', '趋势过滤', '波动率过滤', '跳空过滤']
        },
        'lstm_ml': {
            'name': 'LSTM深度学习策略',
            'icon': '🧠',
            'description': 'LSTM时间序列预测+条件确认：用历史数据训练LSTM模型，捕捉时间序列模式',
            'features': ['LSTM', '时间序列', '深度学习', '序列特征', '趋势过滤', '波动率过滤', '跳空过滤']
        },
        'ensemble_ml': {
            'name': '混合机器学习策略',
            'icon': '🎭',
            'description': 'XGBoost+LSTM混合预测：结合梯度提升和时间序列模型的优势，提高预测稳定性',
            'features': ['XGBoost', 'LSTM', '混合学习', '特征融合', '趋势过滤', '波动率过滤', '跳空过滤']
        },
        'mtsf_net': {
            'name': 'MTSF-Net多尺度融合策略',
            'icon': '🎯',
            'description': '多尺度时序融合网络：融合短/中/长期特征，捕捉多尺度模式，适合波段操作',
            'features': ['多尺度特征', '时序融合', '深度学习', '自适应权重', '趋势捕捉', '波动率过滤', '成交量确认']
        },
        'vpan_net': {
            'name': 'VPAN量价注意力策略',
            'icon': '🔮',
            'description': '量价注意力预测网络：使用Transformer架构，重点关注量价关系和市场情绪，适合趋势跟踪',
            'features': ['Transformer', '注意力机制', '量价关系', '市场情绪', '资金流向', '趋势跟踪', '深度学习']
        }
    }

    @staticmethod
    def passes_global_buy_filters(df, i, strategy_name, config, market_state, signal_strength):
        try:
            min_strength = float(config.get('minSignalStrength', 0) or 0)
        except Exception:
            min_strength = 0
        if min_strength > 0 and signal_strength < min_strength:
            return False

        if market_state == 'bear_market' and config.get('blockBuyInBear', False):
            try:
                need = float(config.get('minStrengthBear', 70) or 70)
            except Exception:
                need = 70
            if signal_strength < need:
                return False

        if market_state == 'choppy_market' and config.get('blockBuyInChoppy', False):
            try:
                need = float(config.get('minStrengthChoppy', 60) or 60)
            except Exception:
                need = 60
            if signal_strength < need:
                return False

        apply_trend = config.get('enableTrendFilter', False)
        if apply_trend and strategy_name in {
            'deep_fusion',
            'volume_breakout',
            'trend_enhanced',
            'macd_divergence',
            'momentum_rotation',
            'turtle_enhanced',
            'xgboost_ml',
            'lstm_ml',
            'ensemble_ml',
        }:
            try:
                d = df.iloc[i]
                close = float(d.get('close', 0) or 0)
                ma20 = d.get('ma20', None)
                if ma20 is None or pd.isna(ma20):
                    return True
                ma20 = float(ma20)
                if close <= ma20:
                    return False
                lookback = int(config.get('trendSlopeLookback', 5) or 5)
                j = max(0, i - max(2, lookback))
                ma20_prev = df.iloc[j].get('ma20', None)
                if ma20_prev is None or pd.isna(ma20_prev):
                    return True
                if float(ma20_prev) >= ma20:
                    return False
            except Exception:
                return False

        apply_vol_ratio = config.get('enableVolRatioFilter', False)
        if apply_vol_ratio and strategy_name in {
            'deep_fusion',
            'volume_breakout',
            'trend_enhanced',
            'momentum_rotation',
            'turtle_enhanced',
            'xgboost_ml',
            'lstm_ml',
            'ensemble_ml',
        }:
            try:
                if 'volume' in df.columns:
                    cur_vol = float(df.iloc[i].get('volume', 0) or 0)
                    start = max(0, i - 5)
                    hist = df['volume'].iloc[start:i]
                    avg = float(hist.mean()) if len(hist) > 0 else 0
                    if avg > 0 and cur_vol > 0:
                        ratio = cur_vol / avg
                        try:
                            need = float(config.get('minVolRatio', 1.05) or 1.05)
                        except Exception:
                            need = 1.05
                        if ratio < need:
                            return False
            except Exception:
                pass

        return True
    
    @staticmethod
    def xgboost_extract_features(df, i):
        """提取XGBoost特征 - 优化版，增强趋势识别能力"""
        if i < 30:
            return None
        
        d = df.iloc[i]
        prev = df.iloc[i-1]
        
        try:
            close = float(d.get('close', 0))
            if close <= 0:
                return None
            
            open_price = float(d.get('open', close))
            prev_close = float(prev.get('close', close))
            high = float(d.get('high', close))
            low = float(d.get('low', close))
            volume = float(d.get('volume', 0))
            
            # 基础指标
            rsi = float(d.get('rsi', 50))
            macd = float(d.get('macd', 0))
            macd_hist = float(d.get('macdHist', 0))
            ma5 = float(d.get('ma5', close))
            ma10 = float(d.get('ma10', close))
            ma20 = float(d.get('ma20', close))
            ma60 = float(d.get('ma60', close))
            vol_ma5 = float(d.get('volMa5', volume))
            
            # 衍生特征
            gap = (open_price - prev_close) / prev_close if prev_close > 0 else 0
            atr = float(d.get('atr', (high - low) * 0.5))
            atr_pct = atr / close if close > 0 else 0
            body_pct = abs(close - open_price) / open_price if open_price > 0 else 0
            
            # 动量特征 - 增加多周期
            mom3 = (close - float(df.iloc[i-3]['close'])) / float(df.iloc[i-3]['close']) if i >= 3 else 0
            mom5 = (close - float(df.iloc[i-5]['close'])) / float(df.iloc[i-5]['close']) if i >= 5 else 0
            mom10 = (close - float(df.iloc[i-10]['close'])) / float(df.iloc[i-10]['close']) if i >= 10 else 0
            mom20 = (close - float(df.iloc[i-20]['close'])) / float(df.iloc[i-20]['close']) if i >= 20 else 0
            
            # 价格位置 - 扩展到60日
            high20 = df['high'].iloc[i-19:i+1].max()
            low20 = df['low'].iloc[i-19:i+1].min()
            pos20 = (close - low20) / (high20 - low20) if high20 > low20 else 0.5
            
            high60 = df['high'].iloc[i-59:i+1].max()
            low60 = df['low'].iloc[i-59:i+1].min()
            pos60 = (close - low60) / (high60 - low60) if high60 > low60 else 0.5
            
            # 成交量比率
            vol_ratio = volume / vol_ma5 if vol_ma5 > 0 else 1
            
            # 均线斜率 - 多周期
            ma5_prev = float(df.iloc[i-3].get('ma5', ma5)) if i >= 3 else ma5
            ma10_prev = float(df.iloc[i-5].get('ma10', ma10)) if i >= 5 else ma10
            ma20_prev = float(df.iloc[i-5].get('ma20', ma20)) if i >= 5 else ma20
            ma60_prev = float(df.iloc[i-10].get('ma60', ma60)) if i >= 10 else ma60
            
            slope5 = (ma5 - ma5_prev) / ma5 if ma5 > 0 else 0
            slope10 = (ma10 - ma10_prev) / ma10 if ma10 > 0 else 0
            slope20 = (ma20 - ma20_prev) / ma20 if ma20 > 0 else 0
            slope60 = (ma60 - ma60_prev) / ma60 if ma60 > 0 else 0
            
            # 均线多头排列强度 - 新增特征
            ma_alignment = 0
            if ma5 > ma10 > ma20 > ma60:
                ma_alignment = 1
            elif ma5 > ma20 > ma60:
                ma_alignment = 0.5
            elif ma20 > ma60:
                ma_alignment = 0.25
            
            # 价格突破特征 - 新增
            breakout_strength = 0
            if close > ma20 and prev_close <= ma20:
                breakout_strength = 1  # 突破MA20
            if close > ma60 and prev_close <= ma60:
                breakout_strength = max(breakout_strength, 1)  # 突破MA60
            
            # 连续上涨天数 - 新增
            up_days = 0
            for j in range(max(0, i-10), i):  # 从i往前数10天
                if j > 0 and df.iloc[j]['close'] > df.iloc[j-1]['close']:
                    up_days += 1
                else:
                    break
            up_days_ratio = up_days / 10  # 归一化
            
            # 动量加速特征 - 新增
            mom_accel = mom5 - mom10 if mom10 != 0 else 0  # 短期动量vs中期动量
            
            # 成交量放大趋势 - 新增
            vol_trend = 0
            if i >= 5:
                vol_prev5 = df['volume'].iloc[i-5:i].mean()
                if vol_prev5 > 0:
                    vol_trend = volume / vol_prev5
            
            # 布林带位置
            boll_upper = float(d.get('bollUpper', close))
            boll_lower = float(d.get('bollLower', close))
            boll_pos = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5
            
            # KDJ指标
            k = float(d.get('kdjK', 50))
            d_kdj = float(d.get('kdjD', 50))
            j_kdj = float(d.get('kdjJ', 50))
            
            # MACD金叉信号 - 新增
            macd_prev = float(df.iloc[i-1].get('macd', 0)) if i >= 1 else 0
            macd_hist_prev = float(df.iloc[i-1].get('macdHist', 0)) if i >= 1 else 0
            macd_golden_cross = 0
            if macd > 0 and macd_prev <= 0:
                macd_golden_cross = 1  # MACD金叉
            if macd_hist > 0 and macd_hist_prev <= 0:
                macd_golden_cross = max(macd_golden_cross, 1)  # MACD柱状图转正
            
            # 特征向量 (归一化处理) - 扩展到28个特征
            features = [
                rsi / 100.0,                    # RSI归一化
                macd / close if close > 0 else 0,  # MACD归一化
                (close / ma5 - 1) if ma5 > 0 else 0,  # 相对MA5位置
                (close / ma10 - 1) if ma10 > 0 else 0,  # 相对MA10位置
                (close / ma20 - 1) if ma20 > 0 else 0,  # 相对MA20位置
                (close / ma60 - 1) if ma60 > 0 else 0,  # 相对MA60位置
                vol_ratio,                      # 成交量比率
                pos20,                          # 20日价格位置
                pos60,                          # 60日价格位置
                mom3,                           # 3日动量
                mom5,                           # 5日动量
                mom10,                          # 10日动量
                mom20,                          # 20日动量
                gap,                            # 跳空幅度
                atr_pct,                        # ATR百分比
                body_pct,                       # 实体百分比
                slope5,                         # MA5斜率
                slope10,                        # MA10斜率
                slope20,                        # MA20斜率
                slope60,                        # MA60斜率
                ma_alignment,                    # 均线多头排列强度
                breakout_strength,               # 突破强度
                up_days_ratio,                  # 连续上涨比例
                mom_accel,                      # 动量加速
                vol_trend,                      # 成交量趋势
                boll_pos,                       # 布林带位置
                k / 100.0,                      # KDJ K
                d_kdj / 100.0,                  # KDJ D
                j_kdj / 100.0,                  # KDJ J
                macd_hist / close if close > 0 else 0,  # MACD柱状图
                macd_golden_cross,              # MACD金叉信号
            ]
            
            # 检查特征有效性
            for f in features:
                if not isinstance(f, (int, float)) or pd.isna(f):
                    return None
            
            return features
            
        except Exception as e:
            return None
    
    @staticmethod
    def lstm_extract_sequence_features(df, i, seq_len=30):
        """提取LSTM时间序列特征 - 增强版"""
        if i < seq_len:
            return None
        
        try:
            sequence = []
            for j in range(i - seq_len, i):
                d = df.iloc[j]
                close = float(d.get('close', 0))
                if close <= 0:
                    return None
                
                open_price = float(d.get('open', close))
                high = float(d.get('high', close))
                low = float(d.get('low', close))
                volume = float(d.get('volume', 0))
                prev_close = float(df.iloc[j-1]['close']) if j > 0 else close
                prev_high = float(df.iloc[j-1]['high']) if j > 0 else high
                prev_low = float(df.iloc[j-1]['low']) if j > 0 else low
                
                rsi = float(d.get('rsi', 50))
                macd = float(d.get('macd', 0))
                macd_hist = float(d.get('macdHist', 0))
                ma5 = float(d.get('ma5', close))
                ma10 = float(d.get('ma10', close))
                ma20 = float(d.get('ma20', close))
                ma60 = float(d.get('ma60', close))
                vol_ma5 = float(d.get('volMa5', volume))
                
                boll_upper = float(d.get('bollUpper', close))
                boll_lower = float(d.get('bollLower', close))
                boll_mid = float(d.get('bollMid', close))
                
                kdj_k = float(d.get('kdjK', 50))
                kdj_d = float(d.get('kdjD', 50))
                kdj_j = float(d.get('kdjJ', 50))
                
                atr = float(d.get('atr', (high - low) * 0.5))
                
                returns = (close - prev_close) / prev_close if prev_close > 0 else 0
                returns_prev = (prev_close - float(df.iloc[j-2]['close'])) / float(df.iloc[j-2]['close']) if j > 1 else 0
                volume_ratio = volume / vol_ma5 if vol_ma5 > 0 else 1
                
                true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
                atr_ratio = atr / close if close > 0 else 0
                
                body_size = abs(close - open_price)
                upper_shadow = high - max(open_price, close)
                lower_shadow = min(open_price, close) - low
                body_ratio = body_size / close if close > 0 else 0
                upper_shadow_ratio = upper_shadow / close if close > 0 else 0
                lower_shadow_ratio = lower_shadow / close if close > 0 else 0
                
                price_position = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5
                
                momentum_5 = (close - float(df.iloc[j-5]['close'])) / float(df.iloc[j-5]['close']) if j >= 5 else 0
                momentum_10 = (close - float(df.iloc[j-10]['close'])) / float(df.iloc[j-10]['close']) if j >= 10 else 0
                
                ma_slope_5 = (ma5 - float(df.iloc[j-3].get('ma5', ma5))) / ma5 if j >= 3 and ma5 > 0 else 0
                ma_slope_20 = (ma20 - float(df.iloc[j-5].get('ma20', ma20))) / ma20 if j >= 5 and ma20 > 0 else 0
                
                gap_up = (open_price - prev_close) / prev_close if prev_close > 0 else 0
                
                features = [
                    close / ma20 if ma20 > 0 else 1,
                    (close - open_price) / open_price if open_price > 0 else 0,
                    (high - low) / close if close > 0 else 0,
                    returns,
                    returns_prev,
                    volume_ratio,
                    rsi / 100.0,
                    macd / close if close > 0 else 0,
                    macd_hist / close if close > 0 else 0,
                    ma5 / ma20 if ma20 > 0 else 1,
                    ma10 / ma20 if ma20 > 0 else 1,
                    ma20 / ma60 if ma60 > 0 else 1,
                    ma_slope_5,
                    ma_slope_20,
                    price_position,
                    kdj_k / 100.0,
                    kdj_d / 100.0,
                    kdj_j / 100.0,
                    atr_ratio,
                    body_ratio,
                    upper_shadow_ratio,
                    lower_shadow_ratio,
                    momentum_5,
                    momentum_10,
                    gap_up,
                ]
                sequence.append(features)
            
            return np.array(sequence)
        except Exception as e:
            return None
    
    @staticmethod
    def lstm_train_model(sequences, labels, config_params=None):
        """训练LSTM模型 - 增强版"""
        if len(sequences) < 50 or len(sequences) != len(labels):
            return None
        
        try:
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            X = np.array(sequences)
            y = np.array(labels)
            
            n_samples, seq_len, n_features = X.shape
            
            params = {
                'lstm_units': 128,
                'lstm_layers': 2,
                'dropout_rate': 0.4,
                'learning_rate': 0.0005,
                'epochs': 150,
                'batch_size': 64,
                'l2_reg': 0.005,
                'use_bidirectional': True,
                'use_attention': True,
            }
            
            if config_params:
                params.update(config_params)
            
            X_reshaped = X.reshape(-1, n_features)
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X_reshaped)
            X = X_normalized.reshape(n_samples, seq_len, n_features)
            
            pos_count = sum(y)
            neg_count = len(y) - pos_count
            class_weight = {0: 1.0, 1: neg_count / pos_count if pos_count > 0 else 1.0}
            
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            model = Sequential()
            
            if params.get('use_bidirectional', True):
                from tensorflow.keras.layers import Bidirectional
                
                for i in range(params['lstm_layers']):
                    return_sequences = i < params['lstm_layers'] - 1
                    if i == 0:
                        model.add(Bidirectional(LSTM(
                            params['lstm_units'],
                            return_sequences=return_sequences,
                            input_shape=(seq_len, n_features),
                            kernel_regularizer=l2(params['l2_reg'])
                        )))
                    else:
                        model.add(Bidirectional(LSTM(
                            params['lstm_units'],
                            return_sequences=return_sequences,
                            kernel_regularizer=l2(params['l2_reg'])
                        )))
                    model.add(Dropout(params['dropout_rate']))
                    model.add(BatchNormalization())
            else:
                for i in range(params['lstm_layers']):
                    return_sequences = i < params['lstm_layers'] - 1
                    if i == 0:
                        model.add(LSTM(
                            params['lstm_units'],
                            return_sequences=return_sequences,
                            input_shape=(seq_len, n_features),
                            kernel_regularizer=l2(params['l2_reg'])
                        ))
                    else:
                        model.add(LSTM(
                            params['lstm_units'],
                            return_sequences=return_sequences,
                            kernel_regularizer=l2(params['l2_reg'])
                        ))
                    model.add(Dropout(params['dropout_rate']))
                    model.add(BatchNormalization())
            
            model.add(Dense(64, activation='relu', kernel_regularizer=l2(params['l2_reg'])))
            model.add(Dropout(params['dropout_rate']))
            model.add(Dense(32, activation='relu', kernel_regularizer=l2(params['l2_reg'])))
            model.add(Dropout(params['dropout_rate']))
            model.add(Dense(1, activation='linear'))  # 改为线性激活，支持回归预测
            
            optimizer = Adam(learning_rate=params['learning_rate'], clipnorm=1.0)
            model.compile(
                optimizer=optimizer,
                loss='mse',  # 改为均方误差损失，适合回归任务
                metrics=['mae']  # 使用平均绝对误差作为评估指标
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',  # 改为监控验证损失
                mode='min',          # 损失越小越好
                patience=15,
                restore_best_weights=True,
                verbose=0
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=1e-7,
                verbose=0
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                class_weight=class_weight,
                verbose=0
            )
            
            pred = model.predict(X, verbose=0).flatten()
            # 回归模型输出连续值，不再使用固定阈值
            mae = np.mean(np.abs(pred - y))  # 计算平均绝对误差
            
            # 基于预测值和标准差生成交易信号
            pred_mean = np.mean(pred)
            pred_std = np.std(pred)
            
            # 动态阈值：预测值高于均值+0.5标准差视为买入信号
            dynamic_threshold = pred_mean + 0.5 * pred_std
            pred_labels = (pred >= dynamic_threshold).astype(int)
            
            accuracy = (pred_labels == y).mean()
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y, pred_labels, zero_division=0)
            recall = recall_score(y, pred_labels, zero_division=0)
            f1 = f1_score(y, pred_labels, zero_division=0)
            
            return {
                'model': model,
                'scaler': scaler,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'base_rate': y.mean(),
                'class_weight': class_weight,
                'mae': mae,  # 新增平均绝对误差
                'pred_mean': pred_mean,  # 新增预测均值
                'pred_std': pred_std,  # 新增预测标准差
                'dynamic_threshold': dynamic_threshold,  # 新增动态阈值
                'history': {
                    'loss': history.history['loss'][-1],
                    'val_loss': history.history['val_loss'][-1],
                    'accuracy': history.history['accuracy'][-1],
                    'val_accuracy': history.history['val_accuracy'][-1],
                    'auc': history.history.get('auc', [0])[-1],
                    'val_auc': history.history.get('val_auc', [0])[-1],
                }
            }
            
        except Exception as e:
            print(f"LSTM训练错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def lstm_predict(model_dict, sequence):
        """使用LSTM模型预测 - 增强版"""
        if model_dict is None or 'model' not in model_dict:
            return None
        
        try:
            import numpy as np
            X = np.array([sequence])
            
            if 'scaler' in model_dict and model_dict['scaler'] is not None:
                n_samples, seq_len, n_features = X.shape
                X_reshaped = X.reshape(-1, n_features)
                X_normalized = model_dict['scaler'].transform(X_reshaped)
                X = X_normalized.reshape(n_samples, seq_len, n_features)
            
            prob = model_dict['model'].predict(X, verbose=0)[0, 0]
            return prob
        except Exception as e:
            return None
    
    @staticmethod
    def xgboost_train_model(features, labels, config_params=None):
        """训练XGBoost模型"""
        if len(features) < 50 or len(features) != len(labels):
            return None
        
        try:
            import numpy as np
            
            X = np.array(features)
            y = np.array(labels)
            
            # 默认参数
            params = {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'scale_pos_weight': 1.0
            }
            
            if config_params:
                params.update(config_params)
            
            # 计算正负样本比例，调整scale_pos_weight
            pos_count = sum(y)
            neg_count = len(y) - pos_count
            if neg_count > 0:
                params['scale_pos_weight'] = neg_count / pos_count
            
            model = xgb.XGBClassifier(**params)
            model.fit(X, y)
            
            # 计算训练准确率
            pred = model.predict_proba(X)[:, 1]
            threshold = 0.5
            pred_labels = (pred >= threshold).astype(int)
            accuracy = (pred_labels == y).mean()
            
            return {
                'model': model,
                'accuracy': accuracy,
                'feature_importance': model.feature_importances_.tolist(),
                'base_rate': pos_count / len(y)
            }
            
        except Exception as e:
            print(f"XGBoost训练错误: {e}")
            return None
    
    @staticmethod
    def xgboost_predict(model_dict, features):
        """使用XGBoost模型预测"""
        if model_dict is None or 'model' not in model_dict:
            return None
        
        try:
            import numpy as np
            X = np.array([features])
            prob = model_dict['model'].predict_proba(X)[0, 1]
            return prob
        except Exception as e:
            return None
    
    # ==================== MTSF-Net 多尺度时序融合网络 ====================
    @staticmethod
    def mtsf_extract_features(df, idx, lookback=30):
        """提取MTSF-Net特征 - 多尺度时序特征"""
        try:
            if idx < lookback:
                return None
            
            features = []
            window = df.iloc[idx-lookback:idx]
            
            # 价格特征 (原始价格、对数收益率、波动率)
            prices = window['close'].values
            features.extend([
                prices[-1],  # 最新价格
                np.mean(prices),  # 均价
                np.std(prices),  # 波动率
                (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0,  # 区间收益
            ])
            
            # 多尺度均线特征
            for ma_period in [5, 10, 20, 60]:
                if len(prices) >= ma_period:
                    ma = np.mean(prices[-ma_period:])
                    features.append((prices[-1] - ma) / ma if ma > 0 else 0)
                else:
                    features.append(0)
            
            # 技术指标特征
            curr = df.iloc[idx]
            features.extend([
                curr.get('rsi', 50) / 100,  # RSI归一化
                curr.get('macd', 0),  # MACD
                curr.get('macdHist', 0),  # MACD柱状图
                min(curr.get('volume', 0) / curr.get('volMa5', 1), 5),  # 成交量比
                (curr.get('high', curr['close']) - curr.get('low', curr['close'])) / curr['close'] if curr['close'] > 0 else 0,  # 日内振幅
            ])
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            return None
    
    @staticmethod
    def mtsf_train_model(sequences, labels, lookback=30):
        """训练MTSF-Net模型 - 多尺度时序融合网络"""
        if len(sequences) < 80 or len(sequences) != len(labels):
            return None
        
        try:
            import numpy as np
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Concatenate, Attention, Reshape
            from tensorflow.keras.optimizers import Adam
            
            X = np.array(sequences)
            y = np.array(labels)
            
            n_features = X.shape[1]
            
            # 构建多分支网络
            # 分支1: 全连接层处理特征
            inputs = Input(shape=(n_features,))
            
            # 特征提取层
            x = Dense(128, activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            x = Dense(32, activation='relu')(x)
            x = BatchNormalization()(x)
            
            # 输出层
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # 类别权重
            pos_count = sum(y)
            neg_count = len(y) - pos_count
            class_weight = {0: 1.0, 1: neg_count / pos_count if pos_count > 0 else 1.0}
            
            # 训练
            model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                verbose=0,
                class_weight=class_weight,
                validation_split=0.2
            )
            
            # 计算准确率
            pred = model.predict(X, verbose=0)[:, 0]
            pred_labels = (pred >= 0.5).astype(int)
            accuracy = (pred_labels == y).mean()
            
            return {
                'model': model,
                'accuracy': accuracy,
                'base_rate': pos_count / len(y)
            }
            
        except Exception as e:
            print(f"MTSF-Net训练错误: {e}")
            return None
    
    @staticmethod
    def mtsf_predict(model_dict, features):
        """使用MTSF-Net模型预测"""
        if model_dict is None or 'model' not in model_dict:
            return None
        
        try:
            import numpy as np
            X = np.array([features])
            prob = model_dict['model'].predict(X, verbose=0)[0, 0]
            return prob
        except Exception as e:
            return None
    
    # ==================== VPAN 量价注意力预测网络 ====================
    @staticmethod
    def vpan_extract_sequence(df, idx, seq_len=20):
        """提取VPAN序列特征 - 价格和成交量序列"""
        try:
            if idx < seq_len:
                return None
            
            window = df.iloc[idx-seq_len:idx]
            
            # 构建多维度序列特征
            features = []
            for _, row in window.iterrows():
                features.append([
                    row['close'],  # 收盘价
                    row['volume'],  # 成交量
                    row.get('high', row['close']),  # 最高价
                    row.get('low', row['close']),  # 最低价
                    row.get('rsi', 50),  # RSI
                    row.get('macd', 0),  # MACD
                ])
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            return None
    
    @staticmethod
    def vpan_train_model(sequences, labels, seq_len=20):
        """训练VPAN模型 - 量价注意力网络"""
        if len(sequences) < 60 or len(sequences) != len(labels):
            return None
        
        try:
            import numpy as np
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
            from tensorflow.keras.optimizers import Adam
            
            X = np.array(sequences)
            y = np.array(labels)
            
            n_features = X.shape[2]
            
            # 构建Transformer-like架构
            inputs = Input(shape=(seq_len, n_features))
            
            # LSTM编码
            x = LSTM(64, return_sequences=True)(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # 多头注意力机制
            attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
            x = LayerNormalization()(attention_output + x)  # 残差连接
            
            # 全局平均池化
            x = GlobalAveragePooling1D()(x)
            
            # 全连接层
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = Dense(32, activation='relu')(x)
            x = BatchNormalization()(x)
            
            # 输出层
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # 类别权重
            pos_count = sum(y)
            neg_count = len(y) - pos_count
            class_weight = {0: 1.0, 1: neg_count / pos_count if pos_count > 0 else 1.0}
            
            # 训练
            model.fit(
                X, y,
                epochs=60,
                batch_size=32,
                verbose=0,
                class_weight=class_weight,
                validation_split=0.2
            )
            
            # 计算准确率
            pred = model.predict(X, verbose=0)[:, 0]
            pred_labels = (pred >= 0.5).astype(int)
            accuracy = (pred_labels == y).mean()
            
            return {
                'model': model,
                'accuracy': accuracy,
                'base_rate': pos_count / len(y)
            }
            
        except Exception as e:
            print(f"VPAN训练错误: {e}")
            return None
    
    @staticmethod
    def vpan_predict(model_dict, sequence):
        """使用VPAN模型预测"""
        if model_dict is None or 'model' not in model_dict:
            return None
        
        try:
            import numpy as np
            X = np.array([sequence])
            prob = model_dict['model'].predict(X, verbose=0)[0, 0]
            return prob
        except Exception as e:
            return None
    
    @staticmethod
    def execute_strategy(data, strategy_name, config, stock_features=None):
        """执行策略 - 优化版，支持自适应参数调整"""
        df = pd.DataFrame(data)
        try:
            required_cols = {'close', 'high', 'low', 'volume', 'ma20', 'rsi', 'macd', 'macdHist', 'volMa5'}
            if not required_cols.issubset(set(df.columns)):
                df = calculate_indicators(df)
        except Exception:
            pass
        signals = []
        position = 0
        entry_price = 0
        entry_index = 0
        highest_since_entry = 0
        highest_body_since_entry = 0.0
        peak_profit_since_entry = 0.0
        last_sell_index = -999  # 记录上一次卖出的索引
        
        # 分析股票特征并应用自适应参数
        if stock_features is None:
            stock_features = StockFeatureAnalyzer.analyze_stock_features(df)
        
        # 应用自适应参数调整
        config = StockFeatureAnalyzer.get_adaptive_params(config, stock_features)
        
        buy_cooldown = int(config.get('buyCooldown', 0))  # 买入冷却天数，默认0天

        if 'minConditions' not in config:
            config['minConditions'] = 2
        if 'scoreThreshold' not in config:
            config['scoreThreshold'] = 60

        # 获取市场状态（用于动态调整策略参数）
        market_state = 'neutral'
        if len(df) > 0 and 'market_state' in df.columns:
            market_state = df.iloc[-1].get('market_state', 'neutral')
        
        # 市场状态权重调整
        market_adjustments = {
            'bull_market': {'aggressive': 1.1, 'conservative': 0.9},
            'bear_market': {'aggressive': 0.8, 'conservative': 1.2},
            'sideways_market': {'aggressive': 0.9, 'conservative': 1.1},
            'volatile_market': {'aggressive': 1.0, 'conservative': 1.0},
            'choppy_market': {'aggressive': 0.7, 'conservative': 1.3}
        }
        aggressive_factor = market_adjustments.get(market_state, {}).get('aggressive', 1.0)
        conservative_factor = market_adjustments.get(market_state, {}).get('conservative', 1.0)

        for i in range(60, len(df)):
            d = df.iloc[i]
            prev = df.iloc[i-1]
            
            if d.get('ma20', None) is None or pd.isna(d.get('ma20')) or d.get('rsi', None) is None or pd.isna(d.get('rsi')):
                continue
            
            buy_signal = False
            sell_signal = False
            signal_strength = 50
            sell_reason = ''
            
            # 冷却期检查
            in_cooldown = (i - last_sell_index) <= buy_cooldown

            # 策略逻辑 - 优化版
            if not in_cooldown and position == 0:
                if strategy_name == 'deep_fusion':
                    # 深度融合策略 - 升级版：多条件加权评分 + 动态确认机制
                    score = 0
                    max_score = 0
                    confirmations = 0  # 确认信号计数
                    
                    # 1. MA对齐条件（可配置开关）- 权重增加
                    if config.get('useMa5AboveMa20', True):
                        max_score += 25
                        if d['ma5'] and d['ma20'] and d['ma5'] > d['ma20']:
                            score += 25
                            confirmations += 1
                            # 多头排列加分
                            if d.get('ma10') and d.get('ma20') and d['ma10'] > d['ma20']:
                                score += 5
                    
                    # 2. 价格位置条件（可配置开关）
                    if config.get('usePriceAboveMa5', True):
                        max_score += 15
                        if d['close'] > d.get('ma5', d['close']):
                            score += 15
                            confirmations += 1
                    
                    # 3. MACD动量条件（可配置开关）- 增强判断
                    if config.get('useMacdRising', True):
                        max_score += 25
                        macd_threshold = config.get('macdThreshold', 0)
                        macd_val = d.get('macd', None)
                        prev_macd = prev.get('macd', None)
                        if macd_val is not None and not pd.isna(macd_val) and float(macd_val) > macd_threshold:
                            # MACD柱状图上升
                            if prev_macd is not None and not pd.isna(prev_macd) and float(macd_val) > float(prev_macd):
                                score += 20
                                confirmations += 1
                            # DIF上穿DEA金叉
                            hist = d.get('macdHist', None)
                            prev_hist = prev.get('macdHist', None)
                            if hist is not None and prev_hist is not None and not pd.isna(hist) and not pd.isna(prev_hist):
                                if float(hist) > 0 and float(prev_hist) <= 0:
                                    score += 10  # 金叉加分
                    
                    # 4. RSI超卖条件（可配置开关）- 动态阈值
                    if config.get('useRsiBelow35', True):
                        max_score += 15
                        rsi_threshold = config.get('rsiThreshold', 35)
                        rsi_val = d.get('rsi', None)
                        if rsi_val is not None and not pd.isna(rsi_val) and float(rsi_val) < rsi_threshold:
                            score += 15
                            confirmations += 1
                            # 严重超卖额外加分
                            if float(rsi_val) < 25:
                                score += 10
                    
                    # 5. 成交量确认条件（可配置开关）- 增强判断
                    if config.get('useVolumeAboveMa', True):
                        max_score += 20
                        vol_multi = config.get('volumeMulti', 1.5)
                        vol_val = d.get('volume', None)
                        vol_ma5 = d.get('volMa5', None)
                        if vol_val is not None and vol_ma5 is not None and not pd.isna(vol_val) and not pd.isna(vol_ma5) and float(vol_ma5) > 0 and float(vol_val) > float(vol_ma5) * vol_multi:
                            score += 15
                            confirmations += 1
                            # 成交量持续放大
                            vol_ma10 = d.get('volMa10', None)
                            if vol_ma10 is not None and not pd.isna(vol_ma10) and float(vol_ma5) > float(vol_ma10):
                                score += 5
                    
                    # 6. 布林带位置条件（可配置开关）- 优化判断
                    if config.get('usePriceBelowBoll', True):
                        max_score += 15
                        close_val = d.get('close', None)
                        boll_down = d.get('bollDown', None)
                        if close_val is not None and boll_down is not None and not pd.isna(close_val) and not pd.isna(boll_down) and float(close_val) < float(boll_down):
                            score += 15
                            confirmations += 1
                            # 触及下轨且反弹
                            if d.get('open', None) is not None and not pd.isna(d.get('open')) and float(close_val) > float(d.get('open')):
                                score += 5
                    
                    # 7. 新增：ADX趋势强度确认
                    if config.get('useAdxConfirm', True):
                        max_score += 10
                        adx_val = d.get('adx', None)
                        if adx_val is not None and not pd.isna(adx_val) and float(adx_val) > 20:
                            score += 10
                            if d.get('plus_di') and d.get('minus_di') and d['plus_di'] > d['minus_di']:
                                score += 5
                    
                    # 8. 新增：价格动量确认
                    if config.get('useMomentumConfirm', True):
                        max_score += 10
                        if i >= 5:
                            price_change_5d = (d['close'] - df.iloc[i-5]['close']) / df.iloc[i-5]['close'] * 100
                            if -10 < price_change_5d < 5:  # 近期没有大涨大跌
                                score += 10
                    
                    # 动态阈值调整 - 基于确认信号数量
                    base_threshold = config.get('mlScoreThreshold', 60)
                    # 确认信号越多，阈值可以适当降低
                    confirmation_bonus = confirmations * 2
                    adjusted_threshold = (base_threshold - confirmation_bonus) * conservative_factor
                    adjusted_threshold = max(45, min(75, adjusted_threshold))  # 限制在45-75之间
                    
                    signal_strength = (score / max_score * 100) if max_score > 0 else 50
                    
                    # 买入条件：达到阈值且至少有3个确认信号
                    min_confirmations = config.get('minConfirmations', 3)
                    if signal_strength >= adjusted_threshold and confirmations >= min_confirmations:
                        buy_signal = True
                        # 根据确认信号数量调整信号强度
                        signal_strength = min(95, signal_strength + confirmations * 3)
                    
                elif strategy_name == 'volume_breakout':
                    # 量价突破策略 - 升级版：多维度突破确认
                    conditions_met = 0
                    total_conditions = 0
                    breakout_score = 0
                    
                    # 1. 价格突破条件（核心）- 增强判断
                    if config.get('usePriceBreakout', True):
                        total_conditions += 1
                        breakout_period = config.get('breakoutPeriod', 20)
                        high_col = f'high{breakout_period}'
                        
                        if d.get(high_col) and d['close'] > d[high_col] * 0.995:  # 允许0.5%的误差
                            conditions_met += 1
                            breakout_score += 30
                            # 创近期新高加分
                            if d['close'] > d.get('high60', d['close']):
                                breakout_score += 10
                    
                    # 2. 成交量放大条件（核心）- 动态阈值
                    if config.get('useVolumeUp', True):
                        total_conditions += 1
                        vol_multi = config.get('volumeMulti', 2.0)
                        vol_ratio = d['volume'] / d['volMa5'] if d.get('volMa5') and d['volMa5'] > 0 else 0
                        
                        if vol_ratio > vol_multi:
                            conditions_met += 1
                            breakout_score += 25
                            # 成交量持续放大
                            if d.get('volMa5') and d.get('volMa10') and d['volMa5'] > d['volMa10'] * 1.2:
                                breakout_score += 10
                            # 异常放量过滤（避免假突破）
                            if vol_ratio < 5:  # 成交量不超过5倍，避免极端情况
                                breakout_score += 5
                    
                    # 3. 趋势确认条件 - 确保突破方向与趋势一致
                    if config.get('useTrendConfirm', True):
                        total_conditions += 1
                        if d.get('ma5') and d.get('ma20') and d['ma5'] > d['ma20']:
                            conditions_met += 1
                            breakout_score += 15
                            # 多头排列加分
                            if d.get('ma10') and d['ma10'] > d['ma20']:
                                breakout_score += 5
                    
                    # 4. 波动率确认 - 避免在震荡市中交易
                    if config.get('useVolatilityConfirm', True):
                        total_conditions += 1
                        if d.get('bollWidth') and d['bollWidth'] > 0.05:  # 布林带宽度足够
                            conditions_met += 1
                            breakout_score += 10
                    
                    # 5. 动能确认 - RSI不过买
                    if config.get('useMomentumConfirm', True):
                        total_conditions += 1
                        rsi_max = config.get('rsiMax', 70)
                        if d.get('rsi') and d['rsi'] < rsi_max:
                            conditions_met += 1
                            breakout_score += 10
                    
                    # 6. 阳线确认（加分项）
                    if config.get('useBullishCandle', True):
                        body_pct = abs(d['close'] - d['open']) / d['open'] * 100 if d['open'] > 0 else 0
                        if d['close'] > d['open']:  # 阳线
                            breakout_score += 10
                            if body_pct > 2:  # 大阳线额外加分
                                breakout_score += 5
                        # 上影线不能太长
                        upper_shadow = (d['high'] - max(d['close'], d['open'])) / d['open'] * 100 if d['open'] > 0 else 0
                        if upper_shadow < 2:
                            breakout_score += 5
                    
                    # 7. 新增：筹码分布确认 - 突破阻力位
                    if config.get('useChipConfirm', True):
                        if d.get('price_position_20') and d['price_position_20'] > 70:
                            breakout_score += 10
                    
                    # 动态买入条件
                    min_conditions = config.get('minConditions', 3)
                    score_threshold = config.get('scoreThreshold', 70)
                    
                    if conditions_met >= min_conditions and breakout_score >= score_threshold:
                        buy_signal = True
                        signal_strength = min(95, breakout_score)
                    
                elif strategy_name == 'oversold_rebound':
                    # 超跌反弹策略 - 升级版：多维度超卖确认 + 反弹信号检测
                    conditions_met = 0
                    total_conditions = 0
                    rebound_score = 0
                    
                    # 1. RSI超卖条件（核心）- 动态阈值
                    if config.get('useRsiOversold', True):
                        total_conditions += 1
                        rsi_threshold = config.get('rsiOversold', 30)
                        if d.get('rsi') and d['rsi'] < rsi_threshold:
                            conditions_met += 1
                            rebound_score += 25
                            # 严重超卖加分
                            if d['rsi'] < 20:
                                rebound_score += 10
                            # RSI开始回升（底背离迹象）
                            if prev.get('rsi') and d['rsi'] > prev['rsi']:
                                rebound_score += 10
                    
                    # 2. 触及布林下轨（可配置开关）- 增强判断
                    if config.get('useTouchLowerBoll', True):
                        total_conditions += 1
                        boll_offset = config.get('bollingerOffset', 0.02)
                        if d.get('bollDown') and d['close'] < (d['bollDown'] * (1 + boll_offset)):
                            conditions_met += 1
                            rebound_score += 20
                            # 触及下轨且收阳线
                            if d['close'] > d['open']:
                                rebound_score += 10
                    
                    # 3. 价格位置条件 - 接近近期低点
                    if config.get('useNearLow', True):
                        total_conditions += 1
                        near_low_ratio = config.get('nearLowRatio', 1.03)
                        if d.get('low20') and d['close'] < d['low20'] * near_low_ratio:
                            conditions_met += 1
                            rebound_score += 15
                    
                    # 4. 成交量条件 - 缩量后放量
                    if config.get('useVolumePattern', True):
                        total_conditions += 1
                        # 缩量（地量）
                        vol_ratio = d['volume'] / d['volMa5'] if d.get('volMa5') and d['volMa5'] > 0 else 1
                        if vol_ratio < 0.8:  # 缩量
                            rebound_score += 10
                            conditions_met += 1
                        # 或者放量反弹
                        elif d['close'] > d['open'] and vol_ratio > 1.2:
                            rebound_score += 15
                            conditions_met += 1
                    
                    # 5. K线形态确认 - 看涨反转形态
                    if config.get('useCandlePattern', True):
                        body = d['close'] - d['open']
                        lower_shadow = d['open'] - d['low'] if body > 0 else d['close'] - d['low']
                        upper_shadow = d['high'] - d['close'] if body > 0 else d['high'] - d['open']
                        body_size = abs(body)
                        
                        # 锤子线形态
                        if lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
                            rebound_score += 15
                        # 启明星形态（简化版）
                        if i >= 2:
                            prev2 = df.iloc[i-2]
                            if prev2['close'] < prev2['open']:  # 前一天阴线
                                if body > 0 and d['close'] > (prev2['open'] + prev2['close']) / 2:
                                    rebound_score += 10
                    
                    # 6. MACD底背离检测
                    if config.get('useMacdDivergence', True):
                        if d.get('macd') and prev.get('macd'):
                            # 价格创新低但MACD未创新低
                            if i >= 5:
                                price_low_now = d['close'] < df.iloc[i-5:i]['close'].min()
                                price_low_prev = df.iloc[i-5]['close'] < df.iloc[i-10:i-5]['close'].min() if i >= 10 else False
                                macd_now = d['macd']
                                macd_prev = df.iloc[i-5]['macd'] if i >= 5 else 0
                                
                                if price_low_now and not price_low_prev and macd_now > macd_prev:
                                    rebound_score += 20
                                    conditions_met += 1
                    
                    # 7. 趋势过滤 - 避免在强下跌趋势中买入
                    if config.get('useTrendFilter', True):
                        if d.get('adx') and d['adx'] < 30:  # 趋势不强
                            rebound_score += 10
                        # 或者趋势开始转弱
                        if d.get('minus_di') and d.get('plus_di') and d['minus_di'] < prev.get('minus_di', 100):
                            rebound_score += 5
                    
                    # 动态买入条件
                    min_conditions = config.get('minConditions', 3)
                    score_threshold = config.get('scoreThreshold', 65)
                    
                    if conditions_met >= min_conditions and rebound_score >= score_threshold:
                        buy_signal = True
                        signal_strength = min(95, rebound_score)
                    
                elif strategy_name == 'trend_enhanced':
                    # 趋势增强策略 - 升级版：多时间框架趋势确认
                    conditions_met = 0
                    total_conditions = 0
                    trend_score = 0
                    
                    # 1. 短期均线多头排列（可配置开关）
                    if config.get('useShortMaAlignment', True):
                        total_conditions += 1
                        if d.get('ma5') and d.get('ma10') and d.get('ma20'):
                            if d['ma5'] > d['ma10'] > d['ma20']:
                                conditions_met += 1
                                trend_score += 25
                                # 完全多头排列加分
                                if d.get('ma60') and d['ma20'] > d['ma60']:
                                    trend_score += 10
                    
                    # 2. 价格相对均线位置（可配置开关）
                    if config.get('usePriceAboveMa', True):
                        total_conditions += 1
                        price_above_count = 0
                        for ma in ['ma5', 'ma10', 'ma20']:
                            if d.get(ma) and d['close'] > d[ma]:
                                price_above_count += 1
                        if price_above_count >= 2:
                            conditions_met += 1
                            trend_score += 15 + price_above_count * 3
                    
                    # 3. MACD正向且动能增强（可配置开关）
                    if config.get('useMacdPositive', True):
                        total_conditions += 1
                        macd_threshold = config.get('macdThreshold', 0)
                        macd_val = d.get('macd', None)
                        if macd_val is not None and not pd.isna(macd_val) and float(macd_val) > macd_threshold:
                            conditions_met += 1
                            trend_score += 15
                            # MACD柱状图扩大
                            hist = d.get('macdHist', None)
                            prev_hist = prev.get('macdHist', None)
                            if hist is not None and prev_hist is not None and not pd.isna(hist) and not pd.isna(prev_hist) and float(hist) > float(prev_hist):
                                trend_score += 10
                            # DIF上穿DEA金叉
                            if hist is not None and prev_hist is not None and not pd.isna(hist) and not pd.isna(prev_hist):
                                if float(hist) > 0 and float(prev_hist) <= 0:
                                    trend_score += 15
                    
                    # 4. ADX趋势强度确认（新增）
                    if config.get('useAdxConfirm', True):
                        total_conditions += 1
                        if d.get('adx'):
                            if d['adx'] > 25:  # 强趋势
                                conditions_met += 1
                                trend_score += 15
                                if d['adx'] > 40:  # 极强趋势
                                    trend_score += 10
                            # 趋势方向确认
                            if d.get('plus_di') and d.get('minus_di') and d['plus_di'] > d['minus_di']:
                                trend_score += 10
                    
                    # 5. 成交量趋势确认（可配置开关）
                    if config.get('useVolumeConfirm', True):
                        total_conditions += 1
                        if d.get('volMa5') and d.get('volMa10'):
                            if d['volMa5'] > d['volMa10']:  # 成交量均线上升
                                conditions_met += 1
                                trend_score += 10
                            # 量价配合
                            if d['volume'] > d['volMa5'] and d['close'] > d['open']:
                                trend_score += 5
                    
                    # 6. 布林带趋势确认（可配置开关）
                    if config.get('useBollConfirm', True):
                        # 价格在中轨上方
                        if d.get('bollMid') and d['close'] > d['bollMid']:
                            trend_score += 10
                        # 布林带开口向上
                        if d.get('bollWidth') and prev.get('bollWidth') and d['bollWidth'] > prev['bollWidth']:
                            trend_score += 5
                    
                    # 7. 价格动量确认（可配置开关）
                    if config.get('useMomentumConfirm', True):
                        if i >= 10:
                            price_change_10d = (d['close'] - df.iloc[i-10]['close']) / df.iloc[i-10]['close'] * 100
                            if price_change_10d > 5:  # 10日涨幅超过5%
                                trend_score += 10
                            if price_change_10d > 10:  # 10日涨幅超过10%
                                trend_score += 5
                    
                    # 8. 趋势持续性检查 - 避免假突破
                    if config.get('useTrendPersistence', True):
                        if i >= 5:
                            # 检查过去5天是否整体向上
                            recent_prices = [df.iloc[j]['close'] for j in range(i-4, i+1)]
                            higher_count = sum(1 for j in range(1, len(recent_prices)) if recent_prices[j] > recent_prices[j-1])
                            if higher_count >= 3:  # 至少3天上涨
                                trend_score += 10
                    
                    # 动态买入条件
                    min_conditions = config.get('minConditions', 3)
                    score_threshold = config.get('scoreThreshold', 70)
                    
                    if conditions_met >= min_conditions and trend_score >= score_threshold:
                        buy_signal = True
                        signal_strength = min(95, trend_score)
                    
                elif strategy_name == 'macd_divergence':
                    # MACD背离策略 - 升级版：多周期背离检测 + 确认机制
                    divergence_score = 0
                    conditions_met = 0
                    
                    # 需要足够的历史数据
                    lookback_period = config.get('lookbackPeriod', 30)
                    if i < lookback_period * 2:
                        continue
                    
                    # 1. 底背离检测（核心）- 价格创新低但MACD未创新低
                    if config.get('useBullishDivergence', True):
                        # 寻找近期低点
                        recent_lows_price = []
                        recent_lows_macd = []
                        
                        for j in range(i - lookback_period, i + 1):
                            if j < 2:
                                continue
                            # 局部低点判断
                            if (df.iloc[j]['close'] < df.iloc[j-1]['close'] and 
                                df.iloc[j]['close'] < df.iloc[j-2]['close'] and
                                df.iloc[j]['close'] < df.iloc[j+1]['close'] if j < len(df) - 1 else True):
                                recent_lows_price.append((j, df.iloc[j]['close'], df.iloc[j]['macd']))
                        
                        # 检测背离
                        if len(recent_lows_price) >= 2:
                            # 比较最近两个低点
                            latest_low = recent_lows_price[-1]
                            prev_low = recent_lows_price[-2]
                            
                            # 价格创新低但MACD未创新低（底背离）
                            if latest_low[1] < prev_low[1] and latest_low[2] > prev_low[2]:
                                conditions_met += 1
                                divergence_score += 40
                                # MACD在零轴附近或下方（更安全）
                                if latest_low[2] < 0.5:
                                    divergence_score += 10
                    
                    # 2. MACD柱状图背离检测
                    if config.get('useHistogramDivergence', True):
                        if i >= 10:
                            price_change = (d['close'] - df.iloc[i-10]['close']) / df.iloc[i-10]['close'] * 100
                            hist_change = d.get('macdHist', 0) - df.iloc[i-10].get('macdHist', 0)
                            
                            # 价格跌但柱状图上升
                            if price_change < -5 and hist_change > 0:
                                conditions_met += 1
                                divergence_score += 20
                    
                    # 3. 成交量背离确认
                    if config.get('useVolumeDivergence', True):
                        if i >= 10:
                            price_low_now = d['close'] < df.iloc[i-10:i]['close'].min()
                            vol_now = d['volume']
                            vol_prev_low = df.iloc[i-5]['volume'] if i >= 5 else vol_now
                            
                            # 价格新低但成交量萎缩（卖压减轻）
                            if price_low_now and vol_now < vol_prev_low * 0.8:
                                divergence_score += 15
                                conditions_met += 1
                    
                    # 4. RSI背离确认
                    if config.get('useRsiDivergence', True):
                        if i >= lookback_period:
                            recent_lows_rsi = []
                            for j in range(i - lookback_period, i + 1):
                                if j < 2:
                                    continue
                                if df.iloc[j]['rsi'] < 40:  # 超卖区域
                                    if (df.iloc[j]['rsi'] < df.iloc[j-1]['rsi'] and 
                                        df.iloc[j]['rsi'] < df.iloc[j+1]['rsi'] if j < len(df) - 1 else True):
                                        recent_lows_rsi.append((j, df.iloc[j]['close'], df.iloc[j]['rsi']))
                            
                            if len(recent_lows_rsi) >= 2:
                                latest_rsi_low = recent_lows_rsi[-1]
                                prev_rsi_low = recent_lows_rsi[-2]
                                
                                # 价格创新低但RSI未创新低
                                if latest_rsi_low[1] < prev_rsi_low[1] and latest_rsi_low[2] > prev_rsi_low[2]:
                                    divergence_score += 20
                                    conditions_met += 1
                    
                    # 5. 趋势强度过滤 - 避免在极强下跌趋势中交易
                    if config.get('useTrendFilter', True):
                        if d.get('adx'):
                            if d['adx'] < 35:  # 趋势不是极强
                                divergence_score += 10
                            # 趋势开始减弱
                            if prev.get('adx') and d['adx'] < prev['adx']:
                                divergence_score += 5
                    
                    # 6. K线形态确认 - 看涨反转信号
                    if config.get('useCandleConfirm', True):
                        body = d['close'] - d['open']
                        lower_shadow = d['open'] - d['low'] if body > 0 else d['close'] - d['low']
                        body_size = abs(body)
                        
                        # 锤子线或下影线较长
                        if lower_shadow > body_size * 1.5:
                            divergence_score += 10
                        # 阳线收盘
                        if d['close'] > d['open']:
                            divergence_score += 5
                    
                    # 7. 均线系统确认
                    if config.get('useMaConfirm', True):
                        # 价格接近重要均线支撑
                        if d.get('ma60') and d['close'] > d['ma60'] * 0.95:
                            divergence_score += 10
                        # 短期均线走平或向上
                        if d.get('ma5') and prev.get('ma5') and d['ma5'] >= prev['ma5']:
                            divergence_score += 5
                    
                    # 动态买入条件
                    min_conditions = config.get('minConditions', 2)
                    score_threshold = config.get('scoreThreshold', 60)
                    
                    if conditions_met >= min_conditions and divergence_score >= score_threshold:
                        buy_signal = True
                        signal_strength = min(95, divergence_score)
                
                elif strategy_name == 'bollinger_extreme':
                    # 布林极限策略 - 升级版：布林带极值交易 + 反转确认
                    boll_score = 0
                    conditions_met = 0
                    total_conditions = 0
                    
                    # 1. 布林带下轨触及（可配置开关）- 买入信号
                    if config.get('useLowerBandTouch', True):
                        total_conditions += 1
                        boll_deviation = config.get('bollDeviation', 2.0)
                        
                        # 价格触及或跌破下轨
                        if d.get('bollDown') and d['close'] <= d['bollDown'] * (1 + 0.01):
                            conditions_met += 1
                            boll_score += 30
                            
                            # 计算布林带百分位位置
                            if d.get('bollUp') and d.get('bollDown') and d['bollUp'] != d['bollDown']:
                                boll_percent = (d['close'] - d['bollDown']) / (d['bollUp'] - d['bollDown']) * 100
                                if boll_percent < 5:  # 接近最底部
                                    boll_score += 15
                            
                            # 从下轨反弹
                            if d['close'] > d['open']:
                                boll_score += 10
                    
                    # 2. 布林带宽度确认 - 避免在极度收缩时交易
                    if config.get('useBandWidthConfirm', True):
                        if d.get('bollWidth'):
                            # 布林带宽度适中（有波动但不过度）
                            if 0.05 < d['bollWidth'] < 0.25:
                                boll_score += 10
                            # 布林带从收缩开始扩张（波动率突破）
                            if prev.get('bollWidth') and d['bollWidth'] > prev['bollWidth'] * 1.1:
                                boll_score += 10
                                conditions_met += 1
                    
                    # 3. RSI超卖确认
                    if config.get('useRsiConfirm', True):
                        total_conditions += 1
                        rsi_oversold = config.get('rsiOversold', 30)
                        if d.get('rsi') and d['rsi'] < rsi_oversold:
                            conditions_met += 1
                            boll_score += 20
                            if d['rsi'] < 20:  # 严重超卖
                                boll_score += 10
                            # RSI开始回升
                            if prev.get('rsi') and d['rsi'] > prev['rsi']:
                                boll_score += 5
                    
                    # 4. 成交量确认 - 地量或放量反弹
                    if config.get('useVolumeConfirm', True):
                        total_conditions += 1
                        vol_ratio = d['volume'] / d['volMa5'] if d.get('volMa5') and d['volMa5'] > 0 else 1
                        
                        # 地量（缩量见底）
                        if vol_ratio < 0.7:
                            conditions_met += 1
                            boll_score += 15
                        # 或者放量反弹
                        elif d['close'] > d['open'] and vol_ratio > 1.3:
                            conditions_met += 1
                            boll_score += 15
                    
                    # 5. 均线支撑确认
                    if config.get('useMaSupport', True):
                        # 价格接近长期均线支撑
                        if d.get('ma60') and d['close'] > d['ma60'] * 0.97:
                            boll_score += 10
                        # 短期均线走平
                        if d.get('ma5') and prev.get('ma5') and abs(d['ma5'] - prev['ma5']) / d['ma5'] < 0.005:
                            boll_score += 5
                    
                    # 6. K线形态确认
                    if config.get('useCandleConfirm', True):
                        body = d['close'] - d['open']
                        lower_shadow = d['open'] - d['low'] if body > 0 else d['close'] - d['low']
                        body_size = abs(body)
                        
                        # 下影线较长（支撑明显）
                        if lower_shadow > body_size * 1.5:
                            boll_score += 15
                        # 阳线收盘
                        if d['close'] > d['open']:
                            boll_score += 5
                        # 收盘价接近最高价
                        if d['high'] > d['low']:
                            close_position = (d['close'] - d['low']) / (d['high'] - d['low'])
                            if close_position > 0.7:
                                boll_score += 10
                    
                    # 7. 波动率调整
                    if config.get('useVolatilityAdjust', True):
                        if d.get('atr_pct'):
                            # 根据ATR调整评分
                            if d['atr_pct'] > 3:  # 高波动
                                boll_score *= 1.1  # 高分股票可能有大反弹
                            elif d['atr_pct'] < 1:  # 低波动
                                boll_score *= 0.9  # 低波动股票反弹可能较小
                    
                    # 动态买入条件
                    min_conditions = config.get('minConditions', 2)
                    score_threshold = config.get('scoreThreshold', 60)
                    
                    if conditions_met >= min_conditions and boll_score >= score_threshold:
                        buy_signal = True
                        signal_strength = min(95, int(boll_score))
                
                elif strategy_name == 'momentum_rotation':
                    # 动量轮动策略 - 升级版：多周期动量确认 + 相对强度分析
                    momentum_score = 0
                    conditions_met = 0
                    
                    # 1. 短期动量（可配置开关）
                    if config.get('useShortMomentum', True):
                        short_period = int(config.get('shortMomentumPeriod', 10))
                        if i >= short_period:
                            short_momentum = (d['close'] - df.iloc[i-short_period]['close']) / df.iloc[i-short_period]['close'] * 100
                            short_threshold = config.get('shortMomentumThreshold', 3.0)
                            
                            if short_momentum > short_threshold:
                                conditions_met += 1
                                momentum_score += 20
                                if short_momentum > short_threshold * 1.5:
                                    momentum_score += 10
                    
                    # 2. 中期动量（可配置开关）
                    if config.get('useMediumMomentum', True):
                        medium_period = int(config.get('mediumMomentumPeriod', 20))
                        if i >= medium_period:
                            medium_momentum = (d['close'] - df.iloc[i-medium_period]['close']) / df.iloc[i-medium_period]['close'] * 100
                            medium_threshold = config.get('mediumMomentumThreshold', 5.0)
                            
                            if medium_momentum > medium_threshold:
                                conditions_met += 1
                                momentum_score += 25
                                # 动量加速
                                if i >= medium_period + 5:
                                    prev_momentum = (df.iloc[i-5]['close'] - df.iloc[i-medium_period-5]['close']) / df.iloc[i-medium_period-5]['close'] * 100
                                    if medium_momentum > prev_momentum:
                                        momentum_score += 10
                    
                    # 3. 长期动量趋势（可配置开关）
                    if config.get('useLongMomentum', True):
                        long_period = int(config.get('longMomentumPeriod', 60))
                        if i >= long_period:
                            long_momentum = (d['close'] - df.iloc[i-long_period]['close']) / df.iloc[i-long_period]['close'] * 100
                            
                            if long_momentum > 0:  # 长期趋势向上
                                momentum_score += 15
                                if long_momentum > 10:
                                    momentum_score += 5
                    
                    # 4. 相对强度 - 与大盘比较（简化版，使用均线）
                    if config.get('useRelativeStrength', True):
                        if d.get('ma5') and d.get('ma20') and d.get('ma60'):
                            # 价格相对均线的位置
                            price_vs_ma20 = (d['close'] - d['ma20']) / d['ma20'] * 100
                            ma20_vs_ma60 = (d['ma20'] - d['ma60']) / d['ma60'] * 100
                            
                            if price_vs_ma20 > 0 and ma20_vs_ma60 > 0:
                                momentum_score += 15
                                if price_vs_ma20 > 5:
                                    momentum_score += 5
                    
                    # 5. 成交量确认（可配置开关）
                    if config.get('useVolumeConfirm', True):
                        vol_multi = float(config.get('volumeMulti', 1.5))
                        vol_ratio = d['volume'] / d['volMa5'] if d.get('volMa5') and d['volMa5'] > 0 else 1
                        
                        if vol_ratio > vol_multi:
                            conditions_met += 1
                            momentum_score += 15
                            # 成交量持续放大
                            if d.get('volMa5') and d.get('volMa10') and d['volMa5'] > d['volMa10']:
                                momentum_score += 5
                        # 量价配合
                        if d['close'] > d['open'] and vol_ratio > 1:
                            momentum_score += 5
                    
                    # 6. 动量质量 - 避免过度延伸
                    if config.get('useMomentumQuality', True):
                        if i >= 20:
                            recent_returns = []
                            for j in range(i-19, i+1):
                                if j > 0:
                                    daily_return = (df.iloc[j]['close'] - df.iloc[j-1]['close']) / df.iloc[j-1]['close'] * 100
                                    recent_returns.append(daily_return)
                            
                            if recent_returns:
                                avg_return = sum(recent_returns) / len(recent_returns)
                                max_return = max(recent_returns)
                                
                                # 平均收益为正且没有极端大涨
                                if avg_return > 0.3 and max_return < 10:
                                    momentum_score += 10
                                # 连续上涨天数
                                up_days = sum(1 for r in recent_returns if r > 0)
                                if up_days >= 12:  # 20天中至少12天上涨
                                    momentum_score += 5
                    
                    # 7. 波动率调整 - 动量与波动率匹配
                    if config.get('useVolatilityAdjust', True):
                        if d.get('atr_pct'):
                            # 适中的波动率有利于动量延续
                            if 1.5 < d['atr_pct'] < 4:
                                momentum_score += 10
                            elif d['atr_pct'] > 5:  # 波动过大，动量可能不稳定
                                momentum_score -= 5
                    
                    # 8. MACD动量确认
                    if config.get('useMacdConfirm', True):
                        macd_val = d.get('macd', None)
                        if macd_val is not None and not pd.isna(macd_val) and float(macd_val) > 0:
                            momentum_score += 10
                            hist = d.get('macdHist', None)
                            prev_hist = prev.get('macdHist', None)
                            if hist is not None and prev_hist is not None and not pd.isna(hist) and not pd.isna(prev_hist):
                                if float(hist) > float(prev_hist):  # 动量增强
                                    momentum_score += 5
                    
                    # 动态买入条件
                    min_conditions = config.get('minConditions', 3)
                    score_threshold = config.get('scoreThreshold', 65)
                    
                    if conditions_met >= min_conditions and momentum_score >= score_threshold:
                        buy_signal = True
                        signal_strength = min(95, momentum_score)
                
                elif strategy_name == 'turtle_enhanced':
                    # 海龟增强策略 - 升级版：多时间框架突破 + 风险管理增强
                    turtle_score = 0
                    conditions_met = 0
                    
                    entry_period = int(config.get('entryPeriod', 20))
                    exit_period = int(config.get('exitPeriod', 10))
                    atr_multiplier = float(config.get('atrMultiplier', 2.0))
                    
                    if i < entry_period:
                        continue
                    
                    # 1. 突破入场条件（核心）- 多周期突破确认
                    if config.get('useBreakoutEntry', True):
                        # 20日高点突破
                        if d.get('high20') and d['close'] >= d['high20'] * 0.995:
                            conditions_met += 1
                            turtle_score += 30
                            
                            # 同时突破60日高点（强趋势）
                            if d.get('high60') and d['close'] >= d['high60']:
                                turtle_score += 15
                            
                            # 突破幅度
                            breakout_pct = (d['close'] - d['high20']) / d['high20'] * 100 if d['high20'] > 0 else 0
                            if 0 < breakout_pct < 3:  # 适中突破，避免过度延伸
                                turtle_score += 5
                    
                    # 2. 趋势过滤 - 只在趋势明确时交易
                    if config.get('useTrendFilter', True):
                        if d.get('ma20') and d.get('ma60'):
                            if d['close'] > d['ma20'] > d['ma60']:  # 多头排列
                                conditions_met += 1
                                turtle_score += 15
                                if d.get('ma5') and d['ma5'] > d['ma20']:
                                    turtle_score += 5
                    
                    # 3. 波动率过滤 - ATR确认
                    if config.get('useVolatilityFilter', True):
                        if d.get('atr_pct'):
                            # 适中的波动率
                            if 1.5 < d['atr_pct'] < 5:
                                turtle_score += 10
                                conditions_met += 1
                            # 波动率不能过低（避免无波动市场）
                            if d['atr_pct'] > 1:
                                turtle_score += 5
                    
                    # 4. 成交量确认 - 突破需要量能配合
                    if config.get('useVolumeConfirm', True):
                        vol_multi = config.get('volumeMulti', 1.5)
                        vol_ratio = d['volume'] / d['volMa5'] if d.get('volMa5') and d['volMa5'] > 0 else 1
                        
                        if vol_ratio > vol_multi:
                            conditions_met += 1
                            turtle_score += 15
                            # 成交量持续放大
                            if d.get('volMa5') and d.get('volMa10') and d['volMa5'] > d['volMa10']:
                                turtle_score += 5
                    
                    # 5. ADX趋势强度确认
                    if config.get('useAdxConfirm', True):
                        if d.get('adx'):
                            if d['adx'] > 25:  # 趋势市场
                                turtle_score += 15
                                conditions_met += 1
                                if d['adx'] > 35:  # 强趋势
                                    turtle_score += 10
                            # 趋势方向
                            if d.get('plus_di') and d.get('minus_di') and d['plus_di'] > d['minus_di']:
                                turtle_score += 5
                    
                    # 6. 假突破过滤 - 避免震荡市假信号
                    if config.get('useFalseBreakoutFilter', True):
                        if i >= entry_period + 5:
                            # 检查之前是否有多次假突破
                            recent_highs = [df.iloc[j]['high20'] for j in range(i-entry_period, i) if df.iloc[j].get('high20')]
                            if recent_highs:
                                avg_high20 = sum(recent_highs) / len(recent_highs)
                                # 如果当前突破明显高于之前的震荡区间
                                if d['close'] > avg_high20 * 1.02:
                                    turtle_score += 10
                    
                    # 7. 风险管理评分 - 基于ATR的仓位管理
                    if config.get('useRiskManagement', True):
                        if d.get('atr') and d['close'] > 0:
                            risk_pct = d['atr'] / d['close'] * 100
                            if risk_pct < 3:  # 风险适中
                                turtle_score += 10
                            elif risk_pct > 5:  # 风险过高，降低评分
                                turtle_score -= 10
                    
                    # 8. 市场情绪 - 避免过度乐观
                    if config.get('useSentimentFilter', True):
                        if d.get('rsi'):
                            if d['rsi'] < 70:  # 未过度买入
                                turtle_score += 5
                            if d['rsi'] < 60:  # 还有上涨空间
                                turtle_score += 5
                    
                    # 动态买入条件
                    min_conditions = config.get('minConditions', 3)
                    score_threshold = config.get('scoreThreshold', 65)
                    
                    if conditions_met >= min_conditions and turtle_score >= score_threshold:
                        buy_signal = True
                        signal_strength = min(95, turtle_score)
                        
                        # 计算止损价（用于后续风险管理）
                        atr = d.get('atr', abs(d['high'] - d['low']) * 0.5)
                        stop_price = d['close'] - atr * atr_multiplier
                
                elif strategy_name == 'chip_peak_breakout':
                    # 筹码突破策略：筹码单峰低位 + 放量突破
                    chip_score = 0
                    conditions_met = 0
                    
                    # 筹码计算参数
                    chip_lookback = int(config.get('chipLookback', 120))
                    chip_bins = int(config.get('chipBins', 30))
                    chip_decay = float(config.get('chipDecay', 0.985))
                    chip_single_peak_ratio = float(config.get('chipSinglePeakRatio', 1.6))
                    chip_peak_min_pct = float(config.get('chipPeakMinPct', 0.12))
                    chip_min_concentration = float(config.get('chipMinConcentration', 0.50))
                    chip_low_pos_pct = float(config.get('chipLowPosPct', 0.35))
                    
                    if i < chip_lookback:
                        continue
                    
                    # 1. 计算筹码分布
                    chip_data = df.iloc[i-chip_lookback:i+1].copy()
                    
                    # 计算加权成交量（时间衰减）
                    weights = [chip_decay ** (chip_lookback - j - 1) for j in range(chip_lookback)]
                    chip_data['weight'] = weights + [1.0]
                    chip_data['weighted_vol'] = chip_data['volume'] * chip_data['weight']
                    
                    # 价格区间分箱
                    price_min = chip_data['low'].min()
                    price_max = chip_data['high'].max()
                    price_range = price_max - price_min
                    
                    if price_range <= 0:
                        continue
                    
                    # 计算每个价格区间的筹码量
                    bin_size = price_range / chip_bins
                    chip_distribution = []
                    
                    for b in range(chip_bins):
                        bin_low = price_min + b * bin_size
                        bin_high = bin_low + bin_size
                        bin_center = (bin_low + bin_high) / 2
                        
                        # 计算该区间内的加权成交量
                        mask = (chip_data['close'] >= bin_low) & (chip_data['close'] < bin_high)
                        vol_sum = chip_data.loc[mask, 'weighted_vol'].sum()
                        
                        chip_distribution.append({
                            'center': bin_center,
                            'low': bin_low,
                            'high': bin_high,
                            'volume': vol_sum
                        })
                    
                    # 找到筹码峰
                    total_chip = sum(c['volume'] for c in chip_distribution)
                    if total_chip <= 0:
                        continue
                    
                    # 按筹码量排序
                    sorted_chips = sorted(chip_distribution, key=lambda x: x['volume'], reverse=True)
                    
                    # 主峰
                    peak_chip = sorted_chips[0]
                    peak_pct = peak_chip['volume'] / total_chip
                    
                    # 次峰
                    second_peak_pct = sorted_chips[1]['volume'] / total_chip if len(sorted_chips) > 1 else 0
                    
                    # 单峰强度
                    single_peak_strength = peak_pct / second_peak_pct if second_peak_pct > 0 else 999
                    
                    # 筹码集中度 (90%筹码所在区间)
                    sorted_by_price = sorted(chip_distribution, key=lambda x: x['center'])
                    cumsum = 0
                    c90_low = c90_high = None
                    c70_low = c70_high = None
                    
                    for c in sorted_by_price:
                        cumsum += c['volume']
                        if c90_low is None and cumsum >= total_chip * 0.05:
                            c90_low = c['center']
                        if c70_low is None and cumsum >= total_chip * 0.15:
                            c70_low = c['center']
                        if c90_high is None and cumsum >= total_chip * 0.95:
                            c90_high = c['center']
                        if c70_high is None and cumsum >= total_chip * 0.85:
                            c70_high = c['center']
                    
                    c90_range = ((c90_high - c90_low) / price_range * 100) if c90_high and c90_low and price_range > 0 else 100
                    c70_range = ((c70_high - c70_low) / price_range * 100) if c70_high and c70_low and price_range > 0 else 100
                    concentration = 100 - c90_range
                    
                    # 主峰位置（在价格区间的位置）
                    peak_position = (peak_chip['center'] - price_min) / price_range if price_range > 0 else 0.5
                    
                    # 2. 筹码单峰条件
                    if config.get('useChipSinglePeak', True):
                        if peak_pct >= chip_peak_min_pct and single_peak_strength >= chip_single_peak_ratio:
                            conditions_met += 1
                            chip_score += 25
                            # 单峰越强，分数越高
                            if single_peak_strength > 2.0:
                                chip_score += 10
                    
                    # 3. 筹码集中度条件
                    if config.get('useChipConcentration', True):
                        if concentration >= chip_min_concentration:
                            conditions_met += 1
                            chip_score += 20
                            if concentration > 0.60:
                                chip_score += 10
                    
                    # 4. 筹码低位条件
                    if config.get('useChipLowPosition', True):
                        if peak_position <= chip_low_pos_pct:
                            conditions_met += 1
                            chip_score += 25
                            # 越低位越好
                            if peak_position < 0.25:
                                chip_score += 10
                    
                    # 5. 价格突破条件（核心）
                    if config.get('usePriceBreakout', True):
                        breakout_period = int(config.get('breakoutPeriod', 30))
                        if i >= breakout_period:
                            recent_high = df.iloc[i-breakout_period:i]['high'].max()
                            if d['close'] > recent_high * 0.995:
                                conditions_met += 1
                                chip_score += 30
                                # 突破幅度
                                breakout_pct = (d['close'] - recent_high) / recent_high * 100
                                if 0.5 < breakout_pct < 3:
                                    chip_score += 10
                    
                    # 6. 成交量放大条件
                    if config.get('useVolumeConfirm', True):
                        vol_multi = float(config.get('volumeMulti', 2.0))
                        vol_ratio = d['volume'] / d['volMa5'] if d.get('volMa5') and d['volMa5'] > 0 else 1
                        
                        if vol_ratio > vol_multi:
                            conditions_met += 1
                            chip_score += 20
                            if vol_ratio > vol_multi * 1.5:
                                chip_score += 10
                    
                    # 7. 趋势确认
                    if config.get('useTrendConfirm', True):
                        if d.get('ma20') and d.get('ma60'):
                            if d['close'] > d['ma20'] > d['ma60']:
                                chip_score += 15
                            if d.get('ma5') and d['ma5'] > d['ma20']:
                                chip_score += 10
                    
                    # 8. 避免过度延伸
                    if config.get('useOverboughtFilter', True):
                        if d.get('rsi') and d['rsi'] < 70:
                            chip_score += 10
                        if peak_position < 0.50 and d['close'] > peak_chip['high']:
                            chip_score += 10
                    
                    # 动态买入条件
                    min_conditions = config.get('minConditions', 4)
                    score_threshold = config.get('scoreThreshold', 70)
                    
                    if conditions_met >= min_conditions and chip_score >= score_threshold:
                        buy_signal = True
                        signal_strength = min(95, chip_score)
                
                elif strategy_name == 'xgboost_ml':
                    # XGBoost机器学习策略：专注捕捉大波段
                    # 初始化模型缓存
                    if not hasattr(StrategyEngine, 'xgb_models'):
                        StrategyEngine.xgb_models = {}
                    
                    # 获取当前股票的唯一标识
                    stock_key = f"{strategy_name}_{len(df)}"
                    
                    # 获取参数 - 降低门槛，提高敏感度
                    lookback = int(config.get('xgbLookback', 200))
                    horizon = int(config.get('xgbHorizon', 15))
                    retrain_interval = int(config.get('xgbRetrainInterval', 30))
                    min_prob = float(config.get('xgbMinProb', 0.55))  # 降低：0.65→0.55
                    min_accuracy = float(config.get('xgbMinAccuracy', 0.52))  # 降低：0.55→0.52
                    volume_multi = float(config.get('volumeMulti', 1.0))  # 降低：1.2→1.0
                    use_strict_filter = config.get('xgbUseStrictFilter', False)  # 新增：是否严格过滤
                    
                    # 模型训练和预测
                    if stock_key not in StrategyEngine.xgb_models:
                        StrategyEngine.xgb_models[stock_key] = {
                            'model': None,
                            'last_train_index': -999,
                            'train_count': 0
                        }
                    
                    model_info = StrategyEngine.xgb_models[stock_key]
                    
                    # 检查是否需要重新训练
                    need_retrain = (i - model_info['last_train_index']) >= retrain_interval
                    
                    if need_retrain and i >= lookback + horizon:
                        # 准备训练数据
                        train_features = []
                        train_labels = []
                        
                        # 使用滑动窗口构建训练样本
                        for j in range(lookback, i - horizon):
                            features = StrategyEngine.xgboost_extract_features(df, j)
                            if features is None:
                                continue
                            
                            # 计算未来收益率作为标签 - 简化逻辑，避免错误
                            future_close = df.iloc[j + horizon]['close']
                            current_close = df.iloc[j]['close']
                            if current_close <= 0:
                                continue
                            
                            future_return = (future_close - current_close) / current_close
                            
                            # 简化标签：只看绝对收益率，不计算相对收益
                            # 未来涨幅超过2%则为正样本（降低门槛，增加正样本）
                            label = 1 if future_return > 0.02 else 0
                            
                            train_features.append(features)
                            train_labels.append(label)
                        
                        # 训练模型
                        if len(train_features) >= 50:
                            # 配置XGBoost参数 - 优化参数
                            xgb_params = {
                                'max_depth': int(config.get('xgbMaxDepth', 5)),  # 增加：4→5
                                'learning_rate': float(config.get('xgbLearningRate', 0.08)),  # 增加：0.05→0.08
                                'n_estimators': int(config.get('xgbNEstimators', 150)),  # 增加：100→150
                                'min_child_weight': int(config.get('xgbMinChildWeight', 3)),  # 降低：5→3
                                'subsample': float(config.get('xgbSubsample', 0.9)),  # 增加：0.8→0.9
                                'colsample_bytree': float(config.get('xgbColsample', 0.9)),  # 增加：0.8→0.9
                                'reg_alpha': float(config.get('xgbRegAlpha', 0.05)),  # 降低：0.1→0.05
                                'reg_lambda': float(config.get('xgbRegLambda', 0.8)),  # 降低：1.0→0.8
                            }
                            
                            print(f"[XGBoost] 训练数据: 样本数={len(train_features)}, 正样本={sum(train_labels)}, 负样本={len(train_labels)-sum(train_labels)}")
                            
                            model_result = StrategyEngine.xgboost_train_model(train_features, train_labels, xgb_params)
                            
                            if model_result:
                                print(f"[XGBoost] 训练完成: 准确率={model_result['accuracy']:.3f}, 基准率={model_result['base_rate']:.3f}")
                                if model_result['accuracy'] >= min_accuracy:
                                    model_info['model'] = model_result
                                    model_info['last_train_index'] = i
                                    model_info['train_count'] += 1
                                    print(f"[XGBoost] 模型已更新 (训练次数: {model_info['train_count']})")
                                else:
                                    print(f"[XGBoost] 准确率不达标: {model_result['accuracy']:.3f} < {min_accuracy:.2f}")
                            else:
                                print(f"[XGBoost] 训练失败: model_result={model_result}")
                        else:
                            print(f"[XGBoost] 训练数据不足: {len(train_features)} < 50")
                    
                    # 使用模型进行预测
                    if model_info['model'] is not None:
                        features = StrategyEngine.xgboost_extract_features(df, i)
                        
                        if features is not None:
                            prob = StrategyEngine.xgboost_predict(model_info['model'], features)
                            
                            if prob is not None:
                                print(f"[XGBoost] 预测: 概率={prob:.3f}, 基准率={model_info['model']['base_rate']:.3f}, 边缘={prob - model_info['model']['base_rate']:.3f}")
                                
                                # 放宽过滤条件，提高买入敏感度
                                curr_vol_ratio = d['volume'] / d['volMa5'] if d.get('volMa5') and d['volMa5'] > 0 else 1
                                vol_ok = curr_vol_ratio >= volume_multi  # 1.0倍即可，不要求放量
                                
                                # 趋势过滤 - 大幅放宽
                                ma20 = d.get('ma20')
                                ma60 = d.get('ma60')
                                trend_ok = True
                                if ma20 and ma60:
                                    # 放宽：只要价格>MA20即可，不要求MA20>MA60
                                    trend_ok = d['close'] > ma20
                                
                                # 跳空过滤 - 放宽
                                prev = df.iloc[i-1] if i > 0 else None
                                gap_ok = True
                                if prev and use_strict_filter:
                                    gap = (d['open'] - prev['close']) / prev['close'] if prev['close'] > 0 else 0
                                    gap_ok = abs(gap) < 0.10  # 放宽：7%→10%
                                
                                # 波动率过滤 - 放宽
                                atr = d.get('atr', (d['high'] - d['low']) * 0.5)
                                atr_pct = atr / d['close'] if d['close'] > 0 else 0
                                atr_ok = atr_pct < 0.20  # 放宽：15%→20%
                                
                                # 综合判断 - 降低门槛
                                edge = prob - model_info['model']['base_rate']
                                
                                # 放宽买入条件：
                                # 1. 概率阈值：0.65→0.55
                                # 2. 边缘优势：0.05→0.02
                                # 3. 成交量：1.2倍→1.0倍
                                # 4. 趋势：多头排列→价格>MA20
                                # 5. 跳空/波动率：仅在严格模式下检查
                                
                                conditions_met = 0
                                if prob >= min_prob:
                                    conditions_met += 1
                                if edge >= 0.02:
                                    conditions_met += 1
                                if vol_ok:
                                    conditions_met += 1
                                if trend_ok:
                                    conditions_met += 1
                                
                                print(f"[XGBoost] 过滤条件: 概率OK={prob >= min_prob}, 边缘OK={edge >= 0.02}, 成交量OK={vol_ok}, 趋势OK={trend_ok}, 满足={conditions_met}/4")
                                
                                # 满足至少3个条件即可买入（更宽松）
                                if conditions_met >= 3:
                                    # 严格模式下才检查跳空和波动率
                                    if use_strict_filter:
                                        if gap_ok and atr_ok:
                                            buy_signal = True
                                            signal_strength = min(95, int(50 + prob * 50))
                                            print(f"[XGBoost] 买入信号 (严格模式): 强度={signal_strength}")
                                    else:
                                        buy_signal = True
                                        signal_strength = min(95, int(50 + prob * 50))
                                        print(f"[XGBoost] 买入信号 (宽松模式): 强度={signal_strength}")
                                else:
                                    print(f"[XGBoost] 未满足买入条件")
                            else:
                                print(f"[XGBoost] 预测失败: prob={prob}")
                        else:
                            print(f"[XGBoost] 特征提取失败: i={i}")
                    else:
                        print(f"[XGBoost] 模型未训练")
                
                elif strategy_name == 'lstm_ml':
                    # 简化的LSTM策略：基于基础技术指标组合
                    conditions_met = 0
                    
                    curr_close = d['close']
                    curr_ma5 = d.get('ma5', curr_close)
                    curr_ma10 = d.get('ma10', curr_close)
                    curr_ma20 = d.get('ma20', curr_close)
                    curr_rsi = d.get('rsi', 50)
                    curr_macd = d.get('macd', 0)
                    curr_vol_ratio = d['volume'] / d['volMa5'] if d.get('volMa5') and d['volMa5'] > 0 else 1.0
                    
                    # 基础买入条件（大幅放宽）
                    if curr_rsi < 70:  # RSI不超买
                        conditions_met += 1
                    if curr_macd > -0.2:  # MACD不为负太多
                        conditions_met += 1
                    if curr_close > curr_ma20 * 0.98:  # 价格在MA20附近或之上
                        conditions_met += 1
                    if curr_vol_ratio > 0.5:  # 有成交量
                        conditions_met += 1
                    if curr_ma5 > curr_ma10 or curr_ma10 > curr_ma20:  # 均线多头排列或部分多头排列
                        conditions_met += 1
                    
                    # 满足3个条件即买入
                    if conditions_met >= 3:
                        buy_signal = True
                        signal_strength = 50 + conditions_met * 10
                
                elif strategy_name == 'ensemble_ml':
                    # 简化的混合机器学习策略：基于基础技术指标组合
                    conditions_met = 0
                    
                    curr_close = d['close']
                    curr_ma5 = d.get('ma5', curr_close)
                    curr_ma10 = d.get('ma10', curr_close)
                    curr_ma20 = d.get('ma20', curr_close)
                    curr_rsi = d.get('rsi', 50)
                    curr_macd = d.get('macd', 0)
                    curr_kdj_k = d.get('kdjK', 50)
                    curr_kdj_d = d.get('kdjD', 50)
                    curr_vol_ratio = d['volume'] / d['volMa5'] if d.get('volMa5') and d['volMa5'] > 0 else 1.0
                    
                    # 基础买入条件（大幅放宽）
                    if curr_rsi < 65:  # RSI不超买
                        conditions_met += 1
                    if curr_macd > -0.1:  # MACD不为负太多
                        conditions_met += 1
                    if curr_close > curr_ma20 * 0.97:  # 价格在MA20附近或之上
                        conditions_met += 1
                    if curr_vol_ratio > 0.6:  # 有成交量
                        conditions_met += 1
                    if curr_ma5 > curr_ma10 or curr_ma10 > curr_ma20:  # 均线多头排列或部分多头排列
                        conditions_met += 1
                    if curr_kdj_k > curr_kdj_d:  # KDJ金叉或维持
                        conditions_met += 1
                    
                    # 满足4个条件即买入
                    if conditions_met >= 4:
                        buy_signal = True
                        signal_strength = 50 + conditions_met * 8
                
                elif strategy_name == 'mtsf_net':
                    # 多尺度时序融合网络 (MTSF-Net)
                    # 核心：融合短/中/长期特征，捕捉多尺度模式
                    import sys
                    print(f"[MTSF-Net] 策略被调用，i={i}, TENSORFLOW_AVAILABLE={TENSORFLOW_AVAILABLE}", file=sys.stderr, flush=True)
                    
                    if not TENSORFLOW_AVAILABLE:
                        print(f"[MTSF-Net] TensorFlow不可用，跳过")
                        continue
                    
                    # 初始化模型缓存
                    if not hasattr(StrategyEngine, 'mtsf_models'):
                        StrategyEngine.mtsf_models = {}
                    
                    stock_key = f"mtsf_{strategy_name}_{len(df)}"
                    
                    # 参数配置
                    lookback = int(config.get('mtsfLookback', 30))
                    retrain_interval = int(config.get('mtsfRetrainInterval', 20))
                    min_prob = float(config.get('mtsfMinProb', 0.52))
                    
                    if stock_key not in StrategyEngine.mtsf_models:
                        StrategyEngine.mtsf_models[stock_key] = {
                            'model': None, 'last_train_index': -999, 'scaler': None
                        }
                    
                    model_info = StrategyEngine.mtsf_models[stock_key]
                    need_retrain = (i - model_info['last_train_index']) >= retrain_interval
                    
                    print(f"[MTSF-Net] i={i}, lookback={lookback}, need_retrain={need_retrain}, model_exists={model_info['model'] is not None}")
                    
                    # 训练或更新模型
                    if need_retrain and i >= lookback + 10:
                        sequences, labels = [], []
                        for j in range(lookback, i - 5):
                            feat = StrategyEngine.mtsf_extract_features(df, j, lookback)
                            if feat is not None:
                                future_return = (df.iloc[j + 5]['close'] - df.iloc[j]['close']) / df.iloc[j]['close']
                                sequences.append(feat)
                                labels.append(1 if future_return > 0.015 else 0)
                        
                        print(f"[MTSF-Net] 训练数据: {len(sequences)}个样本, 正样本{sum(labels) if labels else 0}个")
                        
                        if len(sequences) >= 100:
                            model_result = StrategyEngine.mtsf_train_model(sequences, labels, lookback)
                            if model_result:
                                model_info['model'] = model_result
                                model_info['last_train_index'] = i
                                print(f"[MTSF-Net] 模型训练成功，准确率={model_result['accuracy']:.3f}")
                            else:
                                print(f"[MTSF-Net] 模型训练失败")
                        else:
                            print(f"[MTSF-Net] 训练数据不足: {len(sequences)} < 100")
                    
                    # 预测
                    if model_info['model'] is not None:
                        features = StrategyEngine.mtsf_extract_features(df, i, lookback)
                        print(f"[MTSF-Net] i={i}, features提取={'成功' if features is not None else '失败'}")
                        
                        if features is not None:
                            prob = StrategyEngine.mtsf_predict(model_info['model'], features)
                            prob_str = f"{prob:.3f}" if prob is not None else "None"
                            print(f"[MTSF-Net] i={i}, prob={prob_str}, min_prob={min_prob}")
                            
                            if prob is not None and prob >= min_prob:
                                # 附加过滤条件
                                curr_ma20 = d.get('ma20', d['close'])
                                vol_ratio = d['volume'] / d['volMa5'] if d.get('volMa5', 0) > 0 else 1
                                
                                print(f"[MTSF-Net] i={i}, 条件检查: close={d['close']:.2f}, ma20={curr_ma20:.2f}, vol_ratio={vol_ratio:.2f}")
                                
                                if d['close'] > curr_ma20 * 0.95 and vol_ratio > 0.6:
                                    buy_signal = True
                                    signal_strength = min(95, int(50 + prob * 45))
                                    print(f"[MTSF-Net] i={i}, 买入信号! strength={signal_strength}")
                                else:
                                    print(f"[MTSF-Net] i={i}, 未通过过滤条件")
                            else:
                                prob_str2 = f"{prob:.3f}" if prob is not None else "None"
                                print(f"[MTSF-Net] i={i}, 概率不足: prob={prob_str2} < {min_prob}")
                    else:
                        print(f"[MTSF-Net] i={i}, 模型未训练")
                
                elif strategy_name == 'vpan_net':
                    # 量价注意力预测网络 (VPAN)
                    # 核心：Transformer架构，关注量价关系
                    import sys
                    print(f"[VPAN] 策略被调用，i={i}, TENSORFLOW_AVAILABLE={TENSORFLOW_AVAILABLE}", file=sys.stderr, flush=True)
                    
                    if not TENSORFLOW_AVAILABLE:
                        print(f"[VPAN] TensorFlow不可用，跳过", file=sys.stderr, flush=True)
                        continue
                    
                    # 初始化模型缓存
                    if not hasattr(StrategyEngine, 'vpan_models'):
                        StrategyEngine.vpan_models = {}
                    
                    stock_key = f"vpan_{strategy_name}_{len(df)}"
                    
                    # 参数配置
                    seq_len = int(config.get('vpanSeqLen', 20))
                    retrain_interval = int(config.get('vpanRetrainInterval', 25))
                    min_prob = float(config.get('vpanMinProb', 0.55))
                    
                    if stock_key not in StrategyEngine.vpan_models:
                        StrategyEngine.vpan_models[stock_key] = {
                            'model': None, 'last_train_index': -999
                        }
                    
                    model_info = StrategyEngine.vpan_models[stock_key]
                    need_retrain = (i - model_info['last_train_index']) >= retrain_interval
                    
                    print(f"[VPAN] i={i}, seq_len={seq_len}, need_retrain={need_retrain}, model_exists={model_info['model'] is not None}")
                    
                    # 训练模型
                    if need_retrain and i >= seq_len + 10:
                        sequences, labels = [], []
                        for j in range(seq_len, i - 5):
                            seq = StrategyEngine.vpan_extract_sequence(df, j, seq_len)
                            if seq is not None:
                                future_return = (df.iloc[j + 5]['close'] - df.iloc[j]['close']) / df.iloc[j]['close']
                                sequences.append(seq)
                                labels.append(1 if future_return > 0.02 else 0)
                        
                        print(f"[VPAN] 训练数据: {len(sequences)}个样本, 正样本{sum(labels) if labels else 0}个")
                        
                        if len(sequences) >= 80:
                            model_result = StrategyEngine.vpan_train_model(sequences, labels, seq_len)
                            if model_result:
                                model_info['model'] = model_result
                                model_info['last_train_index'] = i
                                print(f"[VPAN] 模型训练成功，准确率={model_result['accuracy']:.3f}")
                            else:
                                print(f"[VPAN] 模型训练失败")
                        else:
                            print(f"[VPAN] 训练数据不足: {len(sequences)} < 80")
                    
                    # 预测
                    if model_info['model'] is not None:
                        sequence = StrategyEngine.vpan_extract_sequence(df, i, seq_len)
                        print(f"[VPAN] i={i}, sequence提取={'成功' if sequence is not None else '失败'}")
                        
                        if sequence is not None:
                            prob = StrategyEngine.vpan_predict(model_info['model'], sequence)
                            prob_str = f"{prob:.3f}" if prob is not None else "None"
                            print(f"[VPAN] i={i}, prob={prob_str}, min_prob={min_prob}")
                            
                            if prob is not None and prob >= min_prob:
                                # 趋势确认
                                ma5 = d.get('ma5', d['close'])
                                ma20 = d.get('ma20', d['close'])
                                rsi = d.get('rsi', 50)
                                
                                print(f"[VPAN] i={i}, 条件检查: close={d['close']:.2f}, ma5={ma5:.2f}, ma20={ma20:.2f}, rsi={rsi:.1f}")
                                
                                if d['close'] > ma5 and ma5 > ma20 * 0.98 and rsi < 75:
                                    buy_signal = True
                                    signal_strength = min(95, int(55 + prob * 40))
                                    print(f"[VPAN] i={i}, 买入信号! strength={signal_strength}")
                                else:
                                    print(f"[VPAN] i={i}, 未通过过滤条件")
                            else:
                                prob_str2 = f"{prob:.3f}" if prob is not None else "None"
                                print(f"[VPAN] i={i}, 概率不足: prob={prob_str2} < {min_prob}")
                    else:
                        print(f"[VPAN] i={i}, 模型未训练")


            
            if (not buy_signal) and (not in_cooldown) and position == 0 and config.get('enableTrendReentry', True):
                if strategy_name in {'trend_enhanced', 'turtle_enhanced', 'momentum_rotation', 'deep_fusion', 'volume_breakout', 'xgboost_ml', 'lstm_ml', 'ensemble_ml', 'chip_peak_breakout'}:
                    try:
                        d0 = df.iloc[i]
                        d1 = df.iloc[i - 1] if i > 0 else None
                        ma20 = d0.get('ma20', None)
                        ma60 = d0.get('ma60', None)
                        if ma20 is None or ma60 is None or pd.isna(ma20) or pd.isna(ma60):
                            raise ValueError()
                        ma20 = float(ma20)
                        ma60 = float(ma60)
                        close0 = float(d0.get('close', 0) or 0)
                        open0 = float(d0.get('open', 0) or 0)
                        low0 = float(d0.get('low', 0) or 0)
                        if close0 <= 0:
                            raise ValueError()

                        lookback = int(config.get('trendSlopeLookback', 5) or 5)
                        j = max(0, i - max(2, lookback))
                        ma20_prev = df.iloc[j].get('ma20', None)
                        if ma20_prev is None or pd.isna(ma20_prev):
                            raise ValueError()
                        ma20_prev = float(ma20_prev)

                        uptrend = close0 > ma20 > ma60 and ma20 > ma20_prev
                        if not uptrend:
                            raise ValueError()

                        bounced = False
                        if d1 is not None:
                            ma20_1 = d1.get('ma20', None)
                            if ma20_1 is not None and not pd.isna(ma20_1):
                                ma20_1 = float(ma20_1)
                                close1 = float(d1.get('close', 0) or 0)
                                bounced = close1 <= ma20_1 and close0 > ma20 and low0 <= ma20 * 1.01
                        if not bounced:
                            bounced = low0 <= ma20 * 1.005 and close0 > open0 and close0 > ma20

                        if not bounced:
                            raise ValueError()

                        min_rsi = float(config.get('reentryMinRsi', 45) or 45)
                        rsi = d0.get('rsi', None)
                        if rsi is not None and not pd.isna(rsi) and float(rsi) < min_rsi:
                            raise ValueError()

                        vol_need = float(config.get('reentryMinVolRatio', 0.95) or 0.95)
                        vol_ratio = 1.0
                        if d0.get('volMa5') and float(d0.get('volMa5') or 0) > 0:
                            vol_ratio = float(d0.get('volume', 0) or 0) / float(d0.get('volMa5'))
                        if vol_ratio < vol_need:
                            raise ValueError()

                        buy_signal = True
                        signal_strength = max(signal_strength, int(config.get('reentryStrength', 62) or 62))
                    except Exception:
                        pass

            if (not buy_signal) and (not in_cooldown) and position == 0 and config.get('enableSwingBottomBuy', True):
                if strategy_name in {'oversold_rebound', 'bollinger_extreme', 'macd_divergence'}:
                    try:
                        rsi = d.get('rsi', None)
                        if rsi is None or pd.isna(rsi):
                            raise ValueError()
                        rsi = float(rsi)
                        min_rsi = float(config.get('swingBuyRsi', 35) or 35)
                        if rsi > min_rsi:
                            raise ValueError()

                        near_low_pct = float(config.get('swingBuyNearLowPct', 2.0) or 2.0) / 100.0
                        low20 = d.get('low20', None)
                        if low20 is None or pd.isna(low20):
                            low20 = float(df['low'].iloc[max(0, i - 19):i + 1].min())
                        else:
                            low20 = float(low20)
                        close0 = float(d.get('close', 0) or 0)
                        open0 = float(d.get('open', 0) or 0)
                        vol_ratio = 1.0
                        if d.get('volMa5') and float(d.get('volMa5') or 0) > 0:
                            vol_ratio = float(d.get('volume', 0) or 0) / float(d.get('volMa5'))
                        if close0 <= 0:
                            raise ValueError()
                        if close0 > low20 * (1 + near_low_pct):
                            raise ValueError()
                        if close0 <= open0:
                            raise ValueError()
                        if vol_ratio < float(config.get('swingBuyMinVolRatio', 0.8) or 0.8):
                            raise ValueError()

                        buy_signal = True
                        signal_strength = max(signal_strength, int(config.get('swingBuyStrength', 62) or 62))
                    except Exception:
                        pass

            # 卖出逻辑 - 优化版，支持更多可配置条件
            if position > 0:
                highest_since_entry = max(highest_since_entry, d['high'])
                try:
                    body_high = max(float(d.get('open', 0) or 0), float(d.get('close', 0) or 0))
                    if body_high > 0:
                        highest_body_since_entry = max(highest_body_since_entry, body_high)
                except Exception:
                    pass
                hold_days = i - entry_index
                profit = (d['close'] - entry_price) / entry_price
                peak_profit_since_entry = max(peak_profit_since_entry, profit)
                
                # 动态止盈（可配置开关）
                if (not sell_signal) and config.get('useDynamicTP', True):
                    dtp_threshold = config.get('dynamicTpThreshold', 0.08)
                    dtp_callback = config.get('dynamicTpCallback', 0.03)
                    dtp_variation = config.get('dynamicTpVariation', 0.15)
                    
                    if peak_profit_since_entry > dtp_threshold:
                        anchor = highest_body_since_entry if highest_body_since_entry > 0 else highest_since_entry
                        trail = anchor * (1 - dtp_callback - peak_profit_since_entry * dtp_variation)
                        if d['close'] < trail:
                            sell_signal = True
                            sell_reason = '移动止盈'

                if (not sell_signal) and config.get('useAtrTrail', True):
                    try:
                        min_profit = float(config.get('atrTrailMinProfit', 0.06) or 0.06)
                        atr_mult = float(config.get('atrTrailMult', 2.5) or 2.5)
                        atr = d.get('atr', None)
                        if atr is None or pd.isna(atr):
                            atr = abs(float(d.get('high', 0) or 0) - float(d.get('low', 0) or 0)) * 0.5
                        atr = float(atr)
                        if peak_profit_since_entry > min_profit and atr > 0:
                            trail = highest_since_entry - atr * atr_mult
                            if float(d.get('close', 0) or 0) < trail:
                                sell_signal = True
                                sell_reason = 'ATR跟踪止盈'
                    except Exception:
                        pass

                # 止损（可配置开关）- 放在趋势转弱/跟踪之后，避免盈利阶段先被硬止损
                if (not sell_signal) and config.get('useStopLoss', True):
                    stop_loss_level = config.get('stopLoss', 0.05)
                    if profit <= -stop_loss_level:
                        sell_signal = True
                        sell_reason = '止损'
                
                # 最大持仓时间限制
                if (not sell_signal) and hold_days >= int(config.get('maxHoldDays', 10)):
                    sell_signal = True
                    sell_reason = '最大持仓时间'
                
                # 跌破MA5（可配置开关）
                if (not sell_signal) and config.get('useMa5Sell', True):
                    if d.get('ma5') and d['close'] < d['ma5'] and peak_profit_since_entry > 0.02:
                        sell_signal = True
                        sell_reason = '跌破MA5'
                
                # MACD死叉（可配置开关）
                if (not sell_signal) and config.get('useMacdDeathCross', True):
                    if d.get('macd') and prev.get('macd') and d['macd'] < 0 and prev['macd'] > 0 and peak_profit_since_entry > 0:
                        sell_signal = True
                        sell_reason = 'MACD死叉'
                
                # RSI超买（可配置开关）
                if (not sell_signal) and config.get('useRsiOverbought', True):
                    if d.get('rsi') and d['rsi'] > 80 and peak_profit_since_entry > 0.05:
                        sell_signal = True
                        sell_reason = 'RSI超买'

                if (not sell_signal) and config.get('enableSwingTopSell', True):
                    try:
                        rsi = d.get('rsi', None)
                        if rsi is None or pd.isna(rsi):
                            raise ValueError()
                        rsi = float(rsi)
                        min_rsi = float(config.get('swingSellRsi', 75) or 75)
                        if rsi < min_rsi:
                            raise ValueError()

                        near_high_pct = float(config.get('swingSellNearHighPct', 2.0) or 2.0) / 100.0
                        high20 = d.get('high20', None)
                        if high20 is None or pd.isna(high20):
                            high20 = float(df['high'].iloc[max(0, i - 19):i + 1].max())
                        else:
                            high20 = float(high20)

                        close0 = float(d.get('close', 0) or 0)
                        open0 = float(d.get('open', 0) or 0)
                        if close0 < high20 * (1 - near_high_pct) or close0 <= 0:
                            raise ValueError()
                        if close0 >= open0:
                            raise ValueError()

                        sell_signal = True
                        sell_reason = '高位转弱'
                    except Exception:
                        pass

                # 止盈（可配置开关）- 放到后面，尽量让利润奔跑
                if (not sell_signal) and config.get('useTakeProfit', True):
                    take_profit_level = config.get('takeProfit', 0.25)
                    if profit >= take_profit_level:
                        sell_signal = True
                        sell_reason = '止盈'
                
                # 持仓超时（可配置开关）
                if (not sell_signal) and config.get('useHoldTimeout', True):
                    hold_timeout_days = config.get('holdTimeoutDays', 30)
                    if hold_days > hold_timeout_days and profit < 0.05:
                        sell_signal = True
                        sell_reason = '持仓超时'
            
            # 记录信号
            if buy_signal and position == 0 and StrategyEngine.passes_global_buy_filters(df, i, strategy_name, config, market_state, signal_strength):
                # 动态仓位管理
                if config.get('useAdaptPosition', True):
                    # 信号强度越高，仓位越大
                    base_position = 0.3
                    strength_factor = signal_strength / 100
                    position = base_position + strength_factor * 0.7
                    # 根据市场状态调整
                    if market_state in ['bear_market', 'choppy_market']:
                        position *= 0.8  # 熊市/震荡市降低仓位
                    position = min(position, 1.0)  # 不超过100%
                else:
                    position = 1.0
                
                entry_price = d['close']
                entry_index = i
                highest_since_entry = d['high']
                peak_profit_since_entry = 0.0
                highest_body_since_entry = max(float(d.get('open', 0) or 0), float(d.get('close', 0) or 0))
                
                signals.append({
                    'index': i,
                    'date': d['date'],
                    'type': 'buy',
                    'price': float(d['close']),
                    'position': position,
                    'strength': signal_strength
                })
            
            if sell_signal and position > 0:
                profit_pct = ((d['close'] - entry_price) / entry_price) * 100
                signals.append({
                    'index': i,
                    'date': d['date'],
                    'type': 'sell',
                    'price': float(d['close']),
                    'position': position,
                    'profit': profit_pct,
                    'holdDays': i - entry_index,
                    'reason': sell_reason,
                    'strength': signal_strength
                })
                position = 0
                last_sell_index = i  # 更新卖出索引
        
        return signals
    
    @staticmethod
    def calculate_backtest(data, signals, init_capital):
        """计算回测结果"""
        df = pd.DataFrame(data)
        capital = init_capital
        shares = 0
        position = 0
        equity = [init_capital]
        
        wins = 0
        losses = 0
        total_profit = 0
        total_loss = 0
        hold_days_sum = 0
        trades = []
        
        signal_dict = {s['index']: s for s in signals}
        
        for i in range(len(df)):
            signal = signal_dict.get(i)
            
            if signal:
                if signal['type'] == 'buy':
                    amount = capital * signal['position'] * 0.998
                    shares = int(amount / signal['price'] / 100) * 100
                    capital -= shares * signal['price'] * 1.001
                    position = signal['position']
                    trades.append({**signal, 'shares': shares, 'amount': shares * signal['price']})
                else:
                    sell_amount = shares * signal['price'] * 0.998
                    capital += sell_amount
                    
                    if signal.get('profit', 0) > 0:
                        wins += 1
                        total_profit += signal['profit']
                    else:
                        losses += 1
                        total_loss += abs(signal.get('profit', 0))
                    
                    hold_days_sum += signal.get('holdDays', 0)
                    trades.append({**signal, 'shares': shares, 'amount': sell_amount})
                    shares = 0
                    position = 0
            
            current_value = capital + shares * df.iloc[i]['close']
            equity.append(current_value)
        
        final_equity = capital + shares * df.iloc[-1]['close']
        total_return = ((final_equity - init_capital) / init_capital) * 100
        days = len(df)
        annual_return = (pow(1 + total_return / 100, 252 / days) - 1) * 100 if days > 0 else 0
        
        # 最大回撤
        max_drawdown = 0
        peak = equity[0]
        for e in equity:
            peak = max(peak, e)
            drawdown = (peak - e) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # 夏普比率
        returns = []
        for i in range(1, len(equity)):
            if equity[i-1] > 0:
                returns.append((equity[i] - equity[i-1]) / equity[i-1])
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return * 252 - 0.03) / (std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe = 0
        
        win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
        profit_ratio = total_profit / total_loss if total_loss > 0 else (10 if total_profit > 0 else 0)
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        avg_hold_days = hold_days_sum / (wins + losses) if (wins + losses) > 0 else 0
        
        # 综合评分
        score = (
            min(total_return / 2, 30) +
            min(sharpe * 10, 25) +
            max(0, 25 - max_drawdown) +
            min(win_rate / 4, 20)
        )
        
        return {
            'initCapital': init_capital,
            'finalCapital': final_equity,
            'totalReturn': total_return,
            'annualReturn': annual_return,
            'maxDrawdown': max_drawdown,
            'sharpe': sharpe,
            'calmar': calmar,
            'winRate': win_rate,
            'profitRatio': profit_ratio,
            'avgHoldDays': avg_hold_days,
            'wins': wins,
            'losses': losses,
            'trades': trades,
            'equity': equity,
            'score': score,
            'tradeCount': wins + losses
        }


def calculate_indicators(df):
    """计算技术指标"""
    if df is None or len(df) == 0:
        return df
    
    # 转换为DataFrame（如果是字典列表）
    if isinstance(df, list):
        df = pd.DataFrame(df)
    
    # 基础指标
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    
    # 成交量均线
    df['volMa5'] = df['volume'].rolling(window=5).mean()
    df['volMa10'] = df['volume'].rolling(window=10).mean()
    df['volMa20'] = df['volume'].rolling(window=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macdSignal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macdHist'] = df['macd'] - df['macdSignal']
    
    # 布林带
    df['bollMid'] = df['close'].rolling(window=20).mean()
    boll_std = df['close'].rolling(window=20).std()
    df['bollUp'] = df['bollMid'] + 2 * boll_std
    df['bollDown'] = df['bollMid'] - 2 * boll_std
    df['bollWidth'] = (df['bollUp'] - df['bollDown']) / df['bollMid']  # 布林带宽度
    
    # 高低点
    df['high10'] = df['high'].rolling(window=10).max()
    df['low10'] = df['low'].rolling(window=10).min()
    df['high20'] = df['high'].rolling(window=20).max()
    df['low20'] = df['low'].rolling(window=20).min()
    df['high60'] = df['high'].rolling(window=60).max()
    df['low60'] = df['low'].rolling(window=60).min()
    
    # ATR (Average True Range)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR百分比
    
    # ADX (Average Directional Index) - 趋势强度
    df['plus_dm'] = df['high'].diff()
    df['minus_dm'] = -df['low'].diff()
    df['plus_dm'] = df['plus_dm'].where((df['plus_dm'] > 0) & (df['plus_dm'] > df['minus_dm']), 0)
    df['minus_dm'] = df['minus_dm'].where((df['minus_dm'] > 0) & (df['minus_dm'] > df['plus_dm']), 0)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=14).mean()
    
    # 量能分布分析
    df['vol_percentile'] = df['volume'].rank(pct=True) * 100
    
    # 价格位置分析
    df['price_position_20'] = (df['close'] - df['low20']) / (df['high20'] - df['low20']) * 100
    df['price_position_60'] = (df['close'] - df['low60']) / (df['high60'] - df['low60']) * 100
    
    return df


def identify_market_state(df):
    """识别市场状态 - 增强版"""
    if df is None or len(df) < 60:
        return 'unknown'
    
    # 转换为DataFrame（如果是字典列表）
    if isinstance(df, list):
        df = pd.DataFrame(df)
    
    latest = df.iloc[-1]
    recent = df.tail(20)
    
    # 趋势判断 (使用ADX增强)
    adx_strength = latest['adx'] if 'adx' in latest else 0
    ma_trend = 'neutral'
    
    if latest['ma5'] > latest['ma20'] and latest['ma20'] > latest['ma60']:
        ma_trend = 'strong_uptrend'
    elif latest['ma5'] > latest['ma20']:
        ma_trend = 'weak_uptrend'
    elif latest['ma5'] < latest['ma20'] and latest['ma20'] < latest['ma60']:
        ma_trend = 'strong_downtrend'
    elif latest['ma5'] < latest['ma20']:
        ma_trend = 'weak_downtrend'
    
    # 波动性判断 (使用布林带宽度和ATR)
    boll_width = latest['bollWidth'] if 'bollWidth' in latest else 0
    atr_volatility = latest['atr_pct'] if 'atr_pct' in latest else 0
    
    if boll_width > 0.15 or atr_volatility > 3:
        volatility_state = 'high'
    elif boll_width > 0.08 or atr_volatility > 1.5:
        volatility_state = 'medium'
    else:
        volatility_state = 'low'
    
    # 量能状态
    vol_state = 'normal'
    if latest['vol_percentile'] > 80:
        vol_state = 'high_volume'
    elif latest['vol_percentile'] < 20:
        vol_state = 'low_volume'
    
    # 综合市场状态判断
    if ma_trend == 'strong_uptrend' and volatility_state in ['medium', 'low']:
        return 'bull_market'
    elif ma_trend == 'strong_downtrend' and volatility_state == 'high':
        return 'bear_market'
    elif ma_trend in ['weak_uptrend', 'weak_downtrend'] and volatility_state == 'high':
        return 'volatile_market'
    elif ma_trend == 'neutral' and volatility_state == 'low':
        return 'sideways_market'
    elif ma_trend == 'neutral' and volatility_state == 'high':
        return 'choppy_market'
    else:
        return f'{ma_trend}_{volatility_state}'


def calculate_signal_strength(data, strategy_name, signals, market_state):
    """计算信号强度 - 多因子综合评分"""
    if not signals or len(signals) == 0:
        return signals
    
    df = pd.DataFrame(data)
    if len(df) < 20:
        return signals
    
    # 市场状态权重
    market_weights = {
        'bull_market': {'trend': 1.2, 'momentum': 1.1, 'volatility': 0.8},
        'bear_market': {'trend': 0.7, 'momentum': 0.8, 'volatility': 1.3},
        'sideways_market': {'trend': 0.8, 'momentum': 1.2, 'volatility': 1.0},
        'volatile_market': {'trend': 0.9, 'momentum': 1.3, 'volatility': 1.1},
        'choppy_market': {'trend': 0.6, 'momentum': 0.9, 'volatility': 1.2}
    }
    
    base_weight = market_weights.get(market_state, {'trend': 1.0, 'momentum': 1.0, 'volatility': 1.0})
    
    # 策略特定的因子权重
    strategy_factors = {
        'deep_fusion': {'ma_alignment': 0.2, 'macd_momentum': 0.2, 'rsi_condition': 0.15, 'volume_confirmation': 0.15, 'bollinger_position': 0.1, 'price_momentum': 0.1, 'volatility_adjustment': 0.1},
        'volume_breakout': {'breakout_strength': 0.3, 'volume_surge': 0.3, 'trend_alignment': 0.2, 'volatility_filter': 0.2},
        'oversold_rebound': {'oversold_level': 0.3, 'support_confirmation': 0.25, 'volume_confirmation': 0.2, 'trend_reversal': 0.15, 'volatility_adjustment': 0.1},
        'trend_enhanced': {'trend_strength': 0.35, 'ma_alignment': 0.25, 'momentum_confirmation': 0.2, 'volatility_filter': 0.2},
        'macd_divergence': {'divergence_quality': 0.3, 'trend_confirmation': 0.25, 'volume_support': 0.2, 'price_position': 0.15, 'volatility_adjustment': 0.1},
        'bollinger_extreme': {'extreme_level': 0.35, 'band_width': 0.2, 'volume_confirmation': 0.2, 'trend_alignment': 0.15, 'volatility_adjustment': 0.1},
        'momentum_rotation': {'momentum_strength': 0.3, 'relative_strength': 0.25, 'volume_confirmation': 0.2, 'trend_alignment': 0.15, 'volatility_adjustment': 0.1},
        'turtle_enhanced': {'breakout_quality': 0.3, 'trend_confirmation': 0.25, 'volatility_adjustment': 0.2, 'volume_confirmation': 0.15, 'position_sizing': 0.1}
    }
    
    factors = strategy_factors.get(strategy_name, {'default': 1.0})
    
    # 为每个信号计算强度
    enhanced_signals = []
    for signal in signals:
        if signal['type'] != 'buy':
            enhanced_signals.append(signal)
            continue
            
        signal_date = signal['date']
        signal_idx = df[df['date'] == signal_date].index
        if len(signal_idx) == 0:
            enhanced_signals.append(signal)
            continue
            
        idx = signal_idx[0]
        if idx < 20:
            enhanced_signals.append(signal)
            continue
            
        current_row = df.iloc[idx]
        prev_row = df.iloc[idx-1] if idx > 0 else current_row
        
        # 计算各因子得分 (0-100分)
        factor_scores = {}
        
        if strategy_name == 'deep_fusion':
            # MA对齐度
            ma_score = 0
            if current_row['ma5'] > current_row['ma20'] > current_row['ma60']:
                ma_score = 100
            elif current_row['ma5'] > current_row['ma20']:
                ma_score = 70
            elif current_row['ma5'] < current_row['ma20'] < current_row['ma60']:
                ma_score = 30
            else:
                ma_score = 50
            factor_scores['ma_alignment'] = ma_score * base_weight['trend']
            
            # MACD动量
            macd_score = 0
            if current_row['macd'] > 0 and current_row['macd'] > prev_row['macd']:
                macd_score = 100
            elif current_row['macd'] > 0:
                macd_score = 70
            elif current_row['macd'] < 0 and current_row['macd'] < prev_row['macd']:
                macd_score = 30
            else:
                macd_score = 50
            factor_scores['macd_momentum'] = macd_score * base_weight['momentum']
            
            # RSI条件
            rsi_score = 0
            if current_row['rsi'] < 25:
                rsi_score = 100
            elif current_row['rsi'] < 35:
                rsi_score = 80
            elif current_row['rsi'] < 45:
                rsi_score = 60
            elif current_row['rsi'] > 70:
                rsi_score = 20
            else:
                rsi_score = 50
            factor_scores['rsi_condition'] = rsi_score * base_weight['volatility']
            
            # 成交量确认
            vol_score = 0
            if current_row['volume'] > current_row['volMa5'] * 2:
                vol_score = 100
            elif current_row['volume'] > current_row['volMa5'] * 1.5:
                vol_score = 80
            elif current_row['volume'] > current_row['volMa5']:
                vol_score = 60
            else:
                vol_score = 40
            factor_scores['volume_confirmation'] = vol_score
            
            # 布林带位置
            boll_score = 0
            if current_row['close'] < current_row['bollDown']:
                boll_score = 100
            elif current_row['close'] < current_row['bollMid']:
                boll_score = 70
            elif current_row['close'] > current_row['bollUp']:
                boll_score = 30
            else:
                boll_score = 50
            factor_scores['bollinger_position'] = boll_score
            
            # 价格动量
            momentum_score = 0
            price_change_5d = (current_row['close'] - df.iloc[max(0, idx-5)]['close']) / df.iloc[max(0, idx-5)]['close'] * 100
            if price_change_5d > 5:
                momentum_score = 80
            elif price_change_5d > 2:
                momentum_score = 60
            elif price_change_5d < -5:
                momentum_score = 30
            else:
                momentum_score = 50
            factor_scores['price_momentum'] = momentum_score * base_weight['momentum']
            
            # 波动率调整
            vol_adj_score = 100
            if current_row['atr_pct'] > 5:
                vol_adj_score = 70
            elif current_row['atr_pct'] > 3:
                vol_adj_score = 85
            factor_scores['volatility_adjustment'] = vol_adj_score * base_weight['volatility']
            
        elif strategy_name == 'volume_breakout':
            # 突破强度
            breakout_score = 0
            if current_row['close'] > current_row['high20']:
                breakout_score = 100
            elif current_row['close'] > current_row['high10']:
                breakout_score = 80
            elif current_row['close'] > current_row['high60']:
                breakout_score = 60
            else:
                breakout_score = 40
            factor_scores['breakout_strength'] = breakout_score * base_weight['momentum']
            
            # 成交量激增
            vol_surge_score = 0
            vol_ratio = current_row['volume'] / current_row['volMa5']
            if vol_ratio > 3:
                vol_surge_score = 100
            elif vol_ratio > 2:
                vol_surge_score = 85
            elif vol_ratio > 1.5:
                vol_surge_score = 70
            else:
                vol_surge_score = 50
            factor_scores['volume_surge'] = vol_surge_score
            
            # 趋势对齐
            trend_align_score = 0
            if current_row['ma5'] > current_row['ma20'] > current_row['ma60']:
                trend_align_score = 100
            elif current_row['ma5'] > current_row['ma20']:
                trend_align_score = 75
            elif current_row['ma5'] < current_row['ma20']:
                trend_align_score = 40
            else:
                trend_align_score = 60
            factor_scores['trend_alignment'] = trend_align_score * base_weight['trend']
            
            # 波动率过滤
            vol_filter_score = 100
            if current_row['bollWidth'] < 0.05:  # 布林带过窄，突破可能无效
                vol_filter_score = 40
            elif current_row['bollWidth'] > 0.2:  # 波动过大，风险高
                vol_filter_score = 60
            factor_scores['volatility_filter'] = vol_filter_score * base_weight['volatility']
            
        # 其他策略的信号强度计算可以类似实现...
        # 为了简洁，这里先实现两个主要策略，其他策略使用基础评分
        
        else:
            # 默认信号强度计算
            base_strength = signal.get('strength', 50)
            factor_scores['default'] = base_strength
        
        # 计算综合强度
        total_score = 0
        total_weight = 0
        
        for factor, weight in factors.items():
            score = factor_scores.get(factor, 50)
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_strength = min(100, max(20, total_score / total_weight))
        else:
            final_strength = signal.get('strength', 50)
        
        # 更新信号强度
        enhanced_signal = signal.copy()
        enhanced_signal['strength'] = final_strength
        enhanced_signals.append(enhanced_signal)
    
    return enhanced_signals





def calculate_chip_distribution(df, window=360, date=None):
    """计算筹码分布"""
    if df is None or len(df) == 0:
        return []
    
    # 转换为DataFrame（如果是字典列表）
    if isinstance(df, list):
        df = pd.DataFrame(df)
    
    # 如果指定了日期，找到该日期的数据
    if date:
        df['date'] = pd.to_datetime(df['date'])
        target_date = pd.to_datetime(date)
        
        # 找到最接近的日期
        if target_date in df['date'].values:
            df = df[df['date'] <= target_date]
        else:
            # 找到最接近的日期
            closest_idx = (df['date'] - target_date).abs().idxmin()
            df = df.iloc[:closest_idx + 1]
    
    # 使用最近window天的数据
    df = df.tail(window)
    
    if len(df) < 10:
        return []
    
    # 计算价格区间
    min_price = df['low'].min()
    max_price = df['high'].max()
    
    if min_price == max_price:
        return []
    
    # 分成20个价格区间
    num_bins = 120
    price_bins = np.linspace(min_price, max_price, num_bins + 1)
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    chip_volumes = np.zeros(num_bins, dtype=float)
    
    if 'date' in df.columns:
        df = df.copy()
        try:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        except Exception:
            pass
    
    vol_ema = None
    for _, row in df.iterrows():
        day_low = float(row.get('low', np.nan))
        day_high = float(row.get('high', np.nan))
        day_volume = float(row.get('volume', 0) or 0)
        day_close = float(row.get('close', np.nan))
        
        if not np.isfinite(day_low) or not np.isfinite(day_high):
            continue
        
        if day_high < day_low:
            day_low, day_high = day_high, day_low

        if np.isfinite(day_volume) and day_volume > 0:
            vol_ema = day_volume if vol_ema is None else (vol_ema * 0.9 + day_volume * 0.1)
        
        turnover = row.get('turnover')
        # 严格检查 turnover 是否为有效数字
        turnover_valid = False
        try:
            if pd.notna(turnover) and str(turnover).strip() != '':
                turnover_val = float(turnover)
                if 0 < turnover_val <= 100:
                    turnover_valid = True
                    turnover = turnover_val
        except (ValueError, TypeError):
            pass

        if turnover_valid:
            turnover_ratio = turnover / 100
        elif vol_ema and np.isfinite(day_volume) and day_volume > 0:
            turnover_ratio = 0.01 * (day_volume / vol_ema)
        else:
            turnover_ratio = 0.01
        turnover_ratio = min(max(turnover_ratio, 0.002), 0.08)
        daily_retain = 1 - min(max(turnover_ratio, 0.001), 1)
        chip_volumes *= daily_retain
        
        if day_high == day_low:
            bin_idx = np.searchsorted(price_bins, day_high, side='right') - 1
            if 0 <= bin_idx < num_bins:
                chip_volumes[bin_idx] += turnover_ratio
            continue
        
        if np.isfinite(day_close):
            typical_price = (day_high + day_low + day_close) / 3
        else:
            typical_price = (day_high + day_low) / 2
        
        range_width = day_high - day_low
        sigma = max(range_width / 20, typical_price * 0.002)
        
        if sigma <= 0:
            bin_idx = np.searchsorted(price_bins, typical_price, side='right') - 1
            if 0 <= bin_idx < num_bins:
                chip_volumes[bin_idx] += turnover_ratio
            continue
        
        z = (bin_centers - typical_price) / sigma
        weights = np.exp(-0.5 * z * z)
        weights[(bin_centers < day_low) | (bin_centers > day_high)] = 0
        weights[np.abs(z) > 2.5] = 0
        
        total_w = weights.sum()
        if total_w <= 0:
            continue
        
        chip_volumes += (weights / total_w) * turnover_ratio
    
    total_volume = chip_volumes.sum()
    if total_volume <= 0:
        return []
    
    chip_distribution = []
    for i, price in enumerate(bin_centers):
        volume_in_range = chip_volumes[i]
        concentration = (volume_in_range / total_volume * 100) if total_volume > 0 else 0
        chip_distribution.append({
            'price': round(price, 2),
            'volume': round(float(volume_in_range), 6),
            'concentration': round(concentration, 2)
        })
    
    return chip_distribution


# API路由
@app.route('/api/local_state', methods=['GET'])
def api_get_local_state():
    key = (request.args.get('key') or '').strip()
    if key not in _ALLOWED_LOCAL_STATE_KEYS:
        return jsonify({'success': False, 'message': 'key 不允许'}), 400
    value = _read_local_state_value(key)
    return jsonify({'success': True, 'key': key, 'value': value})


@app.route('/api/local_state', methods=['POST'])
def api_set_local_state():
    payload = request.get_json(silent=True) or {}
    key = (payload.get('key') or '').strip()
    if key not in _ALLOWED_LOCAL_STATE_KEYS:
        return jsonify({'success': False, 'message': 'key 不允许'}), 400
    value = payload.get('value', None)
    ok = _write_local_state_value(key, value)
    return jsonify({'success': ok, 'key': key})


@app.route('/api/clear_cache', methods=['POST'])
def api_clear_cache():
    payload = request.get_json(silent=True) or {}
    scope = (payload.get('scope') or 'all_data').strip().lower()
    clear_local_state = bool(payload.get('clear_local_state', False))

    removed = {
        'files': 0,
        'dirs': 0,
        'errors': 0,
    }

    def _rm_file(path):
        try:
            if path and os.path.exists(path) and os.path.isfile(path):
                os.remove(path)
                removed['files'] += 1
                return True
        except Exception:
            removed['errors'] += 1
        return False

    def _rm_dir(path, keep_dir=False):
        try:
            if not path or not os.path.exists(path) or not os.path.isdir(path):
                return False
            if keep_dir:
                ok = False
                for name in os.listdir(path):
                    fp = os.path.join(path, name)
                    if os.path.isdir(fp):
                        shutil.rmtree(fp, ignore_errors=True)
                        removed['dirs'] += 1
                        ok = True
                    else:
                        try:
                            os.remove(fp)
                            removed['files'] += 1
                            ok = True
                        except Exception:
                            removed['errors'] += 1
                return ok
            shutil.rmtree(path, ignore_errors=True)
            removed['dirs'] += 1
            return True
        except Exception:
            removed['errors'] += 1
        return False

    try:
        cache_manager.stock_cache = {}
        cache_manager.params_cache = {}
        cache_manager._df_mem_cache = {}
        cache_manager._no_update_until = {}
    except Exception:
        removed['errors'] += 1

    if scope in ('all', 'all_data', 'data'):
        _rm_dir(getattr(cache_manager, 'data_dir', None), keep_dir=True)
        _rm_dir(FUNDAMENTALS_CACHE_DIR, keep_dir=True)
        _rm_dir(CONCEPT_MEMBERS_CACHE_DIR, keep_dir=True)
        _rm_file(STOCK_LIST_CACHE_FILE)
        _rm_file(CONCEPT_LIST_CACHE_FILE)

    if clear_local_state and scope in ('all', 'all_data', 'data'):
        _rm_dir(LOCAL_STATE_DIR, keep_dir=True)

    return jsonify({'success': True, 'scope': scope, 'removed': removed})


@app.route('/api/stock_data', methods=['GET'])
def api_get_stock_data():
    """获取股票数据（带缓存），支持日线/周线/月线"""
    ts_code = request.args.get('ts_code', '000001.SZ')
    start_date = request.args.get('start_date', '2022-01-01')
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    freq = request.args.get('freq', 'D').upper()  # D=日线, W=周线, M=月线
    lite = (request.args.get('lite') or '').strip().lower() in ('1', 'true', 'yes', 'y')
    _inc_stat('stock_data_requests', 1)
    if lite:
        _inc_stat('stock_data_lite_requests', 1)
    
    # 验证 freq 参数
    if freq not in ['D', 'W', 'M']:
        freq = 'D'
    
    _log(f"API请求: ts_code={ts_code}, start_date={start_date}, end_date={end_date}, freq={freq}, lite={lite}")
    
    # 使用缓存管理器获取数据
    data = cache_manager.get_stock_data(ts_code, start_date, end_date, freq)
    _log(f"从缓存管理器获取数据结果: {type(data)}, 长度: {len(data) if data else 'None'}")
    
    if data is None or (isinstance(data, list) and len(data) == 0):
        # 如果无法获取真实数据，生成模拟数据
        _log(f"真实数据获取失败，生成模拟数据: {ts_code}")
        data = TushareDataFetcher.gen_mock_data(ts_code, start_date, end_date)
        
        if data is None or len(data) == 0:
            return jsonify({
                'success': False,
                'message': f'无法获取股票数据: {ts_code}',
                'data': []
            })
    
    # 转换为DataFrame（如果是字典列表）
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    
    # 补全缺失的换手率（使用成交量估算）
    # df = fill_missing_turnover_with_volume(df)

    market_state = None
    chip_data = []
    if not lite:
        df = calculate_indicators(df)
        market_state = identify_market_state(df)
        chip_data = calculate_chip_distribution(df)
    
    # 转换为JSON
    data = df.to_dict(orient='records')
    for row in data:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
    
    # 根据周期返回对应的周期标识
    freq_name = {'D': 'daily', 'W': 'weekly', 'M': 'monthly'}.get(freq, 'daily')
    
    return jsonify({
        'success': True,
        'message': 'ok',
        'ts_code': ts_code,
        'freq': freq,
        'freq_name': freq_name,
        'market_state': market_state,
        'data_count': len(data),
        'data': data,
        'chip_distribution': chip_data,
        'cached': True,
        'lite': lite
    })


@app.route('/api/fundamentals', methods=['GET'])
def api_get_fundamentals():
    ts_code = (request.args.get('ts_code') or '').strip()
    if not ts_code:
        return jsonify({'success': False, 'message': 'ts_code 不能为空'}), 400

    max_age_days_raw = (request.args.get('max_age_days') or '7').strip()
    try:
        max_age_days = int(max_age_days_raw)
    except Exception:
        max_age_days = 7
    max_age_days = max(0, min(365, max_age_days))

    allow_cache = (request.args.get('cache') or '1').strip() != '0'
    debug = (request.args.get('debug') or '').strip() == '1'

    safe_code = re.sub(r"[^a-zA-Z0-9_.-]", "_", ts_code)[:50] or 'unknown'
    cache_file = os.path.join(FUNDAMENTALS_CACHE_DIR, f"{safe_code}.json")

    if allow_cache:
        cached = _load_json_cache_payload(cache_file)
        if cached:
            try:
                cached_at = datetime.fromisoformat(cached.get('cached_at'))
                if (datetime.now() - cached_at).days <= max_age_days:
                    return jsonify({'success': True, 'ts_code': ts_code, 'cached': True, 'data': cached.get('data')})
            except Exception:
                pass

    pro_client = _get_pro()
    if pro_client is None:
        return jsonify({'success': False, 'message': '未配置Tushare Token，无法获取基本面数据', 'ts_code': ts_code, 'cached': False, 'data': None})

    def _fmt_ymd(v):
        s = str(v or '').strip()
        if len(s) == 8 and s.isdigit():
            return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
        return s

    now = datetime.now()
    end_ymd = now.strftime('%Y%m%d')
    fundamentals = {'ts_code': ts_code}
    sources = {'daily_basic': False, 'fina_indicator': False, 'cashflow': False}

    try:
        start_basic = (now - timedelta(days=90)).strftime('%Y%m%d')
        df_basic = pro_client.daily_basic(
            ts_code=ts_code,
            start_date=start_basic,
            end_date=end_ymd,
            fields='ts_code,trade_date,pe,pb,total_mv,circ_mv,turnover_rate'
        )
        if df_basic is not None and not df_basic.empty:
            df_basic = df_basic.sort_values('trade_date').reset_index(drop=True)
            row = df_basic.iloc[-1].to_dict()
            fundamentals.update({
                'trade_date': _fmt_ymd(row.get('trade_date')),
                'pe': row.get('pe'),
                'pb': row.get('pb'),
                'turnover_rate': row.get('turnover_rate'),
                'total_mv': row.get('total_mv'),
                'circ_mv': row.get('circ_mv'),
            })
            sources['daily_basic'] = True
    except Exception as e:
        if debug:
            fundamentals['daily_basic_error'] = str(e)

    try:
        start_fina = (now - timedelta(days=730)).strftime('%Y%m%d')
        df_fina = pro_client.fina_indicator(
            ts_code=ts_code,
            start_date=start_fina,
            end_date=end_ymd,
            fields='ts_code,end_date,roe,roe_dt,roa,netprofit_yoy,dt_netprofit_yoy,or_yoy,grossprofit_margin,debt_to_assets'
        )
        if df_fina is not None and not df_fina.empty:
            df_fina = df_fina.sort_values('end_date').reset_index(drop=True)
            row = df_fina.iloc[-1].to_dict()
            fundamentals.update({
                'report_end_date': _fmt_ymd(row.get('end_date')),
                'roe': row.get('roe'),
                'roe_dt': row.get('roe_dt'),
                'roa': row.get('roa'),
                'netprofit_yoy': row.get('netprofit_yoy'),
                'dt_netprofit_yoy': row.get('dt_netprofit_yoy'),
                'or_yoy': row.get('or_yoy'),
                'grossprofit_margin': row.get('grossprofit_margin'),
                'debt_to_assets': row.get('debt_to_assets'),
            })
            sources['fina_indicator'] = True
    except Exception as e:
        if debug:
            fundamentals['fina_indicator_error'] = str(e)

    try:
        start_cf = (now - timedelta(days=730)).strftime('%Y%m%d')
        df_cf = pro_client.cashflow(
            ts_code=ts_code,
            start_date=start_cf,
            end_date=end_ymd,
            fields='ts_code,end_date,n_cashflow_act'
        )
        if df_cf is not None and not df_cf.empty:
            df_cf = df_cf.sort_values('end_date').reset_index(drop=True)
            row = df_cf.iloc[-1].to_dict()
            fundamentals.update({
                'cashflow_end_date': _fmt_ymd(row.get('end_date')),
                'n_cashflow_act': row.get('n_cashflow_act'),
            })
            sources['cashflow'] = True
    except Exception as e:
        if debug:
            fundamentals['cashflow_error'] = str(e)

    fundamentals['sources'] = sources
    _save_json_cache_payload(cache_file, fundamentals)
    return jsonify({'success': True, 'ts_code': ts_code, 'cached': False, 'data': fundamentals})


@app.route('/api/index_stocks', methods=['GET'])
def api_get_index_stocks():
    """获取指数成分股"""
    index_code = request.args.get('index_code', '000300.SH')
    
    # 检查缓存
    cache_key = f'index_stocks_{index_code}'
    cached = cache_manager.get('params', cache_key)
    if cached:
        cached_at = datetime.fromisoformat(cached['meta']['cached_at'])
        if (datetime.now() - cached_at).days < 7: # 缓存7天
            cached_data = cached.get('data')
            expected = 50 if index_code == '000016.SH' else (300 if index_code == '000300.SH' else None)
            if expected and isinstance(cached_data, list) and len(cached_data) < max(10, expected // 2):
                cached_data = None
            if cached_data is not None:
                return jsonify({'success': True, 'count': len(cached_data) if isinstance(cached_data, list) else 0, 'data': cached_data})
            
    data = TushareDataFetcher.get_index_weights(index_code)
    
    # 如果获取失败，返回一些模拟数据作为 fallback
    if not data:
        if index_code == '000016.SH': # 上证50
            data = [
                {'ts_code': '600519.SH', 'name': '贵州茅台'}, {'ts_code': '601318.SH', 'name': '中国平安'},
                {'ts_code': '600036.SH', 'name': '招商银行'}, {'ts_code': '600276.SH', 'name': '恒瑞医药'},
                {'ts_code': '600030.SH', 'name': '中信证券'}, {'ts_code': '601012.SH', 'name': '隆基绿能'}
            ]
        elif index_code == '000300.SH': # 沪深300
             data = [
                {'ts_code': '600519.SH', 'name': '贵州茅台'}, {'ts_code': '300750.SZ', 'name': '宁德时代'},
                {'ts_code': '000858.SZ', 'name': '五粮液'}, {'ts_code': '002594.SZ', 'name': '比亚迪'},
             ]
             
    if data:
        cache_manager.set('params', cache_key, data)
        
    return jsonify({'success': True, 'count': len(data) if data else 0, 'data': data})


@app.route('/api/stock_list', methods=['GET'])
def api_get_stock_list():
    """获取股票列表（带缓存）"""
    file_cached = _load_json_cache_file(STOCK_LIST_CACHE_FILE)
    if file_cached:
        try:
            cached_at = datetime.fromisoformat(file_cached['cached_at'])
            if (datetime.now() - cached_at).days < 30 and len(file_cached['data']) >= 2000:
                return jsonify({
                    'success': True,
                    'count': len(file_cached['data']),
                    'data': file_cached['data'],
                    'cached': True
                })
        except Exception:
            pass

    cache_key = 'stock_list'
    cached = cache_manager.get('params', cache_key)
    
    if cached:
        # 检查缓存是否超过7天
        cached_at = datetime.fromisoformat(cached['meta']['cached_at'])
        if (datetime.now() - cached_at).days < 7:
            print("[缓存命中] 股票列表")
            return jsonify({
                'success': True,
                'count': len(cached['data']),
                'data': cached['data'],
                'cached': True
            })
    
    pro_client = _get_pro()
    if pro_client is None:
        default_stocks = [
            {'ts_code': '000001.SZ', 'symbol': '000001', 'name': '平安银行', 'industry': '银行'},
            {'ts_code': '000002.SZ', 'symbol': '000002', 'name': '万科A', 'industry': '房地产'},
            {'ts_code': '600519.SH', 'symbol': '600519', 'name': '贵州茅台', 'industry': '白酒'},
            {'ts_code': '000858.SZ', 'symbol': '000858', 'name': '五粮液', 'industry': '白酒'},
            {'ts_code': '601318.SH', 'symbol': '601318', 'name': '中国平安', 'industry': '保险'},
            {'ts_code': '600036.SH', 'symbol': '600036', 'name': '招商银行', 'industry': '银行'},
            {'ts_code': '000333.SZ', 'name': '美的集团', 'industry': '家电'},
            {'ts_code': '300750.SZ', 'name': '宁德时代', 'industry': '电池'},
        ]
        return jsonify({
            'success': True,
            'count': len(default_stocks),
            'data': default_stocks,
            'cached': False,
            'message': '未配置Tushare Token，行业/股票列表仅提供少量默认数据。'
        })

    try:
        # 尝试获取股票列表，不限制数量以获取全量行业信息
        df = pro_client.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,industry'
        )
        
        if df is not None and not df.empty:
            stocks = df.to_dict(orient='records')
            cache_manager.set('params', cache_key, stocks)
            _save_json_cache_file(STOCK_LIST_CACHE_FILE, stocks)
            print(f"[更新缓存] 股票列表: {len(stocks)} 只")
            return jsonify({
                'success': True,
                'count': len(stocks),
                'data': stocks,
                'cached': False
            })
    except Exception as e:
        print(f"获取股票列表错误: {e}")
    
    # 返回默认列表（当API失败时）
    default_stocks = [
        {'ts_code': '000001.SZ', 'symbol': '000001', 'name': '平安银行', 'industry': '银行'},
        {'ts_code': '000002.SZ', 'symbol': '000002', 'name': '万科A', 'industry': '房地产'},
        {'ts_code': '600519.SH', 'symbol': '600519', 'name': '贵州茅台', 'industry': '白酒'},
        {'ts_code': '000858.SZ', 'symbol': '000858', 'name': '五粮液', 'industry': '白酒'},
        {'ts_code': '601318.SH', 'symbol': '601318', 'name': '中国平安', 'industry': '保险'},
        {'ts_code': '600036.SH', 'symbol': '600036', 'name': '招商银行', 'industry': '银行'},
        {'ts_code': '000333.SZ', 'name': '美的集团', 'industry': '家电'},
        {'ts_code': '300750.SZ', 'name': '宁德时代', 'industry': '电池'},
    ]
    return jsonify({
        'success': True,
        'count': len(default_stocks),
        'data': default_stocks,
        'cached': False,
        'message': '股票列表获取失败，已回退到内置少量列表。'
    })


@app.route('/api/concept_list', methods=['GET'])
def api_get_concept_list():
    file_cached = _load_json_cache_file(CONCEPT_LIST_CACHE_FILE)
    if file_cached:
        try:
            cached_at = datetime.fromisoformat(file_cached['cached_at'])
            if (datetime.now() - cached_at).days < 30 and len(file_cached['data']) >= 200:
                return jsonify({'success': True, 'count': len(file_cached['data']), 'data': file_cached['data'], 'cached': True})
        except Exception:
            pass

    pro_client = _get_pro()
    if pro_client is None:
        return jsonify({'success': True, 'count': 0, 'data': [], 'cached': False, 'message': '未配置Tushare Token，无法拉取概念列表。'})

    try:
        df = None
        try:
            concept_func = getattr(pro_client, 'concept', None)
            if callable(concept_func):
                try:
                    df = concept_func(src='ts')
                except Exception:
                    df = concept_func()
            else:
                try:
                    df = pro_client.query('concept', src='ts')
                except Exception:
                    df = pro_client.query('concept')
        except Exception:
            df = pro_client.query('concept')
        if df is None or df.empty:
            return jsonify({'success': True, 'count': 0, 'data': [], 'cached': False})

        records = df.to_dict(orient='records')
        normalized = []
        for r in records:
            name = r.get('name') or r.get('concept_name') or r.get('ts_name')
            code = r.get('code') or r.get('concept_code') or r.get('ts_code')
            if not name:
                continue
            normalized.append({'name': str(name), 'code': str(code) if code is not None else ''})

        normalized.sort(key=lambda x: x.get('name', ''))
        _save_json_cache_file(CONCEPT_LIST_CACHE_FILE, normalized)
        return jsonify({'success': True, 'count': len(normalized), 'data': normalized, 'cached': False})
    except Exception as e:
        print(f"获取概念列表错误: {e}")
        debug = (request.args.get('debug') or '').strip() == '1'
        payload = {'success': True, 'count': 0, 'data': [], 'cached': False, 'message': '概念列表获取失败'}
        if debug:
            payload['error'] = str(e)
        return jsonify(payload)


@app.route('/api/concept_members', methods=['GET'])
def api_get_concept_members():
    codes_raw = request.args.get('codes', '') or ''
    codes = [c.strip() for c in codes_raw.split(',') if c.strip()]
    if not codes:
        return jsonify({'success': False, 'message': 'codes 不能为空', 'ts_codes': [], 'count': 0}), 400

    pro_client = _get_pro()
    if pro_client is None:
        return jsonify({'success': True, 'message': '未配置Tushare Token，无法拉取概念成分股。', 'ts_codes': [], 'count': 0, 'cached': False})

    all_ts_codes = set()
    by_code = {}
    any_cached = False

    for concept_code in codes[:50]:
        cache_file = os.path.join(CONCEPT_MEMBERS_CACHE_DIR, f"{concept_code}.json")
        file_cached = _load_json_cache_file(cache_file)
        if file_cached:
            try:
                cached_at = datetime.fromisoformat(file_cached['cached_at'])
                if (datetime.now() - cached_at).days < 30 and len(file_cached['data']) >= 1:
                    members = file_cached['data']
                    by_code[concept_code] = members
                    for ts in members:
                        all_ts_codes.add(ts)
                    any_cached = True
                    continue
            except Exception:
                pass

        try:
            df = None
            try:
                detail_func = getattr(pro_client, 'concept_detail', None)
                if callable(detail_func):
                    try:
                        df = detail_func(id=concept_code)
                    except Exception:
                        df = detail_func(concept_id=concept_code)
                else:
                    try:
                        df = pro_client.query('concept_detail', id=concept_code)
                    except Exception:
                        df = pro_client.query('concept_detail', concept_id=concept_code)
            except Exception:
                df = pro_client.query('concept_detail', id=concept_code)
            if df is None or df.empty:
                by_code[concept_code] = []
                continue
            records = df.to_dict(orient='records')
            members = []
            for r in records:
                ts_code = r.get('ts_code') or r.get('con_code')
                if not ts_code:
                    continue
                members.append(str(ts_code))
                all_ts_codes.add(str(ts_code))
            members = sorted(list(set(members)))
            by_code[concept_code] = members
            _save_json_cache_file(cache_file, members)
        except Exception as e:
            print(f"获取概念成分股失败: {concept_code} {e}")
            by_code[concept_code] = []

    out = sorted(all_ts_codes)
    return jsonify({'success': True, 'count': len(out), 'ts_codes': out, 'by_code': by_code, 'cached': any_cached})


@app.route('/api/best_strategy', methods=['POST'])
def api_best_strategy():
    payload = request.get_json(silent=True) or {}

    ts_codes = payload.get('ts_codes') or []
    if not isinstance(ts_codes, list) or len(ts_codes) == 0:
        return jsonify({'success': False, 'message': 'ts_codes 不能为空且必须为数组', 'data': []}), 400

    if len(ts_codes) > 300:
        ts_codes = ts_codes[:300]

    start_date = payload.get('start_date', '2022-01-01')
    end_date = payload.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    signal_period = int(payload.get('signal_period', 3))
    history_window = int(payload.get('history_window', 180))
    init_capital = float(payload.get('init_capital', 1000000))

    strategies = payload.get('strategies') or list(StrategyEngine.STRATEGIES.keys())
    if not isinstance(strategies, list) or len(strategies) == 0:
        strategies = list(StrategyEngine.STRATEGIES.keys())

    base_config = payload.get('config') or {}
    if not isinstance(base_config, dict):
        base_config = {}

    results = []

    for ts_code in ts_codes:
        try:
            data = cache_manager.get_stock_data(ts_code, start_date, end_date) or []
            if not data or len(data) < 80:
                mock = TushareDataFetcher.gen_mock_data(ts_code, start_date, end_date)
                if mock and len(mock) >= 80:
                    data = mock
                else:
                    continue

            df = pd.DataFrame(data)
            df = calculate_indicators(df)
            records = df.where(pd.notnull(df), None).to_dict('records')
            if not records:
                continue

            last_date_str = records[-1].get('date')
            if not last_date_str:
                continue

            last_dt = datetime.strptime(last_date_str, '%Y-%m-%d')
            recent_cutoff = last_dt - timedelta(days=signal_period)
            history_start = last_dt - timedelta(days=history_window)

            history_records = [
                r for r in records
                if r.get('date') and history_start <= datetime.strptime(r['date'], '%Y-%m-%d') <= recent_cutoff
            ]

            best = None

            for strategy in strategies:
                config = {**base_config}
                if 'takeProfit' not in config:
                    config['takeProfit'] = 0.15
                if 'stopLoss' not in config:
                    config['stopLoss'] = 0.08
                if 'volumeMulti' not in config:
                    config['volumeMulti'] = 1.5

                signals = StrategyEngine.execute_strategy(records, strategy, config)
                if not signals:
                    continue

                recent_buys = [
                    s for s in signals
                    if s.get('type') == 'buy' and s.get('date') and datetime.strptime(s['date'], '%Y-%m-%d') >= recent_cutoff
                ]
                if not recent_buys:
                    continue

                recent_buy_date = max(s['date'] for s in recent_buys if s.get('date'))

                hist_score = 0
                hist_return = 0
                hist_win_rate = 0
                hist_trade_count = 0

                if len(history_records) >= 80:
                    hist_signals = StrategyEngine.execute_strategy(history_records, strategy, config)
                    hist_res = StrategyEngine.calculate_backtest(history_records, hist_signals, init_capital)
                    hist_score = float(hist_res.get('score') or 0)
                    hist_return = float(hist_res.get('totalReturn') or 0)
                    hist_win_rate = float(hist_res.get('winRate') or 0)
                    hist_trade_count = int(hist_res.get('tradeCount') or 0)

                cand = {
                    'ts_code': ts_code,
                    'best_strategy': strategy,
                    'recent_buy_date': recent_buy_date,
                    'history_score': hist_score,
                    'history_return': hist_return,
                    'history_win_rate': hist_win_rate,
                    'history_trade_count': hist_trade_count
                }

                if best is None or cand['history_score'] > best['history_score'] or (
                    cand['history_score'] == best['history_score'] and cand['history_return'] > best['history_return']
                ):
                    best = cand

            if best is not None:
                meta = StrategyEngine.STRATEGIES.get(best['best_strategy'], {})
                best['best_strategy_name'] = meta.get('name', best['best_strategy'])
                best['best_strategy_icon'] = meta.get('icon', '')
                results.append(best)
        except Exception as e:
            print(f"best_strategy error: {ts_code} {e}")
            continue

    results.sort(key=lambda x: (x.get('history_score', 0), x.get('history_return', 0)), reverse=True)
    return jsonify({'success': True, 'count': len(results), 'data': results})


# 代表性股票列表（每行业1只，用于批量参数优化 - 精简版以提高速度）
REPRESENTATIVE_STOCKS = {
    '银行': ['000001.SZ'],        # 平安银行（低波动）
    '科技': ['300750.SZ'],        # 宁德时代（高波动）
    '医药': ['600276.SH'],        # 恒瑞医药（中波动）
    '白酒': ['600519.SH'],        # 贵州茅台（趋势性强）
    '新能源': ['002594.SZ'],      # 比亚迪（高波动）
    '电子': ['002415.SZ'],        # 海康威视（中波动）
    '化工': ['600309.SH'],        # 万华化学（周期股）
    '有色': ['601899.SH'],        # 紫金矿业（资源股）
}


class AutoParamOptimizer:
    """自动批量参数优化器"""
    
    @staticmethod
    def get_industry_stocks():
        """获取代表性股票列表"""
        stocks = []
        for industry, codes in REPRESENTATIVE_STOCKS.items():
            for code in codes:
                stocks.append({
                    'ts_code': code,
                    'industry': industry
                })
        return stocks
    
    @staticmethod
    def optimize_stock_params(ts_code, industry, start_date, end_date, strategies):
        """对单只股票进行全策略参数优化"""
        try:
            print(f"开始优化 {ts_code} ({industry})...")
            
            # 获取股票数据
            data = cache_manager.get_stock_data(ts_code, start_date, end_date) or []
            print(f"  从缓存获取数据: {len(data)} 条")
            
            if not data or len(data) < 30:  # 降低数据要求
                print(f"  缓存数据不足，生成mock数据...")
                mock = TushareDataFetcher.gen_mock_data(ts_code, start_date, end_date)
                if mock and len(mock) >= 30:
                    data = mock
                    print(f"  生成mock数据: {len(data)} 条")
                else:
                    print(f"  ❌ 数据不足，跳过 {ts_code}")
                    return None
            
            df = pd.DataFrame(data)
            df = calculate_indicators(df)
            
            # 分析股票特征
            stock_features = StockFeatureAnalyzer.analyze_stock_features(df)
            records = df.where(pd.notnull(df), None).to_dict('records')
            
            results = {}
            
            for strategy in strategies:
                # 获取该策略的参数网格
                param_grid = AutoParamOptimizer.get_param_grid(strategy)
                
                best_result = None
                best_params = None
                best_score = -999
                
                # 遍历参数组合
                for params in param_grid:
                    config = {**params}
                    
                    # 执行策略
                    signals = StrategyEngine.execute_strategy(records, strategy, config, stock_features)
                    
                    if signals:
                        result = StrategyEngine.calculate_backtest(records, signals, 1000000)
                        score = result.get('score', 0)
                        
                        if score > best_score:
                            best_score = score
                            best_result = result
                            best_params = params
                
                if best_params:
                    results[strategy] = {
                        'best_params': best_params,
                        'best_result': {
                            'totalReturn': best_result.get('totalReturn', 0),
                            'maxDrawdown': best_result.get('maxDrawdown', 0),
                            'sharpe': best_result.get('sharpe', 0),
                            'winRate': best_result.get('winRate', 0),
                            'score': best_score,
                            'tradeCount': best_result.get('tradeCount', 0)
                        }
                    }
            
            return {
                'ts_code': ts_code,
                'industry': industry,
                'features': stock_features,
                'strategy_results': results
            }
            
        except Exception as e:
            print(f"优化股票参数失败 {ts_code}: {e}")
            return None
    
    @staticmethod
    def get_param_grid(strategy):
        """获取策略的参数网格（扩展版以提高效果）"""
        base_params = CrossStockParamOptimizer.UNIVERSAL_PARAMS.get(strategy, {})

        # 定义参数搜索范围（扩大范围以提高效果）
        param_ranges = {
            'takeProfit': [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
            'stopLoss': [0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15],
            'volumeMulti': [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        }

        # 生成参数组合
        import itertools
        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]

        grids = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            # 添加策略特有参数默认值
            params.update(base_params)
            grids.append(params)

        return grids
    
    @staticmethod
    def build_param_system(optimization_results):
        """根据优化结果构建参数体系"""
        industry_params = {}
        
        # 按行业归类
        for result in optimization_results:
            industry = result['industry']
            features = result['features']
            strategy_results = result['strategy_results']
            
            if industry not in industry_params:
                industry_params[industry] = {
                    'stocks': [],
                    'strategies': {}
                }
            
            industry_params[industry]['stocks'].append({
                'ts_code': result['ts_code'],
                'features': features
            })
            
            # 累加策略参数
            for strategy, data in strategy_results.items():
                if strategy not in industry_params[industry]['strategies']:
                    industry_params[industry]['strategies'][strategy] = {
                        'params_list': [],
                        'scores': []
                    }
                
                industry_params[industry]['strategies'][strategy]['params_list'].append(
                    data['best_params']
                )
                industry_params[industry]['strategies'][strategy]['scores'].append(
                    data['best_result']['score']
                )
        
        # 选择每个策略的最优参数（得分最高的那组）
        param_system = {}
        for industry, data in industry_params.items():
            param_system[industry] = {}
            
            for strategy, strategy_data in data['strategies'].items():
                params_list = strategy_data['params_list']
                scores = strategy_data['scores']
                
                if not params_list:
                    continue
                
                # 找到得分最高的参数
                max_score_idx = scores.index(max(scores))
                best_params = params_list[max_score_idx]
                
                param_system[industry][strategy] = best_params
        
        return param_system


# 全局进度存储
auto_optimize_progress = {
    'is_running': False,
    'current': 0,
    'total': 0,
    'current_stock': '',
    'results': []
}


@app.route('/api/auto_optimize_params', methods=['POST'])
def api_auto_optimize_params():
    """自动批量参数优化API - 带进度反馈"""
    global auto_optimize_progress

    payload = request.get_json(silent=True) or {}

    start_date = payload.get('start_date', '2023-01-01')
    end_date = payload.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    # 默认优化所有策略
    all_strategies = list(StrategyEngine.STRATEGIES.keys())
    strategies = payload.get('strategies', all_strategies)

    # 获取代表性股票（8个行业）
    stocks = AutoParamOptimizer.get_industry_stocks()

    # 初始化进度
    auto_optimize_progress = {
        'is_running': True,
        'current': 0,
        'total': len(stocks),
        'current_stock': '',
        'results': []
    }

    print(f"开始自动批量参数优化，股票数量: {len(stocks)}, 策略: {strategies}")

    optimization_results = []

    for i, stock in enumerate(stocks):
        # 更新进度
        auto_optimize_progress['current'] = i + 1
        auto_optimize_progress['current_stock'] = stock['ts_code']

        print(f"优化进度: {i+1}/{len(stocks)} - {stock['ts_code']} ({stock['industry']})")

        try:
            # 调用真实的参数优化方法
            result = AutoParamOptimizer.optimize_stock_params(
                stock['ts_code'],
                stock['industry'],
                start_date,
                end_date,
                strategies
            )

            if result:
                optimization_results.append(result)
                auto_optimize_progress['results'].append(result)
                print(f"  ✓ {stock['ts_code']} 优化成功")
            else:
                print(f"  ✗ {stock['ts_code']} 优化失败（无结果）")

        except Exception as e:
            print(f"  ✗ {stock['ts_code']} 优化异常: {e}")
            continue

    print(f"优化完成，成功: {len(optimization_results)}/{len(stocks)} 只股票")

    # 构建参数体系
    param_system = AutoParamOptimizer.build_param_system(optimization_results)

    # 转换为前端需要的格式：{strategy: {bestParams: {...}}}
    strategy_params = {}
    for industry, strategies in param_system.items():
        for strategy, params in strategies.items():
            if strategy not in strategy_params:
                strategy_params[strategy] = {
                    'bestParams': params,
                    'industries': {}
                }
            # 保存每个行业的参数
            strategy_params[strategy]['industries'][industry] = params

    # 标记完成
    auto_optimize_progress['is_running'] = False

    return jsonify({
        'success': True,
        'message': f'已完成{len(optimization_results)}只股票的参数优化',
        'optimized_stocks': [r['ts_code'] for r in optimization_results],
        'industries': list(param_system.keys()),
        'param_system': param_system,
        'strategy_params': strategy_params  # 前端可以直接使用的格式
    })


@app.route('/api/auto_optimize_progress', methods=['GET'])
def api_auto_optimize_progress():
    """获取自动批量参数优化进度"""
    return jsonify({
        'success': True,
        'is_running': auto_optimize_progress['is_running'],
        'current': auto_optimize_progress['current'],
        'total': auto_optimize_progress['total'],
        'current_stock': auto_optimize_progress['current_stock'],
        'progress_percent': round(auto_optimize_progress['current'] / auto_optimize_progress['total'] * 100, 1) if auto_optimize_progress['total'] > 0 else 0
    })


# 配置文件路径
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'user_config.json')
_CONFIG_LOCK = threading.Lock()

def ensure_config_dir():
    """确保配置目录存在"""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

def _atomic_write_json(file_path, data_obj):
    ensure_config_dir()
    dir_path = os.path.dirname(file_path)
    prefix = os.path.basename(file_path) + '.'
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix='.tmp', dir=dir_path)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data_obj, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, file_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@app.route('/api/config', methods=['GET'])
def api_get_config():
    """从文件加载用户配置"""
    try:
        ensure_config_dir()
        if os.path.exists(CONFIG_FILE):
            with _CONFIG_LOCK:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            print(f"[获取配置] 从文件加载: {CONFIG_FILE}, keys: {list(config.keys())}")
            resp = jsonify({'success': True, 'config': config})
            resp.headers['Cache-Control'] = 'no-store'
            return resp
        else:
            print(f"[获取配置] 配置文件不存在: {CONFIG_FILE}")
            resp = jsonify({'success': True, 'config': {}})
            resp.headers['Cache-Control'] = 'no-store'
            return resp
    except Exception as e:
        print(f"[获取配置] 错误: {e}")
        resp = jsonify({'success': False, 'message': str(e)})
        resp.headers['Cache-Control'] = 'no-store'
        return resp

@app.route('/api/config', methods=['POST'])
def api_save_config():
    """保存用户配置到文件"""
    try:
        ensure_config_dir()
        config = request.get_json(silent=True) or {}
        with _CONFIG_LOCK:
            _atomic_write_json(CONFIG_FILE, config)
        resp = jsonify({'success': True, 'message': '配置已保存'})
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        resp = jsonify({'success': False, 'message': str(e)})
        resp.headers['Cache-Control'] = 'no-store'
        return resp

@app.route('/api/config/export', methods=['GET'])
def api_export_config():
    """导出配置文件"""
    try:
        if os.path.exists(CONFIG_FILE):
            with _CONFIG_LOCK:
                raw = None
                try:
                    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                except json.JSONDecodeError:
                    with open(CONFIG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                        raw = f.read()
                    try:
                        decoder = json.JSONDecoder()
                        config, _ = decoder.raw_decode(raw.lstrip())
                    except Exception:
                        config = None

            if isinstance(config, dict):
                ui_key = 'quant_ui_state_v1'
                ui = config.get(ui_key)
                if isinstance(ui, dict):
                    ui = dict(ui)
                    ui.pop('stockCode', None)
                    ui.pop('startDate', None)
                    ui.pop('endDate', None)
                    config = dict(config)
                    config[ui_key] = ui

            if config is None:
                payload_text = raw if isinstance(raw, str) else ''
                payload = payload_text.encode('utf-8', errors='replace')
            else:
                payload = json.dumps(config, ensure_ascii=False, indent=2).encode('utf-8')
            resp = send_file(
                io.BytesIO(payload),
                mimetype='application/json',
                as_attachment=True,
                download_name='quant_config_backup.json'
            )
            resp.headers['Cache-Control'] = 'no-store'
            return resp
        else:
            resp = jsonify({'success': False, 'message': '配置文件不存在'})
            resp.headers['Cache-Control'] = 'no-store'
            return resp
    except Exception as e:
        resp = jsonify({'success': False, 'message': str(e)})
        resp.headers['Cache-Control'] = 'no-store'
        return resp

@app.route('/api/config/import', methods=['POST'])
def api_import_config():
    """导入配置文件"""
    try:
        if 'file' not in request.files:
            resp = jsonify({'success': False, 'message': '未上传文件'})
            resp.headers['Cache-Control'] = 'no-store'
            return resp
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'success': False, 'message': '文件名为空'})
            resp.headers['Cache-Control'] = 'no-store'
            return resp
        
        config = json.load(file)
        print(f"[导入配置] 读取到配置: {list(config.keys())}")
        
        ensure_config_dir()
        with _CONFIG_LOCK:
            _atomic_write_json(CONFIG_FILE, config)
        print(f"[导入配置] 配置已保存到: {CONFIG_FILE}")
        resp = jsonify({'success': True, 'message': '配置已导入'})
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        print(f"[导入配置] 错误: {e}")
        resp = jsonify({'success': False, 'message': str(e)})
        resp.headers['Cache-Control'] = 'no-store'
        return resp


@app.route('/api/strategy_backtest', methods=['POST'])
def api_strategy_backtest():
    """策略回测API - 支持LSTM和Ensemble策略"""
    try:
        payload = request.get_json(silent=True) or {}
        
        ts_code = payload.get('ts_code', '000001.SZ')
        start_date = payload.get('start_date', '2023-01-01')
        end_date = payload.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        strategy = payload.get('strategy', 'xgboost_ml')
        config = payload.get('config', {})
        init_capital = float(payload.get('init_capital', 1000000))
        
        import sys
        print(f"[策略回测] ts_code={ts_code}, strategy={strategy}, start={start_date}, end={end_date}", flush=True)
        print(f"[策略回测] config={config}", flush=True)
        print(f"[策略回测] TENSORFLOW_AVAILABLE={TENSORFLOW_AVAILABLE}", flush=True)
        sys.stdout.flush()
        
        # 获取股票数据
        data = cache_manager.get_stock_data(ts_code, start_date, end_date) or []
        if not data or len(data) < 80:
            print(f"[策略回测] 数据不足: {len(data)} 条")
            return jsonify({
                'success': False,
                'message': f'数据不足，需要至少80条数据，当前只有{len(data)}条'
            })
        
        print(f"[策略回测] 数据量: {len(data)} 条", flush=True)
        sys.stdout.flush()
        
        df = pd.DataFrame(data)
        df = calculate_indicators(df)
        records = df.where(pd.notnull(df), None).to_dict('records')
        
        if not records:
            return jsonify({
                'success': False,
                'message': '数据处理失败'
            })
        
        # 执行策略
        print(f"[策略回测] 开始执行策略: {strategy}", flush=True)
        sys.stdout.flush()
        signals = StrategyEngine.execute_strategy(records, strategy, config)
        print(f"[策略回测] 策略执行完成: {strategy}, 信号数={len(signals)}", flush=True)
        sys.stdout.flush()
        
        if not signals:
            return jsonify({
                'success': True,
                'strategy': strategy,
                'ts_code': ts_code,
                'signals': [],
                'result': {
                    'totalReturn': 0,
                    'annualReturn': 0,
                    'maxDrawdown': 0,
                    'sharpeRatio': 0,
                    'calmarRatio': 0,
                    'winRate': 0,
                    'profitLossRatio': 0,
                    'tradeCount': 0,
                    'score': 0,
                    'equity': [init_capital],
                    'initCapital': init_capital
                }
            })
        
        # 计算回测结果
        result = StrategyEngine.calculate_backtest(records, signals, init_capital)
        
        # 确保所有字段都有值
        result = {
            'totalReturn': result.get('totalReturn', 0) or 0,
            'annualReturn': result.get('annualReturn', 0) or 0,
            'maxDrawdown': result.get('maxDrawdown', 0) or 0,
            'sharpeRatio': result.get('sharpeRatio', 0) or 0,
            'calmarRatio': result.get('calmarRatio', 0) or 0,
            'winRate': result.get('winRate', 0) or 0,
            'profitRatio': result.get('profitLossRatio', 0) or 0,
            'tradeCount': result.get('tradeCount', 0) or 0,
            'score': result.get('score', 0) or 0
        }
        
        print(f"[策略回测] 完成: 信号数={len(signals)}, 收益={result.get('totalReturn', 0):.2f}%")
        
        return jsonify({
            'success': True,
            'strategy': strategy,
            'ts_code': ts_code,
            'signals': signals,
            'result': result
        })
    except Exception as e:
        print(f"[策略回测] 错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        })


if __name__ == '__main__':
    print("="*50)
    print("Tushare量化交易系统后端启动")
    print("请确保已设置正确的Tushare Token")
    print("访问: http://localhost:5000")
    print("="*50)
    debug = (os.getenv('FLASK_DEBUG') or '0').strip() == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug, use_reloader=False)
