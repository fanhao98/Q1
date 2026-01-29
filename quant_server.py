
"""
Tushare量化交易系统后端
需要安装: pip install tushare flask flask-cors pandas numpy
"""

import tushare as ts
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import os
import re
import sys
import threading
import urllib.request
import urllib.parse
import urllib.error

app = Flask(__name__)
CORS(app)

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


LOCAL_STATE_DIR = os.path.join(DATA_ROOT, 'local_state')
try:
    os.makedirs(LOCAL_STATE_DIR, exist_ok=True)
except Exception:
    pass


_LOCAL_STATE_LOCK = threading.Lock()
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
    payload = {'saved_at': datetime.now().isoformat(), 'key': key, 'value': value}
    tmp_path = path + ".tmp"
    try:
        with _LOCAL_STATE_LOCK:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, path)
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

def _load_tushare_token():
    env_token = os.getenv('TUSHARE_TOKEN')
    if env_token:
        return env_token.strip()
    token_file = os.path.join(DATA_ROOT, 'tushare_token.txt')
    try:
        if os.path.exists(token_file):
            with open(token_file, 'r', encoding='utf-8') as f:
                return (f.read() or '').strip()
    except Exception:
        return ''
    return ''


TUSHARE_TOKEN = _load_tushare_token()

print("正在初始化Tushare...")
try:
    if not TUSHARE_TOKEN:
        raise RuntimeError("未配置Tushare Token，请设置环境变量TUSHARE_TOKEN或在数据目录放置tushare_token.txt")
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    print("✓ Tushare初始化成功")
except Exception as e:
    print(f"✗ Tushare初始化失败: {e}")
    pro = None


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
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
            except Exception as e:
                print(f"创建数据目录失败: {e}")
    
    def get_stock_data(self, ts_code, start_date, end_date):
        """获取股票数据（优先读取本地CSV，支持增量更新）"""
        file_path = os.path.join(self.data_dir, f"{ts_code}.csv")
        local_df = None
        
        # 1. 尝试读取本地文件
        if os.path.exists(file_path):
            try:
                # 指定dtype以防止数据类型错误
                local_df = pd.read_csv(file_path, dtype={'date': str})
                if 'date' in local_df.columns:
                    local_df = local_df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
                else:
                    print(f"[数据损坏] {file_path} 缺少date列")
                    local_df = None
            except Exception as e:
                print(f"[读取缓存失败] {file_path}: {e}")
                local_df = None
        
        # 2. 检查并补充数据
        if local_df is not None and not local_df.empty:
            local_min = local_df['date'].min()
            local_max = local_df['date'].max()
            data_changed = False
            
            # A. 向前补充（请求开始时间早于本地最早时间）
            if start_date < local_min:
                pre_end_date = (datetime.strptime(local_min, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
                if start_date <= pre_end_date:
                    print(f"[向前补充] {ts_code}: {start_date} ~ {pre_end_date}")
                    pre_data = TushareDataFetcher.get_stock_data(ts_code, start_date, pre_end_date)
                    if pre_data:
                        pre_df = pd.DataFrame(pre_data)
                        local_df = pd.concat([pre_df, local_df]).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
                        data_changed = True
            
            # B. 向后补充（请求结束时间晚于本地最新时间）
            if end_date > local_max:
                inc_start_date = (datetime.strptime(local_max, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                if inc_start_date <= end_date:
                    print(f"[增量更新] {ts_code}: {inc_start_date} ~ {end_date}")
                    inc_data = TushareDataFetcher.get_stock_data(ts_code, inc_start_date, end_date)
                    if inc_data:
                        inc_df = pd.DataFrame(inc_data)
                        local_df = pd.concat([local_df, inc_df]).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
                        data_changed = True
            
            # 如果有更新，保存回文件
            if data_changed:
                try:
                    local_df.to_csv(file_path, index=False)
                    print(f"[持久化] 已更新 {file_path}, 条数: {len(local_df)}")
                except Exception as e:
                    print(f"[保存失败] {e}")

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
            except Exception as e:
                print(f"补齐pe/pb失败: {e}")

        else:
            # 3. 本地无数据，全量获取
            print(f"[本地无缓存] 全量获取 {ts_code}: {start_date} ~ {end_date}")
            data = TushareDataFetcher.get_stock_data(ts_code, start_date, end_date)
            if data:
                local_df = pd.DataFrame(data)
                try:
                    local_df.to_csv(file_path, index=False)
                    print(f"[持久化] 已保存 {file_path}, 条数: {len(local_df)}")
                except Exception as e:
                    print(f"[保存失败] {e}")
            else:
                # 尝试备用数据源
                try:
                    print("尝试使用备用数据源(Eastmoney)...")
                    data = EastmoneyDataFetcher.get_stock_data(ts_code, start_date, end_date)
                    if data:
                        local_df = pd.DataFrame(data)
                        local_df.to_csv(file_path, index=False)
                        print(f"[持久化] (备用源) 已保存 {file_path}")
                except Exception as e:
                    print(f"备用源获取失败: {e}")

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
    def get_stock_data(ts_code, start_date, end_date):
        """获取股票日线数据"""
        if pro is None:
            print("✗ Tushare API未初始化")
            return None
            
        try:
            print(f"正在获取 {ts_code} 从 {start_date} 到 {end_date} 的数据...")
            
            # 获取日线行情
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            
            print(f"获取到原始数据: {len(df) if df is not None else 'None'} 条")
            
            if df is None or df.empty:
                print("数据为空或None")
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
                print(f"获取换手率数据失败（可能需要更高权限）: {e}")
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
            print("✗ Tushare API未初始化，使用默认股票列表")
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


class StrategyEngine:
    """策略引擎"""
    
    STRATEGIES = {
        'ml_ensemble': {
            'name': 'ML集成策略',
            'icon': '🤖',
            'description': '融合多模型预测，通过加权投票决策'
        },
        'momentum_reversal': {
            'name': '动量反转策略',
            'icon': '⚡',
            'description': '捕捉超跌反弹和动量延续'
        },
        'trend_follow': {
            'name': '趋势跟踪策略',
            'icon': '📈',
            'description': '顺势而为，均线多头排列时入场'
        },
        'breakout': {
            'name': '突破追涨策略',
            'icon': '🔥',
            'description': '关键压力位突破时追涨'
        },
        'mean_reversion': {
            'name': '均值回归策略',
            'icon': '🎯',
            'description': '价格偏离均值过大时反向交易'
        },
        'multi_factor': {
            'name': '多因子策略',
            'icon': '📊',
            'description': '综合多因子打分'
        }
    }
    
    @staticmethod
    def execute_strategy(data, strategy_name, config):
        """执行策略"""
        df = pd.DataFrame(data)
        signals = []
        position = 0
        entry_price = 0
        entry_index = 0
        highest_since_entry = 0
        last_sell_index = -999  # 记录上一次卖出的索引
        
        buy_cooldown = int(config.get('buyCooldown', 1))  # 买入冷却天数，默认1天

        for i in range(60, len(df)):
            d = df.iloc[i]
            prev = df.iloc[i-1]
            
            if pd.isna(d.get('ma20')) or pd.isna(d.get('rsi')):
                continue
            
            buy_signal = False
            sell_signal = False
            signal_strength = 50
            sell_reason = ''
            
            # 冷却期检查：如果距离上次卖出不足冷却天数，则不进行买入判断
            # 例如：buy_cooldown=1，i=100卖出，i=101时 (101-100)=1 <= 1，跳过买入判断
            in_cooldown = (i - last_sell_index) <= buy_cooldown

            # 策略逻辑
            if in_cooldown:
                pass
            elif strategy_name == 'ml_ensemble':
                ml_score = 0
                if d['ma5'] and d['ma20'] and d['ma5'] > d['ma20']:
                    ml_score += 20
                if d['close'] > d.get('ma5', d['close']):
                    ml_score += 15
                if d['macd'] and d['macd'] > 0 and d['macd'] > prev.get('macd', 0):
                    ml_score += 20
                if d['rsi'] and d['rsi'] < 35:
                    ml_score += 15
                if d['rsi'] and d['rsi'] < 25:
                    ml_score += 10
                if d['volume'] and d.get('volMa5') and d['volume'] > d['volMa5'] * config['volumeMulti']:
                    ml_score += 15
                if d['close'] and d.get('bollDown') and d['close'] < d['bollDown']:
                    ml_score += 15
                
                signal_strength = ml_score
                if position == 0 and ml_score >= 60:
                    buy_signal = True
                    
            elif strategy_name == 'momentum_reversal':
                if position == 0:
                    is_near_low = d.get('low10') and d['low'] <= d['low10'] * 1.02
                    is_volume_up = d.get('volMa5') and d['volume'] > d['volMa5'] * config['volumeMulti']
                    is_rsi_oversold = d.get('rsi') and d['rsi'] < 30
                    is_bullish = d['close'] > d['open']
                    
                    if is_near_low and is_volume_up and (is_rsi_oversold or is_bullish):
                        buy_signal = True
                        signal_strength = 70
                        
            elif strategy_name == 'trend_follow':
                if position == 0:
                    ma5_above_ma20 = d.get('ma5') and d.get('ma20') and d['ma5'] > d['ma20']
                    price_above_ma5 = d.get('ma5') and d['close'] > d['ma5']
                    golden_cross = (prev.get('ma5') and prev.get('ma20') and d.get('ma5') and d.get('ma20') and
                                   prev['ma5'] <= prev['ma20'] and d['ma5'] > d['ma20'])
                    macd_positive = d.get('macd') and d['macd'] > 0
                    
                    if (golden_cross or (ma5_above_ma20 and price_above_ma5)) and macd_positive:
                        buy_signal = True
                        signal_strength = 75
                        
            elif strategy_name == 'breakout':
                if position == 0:
                    break_high = d.get('high20') and d['close'] > d['high20']
                    volume_confirm = d.get('volMa5') and d['volume'] > d['volMa5'] * 2
                    
                    if break_high and volume_confirm:
                        buy_signal = True
                        signal_strength = 80
                        
            elif strategy_name == 'mean_reversion':
                if position == 0:
                    touch_lower = d.get('bollDown') and d['close'] < d['bollDown']
                    rsi_oversold = d.get('rsi') and d['rsi'] < 25
                    
                    if touch_lower and rsi_oversold:
                        buy_signal = True
                        signal_strength = 70
                        
            elif strategy_name == 'multi_factor':
                if position == 0 and i >= 80:
                    # 动量因子
                    momentum = (d['close'] - df.iloc[i-20]['close']) / df.iloc[i-20]['close']
                    momentum_score = min(100, max(0, (momentum + 0.1) * 500))
                    
                    # 价值因子
                    value_ratio = d['close'] / d.get('ma60', d['close']) if d.get('ma60') else 1
                    value_score = 80 if value_ratio < 0.95 else (60 if value_ratio < 1 else 40)
                    
                    # 质量因子
                    quality_score = min(100, 100 - (d.get('atr', 0) / d['close']) * 1000) if d.get('atr') else 50
                    
                    signal_strength = momentum_score * 0.3 + value_score * 0.3 + quality_score * 0.4
                    
                    if signal_strength >= 65:
                        buy_signal = True

            elif strategy_name == 'momentum_rotation':
                if position == 0:
                    momentum_period = int(config.get('momentumPeriod', 20))
                    momentum_threshold = float(config.get('momentumThreshold', 0.02))
                    volume_multi = float(config.get('volumeMulti', 1.5))
                    
                    if i >= momentum_period:
                        momentum = (d['close'] - df.iloc[i-momentum_period]['close']) / df.iloc[i-momentum_period]['close']
                        volume_condition = d.get('volMa5') and d['volume'] > d['volMa5'] * volume_multi
                        
                        if momentum > momentum_threshold and volume_condition:
                            signal_strength = min(100, momentum * 100 + 50)
                            buy_signal = True
            
            # 卖出逻辑
            if position > 0:
                highest_since_entry = max(highest_since_entry, d['high'])
                hold_days = i - entry_index
                profit = (d['close'] - entry_price) / entry_price
                
                if profit <= -config['stopLoss']:
                    sell_signal = True
                    sell_reason = '止损'
                elif profit >= config['takeProfit']:
                    sell_signal = True
                    sell_reason = '止盈'
                elif config.get('useMa5Sell') and d.get('ma5') and d['close'] < d['ma5'] and profit > 0.02:
                    sell_signal = True
                    sell_reason = '跌破MA5'
                elif config.get('useDynamicTP') and profit > 0.08:
                    trail = highest_since_entry * (1 - 0.03 - profit * 0.15)
                    if d['close'] < trail:
                        sell_signal = True
                        sell_reason = '移动止盈'
                elif d.get('macd') and prev.get('macd') and d['macd'] < 0 and prev['macd'] > 0 and profit > 0:
                    sell_signal = True
                    sell_reason = 'MACD死叉'
                elif d.get('rsi') and d['rsi'] > 80 and profit > 0.05:
                    sell_signal = True
                    sell_reason = 'RSI超买'
                elif hold_days > 30 and profit < 0.05:
                    sell_signal = True
                    sell_reason = '持仓超时'
            
            # 记录信号
            if buy_signal and position == 0:
                position = 0.3 + (signal_strength / 100) * 0.7 if config.get('useAdaptPosition') else 1
                entry_price = d['close']
                entry_index = i
                highest_since_entry = d['high']
                
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
    
    # 高低点
    df['high10'] = df['high'].rolling(window=10).max()
    df['low10'] = df['low'].rolling(window=10).min()
    df['high20'] = df['high'].rolling(window=20).max()
    df['low20'] = df['low'].rolling(window=20).min()
    
    return df


def identify_market_state(df):
    """识别市场状态"""
    if df is None or len(df) < 20:
        return 'unknown'
    
    # 转换为DataFrame（如果是字典列表）
    if isinstance(df, list):
        df = pd.DataFrame(df)
    
    latest = df.iloc[-1]
    
    # 趋势判断
    if latest['ma5'] > latest['ma20'] and latest['ma20'] > latest['ma60']:
        trend = 'uptrend'
    elif latest['ma5'] < latest['ma20'] and latest['ma20'] < latest['ma60']:
        trend = 'downtrend'
    else:
        trend = 'sideways'
    
    # 波动性判断
    volatility = df['close'].pct_change().std() * 100
    if volatility > 3:
        volatility_state = 'high'
    elif volatility > 1.5:
        volatility_state = 'medium'
    else:
        volatility_state = 'low'
    
    # 综合状态
    if trend == 'uptrend' and volatility_state == 'medium':
        return 'bull_market'
    elif trend == 'downtrend' and volatility_state == 'high':
        return 'bear_market'
    elif trend == 'sideways':
        return 'sideways_market'
    else:
        return f'{trend}_{volatility_state}'





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


@app.route('/api/stock_data', methods=['GET'])
def api_get_stock_data():
    """获取股票数据（带缓存）"""
    ts_code = request.args.get('ts_code', '000001.SZ')
    start_date = request.args.get('start_date', '2022-01-01')
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    print(f"API请求: ts_code={ts_code}, start_date={start_date}, end_date={end_date}")
    
    # 使用缓存管理器获取数据
    data = cache_manager.get_stock_data(ts_code, start_date, end_date)
    print(f"从缓存管理器获取数据结果: {type(data)}, 长度: {len(data) if data else 'None'}")
    
    if data is None or (isinstance(data, list) and len(data) == 0):
        # 如果无法获取真实数据，生成模拟数据
        print(f"真实数据获取失败，生成模拟数据: {ts_code}")
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

    # 计算指标
    df = calculate_indicators(df)
    
    # 识别市场状态
    market_state = identify_market_state(df)
    
    # 计算筹码分布
    chip_data = calculate_chip_distribution(df)
    
    # 转换为JSON
    data = df.to_dict(orient='records')
    for row in data:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
    
    return jsonify({
        'success': True,
        'message': 'ok',
        'ts_code': ts_code,
        'market_state': market_state,
        'data_count': len(data),
        'data': data,
        'chip_distribution': chip_data,
        'cached': True
    })


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
    
    if pro is None:
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
        df = pro.stock_basic(
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

    if pro is None:
        return jsonify({'success': True, 'count': 0, 'data': [], 'cached': False, 'message': '未配置Tushare Token，无法拉取概念列表。'})

    try:
        df = None
        try:
            df = pro.concept(src='ts')
        except Exception:
            df = pro.concept()
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
        return jsonify({'success': True, 'count': 0, 'data': [], 'cached': False, 'message': '概念列表获取失败'})


@app.route('/api/concept_members', methods=['GET'])
def api_get_concept_members():
    codes_raw = request.args.get('codes', '') or ''
    codes = [c.strip() for c in codes_raw.split(',') if c.strip()]
    if not codes:
        return jsonify({'success': False, 'message': 'codes 不能为空', 'ts_codes': [], 'count': 0}), 400

    if pro is None:
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
                df = pro.concept_detail(id=concept_code)
            except Exception:
                df = pro.concept_detail(concept_id=concept_code)
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


if __name__ == '__main__':
    print("="*50)
    print("Tushare量化交易系统后端启动")
    print("请确保已设置正确的Tushare Token")
    print("访问: http://localhost:5000")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=True)
