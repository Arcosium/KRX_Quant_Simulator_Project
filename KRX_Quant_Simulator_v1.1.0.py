import sys
import os
import re
import time
import pandas as pd
import numpy as np
import requests
import sqlite3
import ctypes
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QTableWidget, QTableWidgetItem, QMessageBox, 
                               QHeaderView, QProgressBar, QGridLayout, QScrollArea,
                               QTabWidget, QSplitter, QTextEdit, QFrame, QGroupBox, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal, QEvent
from PySide6.QtGui import QFont, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

YEAR_MAPPING = {
    2024: 'D-1y_data',
    2023: 'D-2y_data',
    2022: 'D-3y_data',
    2021: 'D-4y_data',
    2020: 'D-5y_data'
}

METRICS_LIST = [
    'ROE', 'ROA', 'GPM', 'OPM', 'NPM', 'EBITDA',
    '부채비율', '유동비율', '당좌비율', '자기자본비율',
    '매출액증가율', '영업이익증가율', 'EPS증가율', 'BPS증가율',
    '재고자산회전율', '총자산회전율',
    'PER', 'PBR', 'PSR', 'PCR', 'EV/EBITDA', 'GP/A',
    'OCF', 'FCF'
]

METRIC_UNITS = {
    'ROE': '배', 'ROA': '배', 'GPM': '배', 'OPM': '배', 'NPM': '배',
    '부채비율': '%', '유동비율': '%', '당좌비율': '%', '자기자본비율': '%',
    '매출액증가율': '%', '영업이익증가율': '%', 'EPS증가율': '%', 'BPS증가율': '%',
    '재고자산회전율': '회', '총자산회전율': '회',
    'PER': '배', 'PBR': '배', 'PSR': '배', 'PCR': '배', 'EV/EBITDA': '배', 'GP/A': '배',
    'EBITDA': '억원', 'OCF': '억원', 'FCF': '억원'
}

TECH_INDICATORS_MAP = {
    '골든크로스': 'GC_5_20', '데드크로스': 'DC_5_20',
    '5일선': 'SMA_5', '20일선': 'SMA_20', '60일선': 'SMA_60', 
    '1년선': 'SMA_240', '3년선': 'SMA_720',
    '거래량': 'Volume', '종가': 'Close', '시가': 'Open', '고가': 'High', '저가': 'Low',
    '거래대금': 'TradingValue',
    'RSI': 'RSI', 'MACD_SIGNAL': 'MACD_Signal', 'MACD': 'MACD',
    'STOCH_K': 'Stoch_K', 'STOCH_D': 'Stoch_D',
    'CCI': 'CCI', 'ADX': 'ADX', 'OBV': 'OBV', 'MFI': 'MFI', 'ATR': 'ATR',
    '기관순매수': 'Inst_Net_Volume', '외국인순매수': 'Foreign_Net_Volume'
}

TECH_UNITS = {
    '골든크로스': 'T/F', '데드크로스': 'T/F',
    '5일선': '원', '20일선': '원', '60일선': '원', '1년선': '원', '3년선': '원',
    '거래량': '주', '종가': '원', '시가': '원', '고가': '원', '저가': '원',
    '거래대금': '억원',
    'RSI': '%', 'MACD': '값', 'MACD_SIGNAL': '값',
    'STOCH_K': '%', 'STOCH_D': '%',
    'CCI': '값', 'ADX': '값', 'OBV': '값', 'MFI': '값', 'ATR': '원',
    '기관순매수': '주', '외국인순매수': '주'
}

class DBManager:
    def __init__(self):
        self.dir_name = "Memories"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        
        self.db_path = os.path.join(self.dir_name, "stock_cache.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.create_tables()
        self.hide_db_file()

    def hide_db_file(self):
        if os.name == 'nt':
            try:
                FILE_ATTRIBUTE_HIDDEN = 0x02
                ret = ctypes.windll.kernel32.SetFileAttributesW(self.db_path, FILE_ATTRIBUTE_HIDDEN)
            except Exception:
                pass

    def create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    code TEXT,
                    date TEXT,
                    open INTEGER, high INTEGER, low INTEGER, close INTEGER, volume INTEGER,
                    PRIMARY KEY (code, date)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS investor_data (
                    code TEXT,
                    date TEXT,
                    inst_net INTEGER, foreign_net INTEGER,
                    PRIMARY KEY (code, date)
                )
            """)

    def get_latest_date(self, code, table_name):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT MAX(date) FROM {table_name} WHERE code=?", (code,))
        res = cursor.fetchone()
        return res[0] if res and res[0] else None

    def save_price_data(self, code, df):
        if df.empty: return
        data = []
        for index, row in df.iterrows():
            date_str = index.strftime('%Y-%m-%d') if isinstance(index, pd.Timestamp) else str(index)[:10]
            data.append((
                code, date_str, 
                int(row['Open']), int(row['High']), int(row['Low']), int(row['Close']), int(row['Volume'])
            ))
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO price_data (code, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                data
            )

    def save_investor_data(self, code, df):
        if df.empty: return
        data = []
        for index, row in df.iterrows():
            date_str = index.strftime('%Y-%m-%d') if isinstance(index, pd.Timestamp) else str(index)[:10]
            data.append((
                code, date_str,
                int(row['Inst_Net_Volume']), int(row['Foreign_Net_Volume'])
            ))
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO investor_data (code, date, inst_net, foreign_net) VALUES (?, ?, ?, ?)",
                data
            )

    def load_price_data(self, code):
        query = "SELECT date, open, high, low, close, volume FROM price_data WHERE code=? ORDER BY date ASC"
        df = pd.read_sql(query, self.conn, params=(code,))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df

    def load_investor_data(self, code):
        query = "SELECT date, inst_net, foreign_net FROM investor_data WHERE code=? ORDER BY date ASC"
        df = pd.read_sql(query, self.conn, params=(code,))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df.columns = ['Inst_Net_Volume', 'Foreign_Net_Volume']
        return df

class CrawlerUtil:
    @staticmethod
    def fetch_kospi_index(years=10):
        result = []
        max_pages = int(years * 26) + 10 
        if max_pages < 20: max_pages = 20
        target_date_limit = datetime.today() - timedelta(days=years*365)
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            search_pages = min(max_pages, 400)
            for page in range(1, search_pages + 1):
                url = f"https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI&page={page}"
                res = requests.get(url, headers=headers)
                soup = BeautifulSoup(res.text, 'lxml')
                rows = soup.select('table.type_1 tr')
                valid_rows = 0
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 2: continue
                    try:
                        date_text = cols[0].text.strip()
                        if not date_text or date_text == '.': continue 
                        date = pd.to_datetime(date_text)
                        if date < target_date_limit:
                            return pd.DataFrame(result).drop_duplicates(subset=['Date']).sort_values('Date').set_index('Date')
                        close_text = cols[1].text.strip().replace(',', '')
                        if not close_text: continue
                        close = float(close_text)
                        result.append({'Date': date, 'Close': close})
                        valid_rows += 1
                    except: continue
                if valid_rows == 0 and page > 1: break
        except Exception as e:
            print(f"KOSPI Crawling Error: {e}")
        if not result: return pd.DataFrame()
        return pd.DataFrame(result).drop_duplicates(subset=['Date']).sort_values('Date').set_index('Date')

    @staticmethod
    def fetch_naver_stock_html(code, years=10, stop_date=None):
        result = []
        max_pages = int(years * 26) + 10 
        target_date_limit = datetime.today() - timedelta(days=years*365)
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            search_pages = min(max_pages, 400) 
            for page in range(1, search_pages + 1):
                url = f"https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}"
                res = requests.get(url, headers=headers)
                soup = BeautifulSoup(res.text, 'lxml')
                rows = soup.select('table.type2 tr')
                valid_rows = 0
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) != 7: continue
                    try:
                        date_text = cols[0].text.strip()
                        if not date_text: continue
                        date = pd.to_datetime(date_text)
                        if stop_date and date <= stop_date:
                            return pd.DataFrame(result).drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
                        if date < target_date_limit:
                            return pd.DataFrame(result).drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
                        close = int(cols[1].text.replace(',', ''))
                        open_ = int(cols[3].text.replace(',', ''))
                        high = int(cols[4].text.replace(',', ''))
                        low = int(cols[5].text.replace(',', ''))
                        volume = int(cols[6].text.replace(',', ''))
                        result.append({'Date': date, 'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume})
                        valid_rows += 1
                    except: continue
                if valid_rows == 0 and page > 1: break
        except Exception as e:
            print(f"네이버 시세 크롤링 에러 ({code}): {e}")
        if not result: return pd.DataFrame()
        return pd.DataFrame(result).drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)

    @staticmethod
    def fetch_investor_data(code, years=1, stop_date=None):
        MAX_PAGES = int(years * 26) + 5
        TIMEOUT_LIMIT = 5
        url_base = f"https://finance.naver.com/item/frgn.nhn?code={code}&page="
        result = []
        start_date_dt = datetime.today() - timedelta(days=years*365)
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            for page in range(1, MAX_PAGES + 1):
                url = url_base + str(page)
                try:
                    res = requests.get(url, headers=headers, timeout=TIMEOUT_LIMIT)
                    soup = BeautifulSoup(res.text, 'lxml')
                    rows = soup.select('table.type2 tr')
                    valid_on_page = 0
                    for row in rows:
                        cols = row.find_all('td')
                        if not cols or not cols[0].text.strip(): continue
                        if len(cols) < 7: continue
                        try:
                            date_str = cols[0].text.strip().replace('.', '-')
                            date = pd.to_datetime(date_str)
                            if stop_date and date <= stop_date:
                                df = pd.DataFrame(result)
                                if df.empty: return df
                                return df.sort_values('Date').reset_index(drop=True)
                            if date < start_date_dt:
                                df = pd.DataFrame(result)
                                if df.empty: return df
                                return df.sort_values('Date').reset_index(drop=True)
                            inst_text = cols[5].text.strip()
                            fore_text = cols[6].text.strip()
                            if not inst_text or not fore_text: continue
                            inst_net_vol = int(inst_text.replace(',', '').replace('+', ''))
                            foreign_net_vol = int(fore_text.replace(',', '').replace('+', ''))
                            result.append({'Date': date, 'Inst_Net_Volume': inst_net_vol, 'Foreign_Net_Volume': foreign_net_vol})
                            valid_on_page += 1
                        except Exception: continue
                    if valid_on_page == 0 and page > 5: break
                    next_page = soup.select_one('td.pgR a')
                    if not next_page:
                         if page > 10: break
                except Exception as e:
                    print(f"수급 크롤링 페이지 에러: {e}")
                    continue
        except Exception as e:
            print(f"수급 데이터 전체 에러: {e}")
            return pd.DataFrame()
        df = pd.DataFrame(result)
        if df.empty: return df
        return df.sort_values('Date').reset_index(drop=True)

class TechnicalAnalysis:
    @staticmethod
    def add_indicators(df):
        if df.empty: return df
        df['TradingValue'] = (df['Close'] * df['Volume']) / 100000000
        for window in [5, 20, 60, 120, 240, 720]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        denom = (high_max - low_min).replace(0, np.nan)
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / denom)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        def calculate_mad(x): return np.mean(np.abs(x - np.mean(x)))
        mad = tp.rolling(window=20).apply(calculate_mad, raw=True).replace(0, np.nan)
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        prev_sma5 = df['SMA_5'].shift(1)
        prev_sma20 = df['SMA_20'].shift(1)
        df['GC_5_20'] = (prev_sma5 < prev_sma20) & (df['SMA_5'] > df['SMA_20'])
        df['DC_5_20'] = (prev_sma5 > prev_sma20) & (df['SMA_5'] < df['SMA_20'])
        return df

class BacktestWorker(QThread):
    progress_signal = Signal(int, int, str)
    log_signal = Signal(str)
    result_signal = Signal(dict, pd.DataFrame, dict, dict)
    error_signal = Signal(str)

    def __init__(self, target_companies, buy_logic, sell_logic, period_months):
        super().__init__()
        self.target_companies = target_companies
        self.buy_logic = buy_logic
        self.sell_logic = sell_logic
        self.period_months = period_months 
        self.is_running = True
        self.db_manager = DBManager()

    def run(self):
        fetch_years = (self.period_months / 12) + 2
        processed_data = {} 
        total_companies = len(self.target_companies)
        
        self.progress_signal.emit(2, 100, "KOSPI 지수 크롤링 중...")
        self.log_signal.emit(f"KOSPI 지수 데이터 수집 시작 (최근 {fetch_years:.1f}년)...")
        kospi_df = CrawlerUtil.fetch_kospi_index(years=fetch_years)
        if not kospi_df.empty:
            kospi_df = kospi_df[['Close']].rename(columns={'Close': 'KOSPI'})
        
        self.progress_signal.emit(5, 100, "개별 종목 데이터 수집 중 (캐시 확인)...")
        crawl_start_progress = 5
        crawl_end_progress = 90
        
        for idx, company in enumerate(self.target_companies):
            if not self.is_running: break
            code = str(company['종목코드']).zfill(6)
            name = company['기업명']
            current_prog = crawl_start_progress + int((idx / total_companies) * (crawl_end_progress - crawl_start_progress))
            self.progress_signal.emit(current_prog, 100, f"[{idx+1}/{total_companies}] {name} 데이터 확인 중...")
            
            latest_price_date_str = self.db_manager.get_latest_date(code, 'price_data')
            latest_price_date = pd.to_datetime(latest_price_date_str) if latest_price_date_str else None
            
            if latest_price_date is None or latest_price_date.date() < datetime.today().date():
                self.log_signal.emit(f"-> {name} 시세 업데이트 (Last: {latest_price_date_str or 'None'})...")
                new_price_df = CrawlerUtil.fetch_naver_stock_html(code, years=fetch_years, stop_date=latest_price_date)
                if not new_price_df.empty:
                    new_price_df.set_index('Date', inplace=True)
                    self.db_manager.save_price_data(code, new_price_df)
            
            price_df = self.db_manager.load_price_data(code)
            if price_df.empty:
                self.log_signal.emit(f"   (실패) {name}: 데이터 없음.")
                continue

            latest_inv_date_str = self.db_manager.get_latest_date(code, 'investor_data')
            latest_inv_date = pd.to_datetime(latest_inv_date_str) if latest_inv_date_str else None
            
            if latest_inv_date is None or latest_inv_date.date() < datetime.today().date():
                self.log_signal.emit(f"-> {name} 수급 업데이트...")
                new_inv_df = CrawlerUtil.fetch_investor_data(code, years=fetch_years, stop_date=latest_inv_date)
                if not new_inv_df.empty:
                    new_inv_df.set_index('Date', inplace=True)
                    self.db_manager.save_investor_data(code, new_inv_df)

            investor_df = self.db_manager.load_investor_data(code)
            
            if not investor_df.empty:
                df = price_df.join(investor_df, how='left')
                df['Inst_Net_Volume'] = df['Inst_Net_Volume'].fillna(0)
                df['Foreign_Net_Volume'] = df['Foreign_Net_Volume'].fillna(0)
            else:
                df = price_df
                df['Inst_Net_Volume'] = 0
                df['Foreign_Net_Volume'] = 0

            df = df.sort_index()
            df = TechnicalAnalysis.add_indicators(df)
            processed_data[code] = df
        
        if not processed_data:
            self.error_signal.emit("데이터 수집 실패: DB 및 크롤링 실패")
            return

        self.progress_signal.emit(90, 100, "전략 시뮬레이션 계산 중...")
        self.log_signal.emit("전략 백테스팅 시뮬레이션 시작...")
        
        all_dates_raw = sorted(list(set().union(*[df.index for df in processed_data.values()])))
        end_date = datetime.today()
        start_date = end_date - timedelta(days=self.period_months * 30)
        sim_dates = [d for d in all_dates_raw if start_date <= d <= end_date]
        
        if not sim_dates:
            self.error_signal.emit("선택한 기간에 해당하는 데이터가 충분하지 않습니다.")
            return

        portfolio_curve = [] 
        benchmark_curve = [] 
        
        initial_capital = 100000000
        cash = initial_capital
        holdings = {} 
        buy_prices = {}
        trade_count = 0
        win_count = 0
        
        bnh_holdings = {}
        bnh_cost_basis = 0
        FIXED_INVESTMENT_PER_STOCK = 10000000

        total_sim_days = len(sim_dates)
        for d_idx, date in enumerate(sim_dates):
            if not self.is_running: break
            
            calc_prog = 90 + int((d_idx / total_sim_days) * 10)
            if d_idx % 10 == 0: 
                self.progress_signal.emit(calc_prog, 100, f"시뮬레이션 중... ({date.strftime('%Y-%m-%d')})")
            for code in processed_data.keys():
                if code not in bnh_holdings:
                    df = processed_data[code]
                    if date in df.index:
                        price = df.loc[date, 'Close']
                        if price > 0:
                            qty = int(FIXED_INVESTMENT_PER_STOCK // price)
                            cost = int(qty * price)
                            if qty > 0:
                                bnh_holdings[code] = qty
                                bnh_cost_basis += cost
            bnh_market_value = 0
            for code, qty in bnh_holdings.items():
                if date in processed_data[code].index:
                    current_price = processed_data[code].loc[date, 'Close']
                else:
                    try:
                        current_price = processed_data[code][processed_data[code].index < date].iloc[-1]['Close']
                    except: 
                        current_price = 0
                
                bnh_market_value += qty * current_price

            if bnh_cost_basis > 0:
                bnh_return_rate = (bnh_market_value - bnh_cost_basis) / bnh_cost_basis
                display_bnh_value = int(initial_capital * (1 + bnh_return_rate))
            else:
                display_bnh_value = initial_capital

            benchmark_curve.append({'Date': date, 'BnH': display_bnh_value})
            current_pf_value = cash
            for code, qty in holdings.items():
                if date in processed_data[code].index:
                    price = processed_data[code].loc[date, 'Close']
                else:
                    try:
                        price = processed_data[code][processed_data[code].index < date].iloc[-1]['Close']
                    except: price = 0
                current_pf_value += qty * price
            
            portfolio_curve.append({'Date': date, 'Strategy': current_pf_value})

            for code in processed_data.keys():
                if date not in processed_data[code].index: continue
                row = processed_data[code].loc[date]
                
                try:
                    if code not in holdings:
                        if self.evaluate_logic(self.buy_logic, row):
                            invest_amount = int(current_pf_value * 0.1)
                            if cash >= invest_amount:
                                price = row['Close']
                                qty = int(invest_amount // price)
                                if qty > 0:
                                    holdings[code] = qty
                                    cash -= int(qty * price)
                                    buy_prices[code] = price 
                    elif code in holdings:
                        if self.evaluate_logic(self.sell_logic, row):
                            qty = holdings[code]
                            price = row['Close']
                            buy_price = buy_prices.get(code, price)
                            
                            cash += int(qty * price)
                            trade_count += 1
                            if price > buy_price: win_count += 1
                            
                            del holdings[code]
                            if code in buy_prices: del buy_prices[code]
                except: pass

        result_df = pd.DataFrame(portfolio_curve).set_index('Date')
        bnh_df = pd.DataFrame(benchmark_curve).set_index('Date')
        
        result_df = result_df[~result_df.index.duplicated(keep='last')]
        bnh_df = bnh_df[~bnh_df.index.duplicated(keep='last')]
        
        final_df = result_df.rename(columns={'Strategy': 'Strategy_Value'}).join(bnh_df.rename(columns={'BnH': 'BnH_Value'}), how='outer')
        final_df = final_df.ffill().fillna(initial_capital)
        
        final_df['Strategy'] = ((final_df['Strategy_Value'] - initial_capital) / initial_capital) * 100
        final_df['BnH'] = ((final_df['BnH_Value'] - initial_capital) / initial_capital) * 100
        
        if not kospi_df.empty:
            kospi_cut = kospi_df[kospi_df.index >= sim_dates[0]].copy()
            if not kospi_cut.empty:
                base_k = kospi_cut.iloc[0]['KOSPI']
                kospi_cut['KOSPI_Pct'] = ((kospi_cut['KOSPI'] - base_k) / base_k) * 100
                final_df = final_df.join(kospi_cut[['KOSPI_Pct']], how='left')
                final_df = final_df.rename(columns={'KOSPI_Pct': 'KOSPI'})
        
        final_df = final_df.ffill().fillna(0)
        detailed_stats = self.calculate_advanced_stats(final_df['Strategy_Value'], trade_count, win_count)

        self.progress_signal.emit(99, 100, "최근 신호 종목 검색 중...")
        recent_signals = {'buy': [], 'sell': []}
        
        for code, df in processed_data.items():
            if df.empty or len(df) < 5: continue
            comp_name = next((c['기업명'] for c in self.target_companies if str(c['종목코드']).zfill(6) == code), code)
            recent_df = df.iloc[-5:]
            for date, row in recent_df.iterrows():
                try:
                    if self.evaluate_logic(self.buy_logic, row):
                        recent_signals['buy'].append({'date': date.strftime('%Y-%m-%d'), 'code': code, 'name': comp_name, 'price': row['Close']})
                    if self.evaluate_logic(self.sell_logic, row):
                        recent_signals['sell'].append({'date': date.strftime('%Y-%m-%d'), 'code': code, 'name': comp_name, 'price': row['Close']})
                except: pass

        self.result_signal.emit(processed_data, final_df, detailed_stats, recent_signals)
        self.progress_signal.emit(100, 100, "완료")
        self.log_signal.emit("모든 프로세스 완료.")

    def calculate_advanced_stats(self, value_series, trades, wins):
        if value_series.empty: return {}
        start_val = value_series.iloc[0]
        end_val = value_series.iloc[-1]
        total_return = ((end_val - start_val) / start_val) * 100
        try:
            days = (value_series.index[-1] - value_series.index[0]).days
            years = days / 365.25
            if years > 0 and start_val > 0 and end_val > 0:
                cagr = ((end_val / start_val) ** (1/years) - 1) * 100
            else: cagr = 0
        except: cagr = 0
        rolling_max = value_series.cummax()
        drawdown = (value_series - rolling_max) / rolling_max
        mdd = drawdown.min() * 100 
        win_rate = (wins / trades * 100) if trades > 0 else 0
        return {
            "Total Return": f"{total_return:.2f}%",
            "CAGR": f"{cagr:.2f}%",
            "MDD": f"{mdd:.2f}%",
            "Trades": f"{trades}회",
            "Win Rate": f"{win_rate:.1f}%"
        }

    def evaluate_logic(self, logic_str, row):
        if not logic_str or not logic_str.strip(): return False
        expr = logic_str
        sorted_keys = sorted(TECH_INDICATORS_MAP.keys(), key=len, reverse=True)
        for ko in sorted_keys:
            if ko in expr:
                en = TECH_INDICATORS_MAP[ko]
                val = row.get(en, 0)
                expr = expr.replace(ko, str(val))
        expr = expr.replace(">=", "__GE__").replace("<=", "__LE__").replace("==", "__EQ__")
        expr = expr.replace("=", "==")
        expr = expr.replace("__GE__", ">=").replace("__LE__", "<=").replace("__EQ__", "==")
        expr = expr.replace("AND", "and").replace("OR", "or")
        expr = expr.replace("TRUE", "True").replace("FALSE", "False")
        try:
            return eval(expr)
        except: return False

class QuantApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KRX 퀀트 시뮬레이터 v.1.1.0")
        self.resize(900, 810)
        self.db_manager = DBManager()
        
        self.financial_data = {} 
        self.filtered_companies = []
        self.current_input_widget = None
        
        self.init_ui()
        self.load_csv_files()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10) 

        self.main_scroll = QScrollArea()
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setFrameShape(QFrame.NoFrame)
        
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)
        
        self.tabs = QTabWidget()
        self.content_layout.addWidget(self.tabs)
        
        self.main_scroll.setWidget(content_widget)
        main_layout.addWidget(self.main_scroll)

        tab1 = QWidget()
        self.tabs.addTab(tab1, "1. 재무제표 필터링")
        layout1 = QVBoxLayout(tab1)
        
        metric_group = QGroupBox("재무 지표 선택(클릭 시 입력)")
        metric_layout = QGridLayout(metric_group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(metric_group)
        scroll.setFixedHeight(270)
        
        row, col = 0, 0
        for metric in METRICS_LIST:
            unit = METRIC_UNITS.get(metric, '')
            btn_text = f"{metric} ({unit})" if unit else metric
            
            btn = QPushButton(btn_text)
            btn.setStyleSheet("background-color: #e0e7ff; padding: 6px; text-align: left;")
            btn.clicked.connect(lambda _, t=metric: self.insert_text(self.query_input, t))
            metric_layout.addWidget(btn, row, col)
            col += 1
            if col > 5: col=0; row+=1 
        layout1.addWidget(scroll)

        input_layout = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("예: PER<10 AND PBR<1")
        self.filter_btn = QPushButton("조건 검색 실행")
        self.filter_btn.setStyleSheet("background-color: #2563eb; color: white; padding: 10px; font-weight: bold;")
        self.filter_btn.clicked.connect(self.run_financial_filter)
        input_layout.addWidget(self.query_input)
        input_layout.addWidget(self.filter_btn)
        layout1.addLayout(input_layout)

        self.fin_table = QTableWidget()
        self.fin_table.setColumnCount(3)
        self.fin_table.setHorizontalHeaderLabels(["삭제", "종목코드", "기업명"])
        self.fin_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.fin_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.fin_table.setColumnWidth(0, 50)
        layout1.addWidget(self.fin_table)

        # --- TAB 2: 전략 설정 ---
        tab2 = QWidget()
        self.tabs.addTab(tab2, "2. 전략 설정 및 실행")
        layout2 = QVBoxLayout(tab2)
        
        tech_group = QGroupBox("기술적 지표 & 수급 (클릭 시 입력)")
        tech_layout = QGridLayout(tech_group)
        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setWidget(tech_group)
        scroll2.setFixedHeight(350)
        
        sorted_techs = sorted(TECH_INDICATORS_MAP.keys())
        row, col = 0, 0
        for tech in sorted_techs:
            unit = TECH_UNITS.get(tech, '')
            btn_text = f"{tech} ({unit})" if unit else tech
            
            btn = QPushButton(btn_text)
            btn.setStyleSheet("background-color: #fef3c7; border: 1px solid #fcd34d; padding: 6px; text-align: left;")
            btn.clicked.connect(lambda _, t=tech: self.insert_to_focused(t))
            tech_layout.addWidget(btn, row, col)
            col += 1
            if col > 4: col=0; row+=1
        layout2.addWidget(scroll2)

        logic_frame = QFrame()
        logic_frame.setStyleSheet("background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 5px;")
        logic_layout = QGridLayout(logic_frame)
        
        logic_layout.addWidget(QLabel("매수 로직:"), 0, 0)
        self.buy_input = QLineEdit()
        self.buy_input.setPlaceholderText("예: 골든크로스=True AND 기관순매수>0")
        self.buy_input.installEventFilter(self)
        logic_layout.addWidget(self.buy_input, 0, 1)
        
        logic_layout.addWidget(QLabel("매도 로직:"), 1, 0)
        self.sell_input = QLineEdit()
        self.sell_input.setPlaceholderText("예: 5일선<20일선 OR 외국인순매수<0")
        self.sell_input.installEventFilter(self)
        logic_layout.addWidget(self.sell_input, 1, 1)
        
        logic_layout.addWidget(QLabel("백테스팅 기간:"), 2, 0)
        self.period_combo = QComboBox()
        self.period_combo.addItems(["3개월", "6개월", "1년", "3년", "5년", "10년"])
        self.period_combo.setCurrentText("1년") 
        logic_layout.addWidget(self.period_combo, 2, 1)
        
        layout2.addWidget(logic_frame)
        self.current_input_widget = self.buy_input

        self.backtest_btn = QPushButton("백테스팅 시작")
        self.backtest_btn.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 12px;")
        self.backtest_btn.clicked.connect(self.run_backtest)
        self.backtest_btn.setEnabled(False) 
        layout2.addWidget(self.backtest_btn)
        
        status_container = QWidget()
        status_layout = QVBoxLayout(status_container)
        status_layout.setSpacing(0) 
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(120) 
        self.log_view.setPlaceholderText("진행 로그가 여기에 표시됩니다.")
        status_layout.addWidget(self.log_view)

        self.status_msg = QLabel("준비")
        self.status_msg.setStyleSheet("font-size: 11px; color: #555; margin-left: 2px; margin-bottom: 1px; margin-top: 5px;")
        status_layout.addWidget(self.status_msg)
        
        self.status_bar = QProgressBar()
        self.status_bar.setAlignment(Qt.AlignCenter) 
        self.status_bar.setFixedHeight(20) 
        status_layout.addWidget(self.status_bar)
        
        layout2.addWidget(status_container)

        tab3 = QWidget()
        self.tabs.addTab(tab3, "3. 시뮬레이션 결과")
        layout3 = QVBoxLayout(tab3)
        
        stats_group = QGroupBox("전략 성과 분석")
        stats_group.setStyleSheet("background-color: #ecfdf5; font-weight: bold;")
        stats_layout = QGridLayout(stats_group)
        
        self.stat_labels = {}
        keys = ["Total Return", "CAGR", "MDD", "Trades", "Win Rate"]
        for i, key in enumerate(keys):
            lbl_title = QLabel(key)
            lbl_title.setStyleSheet("color: #047857;")
            lbl_val = QLabel("-")
            lbl_val.setStyleSheet("color: #064e3b; font-size: 16px;")
            stats_layout.addWidget(lbl_title, 0, i)
            stats_layout.addWidget(lbl_val, 1, i)
            self.stat_labels[key] = lbl_val
            
        layout3.addWidget(stats_group)
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout3.addWidget(self.canvas, stretch=3)
        
        signal_group = QGroupBox("매매 신호 로그")
        signal_layout = QVBoxLayout(signal_group)
        self.signal_view = QTextEdit()
        self.signal_view.setReadOnly(True)
        self.signal_view.setFixedHeight(100) 
        self.signal_view.setStyleSheet("font-family: Consolas; font-size: 12px;")
        signal_layout.addWidget(self.signal_view)
        layout3.addWidget(signal_group, stretch=1)

        footer_layout = QVBoxLayout()
        copyright_label = QLabel("Copyright 한양대학교 김현호")
        copyright_label.setAlignment(Qt.AlignRight)
        copyright_label.setStyleSheet("color: #666; font-size: 12px; margin-top: 5px;")
        contact_label = QLabel("문의 및 버그 제보 hh09080@naver.com")
        contact_label.setAlignment(Qt.AlignRight)
        contact_label.setStyleSheet("color: #888; font-size: 11px;")
        footer_layout.addWidget(copyright_label)
        footer_layout.addWidget(contact_label)
        main_layout.addLayout(footer_layout)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusIn:
            if obj == self.buy_input or obj == self.sell_input:
                self.current_input_widget = obj
        return super().eventFilter(obj, event)

    def insert_text(self, widget, text):
        widget.setText(widget.text() + text + " ")
        widget.setFocus()
        
    def insert_to_focused(self, text):
        if self.current_input_widget:
            curr_text = self.current_input_widget.text()
            self.current_input_widget.setText(curr_text + text + " ")
            self.current_input_widget.setFocus()

    def log(self, msg):
        self.log_view.append(msg)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clean_financial_data(self, df):
        pct_cols = [k for k, v in METRIC_UNITS.items() if v == '%']
        for col in pct_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                non_zero = df[df[col] != 0][col].abs()
                if not non_zero.empty:
                    mean_val = non_zero.mean()
                    if mean_val < 1.0: 
                        df[col] = df[col] * 100
        won_to_eok_cols = [k for k, v in METRIC_UNITS.items() if v == '억원']
        for col in won_to_eok_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                non_zero = df[df[col] != 0][col].abs()
                if not non_zero.empty:
                    mean_val = non_zero.mean()
                    if mean_val > 1000000:
                        df[col] = df[col] / 100000000
        return df

    def load_csv_files(self):
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        for year, key_name in YEAR_MAPPING.items():
            file_name = f"{key_name}.csv"
            file_path = os.path.join(base_path, file_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                except:
                    df = pd.read_csv(file_path, encoding='cp949')
                if '종목코드' in df.columns:
                    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
                df = self.clean_financial_data(df)
                self.financial_data[key_name] = df.to_dict('records')
                self.log(f"[{key_name}] 로드 완료")
            else:
                self.log(f"[경고] {file_name} 없음")

    def run_financial_filter(self):
        query = self.query_input.text()
        current_data_key = 'D-1y_data' 
        if current_data_key not in self.financial_data:
            QMessageBox.warning(self, "오류", "데이터 없음")
            return
        if ">=" in query or "<=" in query:
            QMessageBox.critical(self, "문법 오류", ">, <, = 를 사용해주세요.")
            return

        upper_query = query.upper()
        used_metrics = sorted([m for m in METRICS_LIST if m.upper() in upper_query], key=lambda x: query.find(x) if x in query else 999)

        self.filtered_companies = []
        try:
            for comp in self.financial_data[current_data_key]:
                if self.evaluate_financial_query(comp['종목코드'], query):
                    self.filtered_companies.append(comp.copy())
        except Exception as e:
            return

        self.update_filter_table(used_metrics) 
        msg = f"{len(self.filtered_companies)}개 종목 필터링 완료."
        self.log(msg)
        QMessageBox.information(self, "완료", msg)
        
        if self.filtered_companies:
            self.backtest_btn.setEnabled(True)
            self.tabs.setCurrentIndex(1) 

    def update_filter_table(self, used_metrics):
        self.fin_table.clear()
        headers = ["삭제", "종목코드", "기업명"] + used_metrics
        self.fin_table.setColumnCount(len(headers))
        self.fin_table.setHorizontalHeaderLabels(headers)
        self.fin_table.setRowCount(0)
        self.fin_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.fin_table.setColumnWidth(0, 50)
        
        for row_idx, comp in enumerate(self.filtered_companies):
            self.fin_table.insertRow(row_idx)
            del_btn = QPushButton("-")
            del_btn.setStyleSheet("color: red; font-weight: bold;")
            del_btn.clicked.connect(lambda _, r=row_idx: self.delete_company_row(r, used_metrics))
            self.fin_table.setCellWidget(row_idx, 0, del_btn)
            self.fin_table.setItem(row_idx, 1, QTableWidgetItem(str(comp.get('종목코드', ''))))
            self.fin_table.setItem(row_idx, 2, QTableWidgetItem(str(comp.get('기업명', ''))))
            for col_idx, metric in enumerate(used_metrics):
                self.fin_table.setItem(row_idx, 3 + col_idx, QTableWidgetItem(str(comp.get(metric, 0))))

    def delete_company_row(self, row_idx, used_metrics):
        if row_idx < 0 or row_idx >= len(self.filtered_companies): return
        removed_comp = self.filtered_companies.pop(row_idx)
        self.log(f"종목 제외됨: {removed_comp.get('기업명')}")
        self.update_filter_table(used_metrics)

    def evaluate_financial_query(self, code, query):
        if not query.strip(): return True
        query = query.replace("=", "==").replace("AND", " and ").replace("OR", " or ")
        def find_data(key, c): return next((x for x in self.financial_data.get(key, []) if x['종목코드']==c), None)
        def repl_series(match):
            metric, yr_str, op, val_str = match.groups()
            yrs = int(yr_str)
            limit = float(val_str)
            real_metric = next((m for m in METRICS_LIST if m.upper() == metric.upper()), None)
            if not real_metric: return "False"
            for i in range(yrs):
                d = find_data(f"D-{i+1}y_data", code)
                if not d or real_metric not in d: return "False"
                try:
                    if float(d[real_metric]) == 0: return "False"
                    if not eval(f"{d[real_metric]} {op} {limit}"): return "False"
                except: return "False"
            return "True"
        def repl_basic(match):
            metric, op, v_str = match.groups()
            real_metric = next((m for m in METRICS_LIST if m.upper() == metric.upper()), None)
            if not real_metric: return match.group(0)
            d = find_data('D-1y_data', code)
            if not d or real_metric not in d: return "False"
            try:
                if float(d[real_metric]) == 0: return "False"
                return f"{d[real_metric]} {op} {v_str}"
            except: return "False"
        try:
            q = re.sub(r"([가-힣a-zA-Z0-9\/]+)\*(\d+)\s*([><=!]+)\s*(-?[\d\.]+)", repl_series, query)
            q = re.sub(r"([가-힣a-zA-Z0-9\/]+)\s*([><=!]+)\s*(-?[\d\.]+)", repl_basic, q)
            return eval(q)
        except: return False

    def validate_logic_syntax(self, logic_str):
        if not logic_str.strip(): return True
        expr = logic_str
        for ko in sorted(TECH_INDICATORS_MAP.keys(), key=len, reverse=True):
            expr = expr.replace(ko, "100")
        expr = expr.replace(">=", "__GE__").replace("<=", "__LE__").replace("==", "__EQ__")
        expr = expr.replace("=", "==")
        expr = expr.replace("__GE__", ">=").replace("__LE__", "<=").replace("__EQ__", "==")
        expr = expr.replace("AND", "and").replace("OR", "or")
        expr = expr.replace("TRUE", "True").replace("FALSE", "False")
        try:
            eval(expr)
            return True
        except Exception as e:
            QMessageBox.warning(self, "오류", f"수식 오류: {e}")
            return False

    def run_backtest(self):
        buy_logic = self.buy_input.text()
        sell_logic = self.sell_input.text()
        if not self.validate_logic_syntax(buy_logic): return
        if not self.validate_logic_syntax(sell_logic): return

        self.backtest_btn.setEnabled(False)
        self.status_msg.setText("백테스팅 시작...")
        self.log_view.clear()
        self.status_bar.setValue(0) 
        
        period_text = self.period_combo.currentText()
        period_map = {"3개월": 3, "6개월": 6, "1년": 12, "3년": 36, "5년": 60, "10년": 120}
        period_months = period_map.get(period_text, 12)
        
        self.worker = BacktestWorker(self.filtered_companies, buy_logic, sell_logic, period_months)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log)
        self.worker.result_signal.connect(self.display_backtest_result)
        self.worker.error_signal.connect(self.show_error)
        self.worker.start()

    def update_progress(self, curr, total, msg):
        if total > 0: self.status_bar.setValue(int((curr/total)*100))
        self.status_msg.setText(msg)
    
    def show_error(self, msg):
        QMessageBox.critical(self, "오류", msg)
        self.backtest_btn.setEnabled(True)

    def display_backtest_result(self, data_dict, final_df, detailed_stats, recent_signals):
        self.backtest_btn.setEnabled(True)
        self.tabs.setCurrentIndex(2)

        for key, lbl in self.stat_labels.items():
            lbl.setText(detailed_stats.get(key, "-"))

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if not final_df.empty:
            ax.plot(final_df.index, final_df['Strategy'], label='My Strategy', color='#dc2626', linewidth=2)
            ax.plot(final_df.index, final_df['BnH'], label='Benchmark (B&H)', color='#2563eb', alpha=0.6)
            if 'KOSPI' in final_df.columns:
                ax.plot(final_df.index, final_df['KOSPI'], label='KOSPI', color='#059669', linestyle='--')
            ax.set_title("Cumulative Return (%)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        self.canvas.draw()
        
        html = ""
        html += "<h4 style='color:#dc2626'>[최근 매수 신호]</h4><ul>"
        for s in recent_signals['buy']:
            html += f"<li>{s['date']} | <b>{s['name']}</b> ({s['code']}) | {s['price']:,}원</li>"
        html += "</ul><h4 style='color:#2563eb'>[최근 매도 신호]</h4><ul>"
        for s in recent_signals['sell']:
            html += f"<li>{s['date']} | <b>{s['name']}</b> ({s['code']}) | {s['price']:,}원</li>"
        html += "</ul>"
        self.signal_view.setHtml(html)

        try:
            self.save_results_to_excel(final_df, detailed_stats, recent_signals)
            QMessageBox.information(self, "완료", "백테스팅 완료 및 결과 파일 저장됨.")
        except Exception as e:
            QMessageBox.warning(self, "저장 실패", f"결과 파일 저장 중 오류: {e}\n(openpyxl이 설치되어 있는지 확인하세요)")

    def save_results_to_excel(self, final_df, stats, signals):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{now}_Result.xlsx"
        dir_name = "Memories"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        filepath = os.path.join(dir_name, filename)
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            stats_df = pd.DataFrame([stats])
            stats_df.to_excel(writer, sheet_name='Stats', index=False)
            
            if not final_df.empty:
                final_df.to_excel(writer, sheet_name='Returns')

            buy_list = [{'Type': 'Buy', **s} for s in signals['buy']]
            sell_list = [{'Type': 'Sell', **s} for s in signals['sell']]
            all_signals = buy_list + sell_list
            if all_signals:
                sig_df = pd.DataFrame(all_signals)
                sig_df = sig_df.sort_values(by='date')
                sig_df.to_excel(writer, sheet_name='Signals', index=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Malgun Gothic", 10)
    app.setFont(font)
    window = QuantApp()
    window.show()
    sys.exit(app.exec())