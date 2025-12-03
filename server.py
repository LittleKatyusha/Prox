import asyncio
import json
import logging
import uuid
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional, Set, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Response, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from starlette.responses import StreamingResponse
import traceback
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from pathlib import Path
import aiohttp
import requests
import typing
import gzip
import shutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
class Config:
    LOG_DIR = Path("logs")
    REQUEST_LOG_FILE = "requests.jsonl"
    ERROR_LOG_FILE = "errors.jsonl"
    MAX_LOG_SIZE = 50 * 1024 * 1024
    MAX_LOG_FILES = 50 
    
    HOST = "0.0.0.0"
    PORT = 8741
    
    REQUEST_TIMEOUT_SECONDS = 180
    
    STATS_UPDATE_INTERVAL = 5 
    CLEANUP_INTERVAL = 300 
    
    MAX_LOG_MEMORY_ITEMS = 1000 
    MAX_REQUEST_DETAILS = 500 

    V1_URL = os.getenv("V1_URL")
    V1_TOKEN = os.getenv("V1_TOKEN")

    V2_URL = os.getenv("V2_URL")
    V2_TOKEN = os.getenv("V2_TOKEN")

    V3_URL = os.getenv("V3_URL")
    V3_TOKEN = os.getenv("V3_TOKEN")

    V4_URL = os.getenv("V4_URL")
    V4_TOKEN = os.getenv("V4_TOKEN")

    V5_URL = os.getenv("V5_URL")
    V5_TOKEN = os.getenv("V5_TOKEN")

    V6_URL = os.getenv("V6_URL")
    V6_TOKEN = os.getenv("V6_TOKEN")

    V7_URL = os.getenv("V7_URL")
    V7_TOKEN = os.getenv("V7_TOKEN")

    V8_URL = os.getenv("V8_URL")
    V8_TOKEN = os.getenv("V8_TOKEN")

    V9_URL = os.getenv("V9_URL")
    V9_TOKEN = os.getenv("V9_TOKEN")

    V10_URL = os.getenv("V10_URL")
    V10_TOKEN = os.getenv("V10_TOKEN")

    MASTER_KEY = os.getenv("MASTER_KEY")

def get_local_ip():
    return "0.0.0.0"

# --- API KEY IMPLEMENTATION ---

async def verify_api_key(authorization: str = Header(..., alias="Authorization")):
    """Dependency to verify API key and enforce rate limits"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Expected 'Bearer <token>'"
        )
    
    api_key = authorization.replace("Bearer ", "", 1)
    
    if not api_key_manager.is_valid(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    rate_limiter.check_rate_limit(api_key, rate_limit=500)
    
    return api_key

async def verify_master_key(master_key: str = Header(..., alias="Authorization")):
    if master_key != Config.MASTER_KEY:
        raise HTTPException(status_code=403, detail="Invalid master key")
    return True

class APIKeyManager:
    def __init__(self):
        self.keys_info_file = Path("api_keys.json")
        self.valid_keys = set()
        self.key_names = {}  # Store key -> name mapping
        self.load_key_info()
    
    def load_key_info(self):
        """Load API keys and their names from JSON file"""
        if self.keys_info_file.exists():
            try:
                with open(self.keys_info_file, 'r') as f:
                    data = json.load(f)
                    self.key_names = data.get('key_names', {})
                    self.valid_keys = set(self.key_names.keys())
                logging.info(f"Loaded {len(self.valid_keys)} API keys")
            except Exception as e:
                logging.error(f"Error loading API keys: {e}")
                self.key_names = {}
                self.valid_keys = set()
        else:
            logging.warning("api_keys.json not found. Creating empty file.")
            self.save_key_info()

    def get_keys(self):
        """Get all API keys with their names"""
        return [
            {'api_key': key, 'name': self.key_names.get(key, 'Unnamed')}
            for key in self.valid_keys
        ]
    
    def save_key_info(self):
        """Save API key names and metadata"""
        with open(self.keys_info_file, 'w') as f:
            json.dump({'key_names': self.key_names}, f, indent=2)
    
    def add_key(self, api_key: str, name: str = "Unnamed"):
        """Add a new API key with an optional name"""
        self.valid_keys.add(api_key)
        self.key_names[api_key] = name
        self.save_key_info()
    
    def remove_key(self, api_key: str):
        """Remove an API key"""
        self.valid_keys.discard(api_key)
        self.key_names.pop(api_key, None)
        self.save_key_info()
    
    def set_key_name(self, api_key: str, name: str):
        """Set a friendly name for an API key"""
        if api_key in self.valid_keys:
            self.key_names[api_key] = name
            self.save_key_info()
    
    def get_key_name(self, api_key: str) -> str:
        """Get the friendly name for an API key"""
        return self.key_names.get(api_key, "Unnamed")

    def is_valid(self, api_key: str) -> bool:
        """Check if API key is valid"""
        return api_key in self.valid_keys
    
    def reload_keys(self):
        """Reload keys from file (useful for runtime updates)"""
        self.load_key_info()
    
    def get_usage_stats(self, api_key: str) -> dict:
        """Get usage statistics for an API key"""
        # Read logs and filter by API key
        logs = log_manager.read_request_logs(limit=10000)
        current_time = time.time()
        day_ago = current_time - 86400
    
        # Filter logs for this API key in the last 24 hours
        api_key_logs_24h = [
            log for log in logs 
            if log.get('api_key') == api_key and log.get('timestamp', 0) > day_ago
        ]
    
        api_key_logs_all = [
            log for log in logs 
            if log.get('api_key') == api_key
        ]

        total_requests = len(api_key_logs_all)
        daily_requests = len(api_key_logs_24h)
        total_input_tokens = sum(log.get('input_tokens', 0) for log in api_key_logs_all)
        total_output_tokens = sum(log.get('output_tokens', 0) for log in api_key_logs_all)
        daily_input_tokens = sum(log.get('input_tokens', 0) for log in api_key_logs_24h)
        daily_output_tokens = sum(log.get('output_tokens', 0) for log in api_key_logs_24h)
    
        return {
            "daily_requests": daily_requests,
            "total_requests": total_requests,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'daily_input_tokens': daily_input_tokens,
            'daily_output_tokens': daily_output_tokens,
            "rate_limit": 999.999,
            'is_valid': api_key in self.valid_keys
        }

class RateLimiter:
    def __init__(self):
        self.api_key_usage = defaultdict(list)
    
    def check_rate_limit(self, api_key: str, rate_limit: int = 60):
        current_time = time.time()
        
        # Clean up old timestamps (older than 60 seconds)
        self.api_key_usage[api_key] = [t for t in self.api_key_usage[api_key] if current_time - t < 60]
        
        if len(self.api_key_usage[api_key]) >= rate_limit:
            # Calculate seconds until oldest request expires (60 seconds window)
            oldest_timestamp = min(self.api_key_usage[api_key])
            retry_after = int(60 - (current_time - oldest_timestamp))
            retry_after = max(1, retry_after)  # At least 1 second
            
            raise HTTPException(status_code=429, detail=f"Rate limit: RPM ({rate_limit}) rate limit exceeded. Please try again later.")
        
        self.api_key_usage[api_key].append(current_time)

rate_limiter = RateLimiter();
api_key_manager = APIKeyManager();

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler(Config.LOG_DIR / "server.log", encoding='utf-8') 
    ]
)

# --- Prometheus Metrics ---
request_count = Counter(
    'lmarena_requests_total', 
    'Total number of requests',
    ['model', 'status', 'type']
)

request_duration = Histogram(
    'lmarena_request_duration_seconds',
    'Request duration in seconds',
    ['model', 'type'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf"))
)

active_requests_gauge = Gauge(
    'lmarena_active_requests',
    'Number of active requests'
)

token_usage = Counter(
    'lmarena_tokens_total',
    'Total number of tokens used',
    ['model', 'token_type']  # token_type: input/output
)

websocket_status = Gauge(
    'lmarena_websocket_connected',
    'WebSocket connection status (1=connected, 0=disconnected)'
)

error_count = Counter(
    'lmarena_errors_total',
    'Total number of errors',
    ['error_type', 'model']
)

model_registry_gauge = Gauge(
    'lmarena_models_registered',
    'Number of registered models'
)

# --- Request Details Storage ---
@dataclass
class RequestDetails:
    """存储请求的详细信息"""
    request_id: str
    timestamp: float
    model: str
    status: str
    duration: float
    input_tokens: int
    output_tokens: int
    error: Optional[str]
    request_params: dict
    request_messages: list
    response_content: str
    headers: dict
    
class RequestDetailsStorage:
    """管理请求详情的存储"""
    def __init__(self, max_size: int = Config.MAX_REQUEST_DETAILS):
        self.details: Dict[str, RequestDetails] = {}
        self.order: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add(self, details: RequestDetails):
        """添加请求详情"""
        with self._lock:
            if details.request_id in self.details:
                return
            
            if len(self.order) >= self.order.maxlen:
                oldest_id = self.order[0]
                if oldest_id in self.details:
                    del self.details[oldest_id]
            
            self.details[details.request_id] = details
            self.order.append(details.request_id)
    
    def get(self, request_id: str) -> Optional[RequestDetails]:
        """获取请求详情"""
        with self._lock:
            return self.details.get(request_id)
    
    def get_recent(self, limit: int = 100) -> list:
        """获取最近的请求详情"""
        with self._lock:
            recent_ids = list(self.order)[-limit:]
            return [self.details[id] for id in reversed(recent_ids) if id in self.details]

request_details_storage = RequestDetailsStorage()

class LogManager:
    def __init__(self):
        self.request_log_path = Config.LOG_DIR / Config.REQUEST_LOG_FILE
        self.error_log_path = Config.LOG_DIR / Config.ERROR_LOG_FILE
        self._lock = threading.Lock()
        self._check_and_rotate()
    
    def _check_and_rotate(self):
        for log_path in [self.request_log_path, self.error_log_path]:
            if log_path.exists() and log_path.stat().st_size > Config.MAX_LOG_SIZE:
                self._rotate_log(log_path)
    
    def _rotate_log(self, log_path: Path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_path = log_path.with_suffix(f".{timestamp}.jsonl")
        
        shutil.move(log_path, rotated_path)
        
        with open(rotated_path, 'rb') as f_in:
            with gzip.open(f"{rotated_path}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        rotated_path.unlink()
        
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        log_files = sorted(Config.LOG_DIR.glob("*.jsonl.gz"), key=lambda x: x.stat().st_mtime)
        
        while len(log_files) > Config.MAX_LOG_FILES:
            oldest_file = log_files.pop(0)
            oldest_file.unlink()
            logging.info(f": {oldest_file}")
    
    def write_request_log(self, log_entry: dict):
        with self._lock:
            self._check_and_rotate()
            with open(self.request_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def write_error_log(self, log_entry: dict):
        with self._lock:
            self._check_and_rotate()
            with open(self.error_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def read_request_logs(self, limit: int = 100, offset: int = 0, model: str = None) -> list:
        logs = []
        
        if self.request_log_path.exists():
            with open(self.request_log_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                
                for line in reversed(all_lines):
                    try:
                        log = json.loads(line.strip())
                        if log.get('type') == 'request_end':  
                            if model and log.get('model') != model:
                                continue
                            logs.append(log)
                            if len(logs) >= limit + offset:
                                break
                    except json.JSONDecodeError:
                        continue
        
        return logs[offset:offset + limit]
    
    def read_error_logs(self, limit: int = 50) -> list:
        logs = []
        
        if self.error_log_path.exists():
            with open(self.error_log_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                
                for line in reversed(all_lines[-limit:]):
                    try:
                        log = json.loads(line.strip())
                        logs.append(log)
                    except json.JSONDecodeError:
                        continue
        
        return logs

log_manager = LogManager()

class PerformanceMonitor:
    
    def __init__(self):
        self.request_times = deque(maxlen=1000)
        self.model_stats = defaultdict(lambda: {'count': 0, 'errors': 0})
    
    def record_request(self, model: str, duration: float, success: bool):
        self.request_times.append(duration)
        self.model_stats[model]['count'] += 1
        if not success:
            self.model_stats[model]['errors'] += 1
    
    def get_stats(self) -> dict:
        if not self.request_times:
            return {'avg_response_time': 0}
        return {
            'avg_response_time': sum(self.request_times) / len(self.request_times)
        }

    def get_model_stats(self) -> dict:
        """Get statistics per model"""
        result = {}
        for model, stats in self.model_stats.items():
            count = stats['count']
            errors = stats['errors']
            result[model] = {
                'total_requests': count,
                'errors': errors,
                'error_rate': (errors / count * 100) if count > 0 else 0,
                'qps': count  # You may want to calculate this based on time window
            }
        return result

performance_monitor = PerformanceMonitor()

@dataclass
class RealtimeStats:
    active_requests: Dict[str, dict] = field(default_factory=dict)
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=Config.MAX_LOG_MEMORY_ITEMS))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=50))
    model_usage: Dict[str, dict] = field(default_factory=lambda: defaultdict(lambda: {
        'requests': 0, 'tokens': 0, 'errors': 0, 'avg_duration': 0
    }))
    
    def cleanup_old_requests(self):
        current_time = time.time()
        timeout_requests = []
        
        for req_id, req in self.active_requests.items():
            if current_time - req['start_time'] > Config.REQUEST_TIMEOUT_SECONDS:
                timeout_requests.append(req_id)
        
        for req_id in timeout_requests:
            logging.warning(f"Warning: {req_id}")
            del self.active_requests[req_id]

realtime_stats = RealtimeStats()

async def periodic_cleanup():
    while not SHUTTING_DOWN:
        try:
            realtime_stats.cleanup_old_requests()
            
            log_manager._check_and_rotate()
            
            active_requests_gauge.set(len(realtime_stats.active_requests))
            model_registry_gauge.set(len(MODEL_REGISTRY))
            
            logging.info(f"清理任务执行完成. 活跃请求: {len(realtime_stats.active_requests)}")
            
        except Exception as e:
            logging.error(f"Error: {e}")
        
        await asyncio.sleep(Config.CLEANUP_INTERVAL)

# --- Custom Streaming Response with Immediate Flush ---
class ImmediateStreamingResponse(StreamingResponse):
    """Custom streaming response that forces immediate flushing of chunks"""

    async def stream_response(self, send: typing.Callable) -> None:
        await send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": self.raw_headers,
        })

        async for chunk in self.body_iterator:
            if chunk:
                # Send the chunk immediately
                await send({
                    "type": "http.response.body",
                    "body": chunk.encode(self.charset) if isinstance(chunk, str) else chunk,
                    "more_body": True,
                })
                # Force a small delay to ensure the chunk is sent
                await asyncio.sleep(0)

        # Send final empty chunk to close the stream
        await send({
            "type": "http.response.body",
            "body": b"",
            "more_body": False,
        })

# --- Logging Functions ---
def log_request_start(request_id: str, model: str, params: dict, messages: list = None, api_key: str = None):
    request_info = {
        'id': request_id,
        'model': model,
        'start_time': time.time(),
        'status': 'active',
        'params': params,
        'messages': messages or [],
        'api_key': api_key
    }
    
    realtime_stats.active_requests[request_id] = request_info
    
    log_entry = {
        'type': 'request_start',
        'timestamp': time.time(),
        'request_id': request_id,
        'model': model,
        'params': params,
        'api_key': api_key
    }
    log_manager.write_request_log(log_entry)
    
def log_request_end(request_id: str, success: bool, input_tokens: int = 0, 
                   output_tokens: int = 0, error: str = None, response_content: str = "", api_key: str = None):
    if request_id not in realtime_stats.active_requests:
        return
        
    req = realtime_stats.active_requests[request_id]
    duration = time.time() - req['start_time']
    
    req['status'] = 'success' if success else 'failed'
    req['duration'] = duration
    req['input_tokens'] = input_tokens
    req['output_tokens'] = output_tokens
    req['error'] = error
    req['end_time'] = time.time()
    req['response_content'] = response_content
    
    realtime_stats.recent_requests.append(req.copy())
    
    model = req['model']
    stats = realtime_stats.model_usage[model]
    stats['requests'] += 1
    if success:
        stats['tokens'] += input_tokens + output_tokens
    else:
        stats['errors'] += 1
    
    performance_monitor.record_request(model, duration, success)
    
    request_count.labels(model=model, status='success' if success else 'failed', type='chat').inc()
    request_duration.labels(model=model, type='chat').observe(duration)
    token_usage.labels(model=model, token_type='input').inc(input_tokens)
    token_usage.labels(model=model, token_type='output').inc(output_tokens)
    
    details = RequestDetails(
        request_id=request_id,
        timestamp=req['start_time'],
        model=model,
        status='success' if success else 'failed',
        duration=duration,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
        request_params=req.get('params', {}),
        request_messages=req.get('messages', []),
        response_content=response_content[:5000], 
        headers={}
    )
    request_details_storage.add(details)
    
    log_entry = {
        'type': 'request_end',
        'timestamp': time.time(),
        'request_id': request_id,
        'model': model,
        'status': 'success' if success else 'failed',
        'duration': duration,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'error': error,
        'params': req.get('params', {}),
        'api_key': api_key or req.get('api_key')  # Use passed api_key or get from request info
    }
    log_manager.write_request_log(log_entry)

    del realtime_stats.active_requests[request_id]

def log_error(request_id: str, error_type: str, error_message: str, stack_trace: str = ""):
    """记录错误日志"""
    error_data = {
        'timestamp': time.time(),
        'request_id': request_id,
        'error_type': error_type,
        'error_message': error_message,
        'stack_trace': stack_trace
    }
    
    realtime_stats.recent_errors.append(error_data)
    
    # Prometheus
    model = realtime_stats.active_requests.get(request_id, {}).get('model', 'unknown')
    error_count.labels(error_type=error_type, model=model).inc()
   
    log_manager.write_error_log(error_data)

# --- Model Registry ---
MODEL_REGISTRY = {}  # Will be populated dynamically

def load_models_from_file():
    """Load models from allowed_models.txt into MODEL_REGISTRY"""
    global MODEL_REGISTRY
    
    try:
        with open("allowed_models.txt", "r") as f:
            for line in f:
                model_name = line.strip()
                if model_name and not model_name.startswith('#'):
                    MODEL_REGISTRY[model_name] = {
                        "type": "chat",
                        "capabilities": {
                            "outputCapabilities": {}
                        }
                    }
        logging.info(f"Loaded {len(MODEL_REGISTRY)} models from allowed_models.txt")
    except FileNotFoundError:
        logging.warning("allowed_models.txt not found")

# --- Global State ---
response_channels: dict[str, asyncio.Queue] = {}  # Keep for backward compatibility
background_tasks: Set[asyncio.Task] = set()
SHUTTING_DOWN = False
startup_time = time.time()  # 服务器启动时间

# --- FastAPI App and Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_REGISTRY, startup_time
    startup_time = time.time()

    # Use fallback registry on startup - models will be updated by browser script
    load_models_from_file()
    logging.info(f" {len(MODEL_REGISTRY)} ")

    cleanup_task = asyncio.create_task(periodic_cleanup())
    background_tasks.add(cleanup_task)

    try:
        yield
    finally:
        global SHUTTING_DOWN
        SHUTTING_DOWN = True
        logging.info(f"生命周期: 服务器正在关闭。正在取消 {len(background_tasks)} 个后台任务...")

        # Cancel all background tasks
        cancelled_tasks = []
        for task in list(background_tasks):
            if not task.done():
                logging.info(f"生命周期: 正在取消任务: {task}")
                task.cancel()
                cancelled_tasks.append(task)

        # Wait for cancelled tasks to finish
        if cancelled_tasks:
            logging.info(f"生命周期: 等待 {len(cancelled_tasks)} 个已取消的任务完成...")
            results = await asyncio.gather(*cancelled_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.info(f"生命周期: 任务 {i} 完成,结果: {type(result).__name__}")
                else:
                    logging.info(f"生命周期: 任务 {i} 正常完成")

        logging.info("生命周期: 所有后台任务已取消。关闭完成。")


app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Handlers ---
@app.post("/v1/chat/completions")
async def chat_completions_v1(request: Request, api_key: str = Depends(verify_api_key)):
    openai_req = await request.json()
    request_id = str(uuid.uuid4())
    is_streaming = openai_req.get("stream", True)
    model_name = openai_req.get("model")
    
    # Validate model
    model_info = MODEL_REGISTRY.get(model_name)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    # Remove unwanted parameters before sending to backend
    params_to_exclude = ["frequency_penalty", "presence_penalty", "top_p"]
    for param in params_to_exclude:
        openai_req.pop(param, None)  # Use pop with None default to avoid KeyError
    
    # Log request start for stats
    request_params = {
        "temperature": openai_req.get("temperature"),
        "max_tokens": openai_req.get("max_tokens"),
        "streaming": is_streaming
    }
    messages = openai_req.get("messages", [])
    log_request_start(request_id, model_name, request_params, messages, api_key)
    
    try:
        if is_streaming:
            return StreamingResponse(
                stream_from_backend(request_id, openai_req, model_name, api_key),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response
            response_data = await make_backend_request(request_id, openai_req, model_name, api_key)
            return response_data
            
    except Exception as e:
        log_request_end(request_id, False, 0, 0, str(e))
        logging.error(f"API [ID: {request_id}]: Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Exception: Encountered an error. Please try again later or DM norenaboi.")

@app.post("/v2/chat/completions")
async def chat_completions_v2(request: Request, api_key: str = Depends(verify_api_key)):
    """Endpoint for v2 and v3 models combined"""
    openai_req = await request.json()
    request_id = str(uuid.uuid4())
    is_streaming = openai_req.get("stream", True)
    model_name = openai_req.get("model")
    
    # Validate model
    model_info = MODEL_REGISTRY.get(model_name)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    # Remove unwanted parameters before sending to backend
    params_to_exclude = ["frequency_penalty", "presence_penalty", "top_p"]
    for param in params_to_exclude:
        openai_req.pop(param, None)
    
    # Log request start for stats
    request_params = {
        "temperature": openai_req.get("temperature"),
        "max_tokens": openai_req.get("max_tokens"),
        "streaming": is_streaming
    }
    messages = openai_req.get("messages", [])
    log_request_start(request_id, model_name, request_params, messages, api_key)
    
    try:
        if is_streaming:
            return StreamingResponse(
                stream_from_backend_v2(request_id, openai_req, model_name, api_key),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response
            response_data = await make_backend_request_v2(request_id, openai_req, model_name, api_key)
            return response_data
            
    except Exception as e:
        log_request_end(request_id, False, 0, 0, str(e))
        logging.error(f"API [ID: {request_id}]: Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Exception: Encountered an error. Please try again later or DM norenaboi.")

def get_temperature(openai_req: dict):
    temp = openai_req.get("temperature")
    if temp is None:
        return None  # or some default value like 0.7
    elif temp > 1:
        return 1
    else:
        return temp

# --- Backend Request Functions ---
async def stream_from_backend(
    request_id: str, 
    openai_req: dict, 
    model_name: str, 
    api_key: str
) -> AsyncGenerator[str, None]:
    """Stream responses directly from the backend"""
    start_time = time.time()
    accumulated_content = ""

    if model_name.endswith('-v1'):
        backend_url = Config.V1_URL
        backend_token = Config.V1_TOKEN
        actual_model = model_name.replace('-v1', '')
    elif model_name.endswith('-v2'):
        backend_url = Config.V2_URL
        backend_token = Config.V2_TOKEN
        actual_model = model_name.replace('-v2', '')
    elif model_name.endswith('-v3'):
        backend_url = Config.V3_URL
        backend_token = Config.V3_TOKEN
        actual_model = model_name.replace('-v3', '')
    elif model_name.endswith('-v4'):
        backend_url = Config.V4_URL
        backend_token = Config.V4_TOKEN
        actual_model = model_name.replace('-v4', '')
    elif model_name.endswith('-v5'):
        backend_url = Config.V5_URL
        backend_token = Config.V5_TOKEN
        actual_model = model_name.replace('-v5', '')
    elif model_name.endswith('-v6'):
        backend_url = Config.V6_URL
        backend_token = Config.V6_TOKEN
        actual_model = model_name.replace('-v6', '')
    elif model_name.endswith('-v7'):
        backend_url = Config.V7_URL
        backend_token = Config.V7_TOKEN
        actual_model = model_name.replace('-v7', '')
    elif model_name.endswith('-v8'):
        backend_url = Config.V8_URL
        backend_token = Config.V8_TOKEN
        actual_model = model_name.replace('-v8', '')
    elif model_name.endswith('-v9'):
        backend_url = Config.V9_URL
        backend_token = Config.V9_TOKEN
        actual_model = model_name.replace('-v9', '')
    elif model_name.endswith('-v10'):
        backend_url = Config.V10_URL
        backend_token = Config.V10_TOKEN
        actual_model = model_name.replace('-v10', '')
    else:
        error_response = {
            "error": {
                "message": "Error 404: Can't find the model you're looking for.",
                "type": "server_error",
                "code": 404
            }
        }
        raise HTTPException(status_code=404, detail=error_response)

    backend_url = backend_url + "/v1/chat/completions"

    try:
        async with aiohttp.ClientSession() as session:
            # Prepare the request
            data = {
                "model": actual_model,
                "stream": True,
                "messages": openai_req.get("messages", []),
                "max_tokens": openai_req.get("max_tokens")
            }

            BACKEND_HEADERS = {
                "Authorization": f"Bearer {backend_token}",
                "Content-Type": "application/json"
            }
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            async with session.post(
                backend_url,
                headers=BACKEND_HEADERS,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = "Backend JSON/Parameter error."
                    logging.error(f"BACKEND [ID: {request_id}]: {error_text[:200]}")
                    
                    # Return OpenAI-formatted error
                    error_response = {
                        "error": {
                            "message": error_msg,
                            "type": "server_error",
                            "code": response.status
                        }
                    }
                    yield f"data: {json.dumps(error_response)}\n\ndata: [DONE]\n\n"
                    return
                
                # Stream the response
                async for line in response.content:
                    if line:
                        decoded = line.decode("utf-8").strip()
                        if decoded.startswith("data: "):
                            payload = decoded[6:].strip()
                            
                            if payload == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break
                            
                            try:
                                # Parse and accumulate content for logging
                                chunk_data = json.loads(payload)
                                
                                # Check if chunk_data is valid
                                if chunk_data is not None and isinstance(chunk_data, dict):
                                    choices = chunk_data.get("choices", [])
                                    if choices and len(choices) > 0:
                                        delta = choices[0].get("delta", {})
                                        if delta:
                                            content = delta.get("content", "")
                                            if content:
                                                accumulated_content += content
                                
                                # Forward the chunk as-is
                                yield f"data: {payload}\n\n"
                                
                            except json.JSONDecodeError:
                                response_data = {"error": {"message": "Invalid JSON response."}}
                                logging.warning(f"BACKEND [ID: {request_id}]: Invalid JSON in stream.")
                                continue
        
        # Log successful completion
        input_tokens = estimateTokens(json.dumps(openai_req))
        output_tokens = estimateTokens(accumulated_content)
        log_request_end(request_id, True, input_tokens, output_tokens, response_content=accumulated_content, api_key=api_key)
        
    except Exception as e:
        logging.error(f"BACKEND [ID: {request_id}]: Stream error: {e}", exc_info=True)
        
        # Log error
        log_error(request_id, type(e).__name__, str(e), traceback.format_exc())
        log_request_end(request_id, False, 0, 0, str(e))
        
        # Return error in OpenAI format
        error_response = {
            "error": {
                "message": "Server side error. Please try again later or DM norenaboi.",
                "type": "server_error",
                "code": None
            }
        }
        yield f"data: {json.dumps(error_response)}\n\ndata: [DONE]\n\n"

async def make_backend_request(
    request_id: str,
    openai_req: dict,
    model_name: str,
    api_key: str
) -> dict:
    """Make non-streaming request to backend"""
    start_time = time.time()

    if model_name.endswith('-v1'):
        backend_url = Config.V1_URL
        backend_token = Config.V1_TOKEN
        actual_model = model_name.replace('-v1', '')
    elif model_name.endswith('-v2'):
        backend_url = Config.V2_URL
        backend_token = Config.V2_TOKEN
        actual_model = model_name.replace('-v2', '')
    elif model_name.endswith('-v3'):
        backend_url = Config.V3_URL
        backend_token = Config.V3_TOKEN
        actual_model = model_name.replace('-v3', '')
    elif model_name.endswith('-v4'):
        backend_url = Config.V4_URL
        backend_token = Config.V4_TOKEN
        actual_model = model_name.replace('-v4', '')
    elif model_name.endswith('-v5'):
        backend_url = Config.V5_URL
        backend_token = Config.V5_TOKEN
        actual_model = model_name.replace('-v5', '')
    elif model_name.endswith('-v6'):
        backend_url = Config.V6_URL
        backend_token = Config.V6_TOKEN
        actual_model = model_name.replace('-v6', '')
    elif model_name.endswith('-v7'):
        backend_url = Config.V7_URL
        backend_token = Config.V7_TOKEN
        actual_model = model_name.replace('-v7', '')
    elif model_name.endswith('-v8'):
        backend_url = Config.V8_URL
        backend_token = Config.V8_TOKEN
        actual_model = model_name.replace('-v8', '')
    elif model_name.endswith('-v9'):
        backend_url = Config.V9_URL
        backend_token = Config.V9_TOKEN
        actual_model = model_name.replace('-v9', '')
    elif model_name.endswith('-v10'):
        backend_url = Config.V10_URL
        backend_token = Config.V10_TOKEN
        actual_model = model_name.replace('-v10', '')
    else:
        error_response = {
            "error": {
                "message": "Error 404: Can't find the model you're looking for.",
                "type": "server_error",
                "code": 404
            }
        }
        raise HTTPException(status_code=404, detail=error_response)

    backend_url = backend_url + "/v1/chat/completions"

    try:
        async with aiohttp.ClientSession() as session:
            # Prepare the request
            data = {
                "model": actual_model,
                "stream": False,
                "messages": openai_req.get("messages", []),
                "max_tokens": openai_req.get("max_tokens")
            }

            BACKEND_HEADERS = {
                "Authorization": f"Bearer {backend_token}",
                "Content-Type": "application/json"
            }
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            async with session.post(
                backend_url,
                headers=BACKEND_HEADERS,
                json=data
            ) as response:
                # Get response text first, then parse as JSON
                response_text = await response.text()
                
                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError:
                    # If JSON parsing fails, wrap the text response
                    response_data = {"error": {"message": "Invalid JSON response."}}
                
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=response_data.get("Error: Backend JSON/Parameter error. Try disabling prefills or changing presets.")
                    )
                
                # Log successful completion
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                input_tokens = estimateTokens(json.dumps(openai_req))
                output_tokens = estimateTokens(content)
                log_request_end(request_id, True, input_tokens, output_tokens, content, api_key)
                
                return response_data
                
    except Exception as e:
        # Log error
        log_error(request_id, type(e).__name__, str(e), traceback.format_exc())
        log_request_end(request_id, False, 0, 0, str(e))
        
        raise

# --- Backend Request Functions for V2 endpoint ---
async def stream_from_backend_v2(
    request_id: str,
    openai_req: dict,
    model_name: str,
    api_key: str
) -> AsyncGenerator[str, None]:
    """Stream responses directly from the backend for v2/v3 models"""
    start_time = time.time()
    accumulated_content = ""

    # Determine backend based on model suffix
    if model_name.endswith('-v2'):
        backend_url = Config.V2_URL
        backend_token = Config.V2_TOKEN
        actual_model = model_name.replace('-v2', '')
    elif model_name.endswith('-v3'):
        backend_url = Config.V3_URL
        backend_token = Config.V3_TOKEN
        actual_model = model_name.replace('-v3', '')
    else:
        error_response = {
            "error": {
                "message": "Error 404: This endpoint only supports v2 and v3 models.",
                "type": "server_error",
                "code": 404
            }
        }
        raise HTTPException(status_code=404, detail=error_response)

    backend_url = backend_url + "/v1/chat/completions"

    try:
        async with aiohttp.ClientSession() as session:
            # Prepare the request
            data = {
                "model": actual_model,
                "stream": True,
                "messages": openai_req.get("messages", []),
                "max_tokens": openai_req.get("max_tokens")
            }

            BACKEND_HEADERS = {
                "Authorization": f"Bearer {backend_token}",
                "Content-Type": "application/json"
            }
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            async with session.post(
                backend_url,
                headers=BACKEND_HEADERS,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = "Backend JSON/Parameter error."
                    logging.error(f"BACKEND [ID: {request_id}]: {error_text[:200]}")
                    
                    # Return OpenAI-formatted error
                    error_response = {
                        "error": {
                            "message": error_msg,
                            "type": "server_error",
                            "code": response.status
                        }
                    }
                    yield f"data: {json.dumps(error_response)}\n\ndata: [DONE]\n\n"
                    return
                
                # Stream the response
                async for line in response.content:
                    if line:
                        decoded = line.decode("utf-8").strip()
                        if decoded.startswith("data: "):
                            payload = decoded[6:].strip()
                            
                            if payload == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break
                            
                            try:
                                # Parse and accumulate content for logging
                                chunk_data = json.loads(payload)
                                
                                # Check if chunk_data is valid
                                if chunk_data is not None and isinstance(chunk_data, dict):
                                    choices = chunk_data.get("choices", [])
                                    if choices and len(choices) > 0:
                                        delta = choices[0].get("delta", {})
                                        if delta:
                                            content = delta.get("content", "")
                                            if content:
                                                accumulated_content += content
                                
                                # Forward the chunk as-is
                                yield f"data: {payload}\n\n"
                                
                            except json.JSONDecodeError:
                                response_data = {"error": {"message": "Invalid JSON response."}}
                                logging.warning(f"BACKEND [ID: {request_id}]: Invalid JSON in stream.")
                                continue
        
        # Log successful completion
        input_tokens = estimateTokens(json.dumps(openai_req))
        output_tokens = estimateTokens(accumulated_content)
        log_request_end(request_id, True, input_tokens, output_tokens, response_content=accumulated_content, api_key=api_key)
        
    except Exception as e:
        logging.error(f"BACKEND [ID: {request_id}]: Stream error: {e}", exc_info=True)
        
        # Log error
        log_error(request_id, type(e).__name__, str(e), traceback.format_exc())
        log_request_end(request_id, False, 0, 0, str(e))
        
        # Return error in OpenAI format
        error_response = {
            "error": {
                "message": "Server side error. Please try again later or DM norenaboi.",
                "type": "server_error",
                "code": None
            }
        }
        yield f"data: {json.dumps(error_response)}\n\ndata: [DONE]\n\n"

async def make_backend_request_v2(
    request_id: str,
    openai_req: dict,
    model_name: str,
    api_key: str
) -> dict:
    """Make non-streaming request to backend for v2/v3 models"""
    start_time = time.time()

    # Determine backend based on model suffix
    if model_name.endswith('-v2'):
        backend_url = Config.V2_URL
        backend_token = Config.V2_TOKEN
        actual_model = model_name.replace('-v2', '')
    elif model_name.endswith('-v3'):
        backend_url = Config.V3_URL
        backend_token = Config.V3_TOKEN
        actual_model = model_name.replace('-v3', '')
    else:
        error_response = {
            "error": {
                "message": "Error 404: This endpoint only supports v2 and v3 models.",
                "type": "server_error",
                "code": 404
            }
        }
        raise HTTPException(status_code=404, detail=error_response)

    backend_url = backend_url + "/v1/chat/completions"

    try:
        async with aiohttp.ClientSession() as session:
            # Prepare the request
            data = {
                "model": actual_model,
                "stream": False,
                "messages": openai_req.get("messages", []),
                "max_tokens": openai_req.get("max_tokens")
            }

            BACKEND_HEADERS = {
                "Authorization": f"Bearer {backend_token}",
                "Content-Type": "application/json"
            }
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            async with session.post(
                backend_url,
                headers=BACKEND_HEADERS,
                json=data
            ) as response:
                # Get response text first, then parse as JSON
                response_text = await response.text()
                
                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError:
                    # If JSON parsing fails, wrap the text response
                    response_data = {"error": {"message": "Invalid JSON response."}}
                
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail=response_data.get("Error: Backend JSON/Parameter error. Try disabling prefills or changing presets.")
                    )
                
                # Log successful completion
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                input_tokens = estimateTokens(json.dumps(openai_req))
                output_tokens = estimateTokens(content)
                log_request_end(request_id, True, input_tokens, output_tokens, content, api_key)
                
                return response_data
                
    except Exception as e:
        # Log error
        log_error(request_id, type(e).__name__, str(e), traceback.format_exc())
        log_request_end(request_id, False, 0, 0, str(e))
        
        raise

# Simple token estimation function
def estimateTokens(text: str) -> int:
    if not text:
        return 0
    return len(str(text)) // 4

#############################################
################# ENDPOINTS #################
#############################################

@app.get("/api/stats/summary")
async def get_stats_summary():
    current_time = time.time()
    day_ago = current_time - 86400
    
    logs = log_manager.read_request_logs(limit=10000)
    recent_24h_logs = [log for log in logs if log.get('timestamp', 0) > day_ago]

    total_requests = len(logs)
    daily_requests = len(recent_24h_logs)

    successful = sum(1 for log in recent_24h_logs if log.get('status') == 'success')
    failed = total_requests - successful
    
    total_input_tokens = sum(log.get('input_tokens', 0) for log in logs)
    total_output_tokens = sum(log.get('output_tokens', 0) for log in logs)

    daily_input_tokens = sum(log.get('input_tokens', 0) for log in recent_24h_logs)
    daily_output_tokens = sum(log.get('output_tokens', 0) for log in recent_24h_logs)
    
    durations = [log.get('duration', 0) for log in recent_24h_logs if log.get('duration', 0) > 0]
    avg_duration = sum(durations) / len(durations) if durations else 0

    all_api_keys = list(api_key_manager.valid_keys)

    
    return {
        "total_requests": total_requests,
        "daily_requests": daily_requests,
        "successful": successful,
        "failed": failed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "daily_input_tokens": daily_input_tokens,
        "daily_output_tokens": daily_output_tokens,
        "avg_duration": avg_duration,
        "success_rate": (successful / total_requests * 100) if total_requests > 0 else 0,
        "uptime": time.time() - startup_time,
        "total_api_keys": len(all_api_keys)
    }

# Endpoint to verify and get usage without authentication header
@app.post("/api/check-usage")
async def check_usage(request: Request, api_key: str = Depends(verify_api_key)):
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required")
    
    stats = api_key_manager.get_usage_stats(api_key)
    
    if not stats['is_valid']:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {
        "usage": stats
    }

@app.get("/v1/models")
async def get_models_v1():
    """Lists available models excluding v2 and v3 models in an OpenAI-compatible format."""
      
    models_data = []
    try:
        with open("allowed_models.txt", "r") as f:
            for line in f:
                model_name = line.strip()
                # Exclude models with -v2 or -v3 suffix
                if model_name and not model_name.startswith('#'):
                    if not (model_name.endswith('-v2') or model_name.endswith('-v3')):
                        models_data.append({
                            "id": model_name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "norenaboi",
                            "type": "chat"  # Default to chat type
                        })
    except FileNotFoundError:
        # If no file exists, return models from registry (excluding v2/v3)
        for model_name, model_info in MODEL_REGISTRY.items():
            if not (model_name.endswith('-v2') or model_name.endswith('-v3')):
                models_data.append({
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "norenaboi",
                    "type": model_info.get("type", "chat")
                })
      
    return {
        "object": "list",
        "data": models_data
    }

@app.get("/v2/models")
async def get_models_v2():
    """Lists available v2 and v3 models only in an OpenAI-compatible format."""
      
    models_data = []
    try:
        with open("allowed_models.txt", "r") as f:
            for line in f:
                model_name = line.strip()
                # Only include models with -v2 or -v3 suffix
                if model_name and not model_name.startswith('#'):
                    if model_name.endswith('-v2') or model_name.endswith('-v3'):
                        models_data.append({
                            "id": model_name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "norenaboi",
                            "type": "chat"
                        })
    except FileNotFoundError:
        # If no file exists, return v2/v3 models from registry
        for model_name, model_info in MODEL_REGISTRY.items():
            if model_name.endswith('-v2') or model_name.endswith('-v3'):
                models_data.append({
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "norenaboi",
                    "type": model_info.get("type", "chat")
                })
      
    return {
        "object": "list",
        "data": models_data
    }

# -------------------------------------------
# ----------------- USAGE -------------------  
# -------------------------------------------

@app.get("/usage", response_class=HTMLResponse)
async def user_usage():
    html_file_path = Path(__file__).parent / "html/user_usage.html"
    
    if not html_file_path.exists():
        return HTMLResponse(
            content="<h1></h1><p>YOU FORGOT THE HTML STUPID</p>",
            status_code=404
        )
    
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)

@app.get("/admin/login", response_class=HTMLResponse)
async def login_admin_usage():
    html_file_path = Path(__file__).parent / "html/login_admin.html"
    
    if not html_file_path.exists():
        return HTMLResponse(
            content="<h1></h1><p>YOU FORGOT THE HTML STUPID</p>",
            status_code=404
        )
    
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)

@app.get("/admin/usage", response_class=HTMLResponse)
async def admin_usage():
    html_file_path = Path(__file__).parent / "html/admin_usage.html"
    
    if not html_file_path.exists():
        return HTMLResponse(
            content="<h1></h1><p>YOU FORGOT THE HTML STUPID</p>",
            status_code=404
        )
    
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)

# API endpoint for dashboard data
@app.get("/admin/usage-data")
async def get_admin_usage_data(api_key: str = Depends(verify_master_key)):
    """Get usage data for admin dashboard"""
    
    # Get all API keys and their usage
    all_api_keys = list(api_key_manager.valid_keys)
    dashboard_data = []
    
    for api_key in all_api_keys:
        stats = api_key_manager.get_usage_stats(api_key)
        dashboard_data.append({
            "name": api_key_manager.get_key_name(api_key),
            "api_key": api_key[:5] + "..." if len(api_key) > 5 else api_key,
            "total_requests": stats["total_requests"],
            "daily_requests": stats.get("daily_requests", 0),
            "total_input_tokens": stats.get("total_input_tokens", 0),
            "total_output_tokens": stats.get("total_output_tokens", 0),
            "daily_input_tokens": stats.get("daily_input_tokens", 0),
            "daily_output_tokens": stats.get("daily_output_tokens", 0),
        })
    
    # Sort by total requests
    dashboard_data.sort(key=lambda x: x["daily_requests"], reverse=True)

    # Get recent logs
    logs = log_manager.read_request_logs(limit=100)
    
    # Filter for completed requests and format them
    formatted_logs = []
    for log in logs:
        if log.get('type') == 'request_end' and log.get('status') == 'success':
            api_key = log.get('api_key', 'Unknown')
            formatted_logs.append({
                "timestamp": log.get('timestamp', 0),
                "request_id": log.get('request_id', ''),
                "name": api_key_manager.get_key_name(api_key) if api_key != 'Unknown' else 'Unknown',
                "api_key": api_key[:5] + "..." if len(api_key) > 5 else api_key,
                "model": log.get('model', 'Unknown'),
                "input_tokens": log.get('input_tokens', 0),
                "output_tokens": log.get('output_tokens', 0),
                "total_tokens": log.get('input_tokens', 0) + log.get('output_tokens', 0),
                "duration": log.get('duration', 0)
            })
    
    # Sort by timestamp (most recent first) and take top 5
    formatted_logs.sort(key=lambda x: x['timestamp'], reverse=True)
    recent_logs_data = formatted_logs[:50]

    # Calculate totals
    totals = {
        "total_api_keys": len(all_api_keys),
        "total_requests": sum(d["total_requests"] for d in dashboard_data),
        "daily_requests": sum(d["daily_requests"] for d in dashboard_data),
        "total_input_tokens": sum(d["total_input_tokens"] for d in dashboard_data),
        "total_output_tokens": sum(d["total_output_tokens"] for d in dashboard_data),
        "daily_input_tokens": sum(d["daily_input_tokens"] for d in dashboard_data),
        "daily_output_tokens": sum(d["daily_output_tokens"] for d in dashboard_data),
    }
    
    return {
        "summary": totals,
        "api_keys": dashboard_data,
        "recent_logs": recent_logs_data
    }

# -------------------------------------------
# ----------- API KEY MANAGER ---------------  
# -------------------------------------------

@app.get("/admin/manager", response_class=HTMLResponse)
async def admin_manager():
    html_file_path = Path(__file__).parent / "html/admin_manager.html"
    
    if not html_file_path.exists():
        return HTMLResponse(
            content="<h1></h1><p>YOU FORGOT THE HTML STUPID</p>",
            status_code=404
        )
    
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)

@app.get("/admin/keys")
async def get_api_keys(
    authorized: bool = Depends(verify_master_key)
):
    """Get all API keys"""
    try:
        keys = api_key_manager.get_keys()
        return {"keys": keys}
    except Exception as e:
        logging.error(f"Error loading keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/keys/add")
async def add_api_key(
    request: dict,
    authorized: bool = Depends(verify_master_key)
):
    """Add a new API key"""
    try:
        api_key = request.get('api_key', '').strip()
        name = request.get('name', '').strip()
        
        if not api_key or not name:
            raise HTTPException(status_code=400, detail="API key and name are required")
        
        # Check if key already exists
        if api_key in api_key_manager.valid_keys:
            raise HTTPException(status_code=400, detail="API key already exists")
        
        # Add new key (correct signature)
        api_key_manager.add_key(api_key, name)
        logging.info(f"Added new API key: {name}")
        
        return {"message": "API key added successfully", "key": {"api_key": api_key, "name": name}}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error adding key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update API key name
@app.put("/admin/keys/{api_key}")
async def update_api_key(
    request: dict,
    authorized: bool = Depends(verify_master_key)
):
    """Update an API key's name"""
    try:
        new_name = request.get('name', '').strip()
        api_key = request.get('api_key', '').strip()

        if not new_name:
            raise HTTPException(status_code=400, detail="Name is required")
        
        # Find and update the key
        api_key_manager.set_key_name(api_key, new_name)
        
        logging.info(f"Updated API key: {api_key} -> {new_name}")
        
        return {"message": "API key updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Delete API key
@app.delete("/admin/keys/{api_key}")
async def delete_api_key(
    api_key,
    authorized: bool = Depends(verify_master_key)
):
    """Delete an API key"""
    try:
        keys = keys = api_key_manager.get_keys()
        
        # Check if key exists
        if not any(k['api_key'] == api_key for k in keys):
            raise HTTPException(status_code=400, detail="API doesn't exist")
        
        api_key_manager.remove_key(api_key)
        logging.info(f"Deleted API key: {api_key}")
        
        return {"message": "API key deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

print("\n" + "="*60)
print("Kiru Proxy")
print("="*60 + "\n")
if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
