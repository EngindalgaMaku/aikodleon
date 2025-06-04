import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Performance Optimization | Python Web Geliştirme | Kodleon',
  description: 'Python web uygulamaları için performance optimization. Caching, database optimization, async operations ve profiling.',
};

const content = `
# Performance Optimization

Python web uygulamalarında performans optimizasyonu ve ölçeklendirme tekniklerini öğreneceğiz.

## Caching Strategies

\`\`\`python
# caching/redis_cache.py
from typing import Any, Optional
from datetime import timedelta
import json
import redis
from functools import wraps

class RedisCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> None:
        """Set value in cache with optional expiration"""
        serialized = json.dumps(value)
        if expire:
            self.redis.setex(key, expire, serialized)
        else:
            self.redis.set(key, serialized)
    
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        self.redis.delete(key)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.redis.flushdb()

def cache_decorator(
    cache: RedisCache,
    prefix: str = "",
    expire: Optional[int] = None
):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in kwargs.items())
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, expire)
            return result
        
        return wrapper
    return decorator

# Example usage in FastAPI endpoints
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List

app = FastAPI()
cache = RedisCache("redis://localhost:6379/0")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/products")
@cache_decorator(cache, prefix="products", expire=300)
async def get_products(
    db: Session = Depends(get_db),
    category: Optional[str] = None
) -> List[dict]:
    query = db.query(Product)
    if category:
        query = query.filter(Product.category == category)
    products = query.all()
    return [product.to_dict() for product in products]

@app.get("/products/{product_id}")
@cache_decorator(cache, prefix="product", expire=300)
async def get_product(
    product_id: int,
    db: Session = Depends(get_db)
) -> dict:
    product = db.query(Product).get(product_id)
    return product.to_dict()
\`\`\`

## Database Optimization

\`\`\`python
# database/optimization.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import time
import logging

class DatabaseOptimizer:
    def __init__(self, database_url: str):
        # Configure connection pool
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800
        )
        
        self.Session = sessionmaker(bind=self.engine)
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    def analyze_query(self, query: str) -> dict:
        """Analyze query performance using EXPLAIN ANALYZE"""
        with self.session_scope() as session:
            result = session.execute(
                text(f"EXPLAIN ANALYZE {query}")
            )
            return result.fetchall()
    
    def create_index(
        self,
        table: str,
        columns: list,
        index_name: Optional[str] = None
    ):
        """Create an index on specified columns"""
        index_name = index_name or f"idx_{table}_{'_'.join(columns)}"
        with self.session_scope() as session:
            session.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {index_name} "
                    f"ON {table} ({','.join(columns)})"
                )
            )
    
    def bulk_insert(self, table: str, records: List[dict]):
        """Efficiently insert multiple records"""
        with self.session_scope() as session:
            session.execute(
                text(f"INSERT INTO {table} VALUES :values"),
                [{"values": record} for record in records]
            )
    
    def partition_table(
        self,
        table: str,
        partition_key: str,
        partition_type: str = "RANGE"
    ):
        """Set up table partitioning"""
        with self.session_scope() as session:
            session.execute(
                text(
                    f"ALTER TABLE {table} "
                    f"PARTITION BY {partition_type} ({partition_key})"
                )
            )

# Example usage in repository
from sqlalchemy import select
from sqlalchemy.orm import joinedload

class ProductRepository:
    def __init__(self, session: Session):
        self.session = session
    
    def get_products_with_category(self) -> List[Product]:
        # Use joinedload to avoid N+1 queries
        query = (
            select(Product)
            .options(joinedload(Product.category))
            .order_by(Product.name)
        )
        return self.session.execute(query).scalars().all()
    
    def get_products_by_price_range(
        self,
        min_price: float,
        max_price: float
    ) -> List[Product]:
        # Use index on price column
        query = (
            select(Product)
            .where(
                Product.price.between(min_price, max_price)
            )
            .order_by(Product.price)
        )
        return self.session.execute(query).scalars().all()
    
    def bulk_create_products(self, products: List[dict]):
        # Use bulk_insert_mappings for efficiency
        self.session.bulk_insert_mappings(
            Product,
            products
        )
        self.session.commit()
\`\`\`

## Async Operations

\`\`\`python
# async/background_tasks.py
from celery import Celery
from typing import List
import asyncio
from concurrent.futures import ProcessPoolExecutor
import logging

# Configure Celery
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task
def process_image(image_path: str) -> str:
    """CPU-intensive image processing task"""
    # Image processing logic here
    return processed_image_path

@celery_app.task
def send_bulk_emails(emails: List[dict]) -> dict:
    """I/O-bound email sending task"""
    results = {
        'success': 0,
        'failed': 0
    }
    
    for email in emails:
        try:
            # Send email logic here
            results['success'] += 1
        except Exception as e:
            results['failed'] += 1
            logging.error(f"Failed to send email: {e}")
    
    return results

# Async task processing
class TaskProcessor:
    def __init__(self, max_workers: int = None):
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
    
    async def process_cpu_bound(self, func, *args):
        """Process CPU-bound task in separate process"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.process_pool,
            func,
            *args
        )
    
    async def process_io_bound(self, tasks: List[callable]):
        """Process I/O-bound tasks concurrently"""
        return await asyncio.gather(*tasks)

# Example usage in FastAPI endpoints
from fastapi import BackgroundTasks

@app.post("/upload-image")
async def upload_image(
    image: UploadFile,
    background_tasks: BackgroundTasks
):
    # Save image
    image_path = f"uploads/{image.filename}"
    with open(image_path, "wb") as buffer:
        content = await image.read()
        buffer.write(content)
    
    # Process image in background
    task = process_image.delay(image_path)
    
    return {
        "message": "Image upload started",
        "task_id": task.id
    }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None
    }

# Parallel processing example
processor = TaskProcessor()

@app.post("/process-images")
async def process_multiple_images(images: List[UploadFile]):
    # Save images
    image_paths = []
    for image in images:
        path = f"uploads/{image.filename}"
        with open(path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        image_paths.append(path)
    
    # Process images in parallel
    tasks = [
        processor.process_cpu_bound(
            process_image,
            path
        )
        for path in image_paths
    ]
    
    results = await asyncio.gather(*tasks)
    return {"processed_images": results}
\`\`\`

## Performance Profiling

\`\`\`python
# profiling/profiler.py
import cProfile
import pstats
import io
from functools import wraps
from time import time
from typing import Callable, Optional
import tracemalloc
import logging

class Profiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.profiler = cProfile.Profile()
    
    def profile(
        self,
        output_file: Optional[str] = None
    ) -> Callable:
        """Decorator for profiling functions"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Start profiling
                self.profiler.enable()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.profiler.disable()
                    
                    # Print stats
                    s = io.StringIO()
                    stats = pstats.Stats(
                        self.profiler,
                        stream=s
                    ).sort_stats('cumulative')
                    
                    stats.print_stats()
                    
                    # Save to file if specified
                    if output_file:
                        stats.dump_stats(output_file)
                    
                    logging.info(f"Profile results for {func.__name__}:\\n{s.getvalue()}")
            
            return wrapper
        return decorator

class MemoryProfiler:
    def __init__(self):
        self.snapshot = None
    
    def start(self):
        """Start memory profiling"""
        tracemalloc.start()
    
    def take_snapshot(self):
        """Take memory snapshot"""
        self.snapshot = tracemalloc.take_snapshot()
    
    def compare_snapshot(self):
        """Compare with previous snapshot"""
        if not self.snapshot:
            return
        
        current = tracemalloc.take_snapshot()
        stats = current.compare_to(
            self.snapshot,
            'lineno'
        )
        
        for stat in stats[:10]:
            logging.info(str(stat))

def time_it(func: Callable) -> Callable:
    """Decorator for timing function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        
        logging.info(
            f"{func.__name__} took {end_time - start_time:.2f} seconds"
        )
        
        return result
    return wrapper

# Example usage
profiler = Profiler()
memory_profiler = MemoryProfiler()

@app.get("/expensive-operation")
@profiler.profile(output_file="profile_results.prof")
@time_it
async def expensive_operation():
    # Start memory profiling
    memory_profiler.start()
    memory_profiler.take_snapshot()
    
    # Perform operation
    result = await perform_expensive_calculation()
    
    # Compare memory usage
    memory_profiler.compare_snapshot()
    
    return result

# Load testing with locust
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 2.5)
    
    @task(1)
    def get_products(self):
        self.client.get("/products")
    
    @task(2)
    def view_product(self):
        product_id = random.randint(1, 100)
        self.client.get(f"/products/{product_id}")
    
    @task(3)
    def search_products(self):
        self.client.get(
            "/products/search",
            params={"q": "test"}
        )
\`\`\`

## Alıştırmalar

1. **Caching**
   - Redis cache implementasyonu yapın
   - Cache decorators ekleyin
   - Cache invalidation uygulayın
   - Multi-level caching kurun

2. **Database**
   - Query optimization yapın
   - Connection pooling ekleyin
   - Bulk operations uygulayın
   - Indexing stratejisi belirleyin

3. **Async Operations**
   - Background tasks ekleyin
   - Process pools kurun
   - Task queues implementasyonu yapın
   - Concurrent operations yazın

## Sonraki Adımlar

1. [Testing and Quality Assurance](/topics/python/web-gelistirme/testing)
2. [Cloud Deployment](/topics/python/web-gelistirme/deployment)
3. [Microservices Architecture](/topics/python/web-gelistirme/microservices)

## Faydalı Kaynaklar

- [Redis Documentation](https://redis.io/documentation)
- [SQLAlchemy Performance](https://docs.sqlalchemy.org/en/14/faq/performance.html)
- [Celery Documentation](https://docs.celeryproject.org/)
- [Python Profilers](https://docs.python.org/3/library/profile.html)
`;

const learningPath = [
  {
    title: '1. Caching',
    description: 'Caching strategies ve implementation öğrenin.',
    topics: [
      'Redis caching',
      'Cache decorators',
      'Cache invalidation',
      'Multi-level cache',
      'Cache patterns',
    ],
    icon: '🚀',
    href: '/topics/python/web-gelistirme/performance/caching'
  },
  {
    title: '2. Database',
    description: 'Database optimization teknikleri öğrenin.',
    topics: [
      'Query optimization',
      'Connection pooling',
      'Bulk operations',
      'Indexing strategy',
      'Query analysis',
    ],
    icon: '💾',
    href: '/topics/python/web-gelistirme/performance/database'
  },
  {
    title: '3. Async',
    description: 'Async operations ve concurrency öğrenin.',
    topics: [
      'Background tasks',
      'Process pools',
      'Task queues',
      'Concurrent ops',
      'Async patterns',
    ],
    icon: '⚡',
    href: '/topics/python/web-gelistirme/performance/async'
  },
  {
    title: '4. Profiling',
    description: 'Performance profiling ve monitoring öğrenin.',
    topics: [
      'CPU profiling',
      'Memory profiling',
      'Load testing',
      'Bottleneck analysis',
      'Performance metrics',
    ],
    icon: '📊',
    href: '/topics/python/web-gelistirme/performance/profiling'
  }
];

export default function PerformancePage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Button variant="ghost" asChild className="mb-6">
            <Link href="/topics/python/web-gelistirme" className="flex items-center">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Web Geliştirme
            </Link>
          </Button>
        </div>

        <div className="prose prose-lg dark:prose-invert">
          <MarkdownContent content={content} />
        </div>

        <h2 className="text-2xl font-bold mb-6">Öğrenme Yolu</h2>
        
        <div className="grid gap-6 md:grid-cols-2">
          {learningPath.map((topic, index) => (
            <Card key={index} className="p-6 hover:bg-accent transition-colors cursor-pointer">
              <Link href={topic.href}>
                <div className="flex items-start space-x-4">
                  <div className="text-4xl">{topic.icon}</div>
                  <div className="space-y-2">
                    <h3 className="font-bold">{topic.title}</h3>
                    <p className="text-sm text-muted-foreground">{topic.description}</p>
                    <ul className="text-sm space-y-1 list-disc list-inside text-muted-foreground">
                      {topic.topics.map((t, i) => (
                        <li key={i}>{t}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Link>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Kodleon | Python Eğitim Platformu</p>
        </div>
      </div>
    </div>
  );
} 