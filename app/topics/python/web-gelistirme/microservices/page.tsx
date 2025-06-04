import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Microservices Architecture | Python Web Geliştirme | Kodleon',
  description: 'Python web uygulamaları için microservices architecture. Service design, communication patterns, deployment ve monitoring.',
};

const content = `
# Microservices Architecture

Python web uygulamalarında microservices mimarisi ve distributed systems konularını öğreneceğiz.

## Service Design ve Domain-Driven Design

\`\`\`python
# services/order/domain/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

@dataclass
class OrderItem:
    product_id: UUID
    quantity: int
    price: float
    
    @property
    def total(self) -> float:
        return self.quantity * self.price

@dataclass
class Order:
    id: UUID
    customer_id: UUID
    items: List[OrderItem]
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    @classmethod
    def create(cls, customer_id: UUID, items: List[OrderItem]) -> 'Order':
        return cls(
            id=uuid4(),
            customer_id=customer_id,
            items=items,
            status="pending",
            created_at=datetime.utcnow()
        )
    
    @property
    def total_amount(self) -> float:
        return sum(item.total for item in self.items)
    
    def update_status(self, new_status: str) -> None:
        self.status = new_status
        self.updated_at = datetime.utcnow()

# services/order/application/commands.py
from dataclasses import dataclass
from typing import List
from uuid import UUID

@dataclass
class CreateOrderCommand:
    customer_id: UUID
    items: List[dict]

@dataclass
class UpdateOrderStatusCommand:
    order_id: UUID
    new_status: str

# services/order/application/handlers.py
from typing import List
from domain.models import Order, OrderItem
from application.commands import CreateOrderCommand, UpdateOrderStatusCommand
from infrastructure.repositories import OrderRepository
from infrastructure.message_bus import MessageBus

class OrderCommandHandler:
    def __init__(
        self,
        repository: OrderRepository,
        message_bus: MessageBus
    ):
        self.repository = repository
        self.message_bus = message_bus
    
    async def handle_create_order(
        self,
        command: CreateOrderCommand
    ) -> Order:
        # Create order items
        items = [
            OrderItem(
                product_id=item["product_id"],
                quantity=item["quantity"],
                price=item["price"]
            )
            for item in command.items
        ]
        
        # Create order
        order = Order.create(
            customer_id=command.customer_id,
            items=items
        )
        
        # Save order
        await self.repository.save(order)
        
        # Publish event
        await self.message_bus.publish(
            "order_created",
            {
                "order_id": str(order.id),
                "customer_id": str(order.customer_id),
                "total_amount": order.total_amount
            }
        )
        
        return order
    
    async def handle_update_status(
        self,
        command: UpdateOrderStatusCommand
    ) -> Order:
        # Get order
        order = await self.repository.get_by_id(command.order_id)
        
        # Update status
        order.update_status(command.new_status)
        
        # Save changes
        await self.repository.save(order)
        
        # Publish event
        await self.message_bus.publish(
            "order_status_updated",
            {
                "order_id": str(order.id),
                "new_status": order.status
            }
        )
        
        return order
\`\`\`

## Event-Driven Architecture ve RabbitMQ

\`\`\`python
# infrastructure/message_bus.py
import json
import aio_pika
from typing import Any, Dict
import logging

class MessageBus:
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.connection = None
        self.channel = None
    
    async def connect(self):
        if not self.connection:
            self.connection = await aio_pika.connect_robust(
                self.connection_url
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)
    
    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any]
    ):
        await self.connect()
        
        # Create message
        message = aio_pika.Message(
            body=json.dumps({
                "type": event_type,
                "data": data
            }).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        # Declare exchange
        exchange = await self.channel.declare_exchange(
            "events",
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        # Publish message
        await exchange.publish(
            message,
            routing_key=event_type
        )
        
        logging.info(f"Published event: {event_type}")
    
    async def subscribe(
        self,
        event_type: str,
        callback: callable
    ):
        await self.connect()
        
        # Declare exchange
        exchange = await self.channel.declare_exchange(
            "events",
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        # Declare queue
        queue = await self.channel.declare_queue(
            f"{event_type}_queue",
            durable=True
        )
        
        # Bind queue to exchange
        await queue.bind(
            exchange,
            routing_key=event_type
        )
        
        # Start consuming
        async def process_message(message):
            async with message.process():
                try:
                    body = json.loads(message.body.decode())
                    await callback(body["data"])
                except Exception as e:
                    logging.error(f"Error processing message: {e}")
                    await message.reject(requeue=True)
        
        await queue.consume(process_message)
        logging.info(f"Subscribed to event: {event_type}")

# services/payment/handlers.py
from infrastructure.message_bus import MessageBus
from services.payment.service import PaymentService

async def handle_order_created(data: dict):
    order_id = data["order_id"]
    amount = data["total_amount"]
    
    # Process payment
    payment_service = PaymentService()
    payment_result = await payment_service.process_payment(
        order_id=order_id,
        amount=amount
    )
    
    # Publish result
    message_bus = MessageBus("amqp://guest:guest@localhost:5672/")
    await message_bus.publish(
        "payment_processed",
        {
            "order_id": order_id,
            "status": payment_result.status
        }
    )
\`\`\`

## API Gateway ve Service Discovery

\`\`\`python
# gateway/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import consul
from typing import Dict, List
import random

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ServiceRegistry:
    def __init__(self):
        self.consul = consul.Consul()
        self.cache: Dict[str, List[str]] = {}
    
    async def get_service_url(self, service_name: str) -> str:
        # Check cache first
        if service_name in self.cache:
            return random.choice(self.cache[service_name])
        
        # Get service from Consul
        _, services = self.consul.health.service(
            service_name,
            passing=True
        )
        
        if not services:
            raise HTTPException(
                status_code=503,
                detail=f"Service {service_name} not available"
            )
        
        # Update cache
        self.cache[service_name] = [
            f"http://{service['Service']['Address']}:{service['Service']['Port']}"
            for service in services
        ]
        
        return random.choice(self.cache[service_name])

registry = ServiceRegistry()

@app.get("/api/orders/{order_id}")
async def get_order(order_id: str):
    # Get order service URL
    service_url = await registry.get_service_url("order-service")
    
    # Forward request
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{service_url}/orders/{order_id}"
        )
        return response.json()

@app.post("/api/orders")
async def create_order(order: dict):
    # Get order service URL
    service_url = await registry.get_service_url("order-service")
    
    # Forward request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{service_url}/orders",
            json=order
        )
        return response.json()

# Rate limiting middleware
from fastapi import Request
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time
            for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        # Check if under limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        # Add new request
        self.requests[client_ip].append(now)
        return True

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    limiter = RateLimiter()
    client_ip = request.client.host
    
    if not limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests"
        )
    
    return await call_next(request)

# Circuit breaker middleware
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = None
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def record_success(self):
        self.failures = 0
        self.state = CircuitState.CLOSED
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if (datetime.utcnow() - self.last_failure_time) > timedelta(seconds=self.reset_timeout):
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        
        return True

circuit_breakers: Dict[str, CircuitBreaker] = {}

@app.middleware("http")
async def circuit_breaker_middleware(request: Request, call_next):
    service_name = request.url.path.split("/")[2]
    
    if service_name not in circuit_breakers:
        circuit_breakers[service_name] = CircuitBreaker()
    
    breaker = circuit_breakers[service_name]
    
    if not breaker.can_execute():
        raise HTTPException(
            status_code=503,
            detail=f"Service {service_name} is not available"
        )
    
    try:
        response = await call_next(request)
        breaker.record_success()
        return response
    except Exception as e:
        breaker.record_failure()
        raise
\`\`\`

## Distributed Tracing ve Monitoring

\`\`\`python
# monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

def setup_tracing(app: FastAPI, service_name: str):
    # Create tracer provider
    provider = TracerProvider(
        resource=Resource.create({
            "service.name": service_name
        })
    )
    
    # Create Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    # Add span processor
    provider.add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    
    # Set tracer provider
    trace.set_tracer_provider(provider)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument aio-pika
    AioPikaInstrumentor().instrument()
    
    # Instrument SQLAlchemy
    SQLAlchemyInstrumentor().instrument()

# monitoring/metrics.py
from prometheus_client import Counter, Histogram
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

def setup_metrics(app: FastAPI):
    # Initialize Prometheus metrics
    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total count of HTTP requests",
        ["method", "endpoint", "status"]
    )
    
    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"]
    )
    
    # Initialize instrumentator
    instrumentator = Instrumentator().instrument(app)
    
    # Add custom metrics
    @instrumentator.counter(
        "http_requests_total",
        "Total count of HTTP requests",
        labels={"method": True, "endpoint": True, "status": True}
    )
    def count_requests(metric, info):
        metric.inc({
            "method": info.request.method,
            "endpoint": info.request.url.path,
            "status": info.response.status_code
        })
    
    @instrumentator.histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        labels={"method": True, "endpoint": True}
    )
    def track_latency(metric, info):
        metric.observe(
            info.duration,
            {
                "method": info.request.method,
                "endpoint": info.request.url.path
            }
        )
    
    # Expose metrics endpoint
    instrumentator.expose(app, include_in_schema=True)

# Example usage in service
from fastapi import FastAPI
from monitoring.tracing import setup_tracing
from monitoring.metrics import setup_metrics

app = FastAPI()

# Set up monitoring
setup_tracing(app, "order-service")
setup_metrics(app)
\`\`\`

## Alıştırmalar

1. **Service Design**
   - Domain model implementasyonu yapın
   - Event-driven architecture kurun
   - Command handlers ekleyin
   - Service boundaries belirleyin

2. **Communication**
   - RabbitMQ integration yapın
   - API Gateway implementasyonu yapın
   - Service discovery ekleyin
   - Circuit breaker pattern uygulayın

3. **Deployment**
   - Docker Compose ile local deployment yapın
   - Kubernetes manifests yazın
   - Service mesh ekleyin
   - Auto-scaling konfigürasyonu yapın

## Sonraki Adımlar

1. [Performance Optimization](/topics/python/web-gelistirme/performance)
2. [Testing and Quality Assurance](/topics/python/web-gelistirme/testing)
3. [Cloud Deployment](/topics/python/web-gelistirme/deployment)

## Faydalı Kaynaklar

- [Domain-Driven Design](https://www.domainlanguage.com/)
- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
`;

const learningPath = [
  {
    title: '1. Service Design',
    description: 'Domain-driven design ve service architecture öğrenin.',
    topics: [
      'Domain modeling',
      'Bounded contexts',
      'Event sourcing',
      'CQRS pattern',
      'Service boundaries',
    ],
    icon: '🏗️',
    href: '/topics/python/web-gelistirme/microservices/design'
  },
  {
    title: '2. Communication',
    description: 'Service communication patterns öğrenin.',
    topics: [
      'Event-driven arch',
      'Message queues',
      'API Gateway',
      'Service discovery',
      'Circuit breaker',
    ],
    icon: '🔄',
    href: '/topics/python/web-gelistirme/microservices/communication'
  },
  {
    title: '3. Deployment',
    description: 'Microservices deployment strategies öğrenin.',
    topics: [
      'Container orchestration',
      'Service mesh',
      'Config management',
      'Auto-scaling',
      'Load balancing',
    ],
    icon: '🚀',
    href: '/topics/python/web-gelistirme/microservices/deployment'
  },
  {
    title: '4. Observability',
    description: 'Monitoring ve tracing öğrenin.',
    topics: [
      'Distributed tracing',
      'Metrics collection',
      'Log aggregation',
      'Health checks',
      'Alerting',
    ],
    icon: '📊',
    href: '/topics/python/web-gelistirme/microservices/observability'
  }
];

export default function MicroservicesPage() {
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