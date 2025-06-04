import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Cloud Deployment | Python Web Geliştirme | Kodleon',
  description: 'Python web uygulamaları için cloud deployment. Containerization, cloud platforms, CI/CD ve infrastructure as code.',
};

const content = `
# Cloud Deployment

Python web uygulamalarını cloud platformlarda deploy etmeyi öğreneceğiz.

## Containerization ve Docker

\`\`\`dockerfile
# Dockerfile
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/app
      - REDIS_URL=redis://cache:6379/0
    depends_on:
      - db
      - cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=app
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d app"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  cache:
    image: redis:6
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
\`\`\`

## AWS Infrastructure with CDK

\`\`\`python
# infrastructure/app.py
from aws_cdk import (
    App,
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_rds as rds,
    aws_elasticache as elasticache,
    aws_certificatemanager as acm,
    aws_route53 as route53,
)
from constructs import Construct

class WebAppStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # VPC
        vpc = ec2.Vpc(
            self, "WebAppVPC",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24
                )
            ]
        )

        # Security Groups
        app_sg = ec2.SecurityGroup(
            self, "AppSecurityGroup",
            vpc=vpc,
            description="Security group for web application"
        )

        db_sg = ec2.SecurityGroup(
            self, "DBSecurityGroup",
            vpc=vpc,
            description="Security group for database"
        )

        cache_sg = ec2.SecurityGroup(
            self, "CacheSecurityGroup",
            vpc=vpc,
            description="Security group for Redis cache"
        )

        # Allow inbound traffic
        app_sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(80),
            "Allow HTTP traffic"
        )
        
        db_sg.add_ingress_rule(
            app_sg,
            ec2.Port.tcp(5432),
            "Allow PostgreSQL access from app"
        )
        
        cache_sg.add_ingress_rule(
            app_sg,
            ec2.Port.tcp(6379),
            "Allow Redis access from app"
        )

        # RDS Instance
        database = rds.DatabaseInstance(
            self, "Database",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_13
            ),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_ISOLATED
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE3,
                ec2.InstanceSize.MEDIUM
            ),
            security_groups=[db_sg],
            removal_policy=RemovalPolicy.DESTROY,
            deletion_protection=False
        )

        # Redis Cache
        redis_subnet_group = elasticache.CfnSubnetGroup(
            self, "RedisCacheSubnetGroup",
            subnet_ids=vpc.select_subnets(
                subnet_type=ec2.SubnetType.PRIVATE_ISOLATED
            ).subnet_ids,
            description="Subnet group for Redis cache"
        )

        redis_cluster = elasticache.CfnCacheCluster(
            self, "RedisCache",
            cache_node_type="cache.t3.micro",
            engine="redis",
            num_cache_nodes=1,
            vpc_security_group_ids=[cache_sg.security_group_id],
            cache_subnet_group_name=redis_subnet_group.ref
        )

        # ECS Cluster
        cluster = ecs.Cluster(
            self, "WebAppCluster",
            vpc=vpc
        )

        # Fargate Service
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "WebAppService",
            cluster=cluster,
            cpu=256,
            memory_limit_mib=512,
            desired_count=2,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset("."),
                container_port=8000,
                environment={
                    "DATABASE_URL": f"postgresql://{database.instance_endpoint.hostname}",
                    "REDIS_URL": f"redis://{redis_cluster.attr_redis_endpoint_address}"
                }
            ),
            public_load_balancer=True,
            security_groups=[app_sg]
        )

        # Auto Scaling
        scaling = fargate_service.service.auto_scale_task_count(
            max_capacity=4,
            min_capacity=2
        )
        
        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60)
        )

app = App()
WebAppStack(app, "WebAppStack")
app.synth()
\`\`\`

## GitHub Actions CI/CD

\`\`\`yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: webapp
  ECS_CLUSTER: WebAppCluster
  ECS_SERVICE: WebAppService
  CONTAINER_NAME: web

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:6
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://test:test@localhost:5432/test
        REDIS_URL: redis://localhost:6379/0
      run: |
        poetry run pytest --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml

  build-and-deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
    
    - name: Deploy to ECS
      uses: aws-actions/amazon-ecs-deploy-task-definition@v1
      with:
        task-definition: task-definition.json
        service: ${{ env.ECS_SERVICE }}
        cluster: ${{ env.ECS_CLUSTER }}
        wait-for-service-stability: true
        force-new-deployment: true
\`\`\`

## Monitoring ve Logging

\`\`\`python
# monitoring/cloudwatch.py
import boto3
import logging
from datetime import datetime, timedelta
from typing import Dict, List

class CloudWatchMetrics:
    def __init__(self, namespace: str):
        self.client = boto3.client('cloudwatch')
        self.namespace = namespace
    
    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        dimensions: List[Dict[str, str]]
    ):
        try:
            self.client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit,
                    'Dimensions': dimensions,
                    'Timestamp': datetime.utcnow()
                }]
            )
        except Exception as e:
            logging.error(f"Failed to put metric {metric_name}: {str(e)}")
    
    def get_metric_statistics(
        self,
        metric_name: str,
        dimensions: List[Dict[str, str]],
        start_time: datetime,
        end_time: datetime,
        period: int = 300,
        statistics: List[str] = ['Average']
    ):
        try:
            response = self.client.get_metric_statistics(
                Namespace=self.namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=statistics
            )
            return response['Datapoints']
        except Exception as e:
            logging.error(f"Failed to get metric statistics for {metric_name}: {str(e)}")
            return []

# monitoring/logger.py
import json
import logging
import sys
from typing import Any, Dict
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any]
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        
        if not log_record.get('timestamp'):
            log_record['timestamp'] = record.created
        
        if log_record.get('level'):
            log_record['level'] = record.levelname.lower()
        
        if not log_record.get('source'):
            log_record['source'] = record.name

def setup_logging(
    level: str = "INFO",
    service_name: str = "webapp"
) -> None:
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)
    
    # Create JSON handler
    json_handler = logging.StreamHandler(sys.stdout)
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    json_handler.setFormatter(formatter)
    logger.addHandler(json_handler)
    
    # Set default logging fields
    logging.getLogger().addFilter(
        lambda record: record.__setattr__('service', service_name)
    )

# Example usage in FastAPI middleware
from fastapi import FastAPI, Request
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import logging

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        # Calculate request duration
        duration = time.time() - start_time
        
        # Log request details
        logging.info(
            "Request processed",
            extra={
                'method': request.method,
                'path': request.url.path,
                'duration': duration,
                'status_code': response.status_code,
                'client_ip': request.client.host,
                'user_agent': request.headers.get('user-agent')
            }
        )
        
        return response

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

# Set up structured logging
setup_logging(level="INFO", service_name="webapp")

# Initialize CloudWatch metrics
metrics = CloudWatchMetrics(namespace="WebApp")

# Example metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Record request duration
    metrics.put_metric(
        metric_name="RequestDuration",
        value=duration,
        unit="Seconds",
        dimensions=[
            {'Name': 'Path', 'Value': request.url.path},
            {'Name': 'Method', 'Value': request.method}
        ]
    )
    
    return response
\`\`\`

## Alıştırmalar

1. **Containerization**
   - Multi-stage Dockerfile oluşturun
   - Docker Compose ile local environment kurun
   - Container health checks ekleyin
   - Resource limits ayarlayın

2. **Infrastructure**
   - AWS CDK ile VPC oluşturun
   - ECS Fargate service deploy edin
   - RDS ve ElastiCache ekleyin
   - Auto scaling ayarlayın

3. **CI/CD Pipeline**
   - GitHub Actions workflow yazın
   - Test automation ekleyin
   - Docker build ve push yapın
   - ECS deployment otomatize edin

## Sonraki Adımlar

1. [Microservices Architecture](/topics/python/web-gelistirme/microservices)
2. [Performance Optimization](/topics/python/web-gelistirme/performance)
3. [Testing and Quality Assurance](/topics/python/web-gelistirme/testing)

## Faydalı Kaynaklar

- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/latest/guide/home.html)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/intro.html)
`;

const learningPath = [
  {
    title: '1. Containerization',
    description: 'Docker ve container orchestration öğrenin.',
    topics: [
      'Dockerfile creation',
      'Docker Compose',
      'Container networking',
      'Volume management',
      'Resource limits',
    ],
    icon: '🐳',
    href: '/topics/python/web-gelistirme/deployment/containers'
  },
  {
    title: '2. Cloud Platforms',
    description: 'AWS ve cloud services öğrenin.',
    topics: [
      'AWS services',
      'VPC setup',
      'ECS/Fargate',
      'RDS/ElastiCache',
      'Load balancing',
    ],
    icon: '☁️',
    href: '/topics/python/web-gelistirme/deployment/cloud'
  },
  {
    title: '3. CI/CD',
    description: 'Continuous integration ve deployment öğrenin.',
    topics: [
      'GitHub Actions',
      'Test automation',
      'Docker registry',
      'ECS deployment',
      'Blue/green deploy',
    ],
    icon: '🔄',
    href: '/topics/python/web-gelistirme/deployment/cicd'
  },
  {
    title: '4. Monitoring',
    description: 'Monitoring ve logging öğrenin.',
    topics: [
      'CloudWatch metrics',
      'Log aggregation',
      'Request tracking',
      'Performance monitoring',
      'Alerts setup',
    ],
    icon: '📊',
    href: '/topics/python/web-gelistirme/deployment/monitoring'
  }
];

export default function DeploymentPage() {
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