import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'DevOps Practices | Python Web Geliştirme | Kodleon',
  description: 'Python web uygulamaları için DevOps pratikleri. CI/CD, infrastructure as code, monitoring ve automation.',
};

const content = `
# DevOps Practices

Python web uygulamaları için modern DevOps pratiklerini ve araçlarını öğreneceğiz.

## CI/CD Pipeline

\`\`\`yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: --health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install
        
    - name: Run tests
      run: |
        poetry run pytest --cov=app --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
      
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
        
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        push: true
        tags: user/app:latest
        cache-from: type=registry,ref=user/app:buildcache
        cache-to: type=registry,ref=user/app:buildcache,mode=max
        
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.SSH_HOST }}
        username: ${{ secrets.SSH_USERNAME }}
        key: ${{ secrets.SSH_KEY }}
        script: |
          cd /app
          docker-compose pull
          docker-compose up -d
\`\`\`

## Infrastructure as Code

\`\`\`python
# terraform/main.py
from cdktf import App, TerraformStack
from constructs import Construct
from cdktf_cdktf_provider_aws import AwsProvider, ec2

class MyStack(TerraformStack):
    def __init__(self, scope: Construct, id: str):
        super().__init__(scope, id)

        AwsProvider(self, "AWS", region="eu-west-1")

        vpc = ec2.Vpc(self, "MyVPC",
            cidr_block="10.0.0.0/16",
            enable_dns_hostnames=True,
            enable_dns_support=True,
            tags={
                "Name": "MyVPC"
            }
        )

        public_subnet = ec2.Subnet(self, "PublicSubnet",
            vpc_id=vpc.id,
            cidr_block="10.0.1.0/24",
            availability_zone="eu-west-1a",
            map_public_ip_on_launch=True,
            tags={
                "Name": "Public Subnet"
            }
        )

        security_group = ec2.SecurityGroup(self, "WebSecurityGroup",
            name="web-security-group",
            vpc_id=vpc.id,
            description="Security group for web servers",
            ingress=[{
                "protocol": "tcp",
                "from_port": 80,
                "to_port": 80,
                "cidr_blocks": ["0.0.0.0/0"]
            }],
            egress=[{
                "protocol": "-1",
                "from_port": 0,
                "to_port": 0,
                "cidr_blocks": ["0.0.0.0/0"]
            }]
        )

app = App()
MyStack(app, "my-stack")
app.synth()
\`\`\`

## Monitoring ve Logging

\`\`\`python
# monitoring/prometheus.py
from prometheus_client import Counter, Histogram, start_http_server
from functools import wraps
import time

# Metrics
REQUEST_COUNT = Counter(
    'app_request_count',
    'Application Request Count',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Application Request Latency',
    ['method', 'endpoint']
)

def track_requests(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        method = kwargs.get('method', 'GET')
        endpoint = kwargs.get('endpoint', 'unknown')
        
        start_time = time.time()
        
        try:
            response = await func(*args, **kwargs)
            status = response.status_code
        except Exception as e:
            status = 500
            raise e
        finally:
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(time.time() - start_time)
        
        return response
    return wrapper

# logging/logger.py
import logging
import json
from pythonjsonlogger import jsonlogger

def setup_logger():
    logger = logging.getLogger()
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()
\`\`\`

## Automation Scripts

\`\`\`python
# scripts/deploy.py
import click
import docker
import paramiko
import os

@click.command()
@click.option('--env', default='staging', help='Deployment environment')
@click.option('--version', default='latest', help='Application version')
def deploy(env: str, version: str):
    """Deploy application to specified environment"""
    try:
        # Docker build and push
        client = docker.from_env()
        image = client.images.build(
            path=".",
            tag=f"app:{version}",
            rm=True
        )
        client.images.push(f"app:{version}")
        
        # SSH connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=os.getenv(f"{env.upper()}_HOST"),
            username=os.getenv(f"{env.upper()}_USER"),
            key_filename=os.getenv(f"{env.upper()}_KEY_PATH")
        )
        
        # Deploy commands
        commands = [
            f"cd /app",
            f"docker-compose pull",
            f"docker-compose up -d"
        ]
        
        for cmd in commands:
            stdin, stdout, stderr = ssh.exec_command(cmd)
            print(f"Executing: {cmd}")
            print(stdout.read().decode())
            
        click.echo(f"Successfully deployed version {version} to {env}")
        
    except Exception as e:
        click.echo(f"Deployment failed: {str(e)}", err=True)
        raise
    finally:
        ssh.close()

if __name__ == '__main__':
    deploy()
\`\`\`

## Alıştırmalar

1. **CI/CD Pipeline**
   - GitHub Actions workflow oluşturun
   - Test ve build süreçlerini otomatize edin
   - Docker image build ve push işlemlerini ekleyin
   - Deployment automation ekleyin

2. **Infrastructure**
   - Terraform ile AWS altyapısı oluşturun
   - Kubernetes cluster kurun
   - Service mesh implementasyonu yapın
   - Auto-scaling konfigürasyonu yapın

3. **Monitoring**
   - Prometheus metrics ekleyin
   - Grafana dashboard oluşturun
   - Log aggregation kurun
   - Alert rules tanımlayın

## Sonraki Adımlar

1. [Security Best Practices](/topics/python/web-gelistirme/security)
2. [Performance Optimization](/topics/python/web-gelistirme/performance)
3. [Microservices Architecture](/topics/python/web-gelistirme/microservices)

## Faydalı Kaynaklar

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Terraform Documentation](https://www.terraform.io/docs)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Docker Documentation](https://docs.docker.com/)
`;

const learningPath = [
  {
    title: '1. CI/CD Pipeline',
    description: 'Continuous Integration ve Delivery süreçlerini öğrenin.',
    topics: [
      'GitHub Actions',
      'Test automation',
      'Docker builds',
      'Deployment automation',
      'Pipeline monitoring',
    ],
    icon: '🔄',
    href: '/topics/python/web-gelistirme/devops/ci-cd'
  },
  {
    title: '2. Infrastructure',
    description: 'Infrastructure as Code ve cloud yönetimini öğrenin.',
    topics: [
      'Terraform',
      'AWS Services',
      'Kubernetes',
      'Service mesh',
      'Auto-scaling',
    ],
    icon: '☁️',
    href: '/topics/python/web-gelistirme/devops/infrastructure'
  },
  {
    title: '3. Monitoring',
    description: 'Monitoring ve logging sistemlerini öğrenin.',
    topics: [
      'Prometheus',
      'Grafana',
      'Log aggregation',
      'Alerting',
      'Metrics',
    ],
    icon: '📊',
    href: '/topics/python/web-gelistirme/devops/monitoring'
  },
  {
    title: '4. Automation',
    description: 'DevOps automation araçlarını öğrenin.',
    topics: [
      'Shell scripting',
      'Python automation',
      'Configuration management',
      'Backup automation',
      'Deployment scripts',
    ],
    icon: '⚙️',
    href: '/topics/python/web-gelistirme/devops/automation'
  }
];

export default function DevOpsPage() {
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