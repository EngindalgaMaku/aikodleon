import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Cloud Native Development | Python Web Geliştirme | Kodleon',
  description: 'Python ile cloud native uygulama geliştirme. Containerization, Kubernetes, cloud servisleri ve serverless mimari konuları.',
};

const content = `
# Cloud Native Development

Modern cloud native uygulama geliştirme pratiklerini ve Python ile implementasyonunu öğreneceğiz.

## Containerization ve Docker

\`\`\`dockerfile
# Çok aşamalı build örneği
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

# Builder aşamasından Python paketlerini kopyalama
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Uygulama kodlarını kopyalama
COPY . .

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uygulama kullanıcısı
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - AWS_ACCESS_KEY_ID=\${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=\${AWS_SECRET_ACCESS_KEY}
      - AWS_DEFAULT_REGION=eu-west-1
    deploy:
      resources:
        limits:
          cpus: '0.50'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
\`\`\`

## Kubernetes Deployment

\`\`\`yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web-app
        image: web-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20

# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-app
  namespace: production
spec:
  selector:
    app: web-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-app
  namespace: production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - app.example.com
    secretName: app-tls
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-app
            port:
              number: 80
\`\`\`

## Cloud Services Entegrasyonu

\`\`\`python
# cloud/storage.py
import boto3
from botocore.exceptions import ClientError
from typing import BinaryIO

class S3Storage:
    def __init__(self, bucket_name: str):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name
    
    async def upload_file(
        self,
        file: BinaryIO,
        key: str,
        content_type: str = None
    ) -> str:
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            await self.s3.upload_fileobj(
                file,
                self.bucket,
                key,
                ExtraArgs=extra_args
            )
            
            return f"https://{self.bucket}.s3.amazonaws.com/{key}"
        except ClientError as e:
            raise Exception(f"S3 upload error: {str(e)}")
    
    async def generate_presigned_url(
        self,
        key: str,
        expiration: int = 3600
    ) -> str:
        try:
            url = await self.s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': key
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            raise Exception(f"S3 presigned URL error: {str(e)}")

# cloud/queue.py
import json
import boto3
from typing import Any, Dict, List

class SQSQueue:
    def __init__(self, queue_url: str):
        self.sqs = boto3.client('sqs')
        self.queue_url = queue_url
    
    async def send_message(
        self,
        message: Dict[str, Any],
        delay_seconds: int = 0
    ):
        try:
            await self.sqs.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(message),
                DelaySeconds=delay_seconds
            )
        except ClientError as e:
            raise Exception(f"SQS send error: {str(e)}")
    
    async def receive_messages(
        self,
        max_messages: int = 10,
        wait_time: int = 20
    ) -> List[Dict]:
        try:
            response = await self.sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time
            )
            
            messages = response.get('Messages', [])
            return [
                {
                    'id': msg['MessageId'],
                    'body': json.loads(msg['Body']),
                    'receipt_handle': msg['ReceiptHandle']
                }
                for msg in messages
            ]
        except ClientError as e:
            raise Exception(f"SQS receive error: {str(e)}")
\`\`\`

## Serverless Functions

\`\`\`python
# serverless/functions.py
import json
from typing import Dict, Any
import boto3
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.utilities.typing import LambdaContext

logger = Logger()
tracer = Tracer()
app = APIGatewayRestResolver()

@app.get("/products/<product_id>")
@tracer.capture_method
def get_product(product_id: str):
    try:
        # DynamoDB'den ürün bilgisi alma
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('products')
        
        response = table.get_item(
            Key={'id': product_id}
        )
        
        if 'Item' not in response:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Product not found'})
            }
        
        return {
            'statusCode': 200,
            'body': json.dumps(response['Item'])
        }
    except Exception as e:
        logger.exception("Product fetch error")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

@app.post("/orders")
@tracer.capture_method
def create_order(body: Dict[str, Any]):
    try:
        # SQS'e sipariş mesajı gönderme
        sqs = boto3.client('sqs')
        queue_url = 'https://sqs.region.amazonaws.com/account/orders'
        
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(body)
        )
        
        return {
            'statusCode': 202,
            'body': json.dumps({
                'message': 'Order received',
                'order_id': response['MessageId']
            })
        }
    except Exception as e:
        logger.exception("Order creation error")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

@logger.inject_lambda_context
@tracer.capture_lambda_handler
def handler(event: Dict[str, Any], context: LambdaContext) -> Dict[str, Any]:
    return app.resolve(event, context)
\`\`\`

## Alıştırmalar

1. **Container Optimizasyonu**
   - Multi-stage build oluşturun
   - Image boyutunu optimize edin
   - Security scanning yapın
   - Resource limits ayarlayın

2. **Kubernetes Deployment**
   - Deployment stratejileri uygulayın
   - Resource quotas ayarlayın
   - Auto-scaling kurun
   - Monitoring ekleyin

3. **Cloud Integration**
   - S3 file upload implementasyonu yapın
   - SQS queue consumer yazın
   - Lambda function geliştirin
   - CloudWatch metrics ekleyin

## Sonraki Adımlar

1. [Progressive Web Apps](/topics/python/web-gelistirme/pwa)
2. [DevOps Practices](/topics/python/web-gelistirme/devops)
3. [Security Best Practices](/topics/python/web-gelistirme/security)

## Faydalı Kaynaklar

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [AWS Python SDK](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [Serverless Framework](https://www.serverless.com/framework/docs/)
`;

const learningPath = [
  {
    title: '1. Containerization',
    description: 'Container teknolojilerini ve best practice\'leri öğrenin.',
    topics: [
      'Docker temelleri',
      'Multi-stage builds',
      'Resource management',
      'Security scanning',
      'Container orchestration',
    ],
    icon: '🐳',
    href: '/topics/python/web-gelistirme/cloud-native/containers'
  },
  {
    title: '2. Kubernetes',
    description: 'Kubernetes ile container orchestration\'ı öğrenin.',
    topics: [
      'Deployment stratejileri',
      'Service discovery',
      'Auto-scaling',
      'Resource quotas',
      'Monitoring',
    ],
    icon: '☸️',
    href: '/topics/python/web-gelistirme/cloud-native/kubernetes'
  },
  {
    title: '3. Cloud Services',
    description: 'Cloud servislerini ve entegrasyonunu öğrenin.',
    topics: [
      'Storage services',
      'Message queues',
      'Managed databases',
      'CDN',
      'Monitoring',
    ],
    icon: '☁️',
    href: '/topics/python/web-gelistirme/cloud-native/cloud'
  },
  {
    title: '4. Serverless',
    description: 'Serverless mimari ve FaaS\'ı öğrenin.',
    topics: [
      'Lambda functions',
      'API Gateway',
      'DynamoDB',
      'Event triggers',
      'Cold starts',
    ],
    icon: '⚡',
    href: '/topics/python/web-gelistirme/cloud-native/serverless'
  }
];

export default function CloudNativePage() {
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