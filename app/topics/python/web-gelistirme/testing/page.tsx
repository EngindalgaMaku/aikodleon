import { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import MarkdownContent from '@/components/MarkdownContent';
import { Card } from '@/components/ui/card';

export const metadata: Metadata = {
  title: 'Testing and Quality Assurance | Python Web Geliştirme | Kodleon',
  description: 'Python web uygulamaları için testing ve quality assurance. Unit testing, integration testing, end-to-end testing ve code quality.',
};

const content = `
# Testing and Quality Assurance

Python web uygulamalarında testing ve kalite güvence süreçlerini öğreneceğiz.

## Unit Testing ve TDD

\`\`\`python
# tests/unit/test_domain.py
import pytest
from domain.models import User, Order
from domain.commands import CreateOrder
from domain.exceptions import InvalidOrderError

class TestOrder:
    @pytest.fixture
    def user(self):
        return User(
            id="user1",
            name="John Doe",
            email="john@example.com"
        )
    
    @pytest.fixture
    def valid_order_data(self):
        return {
            "product_id": "prod1",
            "quantity": 2,
            "shipping_address": "123 Main St"
        }
    
    def test_create_order_success(self, user, valid_order_data):
        # Arrange
        command = CreateOrder(
            user_id=user.id,
            **valid_order_data
        )
        
        # Act
        order = Order.create(command)
        
        # Assert
        assert order.user_id == user.id
        assert order.product_id == valid_order_data["product_id"]
        assert order.quantity == valid_order_data["quantity"]
        assert order.status == "pending"
    
    def test_create_order_invalid_quantity(self, user, valid_order_data):
        # Arrange
        valid_order_data["quantity"] = 0
        command = CreateOrder(
            user_id=user.id,
            **valid_order_data
        )
        
        # Act & Assert
        with pytest.raises(InvalidOrderError) as exc:
            Order.create(command)
        assert "Quantity must be positive" in str(exc.value)
    
    @pytest.mark.parametrize("field", ["product_id", "quantity", "shipping_address"])
    def test_create_order_missing_required_field(self, user, valid_order_data, field):
        # Arrange
        del valid_order_data[field]
        command = CreateOrder(
            user_id=user.id,
            **valid_order_data
        )
        
        # Act & Assert
        with pytest.raises(InvalidOrderError) as exc:
            Order.create(command)
        assert f"Missing required field: {field}" in str(exc.value)

# tests/unit/test_handlers.py
from unittest.mock import Mock, patch
from domain.models import Order
from handlers.order import OrderHandler
from infrastructure.repositories import OrderRepository

class TestOrderHandler:
    @pytest.fixture
    def order_repo(self):
        return Mock(spec=OrderRepository)
    
    @pytest.fixture
    def handler(self, order_repo):
        return OrderHandler(order_repo)
    
    @patch("handlers.order.send_notification")
    def test_create_order_success(self, mock_notify, handler, order_repo):
        # Arrange
        command = CreateOrder(
            user_id="user1",
            product_id="prod1",
            quantity=2,
            shipping_address="123 Main St"
        )
        order_repo.save.return_value = None
        
        # Act
        result = handler.handle(command)
        
        # Assert
        assert isinstance(result, Order)
        order_repo.save.assert_called_once_with(result)
        mock_notify.assert_called_once_with(
            "order_created",
            {"order_id": result.id}
        )
\`\`\`

## Integration Testing

\`\`\`python
# tests/integration/test_api.py
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import pytest
from main import app
from infrastructure.database import get_db, Base, engine
from infrastructure.repositories import OrderRepository

@pytest.fixture(scope="session")
def db():
    Base.metadata.create_all(bind=engine)
    try:
        db = Session(engine)
        yield db
    finally:
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)

@pytest.fixture
def order_repository(db):
    return OrderRepository(db)

def test_create_order_api(client):
    # Arrange
    order_data = {
        "user_id": "user1",
        "product_id": "prod1",
        "quantity": 2,
        "shipping_address": "123 Main St"
    }
    
    # Act
    response = client.post("/orders", json=order_data)
    
    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["user_id"] == order_data["user_id"]
    assert data["product_id"] == order_data["product_id"]
    assert data["status"] == "pending"

def test_get_order_api(client, order_repository):
    # Arrange
    order = Order.create(CreateOrder(
        user_id="user1",
        product_id="prod1",
        quantity=2,
        shipping_address="123 Main St"
    ))
    order_repository.save(order)
    
    # Act
    response = client.get(f"/orders/{order.id}")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == order.id
    assert data["status"] == order.status

def test_list_user_orders_api(client, order_repository):
    # Arrange
    user_id = "user1"
    orders = [
        Order.create(CreateOrder(
            user_id=user_id,
            product_id=f"prod{i}",
            quantity=i,
            shipping_address="123 Main St"
        ))
        for i in range(1, 4)
    ]
    for order in orders:
        order_repository.save(order)
    
    # Act
    response = client.get(f"/users/{user_id}/orders")
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == len(orders)
    assert all(order["user_id"] == user_id for order in data)
\`\`\`

## End-to-End Testing

\`\`\`python
# tests/e2e/test_flows.py
from playwright.sync_api import Page, expect
import pytest

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {
        **browser_context_args,
        "viewport": {
            "width": 1280,
            "height": 720
        }
    }

def test_create_order_flow(page: Page):
    # Login
    page.goto("/login")
    page.fill("[data-testid=email]", "john@example.com")
    page.fill("[data-testid=password]", "password123")
    page.click("[data-testid=login-button]")
    
    # Navigate to products
    page.click("[data-testid=products-link]")
    expect(page).to_have_url("/products")
    
    # Select product
    page.click("[data-testid=product-1]")
    expect(page).to_have_url("/products/1")
    
    # Add to cart
    page.fill("[data-testid=quantity]", "2")
    page.click("[data-testid=add-to-cart]")
    
    # Checkout
    page.click("[data-testid=cart-icon]")
    page.click("[data-testid=checkout-button]")
    
    # Fill shipping info
    page.fill("[data-testid=address]", "123 Main St")
    page.fill("[data-testid=city]", "New York")
    page.fill("[data-testid=zip]", "10001")
    
    # Place order
    page.click("[data-testid=place-order]")
    
    # Verify success
    success_message = page.locator("[data-testid=success-message]")
    expect(success_message).to_be_visible()
    expect(success_message).to_contain_text("Order placed successfully")
    
    # Verify order in list
    page.click("[data-testid=orders-link]")
    expect(page).to_have_url("/orders")
    
    order_item = page.locator("[data-testid=order-item]").first
    expect(order_item).to_be_visible()
    expect(order_item).to_contain_text("Product 1")
    expect(order_item).to_contain_text("Quantity: 2")

def test_search_and_filter_flow(page: Page):
    # Login
    page.goto("/login")
    page.fill("[data-testid=email]", "john@example.com")
    page.fill("[data-testid=password]", "password123")
    page.click("[data-testid=login-button]")
    
    # Search products
    page.fill("[data-testid=search-input]", "laptop")
    page.press("[data-testid=search-input]", "Enter")
    
    # Verify search results
    results = page.locator("[data-testid=product-card]")
    expect(results).to_have_count(3)
    
    # Apply price filter
    page.click("[data-testid=price-filter]")
    page.fill("[data-testid=min-price]", "500")
    page.fill("[data-testid=max-price]", "1000")
    page.click("[data-testid=apply-filters]")
    
    # Verify filtered results
    filtered_results = page.locator("[data-testid=product-card]")
    expect(filtered_results).to_have_count(2)
    
    # Sort by price
    page.select_option("[data-testid=sort-select]", "price-asc")
    
    # Verify sorted results
    first_price = page.locator("[data-testid=product-price]").first
    last_price = page.locator("[data-testid=product-price]").last
    
    expect(first_price).to_contain_text("$599")
    expect(last_price).to_contain_text("$899")
\`\`\`

## Code Quality ve Static Analysis

\`\`\`python
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
line_length = 88

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true

[[tool.mypy.overrides]]
module = [
    "tests.*",
    "migrations.*"
]
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=app --cov-report=term-missing"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests"
]

# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9
  
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        name: isort (python)
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
  
  - repo: https://github.com/PyCQA/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        additional_dependencies: [
          flake8-bugbear,
          flake8-comprehensions,
          flake8-docstrings,
          flake8-quotes
        ]
\`\`\`

## Alıştırmalar

1. **Unit Testing**
   - Domain model testing yapın
   - Command handler testing yapın
   - Mock kullanımını öğrenin
   - Test fixtures oluşturun

2. **Integration Testing**
   - API endpoint testing yapın
   - Database integration testing yapın
   - External service mocking yapın
   - Test database setup yapın

3. **E2E Testing**
   - User flow testing yapın
   - Visual testing ekleyin
   - Performance testing yapın
   - Cross-browser testing yapın

## Sonraki Adımlar

1. [Cloud Deployment](/topics/python/web-gelistirme/deployment)
2. [Microservices Architecture](/topics/python/web-gelistirme/microservices)
3. [Performance Optimization](/topics/python/web-gelistirme/performance)

## Faydalı Kaynaklar

- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Playwright Python](https://playwright.dev/python/)
- [Python Code Quality](https://realpython.com/python-code-quality/)
`;

const learningPath = [
  {
    title: '1. Unit Testing',
    description: 'TDD ve unit testing prensiplerini öğrenin.',
    topics: [
      'Test-driven development',
      'Mocking ve fixtures',
      'Domain model testing',
      'Command handlers',
      'Test coverage',
    ],
    icon: '🧪',
    href: '/topics/python/web-gelistirme/testing/unit'
  },
  {
    title: '2. Integration Testing',
    description: 'Integration testing ve API testing öğrenin.',
    topics: [
      'Database testing',
      'API testing',
      'Service mocking',
      'Test containers',
      'Test data setup',
    ],
    icon: '🔄',
    href: '/topics/python/web-gelistirme/testing/integration'
  },
  {
    title: '3. E2E Testing',
    description: 'End-to-end testing ve browser testing öğrenin.',
    topics: [
      'Browser automation',
      'User flow testing',
      'Visual testing',
      'Performance testing',
      'Cross-browser testing',
    ],
    icon: '🌐',
    href: '/topics/python/web-gelistirme/testing/e2e'
  },
  {
    title: '4. Code Quality',
    description: 'Code quality tools ve static analysis öğrenin.',
    topics: [
      'Static type checking',
      'Code formatting',
      'Linting tools',
      'Pre-commit hooks',
      'Code coverage',
    ],
    icon: '✨',
    href: '/topics/python/web-gelistirme/testing/quality'
  }
];

export default function TestingPage() {
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