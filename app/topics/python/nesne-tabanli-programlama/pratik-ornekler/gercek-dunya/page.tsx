import { Metadata } from 'next';
import MarkdownContent from "@/components/MarkdownContent";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Globe, Database, Layout, FileSearch } from "lucide-react";

export const metadata: Metadata = {
  title: 'Python OOP Gerçek Dünya Uygulamaları | AIKOD',
  description: 'Python nesne tabanlı programlama ile gerçek dünya uygulama örnekleri ve detaylı açıklamalar.',
};

const content = `
# Gerçek Dünya Uygulamaları

Bu bölümde, nesne tabanlı programlamanın gerçek dünya uygulamalarında nasıl kullanıldığını inceleyeceğiz. Her örnek, profesyonel yazılım geliştirmede karşılaşılan senaryoları içermektedir.

## 1. REST API Servisi

Modern web uygulamalarının temel bileşeni olan REST API servisinin nesne tabanlı implementasyonu.

\`\`\`python
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from flask import Flask, request, jsonify

@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat()
        }

class Repository(ABC):
    @abstractmethod
    def get(self, id: int) -> Optional[User]:
        pass
    
    @abstractmethod
    def get_all(self) -> List[User]:
        pass
    
    @abstractmethod
    def create(self, user: User) -> User:
        pass
    
    @abstractmethod
    def update(self, user: User) -> Optional[User]:
        pass
    
    @abstractmethod
    def delete(self, id: int) -> bool:
        pass

class InMemoryRepository(Repository):
    def __init__(self):
        self._users: Dict[int, User] = {}
        self._next_id = 1
    
    def get(self, id: int) -> Optional[User]:
        return self._users.get(id)
    
    def get_all(self) -> List[User]:
        return list(self._users.values())
    
    def create(self, user: User) -> User:
        user.id = self._next_id
        self._users[user.id] = user
        self._next_id += 1
        return user
    
    def update(self, user: User) -> Optional[User]:
        if user.id in self._users:
            self._users[user.id] = user
            return user
        return None
    
    def delete(self, id: int) -> bool:
        if id in self._users:
            del self._users[id]
            return True
        return False

class UserService:
    def __init__(self, repository: Repository):
        self.repository = repository
    
    def get_user(self, id: int) -> Optional[User]:
        return self.repository.get(id)
    
    def get_all_users(self) -> List[User]:
        return self.repository.get_all()
    
    def create_user(self, username: str, email: str) -> User:
        user = User(
            id=0,  # Will be set by repository
            username=username,
            email=email,
            created_at=datetime.now()
        )
        return self.repository.create(user)
    
    def update_user(self, id: int, username: str, email: str) -> Optional[User]:
        user = self.repository.get(id)
        if user:
            user.username = username
            user.email = email
            return self.repository.update(user)
        return None
    
    def delete_user(self, id: int) -> bool:
        return self.repository.delete(id)

class UserController:
    def __init__(self, service: UserService):
        self.service = service
    
    def get_user(self, id: int):
        user = self.service.get_user(id)
        if user:
            return jsonify(user.to_dict()), 200
        return jsonify({"error": "User not found"}), 404
    
    def get_all_users(self):
        users = self.service.get_all_users()
        return jsonify([user.to_dict() for user in users]), 200
    
    def create_user(self):
        data = request.get_json()
        if not data or "username" not in data or "email" not in data:
            return jsonify({"error": "Invalid request"}), 400
        
        user = self.service.create_user(data["username"], data["email"])
        return jsonify(user.to_dict()), 201
    
    def update_user(self, id: int):
        data = request.get_json()
        if not data or "username" not in data or "email" not in data:
            return jsonify({"error": "Invalid request"}), 400
        
        user = self.service.update_user(id, data["username"], data["email"])
        if user:
            return jsonify(user.to_dict()), 200
        return jsonify({"error": "User not found"}), 404
    
    def delete_user(self, id: int):
        if self.service.delete_user(id):
            return "", 204
        return jsonify({"error": "User not found"}), 404

# Flask uygulama kurulumu
app = Flask(__name__)
repository = InMemoryRepository()
service = UserService(repository)
controller = UserController(service)

# Route tanımlamaları
@app.route("/users/<int:id>", methods=["GET"])
def get_user(id):
    return controller.get_user(id)

@app.route("/users", methods=["GET"])
def get_all_users():
    return controller.get_all_users()

@app.route("/users", methods=["POST"])
def create_user():
    return controller.create_user()

@app.route("/users/<int:id>", methods=["PUT"])
def update_user(id):
    return controller.update_user(id)

@app.route("/users/<int:id>", methods=["DELETE"])
def delete_user(id):
    return controller.delete_user(id)

if __name__ == "__main__":
    app.run(debug=True)
\`\`\`

## 2. ORM Sistemi

Veritabanı işlemlerini nesne tabanlı yaklaşımla yöneten basit bir ORM implementasyonu.

\`\`\`python
from typing import Any, Dict, List, Type, TypeVar
import sqlite3
from dataclasses import dataclass, fields

T = TypeVar('T')

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    def execute(self, query: str, params: tuple = ()) -> Any:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_many(self, query: str, params_list: List[tuple]) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)

class Model:
    _table_name: str
    
    @classmethod
    def create_table(cls) -> str:
        fields_def = []
        for field in fields(cls):
            field_type = "TEXT"
            if field.type == int:
                field_type = "INTEGER"
            elif field.type == float:
                field_type = "REAL"
            
            if field.name == "id":
                fields_def.append(f"{field.name} {field_type} PRIMARY KEY")
            else:
                fields_def.append(f"{field.name} {field_type}")
        
        return f"""
        CREATE TABLE IF NOT EXISTS {cls._table_name} (
            {', '.join(fields_def)}
        )
        """

@dataclass
class Product(Model):
    _table_name = "products"
    
    id: int
    name: str
    price: float
    description: str

class Repository(Generic[T]):
    def __init__(self, db: Database, model_class: Type[T]):
        self.db = db
        self.model_class = model_class
        
        # Create table if not exists
        create_table_query = self.model_class.create_table()
        self.db.execute(create_table_query)
    
    def get(self, id: int) -> Optional[T]:
        query = f"SELECT * FROM {self.model_class._table_name} WHERE id = ?"
        result = self.db.execute(query, (id,))
        if result:
            return self.model_class(*result[0])
        return None
    
    def get_all(self) -> List[T]:
        query = f"SELECT * FROM {self.model_class._table_name}"
        results = self.db.execute(query)
        return [self.model_class(*row) for row in results]
    
    def create(self, model: T) -> T:
        fields_list = [f.name for f in fields(self.model_class) if f.name != "id"]
        placeholders = ", ".join(["?" for _ in fields_list])
        
        query = f"""
        INSERT INTO {self.model_class._table_name}
        ({', '.join(fields_list)})
        VALUES ({placeholders})
        """
        
        values = tuple(getattr(model, f) for f in fields_list)
        self.db.execute(query, values)
        return model
    
    def update(self, model: T) -> Optional[T]:
        fields_list = [f.name for f in fields(self.model_class) if f.name != "id"]
        set_clause = ", ".join([f"{f} = ?" for f in fields_list])
        
        query = f"""
        UPDATE {self.model_class._table_name}
        SET {set_clause}
        WHERE id = ?
        """
        
        values = tuple(getattr(model, f) for f in fields_list)
        values += (model.id,)
        
        self.db.execute(query, values)
        return model
    
    def delete(self, id: int) -> bool:
        query = f"DELETE FROM {self.model_class._table_name} WHERE id = ?"
        self.db.execute(query, (id,))
        return True

# Kullanım örneği
def main():
    db = Database("store.db")
    product_repo = Repository[Product](db, Product)
    
    # Ürün oluştur
    product = Product(
        id=0,
        name="Laptop",
        price=5999.99,
        description="Yüksek performanslı laptop"
    )
    product_repo.create(product)
    
    # Tüm ürünleri listele
    products = product_repo.get_all()
    for p in products:
        print(f"{p.name}: {p.price}₺")

if __name__ == "__main__":
    main()
\`\`\`

## 3. GUI Uygulaması

Tkinter kullanarak nesne tabanlı bir GUI uygulaması örneği.

\`\`\`python
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Task:
    id: int
    title: str
    description: str
    due_date: datetime
    completed: bool = False

class TaskRepository:
    def __init__(self):
        self._tasks: List[Task] = []
        self._next_id = 1
    
    def add(self, task: Task) -> Task:
        task.id = self._next_id
        self._next_id += 1
        self._tasks.append(task)
        return task
    
    def get_all(self) -> List[Task]:
        return self._tasks.copy()
    
    def get(self, id: int) -> Optional[Task]:
        return next((t for t in self._tasks if t.id == id), None)
    
    def update(self, task: Task) -> bool:
        for i, t in enumerate(self._tasks):
            if t.id == task.id:
                self._tasks[i] = task
                return True
        return False
    
    def delete(self, id: int) -> bool:
        task = self.get(id)
        if task:
            self._tasks.remove(task)
            return True
        return False

class TaskManager:
    def __init__(self, repository: TaskRepository):
        self.repository = repository
    
    def create_task(self, title: str, description: str, due_date: datetime) -> Task:
        task = Task(0, title, description, due_date)
        return self.repository.add(task)
    
    def complete_task(self, id: int) -> bool:
        task = self.repository.get(id)
        if task:
            task.completed = True
            return self.repository.update(task)
        return False
    
    def get_all_tasks(self) -> List[Task]:
        return self.repository.get_all()
    
    def delete_task(self, id: int) -> bool:
        return self.repository.delete(id)

class TaskView(tk.Frame):
    def __init__(self, parent, task_manager: TaskManager):
        super().__init__(parent)
        self.task_manager = task_manager
        self.create_widgets()
        self.refresh_task_list()
    
    def create_widgets(self):
        # Task ekleme formu
        form_frame = ttk.LabelFrame(self, text="Yeni Görev", padding=10)
        form_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(form_frame, text="Başlık:").grid(row=0, column=0, sticky=tk.W)
        self.title_entry = ttk.Entry(form_frame)
        self.title_entry.grid(row=0, column=1, sticky=tk.EW)
        
        ttk.Label(form_frame, text="Açıklama:").grid(row=1, column=0, sticky=tk.W)
        self.desc_entry = ttk.Entry(form_frame)
        self.desc_entry.grid(row=1, column=1, sticky=tk.EW)
        
        ttk.Label(form_frame, text="Tarih:").grid(row=2, column=0, sticky=tk.W)
        self.date_entry = ttk.Entry(form_frame)
        self.date_entry.grid(row=2, column=1, sticky=tk.EW)
        self.date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        ttk.Button(form_frame, text="Ekle", command=self.add_task).grid(row=3, column=1, sticky=tk.E)
        
        # Görev listesi
        list_frame = ttk.LabelFrame(self, text="Görevler", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ("id", "title", "description", "due_date", "status")
        self.task_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        
        self.task_tree.heading("id", text="ID")
        self.task_tree.heading("title", text="Başlık")
        self.task_tree.heading("description", text="Açıklama")
        self.task_tree.heading("due_date", text="Tarih")
        self.task_tree.heading("status", text="Durum")
        
        self.task_tree.column("id", width=50)
        self.task_tree.column("title", width=150)
        self.task_tree.column("description", width=200)
        self.task_tree.column("due_date", width=100)
        self.task_tree.column("status", width=100)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.task_tree.yview)
        self.task_tree.configure(yscrollcommand=scrollbar.set)
        
        self.task_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Butonlar
        button_frame = ttk.Frame(self, padding=5)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Tamamla", command=self.complete_task).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Sil", command=self.delete_task).pack(side=tk.LEFT)
    
    def add_task(self):
        try:
            title = self.title_entry.get().strip()
            description = self.desc_entry.get().strip()
            due_date = datetime.strptime(self.date_entry.get(), "%Y-%m-%d")
            
            if not title:
                raise ValueError("Başlık boş olamaz")
            
            self.task_manager.create_task(title, description, due_date)
            self.refresh_task_list()
            
            # Form temizle
            self.title_entry.delete(0, tk.END)
            self.desc_entry.delete(0, tk.END)
            
        except ValueError as e:
            messagebox.showerror("Hata", str(e))
    
    def complete_task(self):
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Uyarı", "Lütfen bir görev seçin")
            return
        
        task_id = int(self.task_tree.item(selection[0])["values"][0])
        if self.task_manager.complete_task(task_id):
            self.refresh_task_list()
    
    def delete_task(self):
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Uyarı", "Lütfen bir görev seçin")
            return
        
        task_id = int(self.task_tree.item(selection[0])["values"][0])
        if messagebox.askyesno("Onay", "Görevi silmek istediğinize emin misiniz?"):
            if self.task_manager.delete_task(task_id):
                self.refresh_task_list()
    
    def refresh_task_list(self):
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
        
        for task in self.task_manager.get_all_tasks():
            status = "Tamamlandı" if task.completed else "Devam Ediyor"
            self.task_tree.insert("", tk.END, values=(
                task.id,
                task.title,
                task.description,
                task.due_date.strftime("%Y-%m-%d"),
                status
            ))

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Görev Yöneticisi")
        self.geometry("800x600")
        
        repository = TaskRepository()
        task_manager = TaskManager(repository)
        
        task_view = TaskView(self, task_manager)
        task_view.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
\`\`\`

## 4. Web Scraping Uygulaması

Web sitelerinden veri çekme işlemlerini nesne tabanlı yaklaşımla gerçekleştiren bir uygulama.

\`\`\`python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import csv
import json

@dataclass
class Article:
    title: str
    author: str
    date: datetime
    content: str
    url: str
    category: str

class Scraper(ABC):
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    @abstractmethod
    def get_articles(self) -> List[Article]:
        pass
    
    def get_soup(self, url: str) -> BeautifulSoup:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

class NewsScraper(Scraper):
    def get_articles(self) -> List[Article]:
        soup = self.get_soup(self.base_url)
        articles = []
        
        # Örnek: Haber sitesinin yapısına göre uyarlanmalı
        for article_elem in soup.select(".article-card"):
            title = article_elem.select_one(".title").text.strip()
            author = article_elem.select_one(".author").text.strip()
            date_str = article_elem.select_one(".date").text.strip()
            content = article_elem.select_one(".content").text.strip()
            url = article_elem.select_one("a")["href"]
            category = article_elem.select_one(".category").text.strip()
            
            date = datetime.strptime(date_str, "%Y-%m-%d")
            
            articles.append(Article(
                title=title,
                author=author,
                date=date,
                content=content,
                url=url,
                category=category
            ))
        
        return articles

class ArticleExporter(ABC):
    @abstractmethod
    def export(self, articles: List[Article], filename: str) -> None:
        pass

class CSVExporter(ArticleExporter):
    def export(self, articles: List[Article], filename: str) -> None:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "Author", "Date", "Content", "URL", "Category"])
            
            for article in articles:
                writer.writerow([
                    article.title,
                    article.author,
                    article.date.strftime("%Y-%m-%d"),
                    article.content,
                    article.url,
                    article.category
                ])

class JSONExporter(ArticleExporter):
    def export(self, articles: List[Article], filename: str) -> None:
        data = []
        for article in articles:
            data.append({
                "title": article.title,
                "author": article.author,
                "date": article.date.strftime("%Y-%m-%d"),
                "content": article.content,
                "url": article.url,
                "category": article.category
            })
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

class ScrapingService:
    def __init__(self, scraper: Scraper, exporter: ArticleExporter):
        self.scraper = scraper
        self.exporter = exporter
    
    def scrape_and_export(self, filename: str) -> None:
        try:
            articles = self.scraper.get_articles()
            self.exporter.export(articles, filename)
            print(f"{len(articles)} makale başarıyla dışa aktarıldı: {filename}")
        except Exception as e:
            print(f"Hata oluştu: {e}")

# Kullanım örneği
def main():
    # Haber sitesi için scraper
    news_scraper = NewsScraper("https://example.com/news")
    
    # CSV formatında dışa aktarma
    csv_service = ScrapingService(news_scraper, CSVExporter())
    csv_service.scrape_and_export("articles.csv")
    
    # JSON formatında dışa aktarma
    json_service = ScrapingService(news_scraper, JSONExporter())
    json_service.scrape_and_export("articles.json")

if __name__ == "__main__":
    main()
\`\`\`

## İyi Pratikler

1. **Modüler Tasarım**
   - Her modül tek bir sorumluluğa sahip olmalı
   - Modüller arası bağımlılıklar minimize edilmeli
   - Interface ve abstract sınıflar kullanılmalı

2. **Hata Yönetimi**
   - Özel exception sınıfları tanımlanmalı
   - Try-except blokları uygun şekilde kullanılmalı
   - Hatalar loglama sistemi ile kaydedilmeli

3. **Test Edilebilirlik**
   - Unit testler yazılmalı
   - Mock ve stub kullanımı
   - Test coverage takibi yapılmalı

4. **Güvenlik**
   - Input validasyonu yapılmalı
   - Hassas veriler şifrelenmeli
   - API güvenliği sağlanmalı

5. **Performans**
   - Kaynak kullanımı optimize edilmeli
   - Caching mekanizmaları kullanılmalı
   - Asenkron programlama teknikleri uygulanmalı
`;

const sections = [
  {
    title: "REST API",
    description: "Web servisleri ve API tasarımı",
    icon: <Globe className="h-6 w-6" />,
    topics: [
      "Controller yapısı",
      "Service katmanı",
      "Repository pattern",
      "HTTP endpoints"
    ]
  },
  {
    title: "ORM Sistemi",
    description: "Veritabanı etkileşimi",
    icon: <Database className="h-6 w-6" />,
    topics: [
      "Model tanımları",
      "CRUD işlemleri",
      "Query builder",
      "Migration sistemi"
    ]
  },
  {
    title: "GUI Uygulaması",
    description: "Grafiksel kullanıcı arayüzü",
    icon: <Layout className="h-6 w-6" />,
    topics: [
      "Tkinter widgets",
      "Event handling",
      "Layout yönetimi",
      "MVC pattern"
    ]
  },
  {
    title: "Web Scraping",
    description: "Veri çekme ve işleme",
    icon: <FileSearch className="h-6 w-6" />,
    topics: [
      "HTML parsing",
      "Data extraction",
      "Export formatları",
      "Async scraping"
    ]
  }
];

export default function GercekDunyaPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <MarkdownContent content={content} />
        
        {/* Feature Cards */}
        <div className="my-12">
          <h2 className="text-3xl font-bold mb-8">Uygulama Türleri</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {sections.map((section, index) => (
              <Card key={index} className="bg-yellow-50 hover:bg-yellow-100 dark:bg-yellow-950/50 dark:hover:bg-yellow-950/70 transition-all duration-300">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg text-yellow-600 dark:text-yellow-400">
                      {section.icon}
                    </div>
                    <CardTitle>{section.title}</CardTitle>
                  </div>
                  <CardDescription className="dark:text-gray-300">{section.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground dark:text-gray-400">
                    {section.topics.map((topic, i) => (
                      <li key={i}>{topic}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Back to Examples Button */}
        <div className="mt-12 flex justify-end">
          <Button asChild variant="outline" className="group">
            <Link href="/topics/python/nesne-tabanli-programlama/pratik-ornekler">
              Örneklere Dön
              <ArrowRight className="h-4 w-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
} 